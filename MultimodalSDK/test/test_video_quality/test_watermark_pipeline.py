#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the MultimodalSDK project.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
#
# MultimodalSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Tests for VideoQualityPipeline."""

from __future__ import annotations

from types import SimpleNamespace

import hashlib

from unittest.mock import MagicMock, patch

import numpy as np
import torch
import pytest

from mm.core.video_quality import (
    VideoQualityPipeline,
    ScorerConfig,
    WatermarkDetector,
)
from mm.core.video_quality.base import (
    MetricResult,
)


MODULE_PATH = "mm.core.video_quality.scorers.watermark_scorer"


@pytest.fixture(autouse=True)
def mock_weight_resolver(tmp_path, monkeypatch):
    with patch(f"{MODULE_PATH}.WeightResolver") as resolver_cls:
        resolver = MagicMock()
        weight_path = tmp_path / "best.pt"
        weight_path.write_bytes(b"test fixture; model loading is mocked")
        resolver.resolve_single.return_value = str(weight_path)
        monkeypatch.setenv(
            "WATERMARK_YOLO_MODEL_SHA256",
            hashlib.sha256(weight_path.read_bytes()).hexdigest(),
        )
        resolver_cls.return_value = resolver
        yield resolver


@pytest.fixture(autouse=True)
def mock_yolo():
    mock_yolo_cls = MagicMock()
    mock_model = MagicMock()

    def _call(frames, **kwargs):
        batch = frames if isinstance(frames, list) else [frames]
        results = []
        for _ in batch:
            result = MagicMock()
            result.boxes = MagicMock()
            result.boxes.__len__ = MagicMock(return_value=0)
            result.boxes.xyxy = torch.empty((0, 4), dtype=torch.float32)
            result.boxes.conf = torch.empty((0,), dtype=torch.float32)
            result.boxes.cls = torch.empty((0,), dtype=torch.float32)
            results.append(result)
        return results

    mock_model.side_effect = _call
    from types import ModuleType

    safe_load_calls = []
    ultralytics_mock = ModuleType("ultralytics")
    ultralytics_mock.__path__ = []
    ultralytics_mock.YOLO = mock_yolo_cls
    nn_mock = ModuleType("ultralytics.nn")
    nn_mock.__path__ = []
    tasks_mock = ModuleType("ultralytics.nn.tasks")

    def torch_safe_load(weight, safe_only=False):
        safe_load_calls.append(safe_only)
        return {}, weight

    tasks_mock.torch_safe_load = torch_safe_load

    class SafeLoad:
        SUPPORTED = True

    tasks_mock._SafeLoad = SafeLoad
    nn_mock.tasks = tasks_mock
    ultralytics_mock.nn = nn_mock

    def construct_yolo(path):
        _, loaded_path = getattr(tasks_mock, 'torch_safe_load')(path)
        assert loaded_path == path
        assert safe_load_calls[-1] is True
        return mock_model

    mock_yolo_cls.side_effect = construct_yolo
    with patch.dict(
        "sys.modules",
        {
            "ultralytics": ultralytics_mock,
            "ultralytics.nn": nn_mock,
            "ultralytics.nn.tasks": tasks_mock,
        },
    ):
        yield mock_model


@pytest.fixture(autouse=True)
def mock_watermark_pipeline_model():
    # Keep pipeline integration tests independent of external YOLO checkpoints.
    def _load_model(scorer):
        model = MagicMock()

        def _predict(frame_batch, **kwargs):
            results = []
            for _ in frame_batch:
                boxes = MagicMock()
                boxes.__len__.return_value = 0
                boxes.xyxy = torch.empty((0, 4), dtype=torch.float32)
                boxes.conf = torch.empty((0,), dtype=torch.float32)
                boxes.cls = torch.empty((0,), dtype=torch.float32)
                results.append(SimpleNamespace(boxes=boxes))
            return results

        model.side_effect = _predict
        scorer._model = model

    with patch.object(
        WatermarkDetector,
        "_load_model",
        autospec=True,
        side_effect=_load_model,
    ):
        yield


def _write_test_video(video_path, frames):
    import cv2 as cv

    writer = cv.VideoWriter(video_path, cv.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 64))
    try:
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer for {video_path}")
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


class TestPipelineInit:
    def test_init_with_scorer_names(self, force_cpu_device):
        pipeline = VideoQualityPipeline(
            scorers=["watermark"],
            device="cpu",
        )
        assert len(pipeline._scorers) == 1
        assert pipeline._scorers[0].name == "watermark"

    def test_init_with_scorer_instances(self, force_cpu_device):
        s1 = WatermarkDetector(config=ScorerConfig(device="cpu"))
        pipeline = VideoQualityPipeline(scorers=[s1], device="cpu")
        assert len(pipeline._scorers) == 1

    def test_init_with_weights(self, force_cpu_device):
        pipeline = VideoQualityPipeline(
            scorers=["watermark"],
            device="cpu",
            weights={"watermark": None},
        )
        assert len(pipeline._scorers) == 1

    def test_pipeline_passes_frame_batch_size(self, force_cpu_device):
        pipeline = VideoQualityPipeline(
            scorers=["watermark"],
            device="cpu",
            frame_batch_size=3,
        )
        assert pipeline._scorers[0].config.frame_batch_size == 3


class TestPipelineScore:
    def test_score_single_video(self, tmp_path, force_cpu_device):
        """Test pipeline with a scorer on a real video file."""

        video_path = str(tmp_path / "test.mp4")
        _write_test_video(video_path, [np.full((64, 64, 3), i * 5, dtype=np.uint8) for i in range(30)])

        pipeline = VideoQualityPipeline(
            scorers=["watermark"],
            device="cpu",
        )
        report = pipeline.score(video_path)

        assert report.video_path == video_path
        assert "watermark" in report.metrics
        assert report.metrics["watermark"].is_success
        assert "watermark_count" in report.all_scores

    def test_score_batch(self, tmp_path, force_cpu_device):
        paths = []
        for idx in range(2):
            video_path = str(tmp_path / f"test_{idx}.mp4")
            _write_test_video(
                video_path,
                [np.full((64, 64, 3), (i + idx * 10) * 5, dtype=np.uint8) for i in range(20)],
            )
            paths.append(video_path)

        pipeline = VideoQualityPipeline(
            scorers=["watermark"],
            device="cpu",
        )
        reports = pipeline.score_batch(paths)
        assert len(reports) == 2
        for r in reports:
            assert "watermark" in r.metrics
            assert r.metrics["watermark"].is_success
            assert "watermark_count" in r.all_scores

    def test_all_failed_raises_runtime_error(self, tmp_path, force_cpu_device):
        """When all scorers fail, pipeline should raise RuntimeError."""

        video_path = str(tmp_path / "test.mp4")
        _write_test_video(video_path, [np.full((64, 64, 3), 100, dtype=np.uint8) for _ in range(20)])

        pipeline = VideoQualityPipeline(
            scorers=["watermark"],
            device="cpu",
        )
        with patch.object(
            pipeline._scorers[0],
            "score_frames",
            return_value=MetricResult(name="watermark", scores={}, error="fail"),
        ):
            with pytest.raises(RuntimeError, match="All scorers failed"):
                pipeline.score(video_path)

    def test_pipeline_close(self, force_cpu_device):
        pipeline = VideoQualityPipeline(
            scorers=["watermark"],
            device="cpu",
        )
        pipeline.close()
