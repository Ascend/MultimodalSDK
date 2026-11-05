#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the MultimodalSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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

"""Tests for WatermarkDetector (YOLO, mocked ultralytics)."""

from __future__ import annotations

import hashlib

from unittest.mock import patch, MagicMock

import pytest
import torch

from mm.core.video_quality import WatermarkDetector, ScorerConfig
from mm.core.video_quality.base import LoadedFrames
from .conftest import make_loaded_frames


MODULE_PATH = "mm.core.video_quality.scorers.watermark_scorer"


@pytest.fixture(autouse=True)
def mock_weight_resolver(tmp_path, monkeypatch):
    with patch(f"{MODULE_PATH}.WeightResolver") as MockResolver:
        resolver = MagicMock()
        weight_path = tmp_path / "best.pt"
        weight_path.write_bytes(b"test fixture; model loading is mocked")
        resolver.resolve_single.return_value = str(weight_path)
        monkeypatch.setenv("WATERMARK_YOLO_MODEL_SHA256", hashlib.sha256(weight_path.read_bytes()).hexdigest())
        MockResolver.return_value = resolver
        yield resolver


@pytest.fixture(autouse=True, name="mock_yolo")
def _mock_yolo_fixture():
    """Mock ultralytics.YOLO."""
    mock_yolo_cls = MagicMock()
    mock_model = MagicMock()
    mock_model.names = {0: "watermark"}

    def _call(frames, **kwargs):
        # YOLO returns one result for each frame in the input batch.
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
    safe_load_calls = []

    def _construct_yolo(path):
        _, loaded_path = getattr(tasks_mock, 'torch_safe_load')(path)
        assert loaded_path == path
        assert safe_load_calls[-1] is True
        return mock_model

    mock_yolo_cls.side_effect = _construct_yolo

    from types import ModuleType

    ultralytics_mock = ModuleType("ultralytics")
    ultralytics_mock.__path__ = []
    ultralytics_mock.YOLO = mock_yolo_cls
    tasks_mock = ModuleType("ultralytics.nn.tasks")

    def torch_safe_load(weight, safe_only=False):
        safe_load_calls.append(safe_only)
        return {}, weight

    tasks_mock.torch_safe_load = torch_safe_load

    class SafeLoad:
        SUPPORTED = True

    tasks_mock._SafeLoad = SafeLoad
    nn_mock = ModuleType("ultralytics.nn")
    nn_mock.__path__ = []
    nn_mock.tasks = tasks_mock
    ultralytics_mock.nn = nn_mock
    with patch.dict(
        "sys.modules",
        {"ultralytics": ultralytics_mock, "ultralytics.nn": nn_mock, "ultralytics.nn.tasks": tasks_mock},
    ):
        yield mock_model


class TestWatermarkDetectorInit:
    def test_name(self):
        scorer = WatermarkDetector(config=ScorerConfig(device="cpu"))
        assert scorer.name == "watermark"

    def test_requires_continuous_frames_false(self):
        scorer = WatermarkDetector(config=ScorerConfig(device="cpu"))
        assert scorer.requires_continuous_frames is False

    def test_rejects_checkpoint_hash_mismatch(self, mock_yolo, monkeypatch):
        monkeypatch.setenv("WATERMARK_YOLO_MODEL_SHA256", "0" * 64)
        scorer = WatermarkDetector(config=ScorerConfig(device="cpu"))
        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            scorer._load_model()

    def test_abstract_base_is_not_star_exported(self):
        from mm.core import video_quality

        assert "BaseQualityScorer" not in video_quality.__all__
        assert hasattr(video_quality, "BaseQualityScorer")


class TestWatermarkBatching:
    def test_score_frames_batches_inputs(self, mock_yolo):
        scorer = WatermarkDetector(config=ScorerConfig(device="cpu", frame_batch_size=4))
        frames = make_loaded_frames(9, 64, 64)
        result = scorer.score_frames(frames)
        assert result.is_success
        assert mock_yolo.call_count == 3


class TestWatermarkScoring:
    def test_score_frames_no_detections(self, mock_yolo):
        scorer = WatermarkDetector(config=ScorerConfig(device="cpu"))
        frames = make_loaded_frames(8, 224, 224)
        result = scorer.score_frames(frames)
        assert result.name == "watermark"
        assert result.is_success
        assert result.scores["watermark_list"] == []
        assert result.scores["watermark_count"] == 0
        assert result.scores["watermark_area_ratio_sum"] == 0.0

    def test_detection_path_uses_tensor_boxes(self, mock_yolo):
        boxes = MagicMock()
        boxes.__len__ = MagicMock(return_value=1)
        boxes.xyxy = torch.tensor([[0.0, 0.0, 32.0, 32.0]])
        boxes.conf = torch.tensor([0.9])
        boxes.cls = torch.tensor([0.0])
        result = MagicMock()
        result.boxes = boxes
        mock_yolo.side_effect = lambda batch, **kwargs: [result for _ in batch]

        scorer = WatermarkDetector(config=ScorerConfig(device="cpu", frame_batch_size=2))
        scored = scorer.score_frames(make_loaded_frames(2, 64, 64))
        assert scored.is_success
        assert scored.scores["watermark_count"] == 2
        assert scored.scores["watermark_area_ratio_sum"] == pytest.approx(0.5)

    def test_empty_frames_returns_default(self, mock_yolo):
        scorer = WatermarkDetector(config=ScorerConfig(device="cpu"))
        frames = LoadedFrames(frames=[], indices=[])
        result = scorer.score_frames(frames)
        assert result.scores["watermark_count"] == 0
        assert result.scores["watermark_area_ratio_sum"] == 0.0

    def test_error_degrades_gracefully(self, mock_yolo):
        """When model raises, scorer returns error."""
        scorer = WatermarkDetector(config=ScorerConfig(device="cpu"))
        scorer._model = None
        scorer._model_loaded = False
        with patch.object(WatermarkDetector, "_load_model", side_effect=RuntimeError("load error")):
            frames = make_loaded_frames(8, 224, 224)
            result = scorer.score_frames(frames)
            assert not result.is_success
            assert result.error == "watermark_inference_failed"

    def test_registered_in_registry(self):
        from mm.core.video_quality import list_scorers

        assert "watermark" in list_scorers()

    def test_conf_threshold_config(self, mock_yolo):
        scorer = WatermarkDetector(config=ScorerConfig(device="cpu", extra={"conf_threshold": 0.5}))
        assert scorer.config.extra["conf_threshold"] == 0.5
