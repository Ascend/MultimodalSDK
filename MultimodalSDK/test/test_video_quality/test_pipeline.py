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

from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import torch

from mm.core.video_quality import (
    VideoQualityPipeline,
    ScorerConfig,
    MotionScorer,
)
from mm.core.video_quality.base import (
    MetricResult,
)


MODULE_PATH = "mm.core.video_quality.scorers.motion_scorer"


@pytest.fixture(autouse=True)
def mock_weight_resolver():
    """Motion scorer needs a Sea-RAFT weight path; stub the resolver."""
    with patch(f"{MODULE_PATH}.WeightResolver") as mock_resolver_cls:
        resolver = MagicMock()
        resolver.resolve_single.return_value = "/fake/flow_ckpt"
        mock_resolver_cls.return_value = resolver
        yield resolver


@pytest.fixture(autouse=True)
def mock_ptlflow():
    """Mock ptlflow.get_model so pipeline scoring works without the real model."""
    mock_model = MagicMock()
    mock_model.eval.return_value = mock_model
    mock_model.to.return_value = mock_model
    mock_model.half.return_value = mock_model

    def _call(data):
        images = data["images"]  # [B, 2, C, H, W]
        b = images.shape[0]
        h, w = images.shape[3], images.shape[4]
        flows = torch.randn(b, 2, h, w)
        return {"flows": flows}

    mock_model.side_effect = _call
    mock_module = MagicMock()
    mock_module.get_model.return_value = mock_model

    fake_ckpt = MagicMock()
    fake_ckpt.is_file.return_value = True
    fake_ckpt.__str__ = MagicMock(return_value="/fake/sea_raft_s-sintel.ckpt")

    with patch.dict("sys.modules", {"ptlflow": mock_module}):
        with patch("pathlib.Path.is_dir", return_value=True):
            with patch("pathlib.Path.glob", return_value=[fake_ckpt]):
                yield mock_module


class TestPipelineInit:
    def test_init_with_scorer_names(self, force_cpu_device):
        pipeline = VideoQualityPipeline(
            scorers=["motion"],
            device="cpu",
        )
        assert len(pipeline._scorers) == 1
        assert pipeline._scorers[0].name == "motion"

    def test_init_with_scorer_instances(self, force_cpu_device):
        s1 = MotionScorer(config=ScorerConfig(device="cpu"))
        pipeline = VideoQualityPipeline(scorers=[s1], device="cpu")
        assert len(pipeline._scorers) == 1

    def test_init_with_weights(self, force_cpu_device):
        pipeline = VideoQualityPipeline(
            scorers=["motion"],
            device="cpu",
            weights={"motion": None},
        )
        assert len(pipeline._scorers) == 1


class TestPipelineScore:
    def test_score_single_video_no_model_scorers(self, tmp_path, force_cpu_device):
        """Test pipeline with pure-CV scorers on a real video file."""
        import cv2 as cv

        # Create a small test video
        video_path = str(tmp_path / "test.mp4")
        fourcc = cv.VideoWriter_fourcc(*"mp4v")
        writer = cv.VideoWriter(video_path, fourcc, 10.0, (64, 64))
        for i in range(30):
            frame = np.full((64, 64, 3), i * 5, dtype=np.uint8)
            writer.write(frame)
        writer.release()

        pipeline = VideoQualityPipeline(
            scorers=["motion"],
            device="cpu",
        )
        report = pipeline.score(video_path)

        assert report.video_path == video_path
        assert "motion" in report.metrics
        assert report.metrics["motion"].is_success
        assert "score_motion_mean" in report.all_scores

    def test_score_batch(self, tmp_path, force_cpu_device):
        import cv2 as cv

        paths = []
        for idx in range(2):
            video_path = str(tmp_path / f"test_{idx}.mp4")
            fourcc = cv.VideoWriter_fourcc(*"mp4v")
            writer = cv.VideoWriter(video_path, fourcc, 10.0, (64, 64))
            for i in range(20):
                frame = np.full((64, 64, 3), (i + idx * 10) * 5, dtype=np.uint8)
                writer.write(frame)
            writer.release()
            paths.append(video_path)

        pipeline = VideoQualityPipeline(
            scorers=["motion"],
            device="cpu",
        )
        reports = pipeline.score_batch(paths)
        assert len(reports) == 2
        for r in reports:
            assert "motion" in r.metrics

    def test_all_failed_raises_runtime_error(self, tmp_path, force_cpu_device):
        """When all scorers fail, pipeline should raise RuntimeError."""
        import cv2 as cv

        video_path = str(tmp_path / "test.mp4")
        fourcc = cv.VideoWriter_fourcc(*"mp4v")
        writer = cv.VideoWriter(video_path, fourcc, 10.0, (64, 64))
        for i in range(20):
            writer.write(np.full((64, 64, 3), 100, dtype=np.uint8))
        writer.release()

        pipeline = VideoQualityPipeline(
            scorers=["motion"],
            device="cpu",
        )
        # Patch score_frames to always fail
        with patch.object(
            pipeline._scorers[0],
            "score_frames",
            return_value=MetricResult(name="motion", scores={}, error="fail"),
        ):
            with pytest.raises(RuntimeError, match="All scorers failed"):
                pipeline.score(video_path)

    def test_partial_failure_continues(self, tmp_path, force_cpu_device):
        """When one scorer fails, others should still produce results."""
        import cv2 as cv

        video_path = str(tmp_path / "test.mp4")
        fourcc = cv.VideoWriter_fourcc(*"mp4v")
        writer = cv.VideoWriter(video_path, fourcc, 10.0, (64, 64))
        for i in range(20):
            writer.write(np.full((64, 64, 3), 100, dtype=np.uint8))
        writer.release()

        # Use two motion scorers with different configs
        pipeline = VideoQualityPipeline(
            scorers=["motion", "motion"],
            device="cpu",
        )
        # Make first motion scorer fail
        with patch.object(
            pipeline._scorers[0],
            "score_frames",
            return_value=MetricResult(name="motion", scores={"motion": None}, error="fail"),
        ):
            report = pipeline.score(video_path)
            # Same-name scorers overwrite each other; the surviving "motion"
            # result is the successful second scorer.
            assert report.metrics["motion"].is_success
            assert report.metrics["motion"].error is None

    def test_pipeline_close(self, force_cpu_device):
        pipeline = VideoQualityPipeline(
            scorers=["motion"],
            device="cpu",
        )
        pipeline.close()  # should not raise
