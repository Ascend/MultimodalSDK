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

from unittest.mock import patch

import numpy as np
import pytest

from mm.core.video_quality import (
    VideoQualityPipeline,
    ScorerConfig,
    AestheticsScorer,
)
from mm.core.video_quality.base import (
    MetricResult,
)


@pytest.fixture(autouse=True, name="mock_pipeline_scorer")
def _mock_pipeline_scorer():
    """Keep pipeline tests independent of external model weights."""
    with patch.object(
        AestheticsScorer,
        "score_frames",
        return_value=MetricResult(name="aesthetics", scores={"score_aes_mean": 0.5}, elapsed_ms=0.0),
    ):
        yield


class TestPipelineInit:
    def test_init_with_scorer_names(self, force_cpu_device):
        pipeline = VideoQualityPipeline(
            scorers=["aesthetics"],
            device="cpu",
        )
        assert len(pipeline._scorers) == 1
        assert pipeline._scorers[0].name == "aesthetics"

    def test_init_with_scorer_instances(self, force_cpu_device):
        s1 = AestheticsScorer(config=ScorerConfig(device="cpu"))
        pipeline = VideoQualityPipeline(scorers=[s1], device="cpu")
        assert len(pipeline._scorers) == 1

    def test_init_with_weights(self, force_cpu_device):
        pipeline = VideoQualityPipeline(
            scorers=["aesthetics"],
            device="cpu",
            weights={"aesthetics": None},
        )
        assert len(pipeline._scorers) == 1


class TestPipelineScore:
    def test_score_single_video(self, tmp_path, force_cpu_device):
        """Test pipeline with a scorer on a real video file."""
        import cv2 as cv

        video_path = str(tmp_path / "test.mp4")
        fourcc = cv.VideoWriter_fourcc(*"mp4v")
        writer = cv.VideoWriter(video_path, fourcc, 10.0, (64, 64))
        for i in range(30):
            frame = np.full((64, 64, 3), i * 5, dtype=np.uint8)
            writer.write(frame)
        writer.release()

        pipeline = VideoQualityPipeline(
            scorers=["aesthetics"],
            device="cpu",
        )
        report = pipeline.score(video_path)

        assert report.video_path == video_path
        assert "aesthetics" in report.metrics
        assert report.metrics["aesthetics"].is_success
        assert "score_aes_mean" in report.all_scores

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
            scorers=["aesthetics"],
            device="cpu",
        )
        reports = pipeline.score_batch(paths)
        assert len(reports) == 2
        for r in reports:
            assert "aesthetics" in r.metrics

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
            scorers=["aesthetics"],
            device="cpu",
        )
        with patch.object(
            pipeline._scorers[0],
            "score_frames",
            return_value=MetricResult(name="aesthetics", scores={}, error="fail"),
        ):
            with pytest.raises(RuntimeError, match="All scorers failed"):
                pipeline.score(video_path)

    def test_pipeline_close(self, force_cpu_device):
        pipeline = VideoQualityPipeline(
            scorers=["aesthetics"],
            device="cpu",
        )
        pipeline.close()
