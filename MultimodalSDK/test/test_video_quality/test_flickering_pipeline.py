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
    TemporalFlickeringScorer,
)
from mm.core.video_quality.base import (
    MetricResult,
)


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
            scorers=["temporal_flickering"],
            device="cpu",
        )
        assert len(pipeline._scorers) == 1
        assert pipeline._scorers[0].name == "temporal_flickering"

    def test_init_with_scorer_instances(self, force_cpu_device):
        s1 = TemporalFlickeringScorer(config=ScorerConfig(device="cpu"))
        pipeline = VideoQualityPipeline(scorers=[s1], device="cpu")
        assert len(pipeline._scorers) == 1

    def test_init_with_weights(self, force_cpu_device):
        pipeline = VideoQualityPipeline(
            scorers=["temporal_flickering"],
            device="cpu",
            weights={"temporal_flickering": None},
        )
        assert len(pipeline._scorers) == 1


class TestPipelineScore:
    def test_score_single_video(self, tmp_path, force_cpu_device):
        """Test pipeline with a scorer on a real video file."""

        video_path = str(tmp_path / "test.mp4")
        _write_test_video(video_path, [np.full((64, 64, 3), i * 5, dtype=np.uint8) for i in range(30)])

        pipeline = VideoQualityPipeline(
            scorers=["temporal_flickering"],
            device="cpu",
        )
        report = pipeline.score(video_path)

        assert report.video_path == video_path
        assert "temporal_flickering" in report.metrics
        assert report.metrics["temporal_flickering"].is_success
        assert "temporal_flickering" in report.all_scores

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
            scorers=["temporal_flickering"],
            device="cpu",
        )
        reports = pipeline.score_batch(paths)
        assert len(reports) == 2
        for r in reports:
            assert "temporal_flickering" in r.metrics

    def test_all_failed_raises_runtime_error(self, tmp_path, force_cpu_device):
        """When all scorers fail, pipeline should raise RuntimeError."""

        video_path = str(tmp_path / "test.mp4")
        _write_test_video(video_path, [np.full((64, 64, 3), 100, dtype=np.uint8) for _ in range(20)])

        pipeline = VideoQualityPipeline(
            scorers=["temporal_flickering"],
            device="cpu",
        )
        with patch.object(
            pipeline._scorers[0],
            "score_frames",
            return_value=MetricResult(name="temporal_flickering", scores={}, error="fail"),
        ):
            with pytest.raises(RuntimeError, match="All scorers failed"):
                pipeline.score(video_path)

    def test_pipeline_close(self, force_cpu_device):
        pipeline = VideoQualityPipeline(
            scorers=["temporal_flickering"],
            device="cpu",
        )
        pipeline.close()
