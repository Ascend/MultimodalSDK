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
    ColorConsistencyScorer,
)
from mm.core.video_quality.base import (
    MetricResult,
)


def _make_color_pipeline(device="cpu"):
    defaults_pipeline = VideoQualityPipeline(scorers=[], device=device)
    scorer = ColorConsistencyScorer(config=defaults_pipeline._build_config("color_consistency"))
    return VideoQualityPipeline(scorers=[scorer], device=device)


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
            scorers=["color_consistency"],
            device="cpu",
        )
        assert len(pipeline._scorers) == 1
        assert pipeline._scorers[0].name == "color_consistency"

    def test_color_continuous_decode_has_explicit_target_fps(self, force_cpu_device):
        pipeline = VideoQualityPipeline(scorers=[], device="cpu")
        config = pipeline._build_config("color_consistency")
        scorer = ColorConsistencyScorer(config=config)
        assert scorer.requires_continuous_frames is True
        assert scorer.config.extra["target_fps"] == 4.0

    def test_init_with_scorer_instances(self, force_cpu_device):
        s1 = ColorConsistencyScorer(config=ScorerConfig(device="cpu"))
        pipeline = VideoQualityPipeline(scorers=[s1], device="cpu")
        assert len(pipeline._scorers) == 1

    def test_init_with_weights(self, force_cpu_device):
        pipeline = VideoQualityPipeline(
            scorers=["color_consistency"],
            device="cpu",
            weights={"color_consistency": None},
        )
        assert len(pipeline._scorers) == 1


class TestPipelineScore:
    def test_score_single_video(self, tmp_path, force_cpu_device):
        """Test pipeline with a scorer on a real video file."""

        video_path = str(tmp_path / "test.mp4")
        _write_test_video(video_path, [np.full((64, 64, 3), i * 5, dtype=np.uint8) for i in range(30)])

        pipeline = _make_color_pipeline()
        report = pipeline.score(video_path)

        assert report.video_path == video_path
        assert "color_consistency" in report.metrics
        assert report.metrics["color_consistency"].is_success
        assert "color_consistency" in report.all_scores

    def test_score_batch(self, tmp_path, force_cpu_device):
        paths = []
        for idx in range(2):
            video_path = str(tmp_path / f"test_{idx}.mp4")
            _write_test_video(
                video_path,
                [np.full((64, 64, 3), (i + idx * 10) * 5, dtype=np.uint8) for i in range(20)],
            )
            paths.append(video_path)

        pipeline = _make_color_pipeline()
        reports = pipeline.score_batch(paths)
        assert len(reports) == 2
        for r in reports:
            assert "color_consistency" in r.metrics

    def test_all_failed_raises_runtime_error(self, tmp_path, force_cpu_device):
        """When all scorers fail, pipeline should raise RuntimeError."""

        video_path = str(tmp_path / "test.mp4")
        _write_test_video(video_path, [np.full((64, 64, 3), 100, dtype=np.uint8) for _ in range(20)])

        pipeline = _make_color_pipeline()
        with patch.object(
            pipeline._scorers[0],
            "score_frames",
            return_value=MetricResult(name="color_consistency", scores={}, error="fail"),
        ):
            with pytest.raises(RuntimeError, match="All scorers failed"):
                pipeline.score(video_path)

    def test_pipeline_close(self, force_cpu_device):
        pipeline = VideoQualityPipeline(
            scorers=["color_consistency"],
            device="cpu",
        )
        pipeline.close()
