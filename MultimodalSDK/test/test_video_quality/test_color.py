#!/usr/bin/env python3
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

"""Tests for ColorConsistencyScorer (pure-CV, no model)."""

from __future__ import annotations

import numpy as np
from mm.core.video_quality import ColorConsistencyScorer, ScorerConfig
from mm.core.video_quality.base import LoadedFrames

from .conftest import make_loaded_frames


class TestColorConsistencyScorerInit:
    def test_name(self):
        scorer = ColorConsistencyScorer(config=ScorerConfig(device="cpu"))
        assert scorer.name == "color_consistency"

    def test_requires_continuous_frames_true(self):
        scorer = ColorConsistencyScorer(config=ScorerConfig(device="cpu"))
        assert scorer.requires_continuous_frames is True


class TestColorConsistencyScoring:
    def test_uniform_frames_high_consistency(self):
        """Uniform-colour frames should have very high colour consistency (~1.0)."""
        scorer = ColorConsistencyScorer(config=ScorerConfig(device="cpu"))
        frames = make_loaded_frames(8, 64, 64)
        for f in frames.frames:
            f[:] = np.array([0, 0, 255], dtype=np.uint8)
        result = scorer.score_frames(frames)
        assert result.is_success
        assert result.scores["color_consistency"] > 0.9

    def test_varying_colors_lower_consistency(self):
        """Frames with different colours should have lower consistency."""
        scorer = ColorConsistencyScorer(config=ScorerConfig(device="cpu"))
        import cv2 as cv

        uniform_frames = make_loaded_frames(8, 64, 64, seed=123)
        for frame in uniform_frames.frames:
            frame[:] = np.array([0, 0, 255], dtype=np.uint8)
        uniform_result = scorer.score_frames(uniform_frames)
        assert uniform_result.is_success

        frames = make_loaded_frames(8, 64, 64, seed=456)
        # Use saturated HSV colors so hue histograms differ across frames.
        for i, frame in enumerate(frames.frames):
            hsv_color = np.uint8([[[int(i * 25), 255, 200]]])
            frame[:] = cv.cvtColor(hsv_color, cv.COLOR_HSV2BGR)[0, 0]
        result = scorer.score_frames(frames)
        assert result.is_success
        score = result.scores["color_consistency"]
        assert 0.0 <= score < uniform_result.scores["color_consistency"]

    def test_single_frame_returns_zero(self):
        scorer = ColorConsistencyScorer(config=ScorerConfig(device="cpu"))
        frames = LoadedFrames(
            frames=[np.zeros((64, 64, 3), dtype=np.uint8)],
            indices=[0],
            meta={},
        )
        result = scorer.score_frames(frames)
        assert result.is_success
        assert result.scores["color_consistency"] == 0.0
        assert result.scores["color_consistency_min"] == 0.0

    def test_consistency_in_range(self):
        """Colour consistency should be in [0, 1]."""
        scorer = ColorConsistencyScorer(config=ScorerConfig(device="cpu"))
        frames = make_loaded_frames(8, 64, 64)
        result = scorer.score_frames(frames)
        assert 0.0 <= result.scores["color_consistency"] <= 1.0
        assert 0.0 <= result.scores["color_consistency_min"] <= 1.0

    def test_min_le_mean(self):
        """color_consistency_min should be <= color_consistency (mean)."""
        scorer = ColorConsistencyScorer(config=ScorerConfig(device="cpu"))
        frames = make_loaded_frames(8, 64, 64)
        result = scorer.score_frames(frames)
        assert result.scores["color_consistency_min"] <= result.scores["color_consistency"]

    def test_custom_bins(self):
        """Custom histogram bin counts should work."""
        scorer = ColorConsistencyScorer(config=ScorerConfig(device="cpu", extra={"hbins": 32, "sbins": 32}))
        frames = make_loaded_frames(8, 64, 64)
        result = scorer.score_frames(frames)
        assert result.is_success

    def test_registered_in_registry(self):
        from mm.core.video_quality import list_scorers

        assert "color_consistency" in list_scorers()

    def test_create_scorer(self):
        from mm.core.video_quality import create_scorer

        scorer = create_scorer("color_consistency", config={"device": "cpu"})
        assert scorer.name == "color_consistency"

    def test_cv_failure_degrades_gracefully(self, monkeypatch):
        import cv2 as cv

        scorer = ColorConsistencyScorer(config=ScorerConfig(device="cpu"))
        frames = make_loaded_frames(2, 64, 64)
        monkeypatch.setattr(cv, "cvtColor", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("cv failure")))
        result = scorer.score_frames(frames)
        assert not result.is_success
        assert result.error == "color_consistency_failed"
        assert not result.scores
