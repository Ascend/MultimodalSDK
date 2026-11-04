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

"""Tests for TemporalFlickeringScorer (pure-CV, no model)."""

from __future__ import annotations

import numpy as np

from mm.core.video_quality import TemporalFlickeringScorer, ScorerConfig
from mm.core.video_quality.base import LoadedFrames
from .conftest import make_loaded_frames


class TestTemporalFlickeringScorerInit:
    def test_name(self):
        scorer = TemporalFlickeringScorer(config=ScorerConfig(device="cpu"))
        assert scorer.name == "temporal_flickering"

    def test_requires_continuous_frames_false(self):
        scorer = TemporalFlickeringScorer(config=ScorerConfig(device="cpu"))
        assert scorer.requires_continuous_frames is False


class TestTemporalFlickeringScoring:
    def test_uniform_frames_zero_flickering(self):
        """Uniform brightness frames should have ~0 flickering."""
        scorer = TemporalFlickeringScorer(config=ScorerConfig(device="cpu"))
        frames = make_loaded_frames(8, 64, 64)
        # Make all frames uniform brightness
        for i, f in enumerate(frames.frames):
            frames.frames[i] = np.full((64, 64, 3), 128, dtype=np.uint8)
        result = scorer.score_frames(frames)
        assert result.is_success
        assert result.scores["temporal_flickering"] == 0.0
        assert result.scores["max_frame_jump"] == 0.0

    def test_varying_brightness_has_flickering(self):
        """Frames with increasing brightness should have non-zero flickering."""
        scorer = TemporalFlickeringScorer(config=ScorerConfig(device="cpu"))
        frames = make_loaded_frames(8, 64, 64)
        for i, f in enumerate(frames.frames):
            frames.frames[i] = np.full((64, 64, 3), i * 20, dtype=np.uint8)
        result = scorer.score_frames(frames)
        assert result.is_success
        assert result.scores["temporal_flickering"] > 0.0
        assert result.scores["max_frame_jump"] > 0.0

    def test_single_frame_returns_zero(self):
        scorer = TemporalFlickeringScorer(config=ScorerConfig(device="cpu"))
        frames = LoadedFrames(
            frames=[np.zeros((64, 64, 3), dtype=np.uint8)],
            indices=[0],
            meta={},
        )
        result = scorer.score_frames(frames)
        assert result.is_success
        assert result.scores["temporal_flickering"] == 0.0

    def test_elapsed_ms_positive(self):
        scorer = TemporalFlickeringScorer(config=ScorerConfig(device="cpu"))
        frames = make_loaded_frames(8, 64, 64)
        result = scorer.score_frames(frames)
        assert result.elapsed_ms >= 0.0

    def test_max_jump_ge_mean(self):
        """max_frame_jump should be >= temporal_flickering (mean)."""
        scorer = TemporalFlickeringScorer(config=ScorerConfig(device="cpu"))
        frames = make_loaded_frames(8, 64, 64)
        result = scorer.score_frames(frames)
        if result.scores["temporal_flickering"] > 0:
            assert result.scores["max_frame_jump"] >= result.scores["temporal_flickering"]

    def test_registered_in_registry(self):
        from mm.core.video_quality import list_scorers

        assert "temporal_flickering" in list_scorers()

    def test_create_scorer(self):
        from mm.core.video_quality import create_scorer

        scorer = create_scorer("temporal_flickering", config={"device": "cpu"})
        assert scorer.name == "temporal_flickering"
