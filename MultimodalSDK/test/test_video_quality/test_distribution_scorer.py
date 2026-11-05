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

"""Tests for DistributionConsistencyScorer (FVD/KVD, dataset-level)."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from mm.core.video_quality import (
    DistributionConsistencyScorer,
    ScorerConfig,
    list_scorers,
)


MODULE_PATH = "mm.core.video_quality.scorers.distribution_scorer"


@pytest.fixture(autouse=True)
def mock_weight_resolver():
    with patch(f"{MODULE_PATH}.WeightResolver") as MockResolver:
        resolver = MagicMock()
        resolver.resolve_single.return_value = "/fake/i3d"
        MockResolver.return_value = resolver
        yield resolver


class TestDistributionConsistencyScorerInit:
    def test_name(self):
        scorer = DistributionConsistencyScorer(config=ScorerConfig(device="cpu"))
        assert scorer.name == "distribution"


class TestDistributionConsistencyScoring:
    def test_score_with_feature_arrays(self):
        """Test scoring with pre-extracted feature arrays (no model needed)."""
        scorer = DistributionConsistencyScorer(config=ScorerConfig(device="cpu"))
        real_feats = np.random.randn(40, 400).astype(np.float32)
        gen_feats = np.random.randn(40, 400).astype(np.float32)
        result = scorer.score(real=real_feats, generated=gen_feats, feature="i3d")
        assert result.name == "distribution"
        assert result.is_success
        assert "fvd" in result.scores
        assert isinstance(result.scores["fvd"], float)
        assert result.scores["fvd"] >= 0.0

    def test_score_with_fewer_than_32_still_works(self):
        """Fewer than 32 samples should still produce a score."""
        scorer = DistributionConsistencyScorer(config=ScorerConfig(device="cpu"))
        real_feats = np.random.randn(5, 400).astype(np.float32)
        gen_feats = np.random.randn(5, 400).astype(np.float32)
        result = scorer.score(real=real_feats, generated=gen_feats, feature="i3d")
        assert result.is_success
        assert "fvd" in result.scores

    def test_score_missing_inputs(self):
        scorer = DistributionConsistencyScorer(config=ScorerConfig(device="cpu"))
        result = scorer.score()
        assert not result.is_success
        assert result.error is not None

    def test_score_frames_not_supported(self):
        """score_frames should return error (dataset-level metric)."""
        from mm.core.video_quality.base import LoadedFrames

        scorer = DistributionConsistencyScorer(config=ScorerConfig(device="cpu"))
        frames = LoadedFrames(frames=[], indices=[])
        result = scorer.score_frames(frames)
        assert not result.is_success
        assert "dataset-level" in result.error

    def test_kvd_key_returns_kvd_key(self):
        """When feature='keyframe', output key should be 'kvd'."""
        scorer = DistributionConsistencyScorer(config=ScorerConfig(device="cpu"))
        real_feats = np.random.randn(40, 512).astype(np.float32)
        gen_feats = np.random.randn(40, 512).astype(np.float32)
        result = scorer.score(real=real_feats, generated=gen_feats, feature="keyframe")
        assert result.is_success
        assert "kvd" in result.scores

    def test_dimension_mismatch_raises(self):
        scorer = DistributionConsistencyScorer(config=ScorerConfig(device="cpu"))
        real_feats = np.random.randn(10, 400).astype(np.float32)
        gen_feats = np.random.randn(10, 512).astype(np.float32)
        result = scorer.score(real=real_feats, generated=gen_feats, feature="i3d")
        assert not result.is_success
        assert "mismatch" in result.error.lower()

    def test_registered_in_registry(self):
        assert "distribution" in list_scorers()

    def test_frechet_distance_identical_distributions(self):
        """Identical distributions should have ~0 Fréchet distance."""
        from mm.core.video_quality.scorers.distribution_scorer import _frechet_distance

        feats = np.random.randn(100, 50).astype(np.float32)
        dist = _frechet_distance(feats, feats)
        assert abs(dist) < 1e-6

    def test_frechet_distance_different_distributions(self):
        """Different distributions should have positive Fréchet distance."""
        from mm.core.video_quality.scorers.distribution_scorer import _frechet_distance

        a = np.random.randn(100, 50).astype(np.float32)
        b = np.random.randn(100, 50).astype(np.float32) + 10.0
        dist = _frechet_distance(a, b)
        assert dist > 0.0
