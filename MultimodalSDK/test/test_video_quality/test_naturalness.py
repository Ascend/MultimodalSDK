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

"""Tests for NaturalnessScorer (CLIP + linear head, mocked transformers)."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest
import torch

from mm.core.video_quality import NaturalnessScorer, ScorerConfig
from mm.core.video_quality.base import LoadedFrames
from .conftest import make_loaded_frames


MODULE_PATH = "mm.core.video_quality.scorers.naturalness_scorer"


@pytest.fixture(autouse=True)
def mock_weight_resolver():
    with patch(f"{MODULE_PATH}.WeightResolver") as MockResolver:
        resolver = MagicMock()
        resolver.resolve_multi.return_value = {
            "clip": "/fake/clip",
            "head": "/fake/head.json",
        }
        MockResolver.return_value = resolver
        yield resolver


@pytest.fixture(autouse=True, name="mock_transformers")
def _mock_transformers_fixture():
    """Mock transformers CLIPModel/CLIPImageProcessor and json.load."""
    mock_model = MagicMock()
    mock_model.eval.return_value = mock_model
    mock_model.to.return_value = mock_model

    def _get_image_features(**kwargs):
        pv = kwargs.get("pixel_values")
        n = pv.shape[0] if hasattr(pv, "shape") else 8
        return torch.randn(n, 768)

    mock_model.get_image_features.side_effect = _get_image_features

    mock_processor = MagicMock()
    mock_processor.crop_size = {"height": 224, "width": 224}
    mock_processor.image_mean = [0.48, 0.46, 0.41]
    mock_processor.image_std = [0.27, 0.26, 0.28]

    import json as json_mod

    mock_open = MagicMock()
    mock_file = MagicMock()
    mock_open.return_value.__enter__ = MagicMock(return_value=mock_file)
    mock_open.return_value.__exit__ = MagicMock(return_value=False)

    with (
        patch("transformers.CLIPModel") as MockModel,
        patch("transformers.CLIPImageProcessor") as MockProcessor,
        patch("builtins.open", mock_open),
        patch(f"{MODULE_PATH}.os.path.isfile", return_value=True),
        patch.object(json_mod, "load", return_value={"coef": [0.1] * 768, "intercept": -2.0}),
    ):
        MockModel.from_pretrained.return_value = mock_model
        MockProcessor.from_pretrained.return_value = mock_processor
        yield {"model": mock_model, "processor": mock_processor}


class TestNaturalnessScorerInit:
    def test_name(self):
        scorer = NaturalnessScorer(config=ScorerConfig(device="cpu"))
        assert scorer.name == "naturalness"

    def test_requires_continuous_frames_false(self):
        scorer = NaturalnessScorer(config=ScorerConfig(device="cpu"))
        assert scorer.requires_continuous_frames is False


class TestNaturalnessScoring:
    def test_score_frames_returns_metric(self, mock_transformers):
        scorer = NaturalnessScorer(config=ScorerConfig(device="cpu"))
        frames = make_loaded_frames(8, 224, 224)
        result = scorer.score_frames(frames)
        assert result.name == "naturalness"
        assert result.is_success
        assert "score_natural_mean" in result.scores
        assert "score_natural_list" in result.scores
        assert len(result.scores["score_natural_list"]) == 8
        assert 0.0 <= result.scores["score_natural_mean"] <= 1.0
        assert 0.0 <= result.scores["score_natural_min"] <= 1.0
        assert 0.0 <= result.scores["score_natural_max"] <= 1.0
        assert result.scores["score_natural_std"] >= 0.0

    def test_empty_frames_returns_default(self, mock_transformers):
        scorer = NaturalnessScorer(config=ScorerConfig(device="cpu"))
        frames = LoadedFrames(frames=[], indices=[])
        result = scorer.score_frames(frames)
        assert result.scores["score_natural_mean"] == -1.0
        assert result.scores["score_natural_list"] == []

    def test_error_degrades_gracefully(self, mock_transformers):
        """When model raises, scorer returns error."""
        mock_transformers["model"].get_image_features.side_effect = RuntimeError("model error")
        scorer = NaturalnessScorer(config=ScorerConfig(device="cpu"))
        frames = make_loaded_frames(8, 224, 224)
        result = scorer.score_frames(frames)
        assert not result.is_success
        assert result.error == "naturalness scoring failed"
        assert result.scores["score_natural_mean"] is None

    def test_registered_in_registry(self):
        from mm.core.video_quality import list_scorers

        assert "naturalness" in list_scorers()
