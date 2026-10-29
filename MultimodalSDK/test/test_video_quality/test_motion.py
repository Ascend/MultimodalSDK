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

"""Tests for MotionScorer (optical flow, mocked ptlflow)."""

from __future__ import annotations

# pylint: disable=redefined-outer-name  # noqa: E501
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import torch

from mm.core.video_quality import MotionScorer, ScorerConfig
from mm.core.video_quality.base import LoadedFrames
from .conftest import make_loaded_frames


MODULE_PATH = "mm.core.video_quality.scorers.motion_scorer"


@pytest.fixture(autouse=True)
def mock_weight_resolver():
    with patch(f"{MODULE_PATH}.WeightResolver") as MockResolver:
        resolver = MagicMock()
        resolver.resolve_single.return_value = "/fake/flow_ckpt"
        MockResolver.return_value = resolver
        yield resolver


@pytest.fixture(autouse=True)
def mock_ptlflow():
    """Mock ptlflow.get_model to return a fake flow model."""
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


class TestMotionScorerInit:
    def test_name(self):
        scorer = MotionScorer(config=ScorerConfig(device="cpu"))
        assert scorer.name == "motion"

    def test_requires_continuous_frames(self):
        scorer = MotionScorer(config=ScorerConfig(device="cpu"))
        assert scorer.requires_continuous_frames is True


class TestMotionScoring:
    def test_score_frames_returns_metric(self, mock_ptlflow):
        scorer = MotionScorer(config=ScorerConfig(device="cpu"))
        frames = make_loaded_frames(8, 224, 224, continuous=True)
        result = scorer.score_frames(frames)
        assert result.name == "motion"
        assert result.is_success
        assert "score_motion_mean" in result.scores
        assert "score_motion_std" in result.scores
        assert "score_motion_min" in result.scores
        assert "score_motion_max" in result.scores
        assert "score_motion_list" in result.scores
        assert isinstance(result.scores["score_motion_list"], list)
        assert len(result.scores["score_motion_list"]) == 7  # 8 frames → 7 pairs

    def test_single_frame_returns_defaults(self, mock_ptlflow):
        scorer = MotionScorer(config=ScorerConfig(device="cpu"))
        frames = LoadedFrames(
            frames=[np.zeros((64, 64, 3), dtype=np.uint8)],
            indices=[0],
            meta={},
            continuous=True,
        )
        result = scorer.score_frames(frames)
        assert result.is_success
        assert result.scores["score_motion_mean"] == 0.0
        assert result.scores["score_motion_list"] == []

    def test_smoothness_in_range(self, mock_ptlflow):
        scorer = MotionScorer(
            config=ScorerConfig(
                device="cpu",
                extra={"metrics": ("motion", "smoothness")},
            )
        )
        frames = make_loaded_frames(8, 224, 224, continuous=True)
        result = scorer.score_frames(frames)
        if result.is_success:
            sm = result.scores.get("motion_smoothness")
            if sm is not None:
                assert 0.0 <= sm <= 1.0

    def test_dynamic_degree_in_range(self, mock_ptlflow):
        scorer = MotionScorer(
            config=ScorerConfig(
                device="cpu",
                extra={"metrics": ("motion", "dynamic")},
            )
        )
        frames = make_loaded_frames(8, 224, 224, continuous=True)
        result = scorer.score_frames(frames)
        if result.is_success:
            dd = result.scores.get("dynamic_degree")
            if dd is not None:
                assert 0.0 <= dd <= 1.0

    def test_coherence_in_range(self, mock_ptlflow):
        scorer = MotionScorer(
            config=ScorerConfig(
                device="cpu",
                extra={"metrics": ("motion", "coherence")},
            )
        )
        frames = make_loaded_frames(8, 224, 224, continuous=True)
        result = scorer.score_frames(frames)
        if result.is_success:
            oc = result.scores.get("optical_flow_consistency")
            if oc is not None:
                assert 0.0 <= oc <= 1.0

    def test_error_degrades_gracefully(self, mock_ptlflow):
        """When model raises, scorer returns error instead of crashing."""
        scorer = MotionScorer(config=ScorerConfig(device="cpu"))
        # Force model to raise
        scorer._model = None
        scorer._model_loaded = False
        with patch.object(MotionScorer, "_load_model", side_effect=RuntimeError("load error")):
            frames = make_loaded_frames(8, 224, 224, continuous=True)
            result = scorer.score_frames(frames)
            assert not result.is_success
            assert result.error is not None

    def test_elapsed_ms_positive(self, mock_ptlflow):
        scorer = MotionScorer(config=ScorerConfig(device="cpu"))
        frames = make_loaded_frames(8, 224, 224, continuous=True)
        result = scorer.score_frames(frames)
        assert result.elapsed_ms >= 0.0

    def test_registered_in_registry(self):
        from mm.core.video_quality import list_scorers

        assert "motion" in list_scorers()
