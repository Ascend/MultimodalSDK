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

"""Shared test fixtures and helpers for video_quality tests."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from mm.core.video_quality.backbones import get_shared_clip_backbone
from mm.core.video_quality.base import (
    BaseQualityScorer,
    LoadedFrames,
    MetricResult,
    QualityScorerRegistry,
)


# ---------------------------------------------------------------------------
# Synthetic frame factory
# ---------------------------------------------------------------------------


def make_frames(num_frames=8, height=224, width=224, seed=42):
    """Return a list of BGR numpy frames (deterministic)."""
    rng = np.random.RandomState(seed)
    return [rng.randint(0, 256, (height, width, 3), dtype=np.uint8) for _ in range(num_frames)]


def make_uniform_frames(num_frames=8, height=224, width=224, base_val=128):
    """Return frames of uniform colour (useful for consistency tests)."""
    return [np.full((height, width, 3), base_val, dtype=np.uint8) for _ in range(num_frames)]


def make_loaded_frames(num_frames=8, height=224, width=224, continuous=False, seed=42):
    """Return a LoadedFrames object with synthetic frames."""
    frames = make_frames(num_frames, height, width, seed)
    indices = list(range(num_frames))
    meta = {"fps": 30.0, "width": width, "height": height, "total_frames": 100, "device": "cpu"}
    return LoadedFrames(frames=frames, indices=indices, meta=meta, continuous=continuous)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_frames():
    """8 synthetic BGR frames (224×224)."""
    return make_frames(8, 224, 224)


@pytest.fixture
def sample_loaded_frames():
    """LoadedFrames with 8 synthetic frames."""
    return make_loaded_frames(8, 224, 224)


@pytest.fixture
def uniform_loaded_frames():
    """LoadedFrames with uniform-colour frames (high consistency expected)."""
    return make_loaded_frames(8, 224, 224, continuous=False, seed=99)


@pytest.fixture(autouse=True)
def force_cpu_device():
    """Force all scorers to CPU for testing.

    NPU suppression is handled entirely by patching
    ``device._npu_available`` below.
    """
    with patch("mm.core.video_quality.device._npu_available", return_value=False):
        yield


@pytest.fixture
def mock_clip_backbone():
    """Mock the SharedClipBackbone with deterministic features."""
    with patch("mm.core.video_quality.backbones.SharedClipBackbone") as MockCls:
        instance = MagicMock()

        # get_image_features returns L2-normalised features [N, D]
        def _get_image_features(frames_bgr):
            n = len(frames_bgr)
            feats = torch.randn(n, 512)
            feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
            return feats

        instance.get_image_features.side_effect = _get_image_features

        # get_text_features returns L2-normalised features [M, D]
        def _get_text_features(texts):
            n = len(texts)
            feats = torch.randn(n, 512)
            feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
            return feats

        instance.get_text_features.side_effect = _get_text_features

        MockCls.return_value = instance
        # Also patch the cache function
        with patch("mm.core.video_quality.backbones.get_shared_clip_backbone", return_value=instance):
            yield instance


@pytest.fixture
def mock_weight_resolver():
    """Mock WeightResolver to return fake paths (no env var needed)."""
    with patch("mm.core.video_quality.weights.WeightResolver") as MockCls:
        resolver = MagicMock()
        resolver.resolve_single.return_value = "/fake/weights"
        resolver.resolve_multi.return_value = {"clip": "/fake/clip", "head": "/fake/head.json"}
        MockCls.return_value = resolver
        yield resolver


# ---------------------------------------------------------------------------
# Scorer stubs
# ---------------------------------------------------------------------------
# This release ships only the video-quality *infrastructure*; the concrete
# scorers (motion, temporal_flickering, subject_consistency, ...) arrive in
# follow-up changes and populate ``QualityScorerRegistry`` at import time.
# The pipeline tests therefore stand in for them with lightweight stubs
# registered under the real names, so the infrastructure can be tested without
# depending on modules that are not part of this change.


class _FrameScorerStub(BaseQualityScorer):
    """Model-free, frame-level scorer stub (mirrors ``temporal_flickering``)."""

    def score_frames(self, frames: LoadedFrames) -> MetricResult:
        return MetricResult(name=self.name, scores={self.name: float(len(frames))})


class _SharedBackboneScorerStub(BaseQualityScorer):
    """Consistency-family stub that pulls the process-level shared backbone."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._backbone = None

    def _load_model(self) -> None:
        self._backbone = get_shared_clip_backbone(
            self.config.extra.get("model_path", "/fake/clip"),
            self.device,
            self.config.dtype,
        )

    def score_frames(self, frames: LoadedFrames) -> MetricResult:
        self._ensure_model()
        return MetricResult(name=self.name, scores={})


class _TemporalFlickeringStub(_FrameScorerStub):
    name = "temporal_flickering"


class _ColorConsistencyStub(_FrameScorerStub):
    name = "color_consistency"


class _SubjectConsistencyStub(_SharedBackboneScorerStub):
    name = "subject_consistency"


class _BackgroundConsistencyStub(_SharedBackboneScorerStub):
    name = "background_consistency"


_STUB_SCORERS = (
    _TemporalFlickeringStub,
    _ColorConsistencyStub,
    _SubjectConsistencyStub,
    _BackgroundConsistencyStub,
)


@pytest.fixture(autouse=True)
def register_stub_scorers():
    """Register the scorer stubs for the duration of a test, then restore."""
    previous = {cls.name: QualityScorerRegistry._registry.get(cls.name) for cls in _STUB_SCORERS}
    for cls in _STUB_SCORERS:
        QualityScorerRegistry.register(cls)
    yield
    for name, original in previous.items():
        if original is None:
            QualityScorerRegistry._registry.pop(name, None)
        else:
            QualityScorerRegistry._registry[name] = original
