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

"""CLIP consistency family — 6 scorers sharing one ``SharedClipBackbone``.

Scorers:
    - SubjectConsistencyScorer     (foreground-mask weighted cosine)
    - BackgroundConsistencyScorer  (background-mask = foreground-inverse)
    - OverallConsistencyScorer    (full-frame, no mask)
    - SceneConsistencyScorer       (CLIP-scene prompt similarity stability)
    - AppearanceStyleScorer        (CLIP-style prompt similarity stability)
    - TemporalStyleScorer          (bucketed CLIP style stability)

All six share the same ``SharedClipBackbone`` instance (CLIP ViT-B/32) and
the same decoded frames, but differ in masking / prompt-set / bucketing.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from ..base import (
    BaseQualityScorer,
    LoadedFrames,
    QualityScorerRegistry,
    ScorerConfig,
)
from ..backbones import SharedClipBackbone, get_shared_clip_backbone
from ..weights import WeightResolver
from ._common import TimedScorerMixin


# ---------------------------------------------------------------------------
# Built-in prompt sets (VBench-compatible scene / style descriptors)
# ---------------------------------------------------------------------------

SCENE_PROMPTS = [
    "a photo of a living room",
    "a photo of a bedroom",
    "a photo of a kitchen",
    "a photo of an office",
    "a photo of a garden",
    "a photo of a city street",
    "a photo of a park",
    "a photo of a forest",
    "a photo of a mountain",
    "a photo of a beach",
    "a photo of a desert",
    "a photo of a snowy landscape",
    "a photo of a rainy scene",
    "a photo of a sunset",
    "a photo of a night scene",
    "a photo of an indoor scene",
    "a photo of an outdoor scene",
]

STYLE_PROMPTS = [
    "a watercolor painting",
    "an oil painting",
    "a sketch",
    "a cartoon",
    "a 3d render",
    "a pixel art image",
    "a photograph",
    "a digital art piece",
    "a charcoal drawing",
    "a pastel painting",
    "a mosaic",
    "a collage",
    "a black and white photo",
    "a sepia photo",
    "a pop art image",
    "an abstract art piece",
    "a surreal image",
    "a minimalist image",
]

_PROMPT_SETS = {
    "scene": SCENE_PROMPTS,
    "style": STYLE_PROMPTS,
}


def _resolve_prompt_set(prompt_set: str | list[str]) -> list[str]:
    if isinstance(prompt_set, str):
        if prompt_set not in _PROMPT_SETS:
            raise ValueError(
                f"Unknown prompt_set: {prompt_set!r}. Available: {sorted(_PROMPT_SETS)} or pass a custom list."
            )
        return _PROMPT_SETS[prompt_set]
    return list(prompt_set)


# ---------------------------------------------------------------------------
# Base for CLIP consistency scorers (shares backbone loading)
# ---------------------------------------------------------------------------


class _ClipConsistencyBase(TimedScorerMixin, BaseQualityScorer):
    """Shared base for the 6 CLIP consistency scorers."""

    requires_continuous_frames: ClassVar[bool] = False

    def __init__(self, config: ScorerConfig | None = None, **kwargs: Any):
        super().__init__(config, **kwargs)
        self._backbone: SharedClipBackbone | None = None
        self._clip_path: str | None = None

    def _ensure_model(self) -> None:
        if self._backbone is not None:
            return
        resolver = WeightResolver(self.name)
        self._clip_path = resolver.resolve_single(self._config.weights, role="clip")
        self._backbone = get_shared_clip_backbone(
            self._clip_path,
            device=self.device,
            dtype=self._config.dtype,
        )

    def _extract_features(self, frames: LoadedFrames) -> np.ndarray:
        """Extract L2-normalised CLIP image features from frames."""
        self._ensure_model()
        feats = self._backbone.get_image_features(frames.frames)
        return feats.cpu().numpy()

    @staticmethod
    def _apply_mask(
        frames: list[np.ndarray],
        strategy: str,
        foreground: bool,
    ) -> list[np.ndarray]:
        """Apply center-crop (foreground) or border-crop (background) mask."""
        import cv2 as cv

        if strategy == "none":
            return frames

        out = []
        for frame in frames:
            h, w = frame.shape[:2]
            if foreground:
                ratio = 0.6
                ch, cw = int(h * ratio), int(w * ratio)
                y0, x0 = (h - ch) // 2, (w - cw) // 2
                crop = frame[y0 : y0 + ch, x0 : x0 + cw]
            else:
                bh, bw = int(h * 0.3), int(w * 0.3)
                crop = np.zeros_like(frame)
                crop[:bh] = frame[:bh]
                crop[h - bh :] = frame[h - bh :]
                crop[:, :bw] = frame[:, :bw]
                crop[:, w - bw :] = frame[:, w - bw :]
            out.append(cv.resize(crop, (224, 224)))
        return out

    def _masked_adjacent_cosine(self, frames: LoadedFrames, foreground: bool) -> float:
        """Mean adjacent-frame cosine over masked CLIP features."""
        strategy = self._config.extra.get("mask_strategy", "saliency")
        masked = self._apply_mask(frames.frames, strategy, foreground)
        self._ensure_model()
        feats = self._backbone.get_image_features(masked).cpu().numpy()
        return self._adjacent_cosine(feats)

    @staticmethod
    def _adjacent_cosine(feats: np.ndarray) -> float:
        sims = (feats[:-1] * feats[1:]).sum(axis=1)
        return round(float(sims.mean()), 6)

    def close(self) -> None:
        """Drop scorer's reference to the shared backbone.

        The backbone stays in the process-level cache for other scorers /
        Pipeline instances to reuse.  Call ``clear_backbone_cache()`` to
        actually free model memory.
        """
        self._backbone = None
        self._model_loaded = False


# ---------------------------------------------------------------------------
# Subject consistency — foreground-mask weighted adjacent cosine
# ---------------------------------------------------------------------------


@QualityScorerRegistry.register
class SubjectConsistencyScorer(_ClipConsistencyBase):
    """Subject consistency (CLIP frame-level foreground-weighted cosine).

    Algorithm: extract CLIP features from center-cropped frames (focusing on
    the subject), compute adjacent-frame cosine similarity, return the mean.
    """

    name: ClassVar[str] = "subject_consistency"
    score_key: ClassVar[str] = "subject_consistency"
    min_frames: ClassVar[int] = 2

    def _compute_scores(self, frames: LoadedFrames) -> dict[str, Any]:
        return {"subject_consistency": self._masked_adjacent_cosine(frames, foreground=True)}


# ---------------------------------------------------------------------------
# Background consistency — border-crop (background) cosine
# ---------------------------------------------------------------------------


@QualityScorerRegistry.register
class BackgroundConsistencyScorer(_ClipConsistencyBase):
    """Background consistency (CLIP border-region adjacent cosine)."""

    name: ClassVar[str] = "background_consistency"
    score_key: ClassVar[str] = "background_consistency"
    min_frames: ClassVar[int] = 2

    def _compute_scores(self, frames: LoadedFrames) -> dict[str, Any]:
        return {"background_consistency": self._masked_adjacent_cosine(frames, foreground=False)}


# ---------------------------------------------------------------------------
# Overall consistency — full-frame CLIP embedding cosine (no mask)
# ---------------------------------------------------------------------------


@QualityScorerRegistry.register
class OverallConsistencyScorer(_ClipConsistencyBase):
    """Overall spatio-temporal consistency (full-frame CLIP cosine)."""

    name: ClassVar[str] = "overall_consistency"
    score_key: ClassVar[str] = "overall_consistency"
    min_frames: ClassVar[int] = 2

    def _compute_scores(self, frames: LoadedFrames) -> dict[str, Any]:
        feats = self._extract_features(frames)
        return {"overall_consistency": self._adjacent_cosine(feats)}


# ---------------------------------------------------------------------------
# Scene consistency — CLIP-scene prompt similarity stability
# ---------------------------------------------------------------------------


@QualityScorerRegistry.register
class SceneConsistencyScorer(_ClipConsistencyBase):
    """Scene consistency (CLIP-scene prompt similarity frame stability).

    Algorithm: extract CLIP image features and scene prompt text features,
    compute per-frame best-prompt similarity, measure frame-to-frame stability
    of the similarity sequence.
    """

    name: ClassVar[str] = "scene_consistency"
    score_key: ClassVar[str] = "scene_consistency"
    min_frames: ClassVar[int] = 2

    def _compute_scores(self, frames: LoadedFrames) -> dict[str, Any]:
        prompt_set = self._resolve_prompt_set()
        self._ensure_model()

        img_feats = self._backbone.get_image_features(frames.frames)
        text_feats = self._backbone.get_text_features(prompt_set)

        sims = (img_feats @ text_feats.t()).max(dim=1).values
        stability = 1.0 - float(np.std(sims.cpu().numpy()))
        stability = max(0.0, min(1.0, stability))
        return {"scene_consistency": round(stability, 6)}

    def _resolve_prompt_set(self) -> list[str]:
        return _resolve_prompt_set(self._config.extra.get("prompt_set", "scene"))


# ---------------------------------------------------------------------------
# Appearance style consistency — CLIP-style prompt similarity stability
# ---------------------------------------------------------------------------


@QualityScorerRegistry.register
class AppearanceStyleScorer(_ClipConsistencyBase):
    """Appearance style consistency (CLIP style-vector frame stability).

    Algorithm: extract CLIP features, apply a sliding window (default 3) to
    compute local style vectors, measure adjacent style-vector cosine.
    """

    name: ClassVar[str] = "appearance_style"
    score_key: ClassVar[str] = "appearance_style_consistency"
    min_frames: ClassVar[int] = 2

    def _compute_scores(self, frames: LoadedFrames) -> dict[str, Any]:
        feats = self._extract_features(frames)
        win = max(1, int(self._config.extra.get("style_window", 3)))

        if len(feats) <= win:
            sims = (feats[:-1] * feats[1:]).sum(axis=1)
        else:
            style_vecs = np.stack([feats[i : i + win].mean(axis=0) for i in range(len(feats) - win + 1)])
            style_vecs /= np.linalg.norm(style_vecs, axis=1, keepdims=True) + 1e-8
            sims = (style_vecs[:-1] * style_vecs[1:]).sum(axis=1)

        return {"appearance_style_consistency": round(float(sims.mean()), 6)}


# ---------------------------------------------------------------------------
# Temporal style consistency — bucketed CLIP style stability
# ---------------------------------------------------------------------------


@QualityScorerRegistry.register
class TemporalStyleScorer(_ClipConsistencyBase):
    """Temporal style consistency (bucketed CLIP style stability).

    Algorithm: extract CLIP features, split into *num_buckets* temporal chunks,
    compute per-bucket mean style vector and measure adjacent-bucket cosine.
    """

    name: ClassVar[str] = "temporal_style"
    score_key: ClassVar[str] = "temporal_style_consistency"
    min_frames: ClassVar[int] = 4

    def _compute_scores(self, frames: LoadedFrames) -> dict[str, Any]:
        feats = self._extract_features(frames)
        k = int(self._config.extra.get("num_buckets", 4))
        k = max(2, min(k, len(feats) // 2))

        chunks = np.array_split(feats, k)
        window_vecs = np.stack([c.mean(axis=0) for c in chunks])
        window_vecs /= np.linalg.norm(window_vecs, axis=1, keepdims=True) + 1e-8

        adj_sims = (window_vecs[:-1] * window_vecs[1:]).sum(axis=1)
        return {"temporal_style_consistency": round(float(adj_sims.mean()), 6)}
