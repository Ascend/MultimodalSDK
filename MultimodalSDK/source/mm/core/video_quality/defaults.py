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

"""Default configurations and weight role registry for video quality scorers.

Each scorer declares its weight roles (dict keys) and corresponding environment
variables.  The WeightResolver uses this registry to resolve paths with a
consistent priority: explicit param > environment variable > error.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class WeightRole:
    """A single weight role declaration for a scorer."""

    role: str
    env_var: str
    description: str = ""
    required: bool = True


@dataclass
class ScorerDefaults:
    """Default configuration for a single scorer.

    Weight roles are intentionally *not* stored here: they are declared once in
    :data:`WEIGHT_ROLES` and read through :func:`get_weight_roles`, so there is
    a single source of truth.
    """

    name: str
    sample_frames: int = 8
    frame_batch_size: int | None = None
    dtype: str = "fp16"
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Weight role registry — each scorer's weight roles and env vars
# ---------------------------------------------------------------------------

WEIGHT_ROLES: dict[str, list[WeightRole]] = {
    "motion": [
        WeightRole("flow", "MOTION_CKPT_DIR", "Sea-RAFT-S optical flow checkpoint directory"),
    ],
    "naturalness": [
        WeightRole("clip", "NATURAL_CLIP_MODEL_ID", "CLIP ViT-L/14 model directory"),
        WeightRole("head", "NATURAL_WEIGHTS_JSON", "Linear regression head weights JSON"),
    ],
    "aesthetics": [
        WeightRole("encoder", "AESTHETIC_SIGLIP_MODEL_DIR", "SigLIP so400m-patch14-384 encoder directory"),
        WeightRole("predictor", "AESTHETIC_PREDICTOR_WEIGHTS_PATH", "AestheticPredictor v2.5 weights path"),
    ],
    "watermark": [
        WeightRole("detector", "WATERMARK_YOLO_MODEL_PATH", "YOLO11x watermark detector .pt path"),
    ],
    "image_quality": [
        WeightRole("liqe", "LIQE_WEIGHTS_PATH", "LIQE-mix weights path"),
    ],
    "distribution": [
        WeightRole("feature", "FVD_FEATURE_MODEL_DIR", "I3D/SVD feature model directory"),
    ],
    "imaging_quality": [
        WeightRole("dover", "DOVER_CKPT_DIR", "DOVER checkpoint directory"),
    ],
    "subject_consistency": [
        WeightRole("clip", "CONSENSUS_CLIP_MODEL_ID", "Consensus CLIP ViT-B/32 model"),
    ],
    "background_consistency": [
        WeightRole("clip", "CONSENSUS_CLIP_MODEL_ID", "Consensus CLIP ViT-B/32 model"),
    ],
    "overall_consistency": [
        WeightRole("clip", "CONSENSUS_CLIP_MODEL_ID", "Consensus CLIP ViT-B/32 model"),
    ],
    "scene_consistency": [
        WeightRole("clip", "CONSENSUS_CLIP_MODEL_ID", "Consensus CLIP ViT-B/32 model"),
    ],
    "appearance_style": [
        WeightRole("clip", "CONSENSUS_CLIP_MODEL_ID", "Consensus CLIP ViT-B/32 model"),
    ],
    "temporal_style": [
        WeightRole("clip", "CONSENSUS_CLIP_MODEL_ID", "Consensus CLIP ViT-B/32 model"),
    ],
}

# Scorers that do NOT require any weights (pure-CV / no-model)
NO_WEIGHT_SCORERS = frozenset(
    {
        "temporal_flickering",
        "color_consistency",
    }
)


# ---------------------------------------------------------------------------
# Scorer default parameters
# ---------------------------------------------------------------------------

SCORER_DEFAULTS: dict[str, ScorerDefaults] = {
    "motion": ScorerDefaults(
        name="motion",
        sample_frames=0,  # uses fps-based continuous decode, not uniform sampling
        dtype="fp16",
        extra={
            "target_fps": 4.0,
            "metrics": ("motion", "smoothness", "dynamic", "coherence"),
            "resize_short": 480,
            "agg_top_percent": 0.05,
        },
    ),
    "naturalness": ScorerDefaults(
        name="naturalness",
        sample_frames=8,
        dtype="fp16",
    ),
    "aesthetics": ScorerDefaults(
        name="aesthetics",
        sample_frames=8,
        dtype="fp16",
        extra={"resize_short": 384},
    ),
    "watermark": ScorerDefaults(
        name="watermark",
        sample_frames=8,
        dtype="fp16",
        extra={"conf_threshold": 0.35},
    ),
    "image_quality": ScorerDefaults(
        name="image_quality",
        sample_frames=8,
        dtype="fp16",
        extra={
            "metrics": ("liqe", "niqe", "brisque"),
            "batch_size": 8,
        },
    ),
    "distribution": ScorerDefaults(
        name="distribution",
        sample_frames=16,
        dtype="fp16",
        extra={
            "feature": "i3d",
            "size": 224,
            "max_videos": 32,
        },
    ),
    "imaging_quality": ScorerDefaults(
        name="imaging_quality",
        sample_frames=16,
        dtype="fp16",
    ),
    "temporal_flickering": ScorerDefaults(
        name="temporal_flickering",
        sample_frames=16,
        dtype="fp32",
    ),
    "subject_consistency": ScorerDefaults(
        name="subject_consistency",
        sample_frames=8,
        dtype="fp16",
        extra={"mask_strategy": "saliency"},
    ),
    "background_consistency": ScorerDefaults(
        name="background_consistency",
        sample_frames=8,
        dtype="fp16",
        extra={"mask_strategy": "saliency"},
    ),
    "overall_consistency": ScorerDefaults(
        name="overall_consistency",
        sample_frames=8,
        dtype="fp16",
    ),
    "color_consistency": ScorerDefaults(
        name="color_consistency",
        sample_frames=8,
        dtype="fp32",
        extra={"hbins": 48, "sbins": 40},
    ),
    "scene_consistency": ScorerDefaults(
        name="scene_consistency",
        sample_frames=8,
        dtype="fp16",
        extra={"prompt_set": "scene"},
    ),
    "appearance_style": ScorerDefaults(
        name="appearance_style",
        sample_frames=8,
        dtype="fp16",
        extra={"prompt_set": "style", "style_window": 3},
    ),
    "temporal_style": ScorerDefaults(
        name="temporal_style",
        sample_frames=32,
        dtype="fp16",
        extra={"prompt_set": "style", "num_buckets": 4},
    ),
}


def get_scorer_defaults(name: str) -> ScorerDefaults:
    """Return the default configuration for *name*; raises KeyError if unknown."""
    if name not in SCORER_DEFAULTS:
        raise KeyError(f"Unknown scorer: {name}. Registered: {sorted(SCORER_DEFAULTS)}")
    return SCORER_DEFAULTS[name]


def get_weight_roles(name: str) -> list[WeightRole]:
    """Return the weight roles declared for scorer *name* (empty list if none)."""
    return WEIGHT_ROLES.get(name, [])
