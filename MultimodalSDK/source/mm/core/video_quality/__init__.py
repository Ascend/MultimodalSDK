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
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE. See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""mm.core.video_quality — video quality scoring infrastructure.

This package provides the **infrastructure** for video quality scoring:

Public API:
    - ``.base.BaseQualityScorer``: abstract extension point for custom scorers;
      import it directly from ``mm.core.video_quality.base`` when subclassing.
    - ``ScorerConfig``, ``MetricResult``, ``VideoQualityReport``,
      ``LoadedFrames``: data contracts.
    - ``QualityScorerRegistry`` / ``create_scorer`` / ``list_scorers``:
      registration & factory.
    - ``DeviceResolver``: unified device resolution (auto → NPU/CPU).
    - ``WeightResolver``: unified weight-path resolution.
    - ``FrameLoader`` / ``load_frames``: uniform-sampling and continuous-fps
      decoding; ``SharedClipBackbone`` / ``clear_backbone_cache`` /
      ``get_cache_info``: process-level backbone pool; ``VideoQualityPipeline``:
      one-decode, shared-frame scoring.

.. note::

   This release ships the ``image_quality`` scorer, the CLIP consistency
   family (subject/background/overall/scene/appearance-style/temporal-style),
   the ``motion`` scorer (optical-flow based), the ``naturalness`` scorer,
   and the ``distribution`` scorer (FVD/KVD, dataset-level).
   Importing this package registers them with ``QualityScorerRegistry``.
"""

from . import base as _base
from .base import (
    LoadedFrames,
    MetricResult,
    ScorerConfig,
    VideoQualityReport,
    QualityScorerRegistry,
    create_scorer,
    list_scorers,
)

from .device import DeviceResolver
from .weights import WeightResolver
from .frame_loader import FrameLoader, load_frames
from .backbones import SharedClipBackbone, get_cache_info
from .backbones import clear_cache as clear_backbone_cache
from .pipeline import VideoQualityPipeline
from .scorers import (
    AppearanceStyleScorer,
    BackgroundConsistencyScorer,
    ColorConsistencyScorer,
    AestheticsScorer,
    DistributionConsistencyScorer,
    ImageQualityScorer,
    MotionScorer,
    NaturalnessScorer,
    OverallConsistencyScorer,
    SceneConsistencyScorer,
    SubjectConsistencyScorer,
    TemporalFlickeringScorer,
    TemporalStyleScorer,
    WatermarkDetector,
)

# Keep the abstract base accessible as a module attribute, but not in __all__.
BaseQualityScorer = _base.BaseQualityScorer

__all__ = [
    "LoadedFrames",
    "MetricResult",
    "VideoQualityReport",
    "ScorerConfig",
    "QualityScorerRegistry",
    "create_scorer",
    "list_scorers",
    "DeviceResolver",
    "WeightResolver",
    "FrameLoader",
    "load_frames",
    "SharedClipBackbone",
    "clear_backbone_cache",
    "get_cache_info",
    "VideoQualityPipeline",
    "AppearanceStyleScorer",
    "BackgroundConsistencyScorer",
    "ColorConsistencyScorer",
    "AestheticsScorer",
    "DistributionConsistencyScorer",
    "ImageQualityScorer",
    "MotionScorer",
    "NaturalnessScorer",
    "OverallConsistencyScorer",
    "SceneConsistencyScorer",
    "SubjectConsistencyScorer",
    "TemporalFlickeringScorer",
    "TemporalStyleScorer",
    "WatermarkDetector",
]
