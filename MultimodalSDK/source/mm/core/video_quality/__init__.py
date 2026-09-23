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

"""mm.core.video_quality — video quality scoring infrastructure.

This package provides the **infrastructure** for video quality scoring:

Public API:
    - ``BaseQualityScorer``: abstract base class — subclass to implement a
      custom scorer (set ``name`` ClassVar and implement ``score_frames``).
    - ``ScorerConfig``, ``MetricResult``, ``VideoQualityReport``,
      ``LoadedFrames``: data contracts.
    - ``QualityScorerRegistry`` / ``create_scorer`` / ``list_scorers``:
      registration & factory (scorer classes are registered via companion
      sub-packages; the registry is populated once all scorer modules are
      delivered).
    - ``DeviceResolver``: unified device resolution (auto → NPU/CPU).
    - ``WeightResolver``: unified weight-path resolution.

.. note::

   Concrete scorer implementations (motion, naturalness, aesthetics, etc.)
   and the ``VideoQualityPipeline`` combiner will be provided in subsequent
   versions.  In this release ``create_scorer`` returns a ``KeyError`` for
   any unknown name; it is prepared for registration from future scorer
   packages.
"""

from .base import (
    BaseQualityScorer,
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

__all__ = [
    # Data contracts
    "LoadedFrames",
    "MetricResult",
    "VideoQualityReport",
    "ScorerConfig",
    # Base & registry
    "BaseQualityScorer",
    "QualityScorerRegistry",
    "create_scorer",
    "list_scorers",
    # Infrastructure
    "DeviceResolver",
    "WeightResolver",
]
