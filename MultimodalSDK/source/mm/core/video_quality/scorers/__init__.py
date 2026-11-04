#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the MultimodalSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MultimodalSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE. See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Concrete video quality scorer implementations.

Importing this package executes the ``QualityScorerRegistry.register``
decorators and therefore populates the registry used by ``create_scorer``
and ``VideoQualityPipeline``.
"""

from .color_scorer import ColorConsistencyScorer
from .consistency_scorer import (
    AppearanceStyleScorer,
    BackgroundConsistencyScorer,
    OverallConsistencyScorer,
    SceneConsistencyScorer,
    SubjectConsistencyScorer,
    TemporalStyleScorer,
)
from .flickering_scorer import TemporalFlickeringScorer
from .image_quality_scorer import ImageQualityScorer
from .motion_scorer import MotionScorer
from .naturalness_scorer import NaturalnessScorer



__all__ = [
    "AppearanceStyleScorer",
    "BackgroundConsistencyScorer",
    "ColorConsistencyScorer",
    "ImageQualityScorer",
    "MotionScorer",
    "NaturalnessScorer",
    "OverallConsistencyScorer",
    "SceneConsistencyScorer",
    "SubjectConsistencyScorer",
    "TemporalFlickeringScorer",
    "TemporalStyleScorer",
]
