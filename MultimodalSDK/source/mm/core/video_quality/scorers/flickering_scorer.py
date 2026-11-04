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

"""Temporal flickering scorer — mean adjacent-frame brightness difference.

Pure-CV operator (no model, no weights).  Measures frame-to-frame brightness
jumps to detect flickering artifacts in generated videos.  Lower is better.
"""

from __future__ import annotations

import time
from typing import ClassVar

import numpy as np

from ..base import (
    BaseQualityScorer,
    LoadedFrames,
    MetricResult,
    QualityScorerRegistry,
)


@QualityScorerRegistry.register
class TemporalFlickeringScorer(BaseQualityScorer):
    """Temporal flickering scorer (pure CV, no model).

    Algorithm: convert each frame to grayscale, compute mean brightness per
    frame, take absolute differences of consecutive means.  The flickering
    score is the mean of these differences; ``max_frame_jump`` is the peak.
    """

    name: ClassVar[str] = "temporal_flickering"
    requires_continuous_frames: ClassVar[bool] = False

    def score_frames(self, frames: LoadedFrames) -> MetricResult:
        import cv2 as cv

        t0 = time.perf_counter()

        if len(frames) < 2:
            return MetricResult(
                name=self.name,
                scores={"temporal_flickering": 0.0, "max_frame_jump": 0.0},
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )

        means = [float(cv.cvtColor(f, cv.COLOR_BGR2GRAY).mean()) for f in frames.frames]
        diffs = np.abs(np.diff(means))

        return MetricResult(
            name=self.name,
            scores={
                "temporal_flickering": round(float(diffs.mean()), 6),
                "max_frame_jump": round(float(diffs.max()), 6),
            },
            elapsed_ms=(time.perf_counter() - t0) * 1000,
        )
