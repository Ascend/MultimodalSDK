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

"""Color consistency scorer — HSV histogram adjacent-frame cosine.

Pure-CV operator (no model, no weights).  Computes HSV colour histograms for
each sampled frame and measures the cosine similarity between adjacent frames'
histograms.  Higher is better (more colour-consistent).
"""

from __future__ import annotations

import logging
import time

import numpy as np

from ..base import (
    BaseQualityScorer,
    LoadedFrames,
    MetricResult,
    QualityScorerRegistry,
)


logger = logging.getLogger(__name__)


@QualityScorerRegistry.register
class ColorConsistencyScorer(BaseQualityScorer):
    """Color consistency scorer (pure CV, no model).

    Algorithm: per frame → resize to 320×180 → convert to HSV → compute 2-D
    histogram (H,S channels) → normalise → adjacent-frame cosine similarity.
    Returns the mean and minimum of adjacent-frame similarities.
    """

    name: str = "color_consistency"
    requires_continuous_frames: bool = True

    def score_frames(self, frames: LoadedFrames) -> MetricResult:
        t0 = time.perf_counter()

        try:
            import cv2 as cv

            hbins = int(self._config.extra.get("hbins", 48))
            sbins = int(self._config.extra.get("sbins", 40))

            if len(frames) < 2:
                return MetricResult(
                    name=self.name,
                    scores={"color_consistency": 0.0, "color_consistency_min": 0.0},
                    elapsed_ms=(time.perf_counter() - t0) * 1000,
                )

            hists = []
            for frame in frames.frames:
                hsv = cv.cvtColor(cv.resize(frame, (320, 180)), cv.COLOR_BGR2HSV)
                hist = cv.calcHist([hsv], [0, 1], None, [hbins, sbins], [0, 180, 0, 256])
                cv.normalize(hist, hist)
                hists.append(hist.flatten())

            hists = np.stack(hists)
            corr = []
            for i in range(len(hists) - 1):
                a, b = hists[i], hists[i + 1]
                denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
                corr.append(float(a.dot(b) / denom))

            return MetricResult(
                name=self.name,
                scores={
                    "color_consistency": round(float(np.mean(corr)), 6),
                    "color_consistency_min": round(float(np.min(corr)), 6),
                },
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )
        except Exception:
            logger.exception("color consistency scoring failed")
            return MetricResult(
                name=self.name,
                scores={},
                error="color_consistency_failed",
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )
