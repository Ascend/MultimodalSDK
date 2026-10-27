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

"""Template-method skeleton shared by the shipped video quality scorers."""

from __future__ import annotations

import time
from typing import Any, ClassVar

from ..base import LoadedFrames, MetricResult


class TimedScorerMixin:
    """Provide the common ``score_frames`` skeleton.

    Subclasses declare ``score_key`` / ``min_frames`` and implement
    ``_compute_scores``; timing, the minimum-frame guard, exception capture and
    ``MetricResult`` construction are handled here.
    """

    score_key: ClassVar[str] = ""
    min_frames: ClassVar[int] = 1
    empty_scores_when_insufficient: ClassVar[bool] = False
    empty_scores_on_error: ClassVar[bool] = False

    def __init__(self, config=None, **kwargs: Any):
        super().__init__(config, **kwargs)
        if kwargs:
            self._config.extra.update(kwargs)

    def score_frames(self, frames: LoadedFrames) -> MetricResult:
        t0 = time.perf_counter()
        score_key = self.score_key or self.name

        if len(frames) < self.min_frames:
            scores = {} if self.empty_scores_when_insufficient else {score_key: 0.0}
            return MetricResult(
                name=self.name,
                scores=scores,
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )

        try:
            scores = self._compute_scores(frames)
            return MetricResult(
                name=self.name,
                scores=scores,
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )
        except Exception as exc:
            scores = {} if self.empty_scores_on_error else {score_key: None}
            return MetricResult(
                name=self.name,
                scores=scores,
                error=str(exc),
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )

    def _compute_scores(self, frames: LoadedFrames) -> dict[str, Any]:
        raise NotImplementedError
