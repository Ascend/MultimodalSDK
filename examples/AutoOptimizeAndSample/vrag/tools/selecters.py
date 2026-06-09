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
# Selection and indexing utilities for video data.


import bisect
from typing import List, Optional, TypeVar

from vrag.logger import logger
from vrag.shared import flatten
from vrag.types import ASRDoc, Timestamp

_T = TypeVar("_T")


def sample_k_uniformly(timestamps: List[Timestamp], start: Timestamp, end: Timestamp, k: int) -> List[int]:
    indices = _select_indices_in_range(timestamps, start, end)

    if len(indices) <= k:
        return indices

    total_points = len(indices)
    selected_indices = []
    for i in range(k):
        idx = int((i + 1) * total_points / (k + 1))
        if idx >= total_points:
            idx = total_points - 1
        selected_indices.append(indices[idx])

    return selected_indices


def select_indices_for_subset(timestamps: List[Timestamp], subset_timestamps: List[Timestamp]) -> List[int]:
    ts_to_idx = {ts: i for i, ts in enumerate(timestamps)}
    return [ts_to_idx[ts] for ts in subset_timestamps if ts in ts_to_idx]


def select_spans(context_span: int, initial_indices: List[int], n_docs: int) -> List[int]:
    if context_span <= 0 and not initial_indices:
        logger.debug("No initial indices provided or content_span <= 0.")
        return []

    selected = sorted(
        set(flatten((range(max(0, i - context_span), min(n_docs, i + context_span + 1)) for i in initial_indices)))
    )

    msg = (
        f"Selected {len(selected)} documents with span of {context_span} in {n_docs} docs in total, "
        f"added {len(selected) - len(initial_indices)} docs."
    )
    logger.debug(msg)

    return selected


def select_related_asr_docs(asr_docs: List[ASRDoc], timestamp: Timestamp) -> Optional[int]:
    return next((i for i, d in enumerate(asr_docs) if d[1][0] <= timestamp <= d[1][1]), None)


def indexed(seq: List[_T], indexes: List[int]) -> List[_T]:
    return [seq[i] for i in indexes]


def _select_indices_in_range(timestamps: List[Timestamp], start: Timestamp, end: Timestamp) -> List[int]:
    left = bisect.bisect_left(timestamps, start)
    right = bisect.bisect_right(timestamps, end)
    return list(range(left, right))
