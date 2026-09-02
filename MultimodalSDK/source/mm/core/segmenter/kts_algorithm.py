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
"""Stateless KTS algorithm functions.

Everything here is a pure function: sampling-time grid, timestamp->frame
mapping, dot-product diff sequence, kernel-matrix KTS dynamic programming,
automatic lambda calibration, boundary alignment, split->segment conversion
and short-segment merging. ``KtsSegmenter`` (kts_segmenter.py) orchestrates
these; keeping them side-effect free makes them directly unit-testable.

Performance contract:
    - ``kts_dp_segment_full`` is O(N^2) in time and memory (N = number of
      sample frames): the cosine kernel ``K = E @ E.T`` and its float64
      integral image ``P`` are preallocated, so peak memory is ~4N^2 bytes
      (kernel, float32) + ~8N^2 bytes (integral image, float64). Recommended
      upper bound N <= 4000 (~1 h at 1 fps): ~64 MB kernel + ~128 MB integral
      image, and a full 19-point lambda scan finishes in a few seconds. With
      the default 1 fps sampling a 5-minute video (N ~= 300) scans the whole
      grid in well under a second.
    - ``auto_select_lambda`` runs the O(N^2) DP once per grid point, but
      ``K``/``P`` are computed only once (they do not depend on lambda_).

Behavior equivalence with the validated baseline is contractual:
    - sampling grid / frame mapping / tail-frame rule;
    - lambda calibration: (normative; the legacy "finest in range"
      branch is unreachable dead code and intentionally not implemented).
"""

from typing import List, Tuple
import time

import numpy as np

from ...comm.log import _Logger

# 19-point log-linear mixed lambda grid (dense over the cliff region 0.05~0.7)
LAMBDA_GRID = (
    1e-4,
    3e-4,
    1e-3,
    3e-3,
    1e-2,
    3e-2,
    5e-2,
    7e-2,
    1e-1,
    1.5e-1,
    2e-1,
    3e-1,
    5e-1,
    7e-1,
    1.0,
    2.0,
    3.0,
    5.0,
    10.0,
)
TIME_EPS = 1e-6  # sampling / mapping tolerance, matches baseline
MIN_BOUNDARY_GAP = 2  # min gap between aligned boundaries (in frames)


def compute_sample_times(duration: float, interval_sec: float) -> List[float]:
    """Sampling grid §4.3-2: t = 0, dt, 2dt, ... < duration - 1e-6, plus a
    tail point max(0, duration - 1e-6); dedup within 1e-6.
    """
    times: List[float] = []
    t = 0.0
    while t < duration - TIME_EPS:
        times.append(t)
        t += interval_sec
    if not times or times[-1] < duration - TIME_EPS:
        times.append(max(0.0, duration - TIME_EPS))
    dedup: List[float] = []
    for x in times:
        if not dedup or abs(x - dedup[-1]) > TIME_EPS:
            dedup.append(x)
    return dedup


def time_to_frame_index(t: float, fps: float, n_frames: int) -> int:
    """§4.3-3: first frame with timestamp >= t - 1e-6 (CFR: ceil), clamped
    to [0, n_frames - 1].
    """
    idx = int(np.ceil((t - TIME_EPS) * fps))
    return max(0, min(idx, n_frames - 1))


def build_diff_sequence(embeds: np.ndarray) -> np.ndarray:
    """Dot-product kernel: diff[i] = 1 - dot(emb_i, emb_{i+1}); requires
    L2-normalized inputs.
    """
    if embeds is None or embeds.shape[0] < 2:
        raise RuntimeError("need at least two frame embeddings to build the diff sequence")
    dots = np.sum(embeds[:-1] * embeds[1:], axis=1)
    return (1.0 - dots).astype(np.float32)


def _precompute_kernel(embeds: np.ndarray) -> Tuple[np.ndarray, int]:
    """Cosine kernel K = E @ E.T plus its float64 integral image P.

    Both depend only on ``embeds`` (not on ``lambda_``), so they are
    computed once and reused by every DP solve during the lambda scan.
    """
    N = embeds.shape[0]
    K = embeds @ embeds.T  # cosine kernel (features already L2-normalized)
    P = np.zeros((N + 1, N + 1), dtype=np.float64)
    P[1:, 1:] = np.cumsum(np.cumsum(K, axis=0), axis=1)
    return P, N


def _dp_solve(P: np.ndarray, N: int, lambda_: float) -> List[int]:
    """O(N^2) vectorized KTS DP over the precomputed integral image P;
    returns split point indices in [1, N-1).
    """
    dp = np.full(N, np.inf)
    prev = np.full(N, -1, dtype=np.int64)
    dp[0] = 0.0
    for j in range(1, N):
        i_arr = np.arange(j, dtype=np.int64)
        length = j - i_arr
        s_block = P[j, j] - P[i_arr, j] - P[j, i_arr] + P[i_arr, i_arr]
        var = length - s_block / length
        np.maximum(var, 0.0, out=var)
        cand = dp[:j] + var + lambda_
        best_i = int(np.argmin(cand))
        dp[j] = cand[best_i]
        prev[j] = best_i

    split_points = []
    ptr = N - 1
    while ptr > 0:
        ptr = int(prev[ptr])
        split_points.append(ptr)
    return sorted({s for s in split_points if 0 < s < N - 1})


def kts_dp_segment_full(embeds: np.ndarray, lambda_: float, quiet: bool = False) -> List[int]:
    """Kernel-matrix KTS with integral-image segment variance, O(n^2)
    vectorized DP; returns split point indices in [1, n-1).
    """
    t0 = time.time()
    P, N = _precompute_kernel(embeds)
    split_points = _dp_solve(P, N, lambda_)
    if not quiet:
        _Logger.info(
            f"kernel KTS DP ({N} frames, lambda={lambda_}): {time.time() - t0:.2f}s, {len(split_points)} splits"
        )
    return split_points


def align_boundaries_to_peaks(
    split_indexes: List[int], diff_seq: np.ndarray, window: int, quiet: bool = False
) -> List[int]:
    """Moves each split to the local diff peak within +/- window; drops
    boundaries closer than 2 frames to the previous one.
    """
    if window <= 0 or not split_indexes or diff_seq is None:
        return split_indexes
    d = diff_seq
    n = len(d)
    aligned: List[int] = []
    for idx in split_indexes:
        lo = max(0, idx - window)
        hi = min(n, idx + window + 1)
        peak = int(np.argmax(d[lo:hi])) + lo
        if aligned and peak - aligned[-1] < MIN_BOUNDARY_GAP:
            continue
        aligned.append(peak)
    if not quiet:
        _Logger.info(f"boundary alignment: {len(split_indexes)} -> {len(aligned)} boundaries (window=±{window}s)")
    return aligned


def splits_to_segments(
    split_indexes: List[int], sample_times: List[float], total_duration: float
) -> List[Tuple[float, float]]:
    """diff[i] sits between sample_times[i] and [i+1] -> boundary time is
    sample_times[i+1]. Segments are gapless and cover [0, duration].
    Valid split indexes are [0, len(sample_times) - 2]; anything else
    raises IndexError instead of corrupting the output silently.
    """
    n = len(sample_times)
    for i in split_indexes:
        if not 0 <= i < n - 1:
            raise IndexError(f"split index {i} out of range [0, {n - 2}]")
    boundaries = [sample_times[i + 1] for i in split_indexes]
    segments = []
    prev_sec = 0.0
    for cur_sec in boundaries:
        if cur_sec > prev_sec + TIME_EPS:
            segments.append((prev_sec, cur_sec))
            prev_sec = cur_sec
    if total_duration > prev_sec + TIME_EPS:
        segments.append((prev_sec, total_duration))
    return segments


def merge_short_segments(
    segments: List[Tuple[float, float]], min_segment_duration: float, quiet: bool = False
) -> List[Tuple[float, float]]:
    """Segments shorter than min_segment_duration merge into the previous
    one; a too-short head segment merges into the next.
    """
    if min_segment_duration <= 0 or len(segments) <= 1:
        return segments
    merged = []
    for start, end in segments:
        if merged and end - start < min_segment_duration:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    if len(merged) >= 2 and merged[0][1] - merged[0][0] < min_segment_duration:
        first_start, _ = merged[0]
        merged = [(first_start, merged[1][1])] + merged[2:]
    if len(merged) == 1 and len(segments) > 1 and not quiet:
        _Logger.warn("all segments merged into one — consider raising lambda_penalty or lowering min_segment_duration")
    return merged


def auto_select_lambda(
    embeds: np.ndarray,
    diff_seq: np.ndarray,
    sample_times: List[float],
    total_duration: float,
    target_segment_duration: float,
    min_segment_duration: float,
    boundary_align_window: int,
) -> float:
    """Scans the 19-point grid with the full evaluation chain (DP ->
    align -> segments -> merge, counting merged segments), groups
    consecutive equal counts, filters to [lo, hi], picks the longest
    plateau (tie: fewer segments) and takes the group's first (smallest)
    lambda. Empty in-range set falls back to closest-to-target (tie:
    fewer segments). The kernel K / integral image P are computed once
    (they do not depend on lambda) and reused by all grid DP solves.
    """
    if total_duration <= 0:
        raise RuntimeError("video duration missing, cannot auto-calibrate lambda")
    n_target = max(1, int(round(total_duration / target_segment_duration)))
    lo = max(1, n_target // 2)
    hi = max(lo + 1, n_target * 2)

    P, N = _precompute_kernel(embeds)
    candidates: List[Tuple[float, int]] = []
    for lam in LAMBDA_GRID:
        splits = _dp_solve(P, N, lam)
        splits = align_boundaries_to_peaks(splits, diff_seq, boundary_align_window, quiet=True)
        segs = splits_to_segments(splits, sample_times, total_duration)
        n_seg = len(merge_short_segments(segs, min_segment_duration, quiet=True))
        candidates.append((lam, n_seg))

    # group consecutive equal segment counts: (run length, count, first index)
    groups = []
    cur_n, cur_start = None, 0
    for i, (_, n) in enumerate(candidates):
        if cur_n is None:
            cur_n, cur_start = n, i
        elif n == cur_n:
            continue
        else:
            groups.append((i - cur_start, cur_n, cur_start))
            cur_n, cur_start = n, i
    groups.append((len(candidates) - cur_start, cur_n, cur_start))

    valid = [g for g in groups if lo <= g[1] <= hi]
    if valid:
        best = max(valid, key=lambda g: (g[0], -g[1]))
        best_lam = candidates[best[2]][0]
        best_n = best[1]
    else:
        best_lam, best_n = min(candidates, key=lambda item: (abs(item[1] - n_target), item[1]))

    _Logger.info(
        f"lambda auto-calibration: {best_lam} "
        f"(target≈{n_target} segments [range {lo}~{hi}], "
        f"{best_n} merged segments, "
        f"avg length≈{total_duration / max(best_n, 1):.0f}s)"
    )
    _Logger.info("  scan: " + ", ".join(f"λ={lam:.0e}→{n}" for lam, n in candidates))
    return best_lam
