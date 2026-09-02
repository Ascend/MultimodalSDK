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
"""Mock-level unit tests for the stateless KTS algorithm functions.

Pure-function tests split out of ``test_kts_segmenter.py`` (PR-split
mapping table §5/PR2): sampling grid / frame-index mapping equivalence
(design doc §4.3) and boundary alignment / segment conversion unit level.
No ``_acc`` extension, video files or embedding services are required.
"""

import numpy as np
import pytest

from mm.core.segmenter import kts_algorithm as ka_mod  # noqa: E402


# --------------------------------------------------------------------------- #
# B. 1 fps sampling equivalence (§4.3)
# --------------------------------------------------------------------------- #
class TestSamplingEquivalence:
    def test_grid_and_tail_point(self):
        # 10.5 s @ 1 fps: [0..10] + tail 10.5-1e-6
        times = ka_mod.compute_sample_times(10.5, 1.0)
        assert times[:11] == [float(i) for i in range(11)]
        assert len(times) == 12
        assert abs(times[-1] - (10.5 - 1e-6)) < 1e-9

    def test_exact_multiple_adds_tail(self):
        # duration=10, dt=1: [0..9] + tail 10-1e-6 (baseline behavior)
        times = ka_mod.compute_sample_times(10.0, 1.0)
        assert len(times) == 11
        assert abs(times[-1] - (10.0 - 1e-6)) < 1e-9

    def test_short_video_single_frame(self):
        # 0.5 s @ 1 fps: t=0 plus the tail point 0.5-1e-6 (baseline semantics)
        times = ka_mod.compute_sample_times(0.5, 1.0)
        assert times == [0.0, pytest.approx(0.5 - 1e-6)]

    def test_interval_two_seconds(self):
        times = ka_mod.compute_sample_times(5.0, 2.0)
        assert times[0] == 0.0
        assert times[1] == 2.0
        assert times[2] == 4.0
        assert len(times) == 4  # + tail 5-1e-6

    def test_dedup_within_eps(self):
        times = ka_mod.compute_sample_times(2.0, 1.0)
        # t=0,1 then t=2 stops the loop (2 >= 2-eps); tail = 2-1e-6
        assert all(abs(b - a) > 1e-6 for a, b in zip(times, times[1:]))

    def test_time_to_frame_index_mapping(self):
        # fps=25, n=2500: first frame with ts >= t - 1e-6 (CFR: ceil)

        def f(t):
            return ka_mod.time_to_frame_index(t, 25.0, 2500)

        assert f(0.0) == 0
        assert f(1.0) == 25  # exact boundary -> frame at t
        assert f(1.04) == 26  # ceil(1.04*25)=26
        assert f(0.001) == 1  # sub-frame -> next frame
        assert f(99.999999) == 2499  # clamped tail
        assert f(1e9) == 2499  # clamp high
        assert f(-1.0) == 0  # clamp low


# --------------------------------------------------------------------------- #
# H. boundary alignment / segment conversion (unit level)
# --------------------------------------------------------------------------- #
class TestBoundaryAlignment:
    # 100 s @ 1 fps sample grid shared by the tests below
    TIMES = [float(i) for i in range(100)] + [100.0 - 1e-6]

    def test_aligns_to_local_peak(self):
        d = np.zeros(101, dtype=np.float32)
        d[49] = 1.0  # diff peak at 49
        aligned = ka_mod.align_boundaries_to_peaks([47], d, window=3)
        assert aligned == [49]  # pulled to the peak within ±3

    def test_min_gap_dedup(self):
        d = np.zeros(101, dtype=np.float32)
        d[50] = 1.0
        d[52] = 0.9
        # both map to peak 50 -> second dropped (< 2 frames apart)
        assert ka_mod.align_boundaries_to_peaks([49, 51], d, window=3) == [50]

    def test_window_zero_disables(self):
        d = np.ones(101, dtype=np.float32)
        assert ka_mod.align_boundaries_to_peaks([47, 60], d, window=0) == [47, 60]

    def test_splits_to_segments_boundary_time(self):
        # split i sits between sample i and i+1 -> boundary = times[i+1]
        segs = ka_mod.splits_to_segments([49], self.TIMES, 100.0)
        assert segs == [(0.0, 50.0), (50.0, 100.0)]

    def test_merge_short_segments(self):
        # (80, 88) is short -> merges into previous; head (0, 5) is short -> merges next
        segs = ka_mod.merge_short_segments([(0.0, 5.0), (5.0, 80.0), (80.0, 88.0), (88.0, 100.0)], 10.0)
        assert segs == [(0.0, 88.0), (88.0, 100.0)]

    def test_merge_head_into_next(self):
        segs = ka_mod.merge_short_segments([(0.0, 4.0), (4.0, 60.0), (60.0, 100.0)], 10.0)
        assert segs == [(0.0, 60.0), (60.0, 100.0)]


# --------------------------------------------------------------------------- #
# I. kernel KTS DP / lambda auto-calibration (core algorithm, constructive)
# --------------------------------------------------------------------------- #
def _three_block_embeds() -> np.ndarray:
    """30 frames = 3 homogeneous blocks of 10 (one-hot rows): the cosine
    kernel is perfectly separable (within-block dot = 1, cross-block = 0),
    so the unique zero-variance optimum splits at both block ends.
    """
    seg_a = np.eye(4)[np.zeros(10, dtype=int)]
    seg_b = np.eye(4)[np.ones(10, dtype=int)]
    seg_c = np.eye(4)[np.full(10, 2)]
    return np.vstack([seg_a, seg_b, seg_c]).astype(np.float32)


class TestKtsDP:
    def test_dp_known_splits(self):
        # 3 blocks of 10 -> splits at the block ends (indices 10 and 20),
        # each segment has zero variance -> unique cost-3 optimum for lambda=1
        embeds = _three_block_embeds()
        splits = ka_mod.kts_dp_segment_full(embeds, lambda_=1.0, quiet=True)
        assert splits == [10, 20]

    def test_dp_no_split_for_huge_lambda(self):
        # lambda -> inf: one segment always wins
        embeds = _three_block_embeds()
        assert ka_mod.kts_dp_segment_full(embeds, lambda_=1e9, quiet=True) == []


class TestAutoSelectLambda:
    def test_recovers_three_segments(self):
        # auto-calibration must pick a lambda whose full chain yields the
        # 3 known blocks (30 s @ 1 fps, target 10 s per segment)
        embeds = _three_block_embeds()
        times = ka_mod.compute_sample_times(30.0, 1.0)
        diff = ka_mod.build_diff_sequence(embeds)
        lam = ka_mod.auto_select_lambda(
            embeds,
            diff,
            times,
            30.0,
            target_segment_duration=10.0,
            min_segment_duration=5.0,
            boundary_align_window=0,
        )
        splits = ka_mod.kts_dp_segment_full(embeds, lam, quiet=True)
        segs = ka_mod.splits_to_segments(splits, times, 30.0)
        segs = ka_mod.merge_short_segments(segs, 5.0, quiet=True)
        assert len(segs) == 3
