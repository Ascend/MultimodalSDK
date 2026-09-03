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
"""Mock-level unit tests for KtsSegmenter (PR4 subset).

All video / embedding dependencies are faked: ``video_info`` /
``video_decode`` are monkeypatched inside the kts_segmenter module, the
embedding backend is a deterministic in-process fake whose image vectors
depend on the global frame index (recovered from the f000049.jpg file name),
so results are independent of chunk/batch ordering.

NOTE (PR4 form): the mm.acc pre-stub block lives in test/conftest.py
(shared by all KTS test modules); pure-function test cases for
kts_algorithm / kts_cache live in their own modules.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from mm.core.segmenter import kts_algorithm as ka_mod  # noqa: E402
from mm.core.segmenter import kts_segmenter as ks_mod  # noqa: E402
from mm.core.segmenter.embedding_backend import EmbeddingBackend  # noqa: E402
from mm.core.segmenter.kts_segmenter import KtsSegmenter, SegmentResult, VideoSegment  # noqa: E402

LAMBDA_GRID = ka_mod.LAMBDA_GRID


# --------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------- #
class FakePillowImage:
    def __init__(self, payload):
        self.payload = payload

    # `format` mirrors the PIL save() signature
    # pylint: disable=redefined-builtin
    def save(self, path, format=None, quality=None):  # noqa: A002
        Path(path).write_bytes(self.payload)


class FakeDecodedImage:
    def __init__(self, frame_idx):
        self.frame_idx = frame_idx

    def pillow(self):
        return FakePillowImage(f"jpeg-frame-{self.frame_idx}".encode())


class FakeBackend(EmbeddingBackend):
    """Deterministic backend: frame i < split_at -> unit vector e0, else e1;
    text containing 'A' -> e0, otherwise e1.
    """

    def __init__(self, split_at=50, dim=8, backend_id="fake:v1"):
        self.split_at = split_at
        self.dim = dim
        self._backend_id = backend_id
        self.image_calls = 0
        self.text_calls = 0
        self.closed = False

    @property
    def backend_id(self):
        return self._backend_id

    def encode_images(self, image_paths):
        self.image_calls += 1
        vecs = np.zeros((len(image_paths), self.dim), dtype=np.float32)
        for i, p in enumerate(image_paths):
            frame_idx = int(Path(p).stem[1:])  # f000049.jpg -> 49
            vecs[i, 0 if frame_idx < self.split_at else 1] = 1.0
        return vecs

    def encode_text(self, text):
        self.text_calls += 1
        v = np.zeros(self.dim, dtype=np.float32)
        v[0 if "A" in text else 1] = 1.0
        return v

    def close(self):
        self.closed = True


def make_video_info(fps, n_frames, duration):
    def _info(path):
        return {"fps": fps, "n_frames": n_frames, "duration_sec": duration}

    return _info


class DecodeRecorder:
    def __init__(self, n_frames):
        self.n_frames = n_frames
        self.calls = []  # list of sorted frame-index lists

    def __call__(self, path, device, frame_indices=None, sample_num=-1):
        assert device == "cpu"
        idx = sorted(frame_indices)
        assert all(0 <= i < self.n_frames for i in idx)
        self.calls.append(idx)
        return [FakeDecodedImage(i) for i in idx]


@pytest.fixture(name="fake_video")
def _fake_video_factory(tmp_path):
    p = tmp_path / "video.mp4"
    p.write_bytes(b"fake-mp4-content")
    # Linux umask 022 yields 0o644; SDK baseline rejects video files >0o640
    # (Windows skips the POSIX check, so chmod is a no-op there).
    os.chmod(p, 0o640)
    return str(p)


@pytest.fixture(name="wired")
def _wired_factory(fake_video, monkeypatch):
    """Standard wiring: 100 s video @25 fps, block embeddings split at frame 50."""
    info = make_video_info(fps=25.0, n_frames=2500, duration=100.0)
    decoder = DecodeRecorder(n_frames=2500)
    monkeypatch.setattr(ks_mod, "video_info", info)
    monkeypatch.setattr(ks_mod, "video_decode", decoder)
    backend = FakeBackend(split_at=50)
    return {"path": fake_video, "info": info, "decoder": decoder, "backend": backend}


def make_segmenter(wired, **kwargs):
    defaults = dict(embed_backend=wired["backend"])
    defaults.update(kwargs)
    return KtsSegmenter(wired["path"], **defaults)


# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# A. constructor validation
# --------------------------------------------------------------------------- #
class TestConstructorValidation:
    def test_video_path_type_and_emptiness(self):
        with pytest.raises(TypeError):
            KtsSegmenter(123, embed_backend=FakeBackend())
        with pytest.raises(TypeError):
            KtsSegmenter("", embed_backend=FakeBackend())

    def test_video_path_suffix(self, tmp_path):
        p = tmp_path / "video.avi"
        p.write_bytes(b"x")
        with pytest.raises(ValueError):
            KtsSegmenter(str(p), embed_backend=FakeBackend())

    def test_video_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            KtsSegmenter(str(tmp_path / "nope.mp4"), embed_backend=FakeBackend())

    def test_backend_or_base_url_required(self, fake_video):
        with pytest.raises(ValueError):
            KtsSegmenter(fake_video)  # neither backend nor base_url

    def test_base_url_format(self, fake_video):
        for bad in ("localhost:8000/v1", "http://host:8000/api", "ftp://x/v1"):
            with pytest.raises(ValueError):
                KtsSegmenter(fake_video, embed_base_url=bad)

    def test_embed_backend_type(self, fake_video):
        with pytest.raises(TypeError):
            KtsSegmenter(fake_video, embed_backend="not-a-backend")

    def test_numeric_ranges(self, fake_video):
        bad_cases = [
            dict(sample_interval_sec=0),
            dict(sample_interval_sec=-1),
            dict(sample_interval_sec=3601),
            dict(target_segment_duration=0),
            dict(target_segment_duration=86401),
            dict(min_segment_duration=-1),
            dict(min_segment_duration=100),
            dict(lambda_penalty=0),
            dict(lambda_penalty=-1e-3),
            dict(boundary_align_window=-1),
            dict(boundary_align_window=11),
            dict(jpeg_quality=49),
            dict(jpeg_quality=101),
            dict(batch_size=0),
            dict(embed_workers=0),
            dict(embed_timeout=0),
        ]
        for kwargs in bad_cases:
            with pytest.raises(ValueError):
                KtsSegmenter(fake_video, embed_backend=FakeBackend(), **kwargs)

    def test_bool_type_params(self, fake_video):
        for kwargs in (dict(use_cache=1), dict(keep_frames="yes")):
            with pytest.raises(TypeError):
                KtsSegmenter(fake_video, embed_backend=FakeBackend(), **kwargs)

    def test_cache_dir_missing_and_unwritable(self, fake_video, tmp_path, monkeypatch):
        with pytest.raises(FileNotFoundError):
            KtsSegmenter(
                fake_video,
                embed_backend=FakeBackend(),
                cache_dir=str(tmp_path / "nope"),
            )
        d = tmp_path / "ro"
        d.mkdir()
        monkeypatch.setattr(os, "access", lambda p, m: False)
        with pytest.raises(PermissionError):
            KtsSegmenter(fake_video, embed_backend=FakeBackend(), cache_dir=str(d))

    def test_progress_callback_not_callable(self, fake_video):
        with pytest.raises(TypeError):
            KtsSegmenter(fake_video, embed_backend=FakeBackend(), progress_callback=42)

    def test_posix_owner_and_permission(self, fake_video, monkeypatch):
        real_stat = os.stat

        class St:
            st_mode = 0o100644
            st_uid = 1000

        monkeypatch.setattr(os, "name", "posix")
        monkeypatch.setattr(os, "getuid", lambda: 1000, raising=False)
        monkeypatch.setattr(os, "stat", lambda p: St())
        # owner mismatch
        monkeypatch.setattr(os, "getuid", lambda: 2000, raising=False)
        with pytest.raises(PermissionError):
            KtsSegmenter(fake_video, embed_backend=FakeBackend())
        # permission 0644 > 0640 (other has read)
        monkeypatch.setattr(os, "getuid", lambda: 1000, raising=False)
        with pytest.raises(PermissionError):
            KtsSegmenter(fake_video, embed_backend=FakeBackend())
        # 0640 itself is fine
        St.st_mode = 0o100640
        KtsSegmenter(fake_video, embed_backend=FakeBackend())
        # 0600 (subset) is fine
        St.st_mode = 0o100600
        KtsSegmenter(fake_video, embed_backend=FakeBackend())
        assert real_stat(fake_video).st_size > 0

    def test_symlink_rejected(self, tmp_path):
        target = tmp_path / "real.mp4"
        target.write_bytes(b"x")
        link = tmp_path / "link.mp4"
        try:
            link.symlink_to(target)
        except OSError:  # privilege-restricted on Windows
            pytest.skip("symlink creation not permitted")
        with pytest.raises(PermissionError):
            KtsSegmenter(str(link), embed_backend=FakeBackend())


# --------------------------------------------------------------------------- #
# B. 1 fps sampling equivalence — pipeline-level cases
# (pure grid / mapping cases live in test_kts_algorithm.py)
# --------------------------------------------------------------------------- #
class TestSamplingEquivalence:
    def test_full_flow_writes_one_jpeg_per_sample_point(self, wired):
        seg = make_segmenter(wired, lambda_penalty=0.5)
        seg.segment()
        n = len(seg.sample_times)
        frame_dir = Path(seg.sample_frame_paths[0]).parent
        files = sorted(frame_dir.glob("*.jpg"))
        assert len(files) == n
        assert [p.name for p in files] == [f"f{i:06d}.jpg" for i in range(n)]

    def test_tail_frame_duplicated(self, wired, monkeypatch):
        # duration 2.5 s @ 2 fps (5 frames): times [0, 1, 2, 2.5-1e-6]
        #   -> mapped [0, 2, 4, 4]: the last sample duplicates frame 4's JPEG
        monkeypatch.setattr(ks_mod, "video_info", make_video_info(fps=2.0, n_frames=5, duration=2.5))
        wired["decoder"].n_frames = 5
        seg = make_segmenter(wired, lambda_penalty=0.5)
        seg.segment()
        paths = seg.sample_frame_paths
        assert len(paths) == 4
        assert Path(paths[2]).read_bytes() == Path(paths[3]).read_bytes()
        assert Path(paths[0]).read_bytes() != Path(paths[2]).read_bytes()
        assert wired["decoder"].calls == [[0, 2, 4]]  # frame 4 decoded once

    def test_decode_batches_of_64(self, wired):
        seg = make_segmenter(wired, lambda_penalty=0.5)
        seg.segment()
        # 101 unique frame indices (100 s @ 1 fps) -> 64 + 37
        assert len(wired["decoder"].calls) == 2
        assert len(wired["decoder"].calls[0]) == 64
        assert len(wired["decoder"].calls[1]) == 37
        all_idx = sorted(i for c in wired["decoder"].calls for i in c)
        assert all_idx == sorted(set(all_idx))  # each target decoded exactly once


# --------------------------------------------------------------------------- #
# C. lambda auto-calibration
# --------------------------------------------------------------------------- #
class TestLambdaCalibration:
    def _seg_with_table(
        self,
        wired,
        monkeypatch,
        seg_count_by_lambda,
        duration=600.0,
        target=60.0,
        **kwargs,
    ):
        """Patches the DP so lambda L yields len(splits) = seg_count-1."""
        info = make_video_info(fps=25.0, n_frames=int(duration * 25), duration=duration)
        monkeypatch.setattr(ks_mod, "video_info", info)
        wired["decoder"].n_frames = int(duration * 25)

        def fake_dp(P, N, lambda_):  # pylint: disable=invalid-name
            n = seg_count_by_lambda[lambda_]
            return list(range(1, n))  # n-1 splits -> n segments after conversion

        # Patch the shared low-level solver, not kts_dp_segment_full:
        # auto_select_lambda calls _dp_solve directly (the kernel matrix is
        # precomputed once and reused across the whole grid), and
        # kts_dp_segment_full delegates to _dp_solve too, so this single
        # patch covers both the calibration scan and the final DP.
        monkeypatch.setattr(ka_mod, "_dp_solve", fake_dp)
        return make_segmenter(
            wired,
            target_segment_duration=target,
            min_segment_duration=0.0,
            boundary_align_window=0,
            **kwargs,
        )

    def test_longest_plateau_first_lambda(self, wired, monkeypatch):
        # target 60 s / 600 s -> n_target=10, range [5, 20]
        # plateau of 8-segment lambdas (4 wide) beats the 15-segment one (2 wide)
        counts = [3, 3, 3, 8, 8, 8, 8, 15, 15, 25, 30, 30, 40, 40, 40, 40, 40, 40, 40]
        table = dict(zip(LAMBDA_GRID, counts))
        seg = self._seg_with_table(wired, monkeypatch, table)
        result = seg.segment()
        assert result.lambda_penalty == LAMBDA_GRID[3]
        assert len(result.segments) == 8

    def test_plateau_tie_fewer_segments(self, wired, monkeypatch):
        # two in-range plateaus of equal width -> fewer segments wins
        counts = [
            5,
            5,
            5,
            5,
            12,
            12,
            12,
            12,
            25,
            25,
            25,
            25,
            25,
            25,
            25,
            25,
            25,
            25,
            25,
        ]
        table = dict(zip(LAMBDA_GRID, counts))
        seg = self._seg_with_table(wired, monkeypatch, table)
        result = seg.segment()
        assert result.lambda_penalty == LAMBDA_GRID[0]
        assert len(result.segments) == 5

    def test_empty_in_range_falls_back_to_closest(self, wired, monkeypatch):
        # all counts out of range [5, 20]; |25-10|=15 is closest (|30-10|=20, |40-10|=30)
        counts = [30] * 5 + [25] * 6 + [40] * 8
        table = dict(zip(LAMBDA_GRID, counts))
        seg = self._seg_with_table(wired, monkeypatch, table)
        result = seg.segment()
        assert result.lambda_penalty == LAMBDA_GRID[5]
        assert len(result.segments) == 25

    def test_explicit_lambda_skips_calibration(self, wired, monkeypatch):
        calls = {"n": 0}

        def fake_dp(embeds, lambda_, quiet=False):
            calls["n"] += 1
            return [len(embeds) // 2]

        monkeypatch.setattr(ka_mod, "kts_dp_segment_full", fake_dp)
        seg = make_segmenter(wired, lambda_penalty=0.5)
        seg.segment()
        assert calls["n"] == 1  # single DP run, no 19-point scan
        assert seg.lambda_penalty == 0.5


# --------------------------------------------------------------------------- #
# D. segmentation pipeline
# --------------------------------------------------------------------------- #
class TestSegmentPipeline:
    def test_full_pipeline_two_blocks(self, wired):
        seg = make_segmenter(wired, lambda_penalty=0.5)
        result = seg.segment()

        assert isinstance(result, SegmentResult)
        assert result.video_duration_sec == 100.0
        assert result.fps == 25.0
        assert result.n_sample_frames == 101
        assert result.lambda_penalty == 0.5

        # block boundary at t=50 -> two segments [0, 50), [50, 100]
        assert len(result.segments) == 2
        s1, s2 = result.segments
        assert s1.start_sec == 0.0 and s1.end_sec == 50.0
        assert s2.start_sec == 50.0 and s2.end_sec == 100.0
        assert s1.n_frames == 50 and s2.n_frames == 51  # tail frame belongs to s2
        assert s1.frame_indices == list(range(50))
        assert s2.frame_indices == list(range(50, 101))

        # stats diagnostics
        stats = result.stats
        for key in ("diff_min", "diff_max", "diff_mean", "diff_std"):
            assert key in stats
        timing = stats["timing"]
        for key in ("decode_s", "embed_s", "kts_dp_s", "total_s"):
            assert key in timing

    def test_segments_are_time_ordered_and_complete(self, wired):
        seg = make_segmenter(wired, lambda_penalty=0.5)
        result = seg.segment()
        segs = result.segments
        assert segs[0].start_sec == 0.0
        assert segs[-1].end_sec == pytest.approx(result.video_duration_sec)
        for a, b in zip(segs, segs[1:]):
            assert a.end_sec == pytest.approx(b.start_sec)
        assert sum(s.n_frames for s in segs) == result.n_sample_frames

    def test_segment_idempotent(self, wired):
        seg = make_segmenter(wired, lambda_penalty=0.5)
        r1 = seg.segment()
        decode_calls_before = len(wired["decoder"].calls)
        r2 = seg.segment()
        assert r1 is r2
        assert len(wired["decoder"].calls) == decode_calls_before
        assert wired["backend"].image_calls == 2  # 101 frames / 64-batch -> 2 chunks

    def test_progress_callback(self, wired):
        events = []

        def cb(done, total):
            events.append((done, total))
            if done == total:
                raise RuntimeError("boom on last batch")  # must be swallowed

        seg = make_segmenter(
            wired,
            lambda_penalty=0.5,
            batch_size=50,
            embed_workers=1,
            progress_callback=cb,
        )
        seg.segment()
        assert events == [(50, 101), (100, 101), (101, 101)]
        assert all(t == 101 for _, t in events)

    def test_embedding_chunking_with_workers(self, wired):
        seg = make_segmenter(wired, lambda_penalty=0.5, batch_size=30, embed_workers=3)
        seg.segment()
        # 101 frames / 30 per chunk -> 4 chunks; each backend call is one chunk
        assert wired["backend"].image_calls == 4
        assert seg.embeds.shape == (101, 8)

    def test_backend_failure_propagates(self, wired):
        def boom(paths):
            raise RuntimeError("service down")

        wired["backend"].encode_images = boom
        seg = make_segmenter(wired, lambda_penalty=0.5)
        with pytest.raises(RuntimeError, match="service down"):
            seg.segment()

    def test_invalid_video_metadata(self, wired, monkeypatch):
        monkeypatch.setattr(ks_mod, "video_info", make_video_info(fps=0.0, n_frames=0, duration=0.0))
        seg = make_segmenter(wired, lambda_penalty=0.5)
        with pytest.raises(RuntimeError, match="invalid video metadata"):
            seg.segment()


# --------------------------------------------------------------------------- #
# select_frames total-cap branch (regression)
# --------------------------------------------------------------------------- #
class TestSelectFramesCap:
    def test_total_cap_trims_across_segments(self, wired):
        """Regression: with len(idx) > max_frames the trim must map the evenly
        spaced positions back to the real sample indices of the given segments
        (the old code returned the video-head frames paths[0..max_frames-1]).
        """
        seg = make_segmenter(wired, lambda_penalty=0.5)
        seg.segment()  # 101 sample frames (t = 0..100) exist in the cache dir

        # 30 segments over [20, 100): m=30 > max_frames=24 -> k_eff = 1 and
        # one candidate per segment, so len(idx) = 30 > 24 hits the cap branch
        bounds = np.linspace(20.0, 100.0, 31)
        times = seg.sample_times
        segments = []
        candidates = []
        for lo, hi in zip(bounds, bounds[1:]):
            in_seg = [i for i, t in enumerate(times) if lo <= t < hi]
            assert in_seg  # every segment must hold at least one frame
            segments.append(
                VideoSegment(
                    start_sec=float(lo),
                    end_sec=float(hi),
                    n_frames=len(in_seg),
                    frame_indices=in_seg,
                )
            )
            candidates.append(in_seg[0])

        picked = seg.select_frames(segments, seg_frames=4, max_frames=24)
        picked_idx = sorted(int(Path(p).stem[1:]) for p in picked)

        # 24 frames evenly drawn from the 30 candidates, all inside [20, 100);
        # a video-head trim (positions 0..23) fails the range check below
        keep = np.linspace(0, len(candidates) - 1, 24, dtype=int)
        assert picked_idx == sorted(candidates[int(j)] for j in keep)
        assert len(picked_idx) == 24
        assert picked_idx[0] >= 20


# E. caching
# --------------------------------------------------------------------------- #
class TestCaching:
    def test_cache_roundtrip(self, wired):
        seg1 = make_segmenter(wired, lambda_penalty=0.5)
        seg1.segment()
        assert wired["backend"].image_calls == 2

        backend2 = FakeBackend(split_at=50)  # same backend_id
        seg2 = KtsSegmenter(wired["path"], embed_backend=backend2, lambda_penalty=0.5, use_cache=True)
        result = seg2.segment()
        assert backend2.image_calls == 0  # embeddings from cache
        assert len(wired["decoder"].calls) == 2  # no extra decode
        assert len(result.segments) == 2
        assert len(seg2.sample_frame_paths) == 101  # frame files reused
        assert all(Path(p).is_file() for p in seg2.sample_frame_paths)

    def test_cache_key_contains_backend_id(self, wired):
        seg1 = make_segmenter(wired, lambda_penalty=0.5)
        seg1.segment()
        key1 = seg1._cache.key

        backend2 = FakeBackend(split_at=50, backend_id="fake:v2")
        seg2 = KtsSegmenter(wired["path"], embed_backend=backend2, lambda_penalty=0.5)
        assert seg2._cache.key != key1
        seg2.segment()
        assert backend2.image_calls == 2  # cache miss -> recompute

    def test_use_cache_false_recomputes(self, wired):
        seg1 = make_segmenter(wired, lambda_penalty=0.5)
        seg1.segment()
        backend2 = FakeBackend(split_at=50)
        seg2 = KtsSegmenter(wired["path"], embed_backend=backend2, lambda_penalty=0.5, use_cache=False)
        seg2.segment()
        assert backend2.image_calls == 2

    def test_keep_frames_false_cleans_files(self, wired):
        seg = make_segmenter(wired, lambda_penalty=0.5, keep_frames=False)
        result = seg.segment()
        assert len(result.segments) == 2
        assert seg.sample_frame_paths == []
        frame_dirs = list(Path(wired["path"]).parent.glob(".kts_cache/frames_*"))
        leftover = [p for d in frame_dirs for p in d.glob("*.jpg")]
        assert frame_dirs and not leftover

    def test_keep_frames_false_blocks_select_frames(self, wired):
        seg = make_segmenter(wired, lambda_penalty=0.5, keep_frames=False)
        seg.segment()
        with pytest.raises(RuntimeError, match="frame files are unavailable"):
            seg.select_frames(seg._result.segments)


# --------------------------------------------------------------------------- #
# F. retrieve / select_frames
# --------------------------------------------------------------------------- #
class TestRetrieve:
    def test_requires_segment_first(self, wired):
        seg = make_segmenter(wired, lambda_penalty=0.5)
        with pytest.raises(RuntimeError, match="segment\\(\\) must be called"):
            seg.retrieve("query")

    def test_retrieval_ranking_and_time_order(self, wired):
        seg = make_segmenter(wired, lambda_penalty=0.5)
        seg.segment()
        # "A" -> e0 -> segment 1 (frames < 50) ranks first; result is time-ordered
        hits = seg.retrieve("A question", top_k=2)
        assert len(hits) == 2
        assert hits[0].start_sec == 0.0  # best match, but time order preserved
        assert hits[1].start_sec == 50.0
        hits_b = seg.retrieve("B question", top_k=1)  # -> e1 -> segment 2
        assert hits_b[0].start_sec == 50.0
        assert wired["backend"].text_calls == 2

    def test_top_k_above_segment_count_returns_all(self, wired):
        seg = make_segmenter(wired, lambda_penalty=0.5)
        seg.segment()
        hits = seg.retrieve("A", top_k=99)
        assert len(hits) == 2

    def test_query_validation(self, wired):
        seg = make_segmenter(wired, lambda_penalty=0.5)
        seg.segment()
        with pytest.raises(TypeError):
            seg.retrieve(123)
        with pytest.raises(ValueError):
            seg.retrieve("")
        with pytest.raises(ValueError):
            seg.retrieve("q", top_k=0)
        with pytest.raises(ValueError):
            seg.retrieve("q", top_k=True)  # bool is not an int here
