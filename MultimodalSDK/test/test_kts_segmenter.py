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
kts_algorithm / kts_cache live in their own modules. Only the wired
pipeline-level cases below ship with PR4. The A/B/C/E/F groups were
deferred to PR5 (test completion).
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
