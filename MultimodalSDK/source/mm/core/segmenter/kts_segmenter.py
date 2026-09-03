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
"""Kernel Temporal Segmentation (KTS) based video semantic segmenter.

``KtsSegmenter`` splits a long video into semantically coherent segments
without any query (contrast to ``mm.core.frame_selector`` which is
query-driven). It is the **orchestrator** of the pipeline:

    1 fps time sampling via ``mm.video_decode`` -> remote frame embedding
    (L2-normalized, cached) -> kernel-matrix KTS dynamic programming with
    automatic lambda calibration -> boundary alignment to local diff peaks
    -> short-segment merging. Optional query-based segment retrieval and
    per-segment frame selection feed a downstream VLM.

The stateless algorithm steps live in ``kts_algorithm`` (pure functions) and
the on-disk analysis cache in ``kts_cache.KtsCache``; this module owns
validation, decoding, embedding orchestration and the public API.

Behavior equivalence with the validated baseline implementation is
contractual: sampling grid / frame-index mapping / tail-frame duplication
follow design doc §4.3; the lambda calibration follows §3.1.1.
"""

import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from ...acc.wrapper.video_wrapper import video_decode, video_info
from ...comm.log import _Logger
from . import kts_algorithm
from .embedding_backend import EmbeddingBackend, OpenAIEmbeddingBackend
from .kts_cache import KtsCache

_DECODE_BATCH_SIZE = 64  # target frames per video_decode call (memory bound)
_FILE_MODE_MAX = 0o640  # SDK file security baseline (<= 0640, per-group)


@dataclass
class VideoSegment:
    """One semantic segment of the video."""

    start_sec: float  # segment start time (seconds)
    end_sec: float  # segment end time (seconds)
    n_frames: int  # number of sampled (1 fps) frames inside
    frame_indices: List[int]  # ascending sample-sequence indices of those frames


@dataclass
class SegmentResult:
    """Full result of KtsSegmenter.segment()."""

    segments: List[VideoSegment]  # time-ascending, no gap/overlap, covers [0, duration]
    video_duration_sec: float
    fps: float
    n_sample_frames: int  # total sampled frames (incl. duplicated tail frame)
    lambda_penalty: float  # effective lambda (calibrated or user-specified)
    stats: Dict[str, object] = field(default_factory=dict)  # diagnostics, see design doc §6.2


class KtsSegmenter:
    """Query-free semantic video segmenter based on kernel temporal segmentation.

    Lifecycle: ``__init__`` (validation only) -> ``segment()`` (decode +
    embedding + KTS, idempotent) -> ``retrieve()`` / ``select_frames()`` ->
    ``close()``.
    """

    def __init__(
        self,
        video_path: str,
        embed_backend: Optional[EmbeddingBackend] = None,
        embed_base_url: Optional[str] = None,
        embed_model_name: str = "Qwen3-VL-Embedding-2B",
        sample_interval_sec: float = 1.0,
        target_segment_duration: float = 60.0,
        min_segment_duration: float = 10.0,
        lambda_penalty: Optional[float] = None,
        boundary_align_window: int = 3,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        keep_frames: bool = True,
        jpeg_quality: int = 80,
        batch_size: int = 64,
        embed_workers: int = 4,
        embed_timeout: float = 60.0,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ):
        """Initializes the segmenter (no network access, no decoding).

        Args:
            video_path: MP4 video file path; validated against the SDK file
                security baseline (existing regular non-symlink file, owned by
                the current user, permission <= 0640, mp4/MP4 suffix).
            embed_backend: pre-built embedding backend; when given,
                embed_base_url/embed_model_name/embed_timeout are ignored.
            embed_base_url: OpenAI-compatible embedding service address
                (http(s)://ip:port/v1); required when embed_backend is None.
            embed_model_name: served model name of the embedding service.
            sample_interval_sec: time sampling interval in seconds (1.0 = 1 fps,
                the paper baseline).
            target_segment_duration: target average segment length in seconds,
                drives automatic lambda calibration.
            min_segment_duration: minimum segment length in seconds; shorter
                segments are merged into the previous one.
            lambda_penalty: KTS cost per cut; None = automatic calibration.
            boundary_align_window: boundary alignment window in seconds
                (each cut moves to the local diff peak within +/- window);
                0 disables alignment.
            cache_dir: cache directory for frame JPEGs and embeddings;
                defaults to <video dir>/.kts_cache.
            use_cache: whether to reuse the on-disk cache.
            keep_frames: keep frame JPEGs after embedding (needed by
                select_frames); False deletes them after embedding.
            jpeg_quality: JPEG quality for sampled frames (50-100); 80 is the
                validated baseline value.
            batch_size: frames per embedding batch (memory control).
            embed_workers: parallel workers calling the backend (validated
                optimum is 4; each worker issues up to 32 concurrent requests
                inside the backend).
            embed_timeout: per-request timeout in seconds; passed to a
                self-built OpenAIEmbeddingBackend (ignored when embed_backend
                is injected).
            progress_callback: called as (done_frames, total_frames) after each
                embedding batch; exceptions are logged and swallowed.
        """
        self._validate_video_path(video_path)
        if embed_backend is not None and not isinstance(embed_backend, EmbeddingBackend):
            raise TypeError("embed_backend must be an EmbeddingBackend instance")
        if embed_backend is None:
            if not isinstance(embed_base_url, str) or not embed_base_url:
                raise ValueError("embed_base_url is required when embed_backend is None")
            if not (embed_base_url.startswith("http://") or embed_base_url.startswith("https://")):
                raise ValueError(f"embed_base_url must start with http:// or https://: {embed_base_url}")
            if not embed_base_url.endswith("/v1"):
                raise ValueError(f"embed_base_url must end with /v1: {embed_base_url}")
        if not isinstance(embed_model_name, str) or not embed_model_name:
            raise ValueError("embed_model_name must be a non-empty string")
        if (
            isinstance(sample_interval_sec, bool)
            or not isinstance(sample_interval_sec, (int, float))
            or not (0 < sample_interval_sec <= 3600)
        ):
            raise ValueError("sample_interval_sec must be in (0, 3600]")
        if (
            isinstance(target_segment_duration, bool)
            or not isinstance(target_segment_duration, (int, float))
            or not (0 < target_segment_duration <= 86400)
        ):
            raise ValueError("target_segment_duration must be in (0, 86400]")
        if (
            isinstance(min_segment_duration, bool)
            or not isinstance(min_segment_duration, (int, float))
            or not (0 <= min_segment_duration <= target_segment_duration)
        ):
            raise ValueError("min_segment_duration must be in [0, target_segment_duration]")
        if lambda_penalty is not None:
            if isinstance(lambda_penalty, bool) or not isinstance(lambda_penalty, (int, float)) or lambda_penalty <= 0:
                raise ValueError("lambda_penalty must be None or a positive number")
        if (
            isinstance(boundary_align_window, bool)
            or not isinstance(boundary_align_window, int)
            or not (0 <= boundary_align_window <= 10)
        ):
            raise ValueError("boundary_align_window must be an int in [0, 10]")
        if cache_dir is not None:
            if not isinstance(cache_dir, str) or not cache_dir:
                raise TypeError("cache_dir must be a non-empty str or None")
            if not os.path.isdir(cache_dir):
                raise FileNotFoundError(f"cache_dir does not exist: {cache_dir}")
            if not os.access(cache_dir, os.W_OK):
                raise PermissionError(f"cache_dir is not writable: {cache_dir}")
        if not isinstance(use_cache, bool):
            raise TypeError("use_cache must be bool")
        if not isinstance(keep_frames, bool):
            raise TypeError("keep_frames must be bool")
        if isinstance(jpeg_quality, bool) or not isinstance(jpeg_quality, int) or not (50 <= jpeg_quality <= 100):
            raise ValueError("jpeg_quality must be an int in [50, 100]")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if isinstance(embed_workers, bool) or not isinstance(embed_workers, int) or embed_workers < 1:
            raise ValueError("embed_workers must be a positive integer")
        if isinstance(embed_timeout, bool) or not isinstance(embed_timeout, (int, float)) or embed_timeout <= 0:
            raise ValueError("embed_timeout must be a positive number")
        if progress_callback is not None and not callable(progress_callback):
            raise TypeError("progress_callback must be callable")

        self.video_path = str(video_path)
        self.sample_interval_sec = float(sample_interval_sec)
        self.target_segment_duration = float(target_segment_duration)
        self.min_segment_duration = float(min_segment_duration)
        self.lambda_penalty = lambda_penalty
        self.boundary_align_window = boundary_align_window
        self.use_cache = use_cache
        self.keep_frames = keep_frames
        self.jpeg_quality = jpeg_quality
        self.batch_size = batch_size
        self.embed_workers = embed_workers
        self.progress_callback = progress_callback

        if embed_backend is not None:
            self.embed_backend = embed_backend
            self._owns_backend = False
        else:
            self.embed_backend = OpenAIEmbeddingBackend(
                base_url=embed_base_url,
                model_name=embed_model_name,
                timeout=float(embed_timeout),
            )
            self._owns_backend = True

        self._cache = KtsCache(
            self.video_path,
            backend_id=self.embed_backend.backend_id,
            sample_interval_sec=self.sample_interval_sec,
            cache_dir=cache_dir,
            use_cache=use_cache,
        )

        # runtime state
        self.fps: float = 0.0
        self.total_duration: float = 0.0
        self.n_video_frames: int = 0
        self.sample_times: List[float] = []
        self.sample_frame_paths: List[str] = []
        self.embeds: Optional[np.ndarray] = None
        self.diff_seq: Optional[np.ndarray] = None
        self._segment_vecs: List[Optional[np.ndarray]] = []
        self._result: Optional[SegmentResult] = None
        self._closed = False
        self._segmented = False

        _Logger.info(
            f"KtsSegmenter initialized: video={self.video_path}, "
            f"interval={self.sample_interval_sec}s, backend={self.embed_backend.backend_id}"
        )

    # ------------------------------------------------------------------ #
    # input validation (SDK file security baseline)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _validate_video_path(video_path):
        if isinstance(video_path, bytes):
            video_path = video_path.decode("utf-8")
        if not isinstance(video_path, str) or not video_path:
            raise TypeError("video_path must be a non-empty str")
        if "\x00" in video_path:
            raise ValueError("video_path must not contain null characters")
        if not video_path.lower().endswith((".mp4",)):
            raise ValueError(f"video_path must be an mp4 file: {video_path}")
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"video file not found: {video_path}")
        if os.path.islink(video_path):
            raise PermissionError(f"video file must not be a symbolic link: {video_path}")
        if not os.path.isfile(video_path):
            raise PermissionError(f"video path is not a regular file: {video_path}")
        st = os.stat(video_path)
        if os.name == "posix":  # ownership/permission checks follow the Linux SDK baseline
            # getuid() is POSIX-only; pylint may run on a Windows host
            current_uid = os.getuid()  # pylint: disable=no-member
            if st.st_uid != current_uid:
                raise PermissionError(f"video file owner mismatch: process uid={current_uid}, file uid={st.st_uid}")
            perm = st.st_mode & 0o777
            for shift in (
                6,
                3,
                0,
            ):  # owner / group / other, per-group numeric compare (C++ baseline)
                cur = (perm >> shift) & 0o7
                maxp = (_FILE_MODE_MAX >> shift) & 0o7
                if cur > maxp:
                    raise PermissionError(
                        f"video file permission {oct(perm)} exceeds the SDK baseline "
                        f"{oct(_FILE_MODE_MAX)}: {video_path}"
                    )

    # ------------------------------------------------------------------ #
    # cache load (hit -> skip decode + embedding)
    # ------------------------------------------------------------------ #
    def _load_cache(self) -> bool:
        """Loads analysis cache; True when times/embeds are restored."""
        cached = self._cache.load()
        if cached is None:
            return False
        self.sample_times, self.embeds = cached
        paths = self._cache.frame_paths_if_complete(len(self.sample_times))
        self.sample_frame_paths = paths
        if not paths:
            _Logger.warn("frame files missing (keep_frames=False or cleaned); sample_frame_paths is empty")
        _Logger.info(
            f"analysis cache hit: {len(self.sample_times)} frame embeddings reused ({len(paths)} frame files available)"
        )
        return True

    # ------------------------------------------------------------------ #
    # 1 fps sampling (equivalence spec §4.3)
    # ------------------------------------------------------------------ #
    def _read_video_meta(self):
        info = video_info(self.video_path)
        self.fps = float(info["fps"])
        self.n_video_frames = int(info["n_frames"])
        self.total_duration = float(info["duration_sec"])
        if self.fps <= 0 or self.n_video_frames <= 0 or self.total_duration <= 0:
            raise RuntimeError(
                f"invalid video metadata: fps={self.fps}, n_frames={self.n_video_frames}, "
                f"duration={self.total_duration}s ({self.video_path})"
            )
        _Logger.info(
            f"video meta: fps={self.fps:.2f}, n_frames={self.n_video_frames}, duration={self.total_duration:.2f}s"
        )

    def _decode_sample_frames(self):
        """Decodes 1 fps frames via mm.video_decode in batches and writes JPEGs.

        §4.3-4/5: when consecutive sample points map to the same frame index
        (the tail fallback point does so on every real video), the previous
        JPEG is duplicated so that N sample times -> N JPEG files; encoding
        goes through Image.pillow() -> PIL save(JPEG, quality).
        """
        t0 = time.time()
        times = kts_algorithm.compute_sample_times(self.total_duration, self.sample_interval_sec)
        mapped = [kts_algorithm.time_to_frame_index(t, self.fps, self.n_video_frames) for t in times]

        frame_dir = self._cache.ensure_frame_dir()
        existing = sorted(frame_dir.glob("*.jpg"))
        if len(existing) == len(times):
            self.sample_frame_paths = [str(p) for p in existing]
            self.sample_times = times
            _Logger.info(f"reusing {len(existing)} existing frame files ({time.time() - t0:.2f}s)")
            return
        for p in existing:
            p.unlink()

        # decode each unique target frame once, in batches
        unique_indices = sorted(set(mapped))
        frame_by_index: Dict[int, object] = {}
        for start in range(0, len(unique_indices), _DECODE_BATCH_SIZE):
            batch = set(unique_indices[start : start + _DECODE_BATCH_SIZE])
            frames = video_decode(self.video_path, "cpu", frame_indices=batch)
            if len(frames) != len(batch):
                raise RuntimeError(f"video_decode returned {len(frames)} frames, expected {len(batch)}")
            for idx, img in zip(sorted(batch), frames):
                frame_by_index[idx] = img

        # write JPEG per sample point (duplicate file for repeated frame indices)
        n_duplicated = 0
        for i, idx in enumerate(mapped):
            out_path = frame_dir / f"f{i:06d}.jpg"
            if i > 0 and mapped[i - 1] == idx:
                shutil.copyfile(frame_dir / f"f{i - 1:06d}.jpg", out_path)
                n_duplicated += 1
            else:
                pil_img = frame_by_index[idx].pillow()
                pil_img.save(out_path, format="JPEG", quality=self.jpeg_quality)

        self.sample_times = times
        self.sample_frame_paths = self._cache.frame_paths(len(times))
        _Logger.info(
            f"sampled {len(times)} frames ({n_duplicated} duplicated tail copies), decode time: {time.time() - t0:.2f}s"
        )

    # ------------------------------------------------------------------ #
    # embedding (batch chunks x embed_workers, progress callback)
    # ------------------------------------------------------------------ #
    def _compute_embeddings(self):
        t0 = time.time()
        paths = self.sample_frame_paths
        n = len(paths)
        if n == 0:
            raise RuntimeError("no sampled frames to embed")
        workers = max(1, int(self.embed_workers))
        chunks = [(b, paths[b : b + self.batch_size]) for b in range(0, n, self.batch_size)]
        results: List[Optional[np.ndarray]] = [None] * n
        total_batches = len(chunks)
        done_batches = 0
        done_frames = 0

        _Logger.info(f"embedding {n} frames: batch={self.batch_size}, workers={workers}")

        def run_chunk(ci: int) -> None:
            b, chunk = chunks[ci]
            mat = self.embed_backend.encode_images(list(chunk))
            if mat.shape[0] != len(chunk):
                raise RuntimeError(f"backend returned {mat.shape[0]} vectors, expected {len(chunk)}")
            results[b : b + len(chunk)] = [mat[k] for k in range(len(chunk))]

        def report(n_frames_in_batch: int):
            nonlocal done_batches, done_frames
            done_batches += 1
            done_frames += n_frames_in_batch
            if done_batches % 5 == 0 or done_batches == total_batches:
                _Logger.info(f"  [progress] embedding batch {done_batches}/{total_batches} ({time.time() - t0:.1f}s)")
            if self.progress_callback is not None:
                try:
                    self.progress_callback(done_frames, n)
                except Exception as e:  # noqa: BLE001  callback must not interrupt
                    _Logger.warn(f"progress_callback raised, ignored: {e}")

        if workers == 1:
            for ci in range(total_batches):
                run_chunk(ci)
                report(len(chunks[ci][1]))
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futures = {ex.submit(run_chunk, ci): ci for ci in range(total_batches)}
                for fut in as_completed(futures):
                    fut.result()  # propagate the first failure
                    report(len(chunks[futures[fut]][1]))

        missing = [i for i, v in enumerate(results) if v is None]
        if missing:
            raise RuntimeError(f"{len(missing)} frame embeddings missing")
        self.embeds = np.stack(results, axis=0)
        _Logger.info(f"embedding shape: {self.embeds.shape}, time: {time.time() - t0:.2f}s")

        self._cache.save(self.sample_times, self.embeds, self.fps, self.total_duration)
        if not self.keep_frames:
            self._cache.clean_frames()
            self.sample_frame_paths = []
            _Logger.info("frame files cleaned (keep_frames=False)")

    # ------------------------------------------------------------------ #
    # main pipeline
    # ------------------------------------------------------------------ #
    def segment(self) -> SegmentResult:
        """Runs the full segmentation pipeline. Idempotent: repeated calls
        return the cached in-memory result without recomputation.
        """
        self._ensure_state(require_segmented=False)
        if self._result is not None:
            return self._result

        timings: Dict[str, float] = {}
        t_total = time.time()

        t = time.time()
        self._read_video_meta()
        if self._load_cache():
            _Logger.info("cache hit, skipping decode and embedding")
        else:
            t2 = time.time()
            self._decode_sample_frames()
            timings["decode_s"] = time.time() - t2
            t2 = time.time()
            self._compute_embeddings()
            timings["embed_s"] = time.time() - t2
        meta_s = time.time() - t
        if meta_s > 0.01:
            _Logger.info(f"preprocessing total: {meta_s:.2f}s")

        t = time.time()
        self.diff_seq = kts_algorithm.build_diff_sequence(self.embeds)
        diff_stats = {
            "diff_min": float(self.diff_seq.min()),
            "diff_max": float(self.diff_seq.max()),
            "diff_mean": float(self.diff_seq.mean()),
            "diff_std": float(self.diff_seq.std()),
        }

        t = time.time()
        if self.lambda_penalty is None:
            lambda_ = kts_algorithm.auto_select_lambda(
                self.embeds,
                self.diff_seq,
                self.sample_times,
                self.total_duration,
                self.target_segment_duration,
                self.min_segment_duration,
                self.boundary_align_window,
            )
            timings["lambda_s"] = time.time() - t
        else:
            lambda_ = float(self.lambda_penalty)
        self.lambda_penalty = lambda_

        t = time.time()
        splits = kts_algorithm.kts_dp_segment_full(self.embeds, lambda_)
        timings["kts_dp_s"] = time.time() - t
        splits = kts_algorithm.align_boundaries_to_peaks(splits, self.diff_seq, self.boundary_align_window)
        seg_pairs = kts_algorithm.splits_to_segments(splits, self.sample_times, self.total_duration)
        seg_pairs = kts_algorithm.merge_short_segments(seg_pairs, self.min_segment_duration)

        # per-segment frame indices and representation vectors
        video_segments: List[VideoSegment] = []
        seg_vecs: List[Optional[np.ndarray]] = []
        for start, end in seg_pairs:
            idx = [i for i, ts in enumerate(self.sample_times) if start <= ts < end]
            video_segments.append(
                VideoSegment(
                    start_sec=float(start),
                    end_sec=float(end),
                    n_frames=len(idx),
                    frame_indices=idx,
                )
            )
            if idx:
                v = self.embeds[idx].mean(axis=0)
                norm = float(np.linalg.norm(v))
                seg_vecs.append((v / norm).astype(np.float32) if norm > 0 else v)
            else:
                seg_vecs.append(None)
        self._segment_vecs = seg_vecs

        timings["total_s"] = time.time() - t_total
        stats = dict(diff_stats)
        stats["timing"] = timings
        self._segmented = True
        self._result = SegmentResult(
            segments=video_segments,
            video_duration_sec=self.total_duration,
            fps=self.fps,
            n_sample_frames=len(self.sample_times),
            lambda_penalty=lambda_,
            stats=stats,
        )
        _Logger.info(
            f"segmentation done: {len(video_segments)} segments, lambda={lambda_}, total {timings['total_s']:.2f}s"
        )
        return self._result

    # ------------------------------------------------------------------ #
    # retrieval / frame selection
    # ------------------------------------------------------------------ #
    def retrieve(self, query: str, top_k: int = 3) -> List[VideoSegment]:
        """Retrieves the top-k segments most relevant to the query (returned
        in time order).

        Args:
            query: non-empty query text. Callers that need option-style
                queries (e.g. multiple-choice questions) should concatenate
                the option texts into query themselves.
            top_k: positive integer; values above the segment count return all
                segments without error.
        """
        self._ensure_state(require_segmented=True)
        if not isinstance(query, str):
            raise TypeError("query must be str")
        if not query:
            raise ValueError("query must not be empty")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer")

        query_vec = self.embed_backend.encode_text(query)
        seg_vecs = self._segment_vecs
        sims = np.full(len(seg_vecs), -np.inf, dtype=np.float64)
        for i, v in enumerate(seg_vecs):
            if v is not None:
                sims[i] = float(query_vec @ v)
        k = min(top_k, len(seg_vecs))
        order = np.argsort(-sims)[:k]
        return [self._result.segments[int(i)] for i in sorted(order)]

    def select_frames(
        self,
        segments: Sequence[VideoSegment],
        seg_frames: int = 4,
        max_frames: int = 24,
    ) -> List[str]:
        """Uniformly selects frames inside the given segments.

        Args:
            segments: non-empty list of VideoSegment (typically from retrieve).
            seg_frames: frames per segment (positive integer).
            max_frames: total frame cap (positive integer); when exceeded,
                the per-segment budget shrinks to max(1, max_frames // m).

        Returns:
            Frame file paths in time order (deduplicated). Raises RuntimeError
            when frame files were cleaned (keep_frames=False) or are missing.
        """
        self._ensure_state(require_segmented=True)
        if isinstance(segments, (str, bytes)) or not isinstance(segments, Sequence) or len(segments) == 0:
            raise ValueError("segments must be a non-empty list of VideoSegment")
        if isinstance(seg_frames, bool) or not isinstance(seg_frames, int) or seg_frames < 1:
            raise ValueError("seg_frames must be a positive integer")
        if isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames < 1:
            raise ValueError("max_frames must be a positive integer")
        if not self.sample_frame_paths:
            raise RuntimeError(
                "frame files are unavailable (keep_frames=False or cache cleaned); call segment() again to rebuild them"
            )

        m = max(1, len(segments))
        k_eff = max(1, min(seg_frames, max_frames // m))
        times = self.sample_times
        paths = self.sample_frame_paths
        idx: List[int] = []
        for seg in segments:
            in_seg = [i for i, t in enumerate(times) if seg.start_sec <= t < seg.end_sec]
            if not in_seg:
                continue
            sel = np.linspace(0, len(in_seg) - 1, min(k_eff, len(in_seg)), dtype=int)
            idx.extend(in_seg[j] for j in sel)
        if len(idx) > max_frames:
            # trim evenly by position, then map back to the real sample
            # indices those positions hold (positions must never replace
            # sample indices, or the video-head frames would be returned)
            sel = np.linspace(0, len(idx) - 1, max_frames, dtype=int)
            idx = [idx[int(j)] for j in sel]
        picked = sorted(set(int(i) for i in idx))
        missing = [i for i in picked if i >= len(paths) or not os.path.isfile(paths[i])]
        if missing:
            raise RuntimeError(
                f"{len(missing)} selected frame files are missing (first: {paths[missing[0]] if paths else 'n/a'})"
            )
        return [paths[i] for i in picked]

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    def _ensure_state(self, require_segmented: bool):
        if self._closed:
            raise RuntimeError("KtsSegmenter is closed")
        if require_segmented and not self._segmented:
            raise RuntimeError("segment() must be called before retrieve()/select_frames()")

    def close(self) -> None:
        """Releases backend resources (only the self-built backend). Idempotent."""
        if self._closed:
            return
        self._closed = True
        if self._owns_backend:
            try:
                self.embed_backend.close()
            except Exception as e:  # noqa: BLE001
                _Logger.warn(f"backend close warning: {e}")
        self.embeds = None
        self.diff_seq = None
        self._segment_vecs = []
        _Logger.info("KtsSegmenter closed")

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001  # nosec B110  best-effort cleanup must never raise
            pass
