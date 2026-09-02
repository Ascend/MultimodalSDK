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
"""On-disk analysis cache for KTS segmentation (design doc §8.3).

``KtsCache`` owns the cache key
(``md5(abspath|size|mtime|interval|backend_id)``), the per-key file layout
(``kts_<key>_meta.json`` / ``kts_<key>_times.npy`` / ``kts_<key>_embeds.npy``
/ ``frames_<key>/``) and frame-directory housekeeping. Differences from the
legacy key: ``embed_backend_id`` added (never mix backends), the ``mask``
field dropped (subtitle masking is not part of the SDK, §3.4), lambda / batch
parameters excluded (lambda is computed after caching).

Construction is side-effect free (no stat / mkdir); the key and paths are
computed lazily on first use, so building a segmenter never touches the
filesystem beyond what its own validation already did.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from ...comm.log import _Logger

CACHE_DIR_NAME = ".kts_cache"


class KtsCache:
    """Per-video analysis cache: sample times + frame embeddings + frame JPEGs."""

    def __init__(
        self,
        video_path: str,
        backend_id: str,
        sample_interval_sec: float,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
    ):
        """
        Args:
            video_path: video file path (must exist; owner validation is the
                segmenter's job).
            backend_id: embedding backend identity tag (part of the key).
            sample_interval_sec: sampling interval (part of the key).
            cache_dir: explicit cache root; defaults to <video dir>/.kts_cache.
            use_cache: False disables both read and write.
        """
        self.video_path = str(video_path)
        self.backend_id = backend_id
        self.sample_interval_sec = float(sample_interval_sec)
        self.cache_dir = cache_dir
        self.use_cache = bool(use_cache)
        self._key: Optional[str] = None

    # ------------------------------------------------------------------ #
    # key / paths (lazy)
    # ------------------------------------------------------------------ #
    @property
    def key(self) -> str:
        """Cache key = md5(abspath|size|mtime|interval|backend_id); file
        changes / model / interval changes invalidate automatically.
        """
        if self._key is None:
            st = os.stat(self.video_path)
            material = (
                f"{os.path.abspath(self.video_path)}|{st.st_size}|{st.st_mtime_ns}"
                f"|{self.sample_interval_sec}|{self.backend_id}"
            )
            # cache fingerprint (not security-related): usedforsecurity=False
            self._key = hashlib.md5(material.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]
        return self._key

    @property
    def base(self) -> Path:
        if self.cache_dir:
            return Path(self.cache_dir)
        return Path(self.video_path).parent / CACHE_DIR_NAME

    @property
    def meta_path(self) -> Path:
        return self.base / f"kts_{self.key}_meta.json"

    @property
    def times_path(self) -> Path:
        return self.base / f"kts_{self.key}_times.npy"

    @property
    def embeds_path(self) -> Path:
        return self.base / f"kts_{self.key}_embeds.npy"

    @property
    def frame_dir(self) -> Path:
        return self.base / f"frames_{self.key}"

    # ------------------------------------------------------------------ #
    # read / write
    # ------------------------------------------------------------------ #
    def load(self) -> Optional[Tuple[List[float], np.ndarray]]:
        """Loads cached (sample_times, embeds); None on miss / corruption."""
        if not self.use_cache:
            return None
        if not (self.meta_path.is_file() and self.times_path.is_file() and self.embeds_path.is_file()):
            return None
        try:
            with open(self.meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            times = np.load(self.times_path, allow_pickle=False)
            embeds = np.load(self.embeds_path, allow_pickle=False)
            if len(times) != embeds.shape[0]:
                return None
            if meta.get("sample_interval_sec") != self.sample_interval_sec:
                return None
        except Exception as e:  # noqa: BLE001  corrupted cache -> recompute
            _Logger.warn(f"cache load failed, will recompute: {e}")
            return None
        return times.tolist(), embeds

    def save(self, times: List[float], embeds: np.ndarray, fps: float, total_duration: float) -> None:
        """Writes meta + times + embeds. Skipped entirely when use_cache=False."""
        if not self.use_cache:
            return
        self.base.mkdir(parents=True, exist_ok=True)
        meta = {
            "sample_interval_sec": self.sample_interval_sec,
            "fps": fps,
            "total_duration": total_duration,
            "n_frames": len(times),
            "embed_dim": None if embeds is None else int(embeds.shape[1]),
        }
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        np.save(self.times_path, np.asarray(times, dtype=np.float64))
        if embeds is not None:
            np.save(self.embeds_path, embeds)
        _Logger.info(f"analysis cache saved: {self.base}")

    # ------------------------------------------------------------------ #
    # frame directory housekeeping
    # ------------------------------------------------------------------ #
    def ensure_frame_dir(self) -> Path:
        """Creates (if needed) and returns the frame JPEG directory."""
        self.frame_dir.mkdir(parents=True, exist_ok=True)
        return self.frame_dir

    def frame_paths_if_complete(self, n_times: int) -> List[str]:
        """Sorted frame JPEG paths when the on-disk count matches n_times,
        else [] (incomplete directory, e.g. cleaned or partially deleted).
        """
        existing = sorted(self.frame_dir.glob("*.jpg"))
        if len(existing) == n_times:
            return [str(p) for p in existing]
        return []

    def frame_paths(self, n_times: int) -> List[str]:
        """Expected frame paths for n_times sample points (f{i:06d}.jpg)."""
        return [str(self.frame_dir / f"f{i:06d}.jpg") for i in range(n_times)]

    def clean_frames(self) -> None:
        """Deletes all frame JPEGs (keep_frames=False)."""
        for p in list(self.frame_dir.glob("*.jpg")):  # materialize: deleting
            p.unlink()  # while iterating may skip
