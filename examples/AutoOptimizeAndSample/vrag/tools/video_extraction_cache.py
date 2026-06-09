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
# Video extraction cache for persistent storage.

import asyncio
import hashlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from cachetools import LRUCache
from filelock import AsyncUnixFileLock
from pydantic import BaseModel

from vrag.logger import logger
from vrag.shared import execute_time, into_u8_frames
from vrag.tools.decord import FrameExtraction, compute_frame_idx, process_video
from vrag.tools.ffmpeg import probe_video
from vrag.tools.audio import AudioChunkExtraction, chunk_audio


@dataclass
class VideoExtractionCache:
    root: Path
    lock_timeout: float = 600.0
    max_workers: int = 40

    LOCK_SUFFIX = ".lock"
    METADATA_FILE = "metadata.json"
    VIDEO_FRAMES_DIR = "frames"
    AUDIO_CHUNKS_DIR = "audios"
    _frame_extraction_cache: LRUCache = field(init=False, default_factory=lambda: LRUCache(maxsize=10))
    _audio_extraction_cache: LRUCache = field(init=False, default_factory=lambda: LRUCache(maxsize=10))
    _executor: ThreadPoolExecutor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="vec_")

    @staticmethod
    def _compute_video_checksum(video_path: Path, chunk_size: int = 65536) -> str:
        sha256 = hashlib.sha256()

        with video_path.open(mode="rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                sha256.update(chunk)

        return sha256.hexdigest()

    @staticmethod
    def _compute_video_cache_dir(video_path: Path, store_dir: Path) -> Path:
        return store_dir.joinpath(VideoExtractionCache._compute_video_checksum(video_path))

    @staticmethod
    def _count_continuous_frames(path: Path, max_limit: int) -> int:
        if not path.exists():
            return 0
        return min(max_limit, len(list(path.glob("frame_*.png"))))

    @staticmethod
    def _frame_of(i: int) -> str:
        return f"frame_{i}.png"

    @staticmethod
    def _get_cache_key(fps: float, max_sample_num: int, force_sample: bool) -> str:
        if force_sample:
            return str(max_sample_num)
        fps_str = f"{fps:.2f}".replace(".", "_")
        return f"fps_{fps_str}"

    @classmethod
    def from_cache_key(
        cls, video_path: Path, store_dir: Path, /, lock_timeout: Optional[float] = None
    ) -> "VideoExtractionCache":
        store_dir = VideoExtractionCache._compute_video_cache_dir(video_path, store_dir)
        store_dir.mkdir(parents=True, exist_ok=True)

        if lock_timeout:
            return cls(root=store_dir.resolve(), lock_timeout=lock_timeout)
        else:
            return cls(root=store_dir.resolve())

    async def get_with_frames_from_video(
        self, video_path: Path, fps: float, max_sample_num: int, force_sample: bool, **kwargs
    ) -> Optional[FrameExtraction]:
        probe_result = await asyncio.to_thread(probe_video, video_path)

        if not probe_result.has_video:
            msg = f"Cannot get video from {video_path.resolve().as_posix()}, Skipping..."
            logger.warning(msg)
            return None

        if probe_result.is_hevc:
            logger.warning("Cannot process video from H.265, Skipping...")
            return None

        return await self._get_with_frames(video_path, fps, max_sample_num, force_sample, **kwargs)

    async def get_with_audio_from_video(
        self, video_path: Path, chunk_length: int, min_chunk_threshold: float = 1.0
    ) -> AudioChunkExtraction:
        async with AsyncUnixFileLock(
            self._audio_lock_path(chunk_length, min_chunk_threshold), timeout=self.lock_timeout
        ):
            chunks = self._audio_extraction_cache.get(self._audio_cache_key(chunk_length, min_chunk_threshold))
            if chunks:
                msg = f"Audio chunk cache hit, returning chunks of {chunk_length}"
                logger.debug(msg)
                return chunks

            probe_result = await asyncio.to_thread(probe_video, video_path)

            if not probe_result.has_audio:
                msg = f"Cannot get audio from {video_path.resolve().as_posix()}, Skipping..."
                logger.warning(msg)
                return AudioChunkExtraction()

            chunks = await asyncio.to_thread(chunk_audio, video_path, chunk_length, min_chunk_threshold)

            if not chunks:
                logger.warning("Audio chucking returned empty list.")
                return AudioChunkExtraction()

            self._audio_extraction_cache[self._audio_cache_key(chunk_length, min_chunk_threshold)] = chunks
            return chunks

    async def get_all(
        self,
        video_path: Path,
        fps: float,
        max_sample_num: int,
        force_sample: bool,
        chunk_length: int,
        min_chunk_threshold: float,
        **kwargs,
    ) -> Tuple[Optional[FrameExtraction], Optional[AudioChunkExtraction]]:
        get_video = self.get_with_frames_from_video(video_path, fps, max_sample_num, force_sample, **kwargs)
        get_audio = self.get_with_audio_from_video(video_path, chunk_length, min_chunk_threshold)
        return await asyncio.gather(get_video, get_audio)

    def _frames_cache_path(self, fps: float, max_sample_num: int, force_sample: bool) -> Path:
        cache_key = self._get_cache_key(fps, max_sample_num, force_sample)
        return self.root / self.VIDEO_FRAMES_DIR / cache_key

    def _frames_lock_path(self, fps: float, max_sample_num: int, force_sample: bool) -> Path:
        return self._frames_cache_path(fps, max_sample_num, force_sample).with_suffix(self.LOCK_SUFFIX)

    def _audio_chunk_path(self, chunk_length: int, min_chunk_threshold: float) -> Path:
        return self.root / self.AUDIO_CHUNKS_DIR / f"{chunk_length}-{min_chunk_threshold}"

    def _audio_lock_path(self, chunk_length: int, min_chunk_threshold: float) -> Path:
        return self._audio_chunk_path(chunk_length, min_chunk_threshold).with_suffix(self.LOCK_SUFFIX)

    def _frame_cache_key(self, fps: float, max_sample_num: int, force_sample: bool) -> str:
        return f"{self.root.stem}_{self._get_cache_key(fps, max_sample_num, force_sample)}"

    def _audio_cache_key(self, chunk_length: int, min_chunk_threshold: float) -> str:
        return f"{self.root.stem}_{chunk_length}_{min_chunk_threshold}"

    async def _read_metadata_unsafe(self) -> Optional["_VideoMetadata"]:
        p = self.root / self.METADATA_FILE

        if p.exists():
            return _VideoMetadata.model_validate_json(p.read_text(encoding="utf-8"))

        return None

    async def _write_metadata_unsafe(self, metadata: "_VideoMetadata") -> "_VideoMetadata":
        p = self.root / self.METADATA_FILE

        if not p.exists():
            p.write_text(metadata.model_dump_json(indent=2), encoding="utf-8")

        return metadata

    async def _load_frames_range(self, path: Path, count: int) -> np.ndarray:
        if count < 0:
            return np.empty((), dtype=np.uint8)

        async def _load_single(i: int) -> Optional[np.ndarray]:
            img = await asyncio.to_thread(cv2.imread, (path / self._frame_of(i)).resolve().as_posix(), cv2.IMREAD_COLOR)
            if img is None:
                msg = f"Failed to load cache frame {i}"
                logger.error(msg)
                return None
            return await asyncio.to_thread(cv2.cvtColor, img, cv2.COLOR_BGR2RGB)

        tasks = [_load_single(i) for i in range(count)]
        results = await asyncio.gather(*tasks)

        valid_frames = [f for f in results if f is not None]

        if not valid_frames:
            return np.empty((), dtype=np.uint8)

        if len(valid_frames) < count:
            msg = f"Partial load: {len(valid_frames)}/{count} frames recovered."
            logger.warning(msg)

        return np.stack(valid_frames)

    @execute_time
    def _save_frames_incremental(self, path: Path, frames: np.ndarray) -> None:
        if frames.size == 0:
            return

        path.mkdir(parents=True, exist_ok=True)

        frames = into_u8_frames(frames)

        def _worker(i: int) -> bool:
            frame_path = path / self._frame_of(i)

            try:
                if frame_path.exists() and frame_path.stat().st_size > 0:
                    return False
            except OSError:
                msg = f"Skip to save frame {i}, file existed."
                logger.warning(msg)
                pass

            try:
                return cv2.imwrite(frame_path.resolve().as_posix(), cv2.cvtColor((frames[i]), cv2.COLOR_RGB2BGR))
            except (OSError, RuntimeError) as e:
                msg = f"Failed to save frame {i}: {e}"
                logger.error(msg)
                return False

        results = list(self._executor.map(_worker, range(len(frames))))
        saved_count = sum(results)
        skipped_count = len(frames) - saved_count

        if saved_count > 0:
            msg = f"Cache saved: wrote {saved_count} new frames."
            logger.debug(msg)
        if skipped_count > 0:
            msg = f"Cache saved: skipped {skipped_count} frames for existing or error."
            logger.debug(msg)

    async def _read_frames_unsafe(
        self, fps: float, max_sample_num: int, force_sample: bool
    ) -> Optional[FrameExtraction]:
        p = self._frames_cache_path(fps, max_sample_num, force_sample)
        frame_extraction = self._frame_extraction_cache.get(self._frame_cache_key(fps, max_sample_num, force_sample))
        if frame_extraction:
            msg = f"Frame LRU cache hit, returning {max_sample_num} frames at most."
            logger.debug(msg)
            return frame_extraction

        existing_count = self._count_continuous_frames(p, max_sample_num)

        metadata = await self._read_metadata_unsafe()
        if metadata:
            needed = metadata.needed_samples(fps, max_sample_num, force_sample)
            if existing_count < needed:
                msg = f"Cache Miss/Incomplete: Found {existing_count} files, need {needed}."
                logger.debug(msg)
                return None

            msg = f"Cache Hit at {self.root}: Found {existing_count} files, loading {needed} frames."
            logger.debug(msg)
            frames = await self._load_frames_range(p, needed)
            fe = metadata.rewind_to_extraction(frames, fps, max_sample_num, force_sample)
            msg = f"Cache Hit at {self.root}: Found {existing_count} files, load {needed} frames"
            logger.debug(msg)
            self._frame_extraction_cache[self._frame_cache_key(fps, max_sample_num, force_sample)] = fe
            return fe

        msg = f"Cache Miss/Incomplete: Found {existing_count} files, need at most {max_sample_num}"
        logger.debug(msg)
        return None

    async def _write_frames_unsafe(
        self, video_path: Path, fps: float, max_sample_num, force_sample: bool, **kwargs
    ) -> FrameExtraction:
        frames_path = self._frames_cache_path(fps, max_sample_num, force_sample)

        msg = f"Generate frames cache for {max_sample_num} frames."
        logger.info(msg)

        extraction = await asyncio.to_thread(
            lambda: process_video(video_path, max_sample_num, fps, force_sample, **kwargs)
        )

        await asyncio.to_thread(self._save_frames_incremental, frames_path, extraction.frames)
        await self._write_metadata_unsafe(_VideoMetadata.from_extraction(extraction))

        self._frame_extraction_cache[self._frame_cache_key(fps, max_sample_num, force_sample)] = extraction
        return extraction

    async def _get_with_frames(
        self, video_path: Path, fps: float, max_sample_num: int, force_sample: bool, **kwargs
    ) -> FrameExtraction:
        async with AsyncUnixFileLock(
            self._frames_lock_path(fps, max_sample_num, force_sample), timeout=self.lock_timeout
        ):
            frame_extraction = await self._read_frames_unsafe(fps, max_sample_num, force_sample)
            if frame_extraction is not None:
                return frame_extraction
            return await self._write_frames_unsafe(video_path, fps, max_sample_num, force_sample, **kwargs)


class _VideoMetadata(BaseModel):
    video_duration: float
    avg_fps: float
    total_frame_count: int

    @classmethod
    def from_extraction(cls, extraction: FrameExtraction) -> "_VideoMetadata":
        return cls(
            video_duration=extraction.video_duration,
            avg_fps=extraction.avg_fps,
            total_frame_count=extraction.total_frame_num,
        )

    def rewind_to_extraction(
        self, frames: np.ndarray, fps: float, max_sample_num: int, force_sample
    ) -> FrameExtraction:
        frame_idx = compute_frame_idx(force_sample, self.avg_fps, self.total_frame_count, fps, max_sample_num)
        return FrameExtraction(
            frames=frames,
            avg_fps=self.avg_fps,
            video_duration=self.video_duration,
            total_frame_num=self.total_frame_count,
            frame_timestamps=[idx / self.avg_fps for idx in frame_idx],
        )

    def needed_samples(self, fps: float, max_sample_num: int, force_sample: bool) -> int:
        return len(compute_frame_idx(force_sample, self.avg_fps, self.total_frame_count, fps, max_sample_num))
