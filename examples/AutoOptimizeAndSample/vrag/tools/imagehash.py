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
# Perceptual hashing utilities for video frame duplication.


import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Optional, Tuple, TypeAlias

import cv2
import numpy as np

from vrag.logger import logger
from vrag.shared import into_u8_frames, once
from vrag.tools.np_cacher import CacherBase, get_cacher

FrameFingerPrint: TypeAlias = int


@dataclass
class ImageHasher:
    thread_pool: ThreadPoolExecutor
    cacher: CacherBase

    @staticmethod
    @once
    def instance() -> "ImageHasher":
        return ImageHasher.new()

    @classmethod
    def new(cls, workers: Optional[int] = None, cap: int = 4096) -> "ImageHasher":
        return cls(thread_pool=ThreadPoolExecutor(max_workers=workers), cacher=get_cacher(cap))

    @classmethod
    def with_cacher(cls, cacher: CacherBase, workers: Optional[int] = None):
        return cls(thread_pool=ThreadPoolExecutor(max_workers=workers), cacher=cacher)

    def get_unique_frame_indices(self, frames: np.ndarray, threshold: int = 5, block_size: int = 16) -> List[int]:
        """
        Identify indices of unique frames from a video stream array.

        Processes a 4D numpy array (N, H, W, C) to extract frames that differ significantly from their immediate
            predecessors using perceptual hashing.

        Args:
            frames: video frames.
            threshold: Hamming distance threshold for considering different.
            block_size: Grid size for fingerprint calculation.

        Returns:
            List[int]: Indices of frames selected to represent unique visual content.
        """
        if frames.ndim != 4:
            raise ValueError(f"Frames expected 4D array (N, H, C, W), but get {frames.ndim}D")

        frames_num = frames.shape[0]
        if frames_num < 2:
            return []

        frames_hashes = self._compute_hashes(frames, block_size)

        keep_flags = [False] * frames_num
        keep_flags[0] = True

        for i in range(1, frames_num):
            prev_hash = frames_hashes[i - 1]
            curr_hash = frames_hashes[i]

            if prev_hash is not None and curr_hash is not None:
                distance = _hamming_distance(prev_hash, curr_hash)
                if distance > threshold:
                    keep_flags[i] = True

        keep_indices = [idx for idx, keep in enumerate(keep_flags) if keep]
        msg = f"Discard duplicated frames: {len(frames) - len(keep_indices)}, remaining {len(keep_indices)}"
        logger.debug(msg)
        return keep_indices

    async def get_unique_frame_indices_async(
        self, frames: np.ndarray, threshold: int = 5, block_size: int = 16
    ) -> List[int]:
        return await asyncio.to_thread(self.get_unique_frame_indices, frames, threshold, block_size)

    def _compute_hashes(self, frames: np.ndarray, block_size: int) -> List[int]:
        def _cache_suffix(block_size: int) -> str:
            return f"{block_size}"

        @self.cacher.cached_sync_with(_cache_suffix)
        def _compute(spare_frames: np.ndarray, block_size: int) -> List[int]:
            return self._compute_hashes_inner(block_size, spare_frames)

        return _compute(frames, block_size)

    def _compute_hashes_inner(self, block_size: int, frames: np.ndarray) -> List[Optional[int]]:
        frames_num = frames.shape[0]
        tasks = [(i, frames[i], block_size) for i in range(frames_num)]
        computed_hashes: List[Optional[int]] = [None] * frames_num

        futures = {self.thread_pool.submit(_compute_fingerprint_task, t): t[0] for t in tasks}
        for future in futures:
            frame_idx, h = future.result()
            computed_hashes[frame_idx] = h
        return computed_hashes


def _get_frame_fingerprint(frame: np.ndarray, block_size: int = 16) -> FrameFingerPrint:
    """
    Generate a robust integer fingerprint using OpenCV DCT perceptual hash.

    Uses doubled resolution grid based on block_size to enhance precision.
    Convert input to grayscale float32 before processing.

    Args:
        frame: Input images, support 2D grayscale or 3D BGR arrays.
        block_size: Grid dimension for low-frequency extraction.

    Returns:
        int: Binary hash converted to an unsigned integer.
    """
    if frame.ndim == 3:
        gray = cv2.cvtColor(into_u8_frames(frame), cv2.COLOR_BGR2GRAY)
    elif frame.ndim == 2:
        gray = frame
    else:
        raise ValueError(f"Input frame ndim: {frame.ndim} is not valid")

    if gray.dtype != np.float32:
        gray = gray.astype(np.float32)

    target_size = block_size * 8
    resized = cv2.resize(gray, (target_size, target_size))

    dct = cv2.dct(resized)

    low_freq = dct[:block_size, :block_size]

    median = np.median(low_freq)

    binary_hash = (low_freq > median).astype(np.uint8)

    return int("".join(map(str, binary_hash.flatten())), 2)


def _compute_fingerprint_task(task_args: Tuple[int, np.ndarray, int]) -> Tuple[int, FrameFingerPrint]:
    i, frame, bs = task_args
    hash_int = _get_frame_fingerprint(frame, block_size=bs)
    return i, hash_int


def _hamming_distance(hash0: FrameFingerPrint, hash1: FrameFingerPrint) -> int:
    """Calculate bitwise difference between two integer."""
    return bin(hash0 ^ hash1).count("1")
