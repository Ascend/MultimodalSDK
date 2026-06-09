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
# Video frame extraction using library decord.

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import List

import cv2
import numpy as np
from decord import VideoReader, cpu

from vrag.logger import logger
from vrag.shared import into_u8_frames
from vrag.types import FrameExtraction


def smart_resize_batch(frames: np.ndarray, target_h: int = 720) -> np.ndarray:
    """
    Batch smart resize video frames.

    Args:
        frames: Numpy array of shape (N, H, W, C).
        target_h: Target height of frames.

    Returns:
        Numpy array of shape (N, NEW_H, NEW_W, C).
    """

    if frames is None or len(frames) == 0:
        return np.empty((0,))

    if frames.ndim != 4:
        raise ValueError("Frames must in shape of (N, H, W, C)")

    if target_h <= 0:
        raise ValueError(f"Target frame height must be positive, but get target_h: {target_h}")

    first_h, first_w = frames.shape[1:3]

    if first_h <= 0 or first_w <= 0:
        raise ValueError(f"Frames height and width must be positive, but get height: {first_h} and width: {first_w}")

    baseline_width = target_h * 16 // 9
    max_pixels = target_h * baseline_width

    original_pixels = first_h * first_w
    if original_pixels <= max_pixels:
        msg = f"Original pixels {original_pixels} <= limit {max_pixels}, keep original sizes."
        logger.debug(msg)
        return frames

    msg = f"Original pixels {original_pixels} exceed limit {max_pixels}, need resizing."
    logger.debug(msg)

    k = (float(max_pixels) / original_pixels) ** 0.5
    new_h = int(first_h * k)
    new_w = int(first_w * k)

    def process_single_frame(img: np.ndarray) -> np.ndarray:
        return cv2.resize(into_u8_frames(img), (new_w, new_h), interpolation=cv2.INTER_AREA)

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_single_frame, frame) for frame in frames]
        results = [f.result() for f in futures]
        return np.stack(results, axis=0)


def process_video(
    video_path: Path,
    max_frames_num: int,
    fps: float = 1.0,
    force_sample: bool = False,
    decord_workers: int = 0,
    resolution: int = 720,
) -> FrameExtraction:
    """
    Extraction frames from a video.

    Args:
        video_path: Path to video file.
        max_frames_num: Maximum number of frames sampled from video.
        fps: Target frames per second for extraction.
        force_sample: If force uniform sampling to max_frames_num.
        decord_workers: Num of decord running threads.
        resolution: Target height for resizing frames.

    Returns:
        FrameExtraction containing sampled frames.
    """

    msg = f"Extraction frames from {video_path} with FPS={fps}."
    logger.debug(msg)

    vr = VideoReader(video_path.resolve().as_posix(), ctx=cpu(), num_threads=decord_workers)
    total_frame_num: int = len(vr)
    avg_fps: float = vr.get_avg_fps()
    video_time: float = total_frame_num / avg_fps

    frame_idx = compute_frame_idx(force_sample, avg_fps, total_frame_num, fps, max_frames_num)

    spare_frames = vr.get_batch(frame_idx).asnumpy()
    spare_frames = spare_frames.astype(np.uint8) if spare_frames.dtype != np.uint8 else spare_frames

    return FrameExtraction(
        frames=smart_resize_batch(spare_frames, resolution),
        frame_timestamps=[i / avg_fps for i in frame_idx],
        avg_fps=avg_fps,
        video_duration=video_time,
        total_frame_num=total_frame_num,
    )


@lru_cache
def compute_frame_idx(
    force_sample: bool,
    avg_fps: float,
    total_frame_num: int,
    sample_fps: float,
    max_frame_num: int,
) -> List[int]:
    """
    Compute frame indices with fps and frame num.

    1. If force_sample, just sample max_frame_num indices.
    2. If not force_sample, use sample_fps as interval to sample indices with no more than max_frame_num frames.

    Args:
        force_sample: Force sample frames.
        avg_fps: Average fps from raw video.
        total_frame_num: Total frame num from raw video.
        sample_fps: Target fps to sample frames.
        max_frame_num: Max frame num to sample frames.
    """
    if force_sample:
        uniform_sampled_frames: np.ndarray = np.linspace(0, total_frame_num - 1, max_frame_num, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
    else:
        step: int = max(1, round(avg_fps / sample_fps))
        limix_index = min(total_frame_num, max_frame_num * step)
        frame_idx = list(range(0, limix_index, step))
        frame_idx = frame_idx[:max_frame_num]
    return frame_idx
