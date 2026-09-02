#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the MultimodalSDK project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
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
# `acc` is provided by the SWIG `_acc` extension at runtime
from .._impl import acc as _acc  # pylint: disable=no-name-in-module
from .image_wrapper import Image
from .util import _ensure_bytes

_SUPPORT_PILLOW_MODE = "RGB"
_RESIZED_SIZE_LEN = 2


def video_decode(video_path: str | bytes, device: str | bytes, frame_indices: set = None, sample_num: int = -1) -> list:
    """Video decode api. Device check will be performed in C++ code


    Args:
        video_path (str | bytes): given video path
        device (str | bytes): only support cpu now
        frame_indices (set):
        sample_num (int):

    Returns:
        list: decoded image list
    """
    path_bytes = _ensure_bytes(video_path, "path")
    device_bytes = _ensure_bytes(device, "device")
    frames = []
    result = []
    if frame_indices is None:
        frame_indices = set()
    frames = _acc.video_decode(path_bytes, device_bytes, frame_indices, sample_num)
    for frame in frames:
        result.append(Image._from_acc(frame))
    return result


def video_info(video_path: str | bytes) -> dict:
    """Video metadata api (no decoding). Validation rules are identical to
    video_decode: mp4/MP4 suffix, permission <= 0640, not a symbolic link,
    owned by the current user.

    Args:
        video_path (str | bytes): given video path

    Returns:
        dict: {"fps": float, "n_frames": int, "duration_sec": float}.
            duration_sec uses the video stream duration (container duration as
            fallback), not n_frames / fps.
    """
    path_bytes = _ensure_bytes(video_path, "path")
    info = _acc.video_info(path_bytes)
    return {
        "fps": float(info.fps),
        "n_frames": int(info.nFrames),
        "duration_sec": float(info.durationSec),
    }
