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
"""
vLLM media IO monkey-patches.

This file replaces vLLM's media loading entry points so that video/image
decoding is routed through MultimodalSDK's own decoder stack (AccSDK
backend) instead of the upstream defaults:

* ``VideoMediaIO.load_file`` -> SDK video decode with target frame indices
* ``ImageMediaIO.load_file`` -> SDK image open
"""

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from vllm.multimodal.media.base import MediaWithBytes
from vllm.multimodal.media.video import VideoMediaIO
from vllm.multimodal.media.image import ImageMediaIO
from vllm.multimodal.video import VideoTargetMetadata, VideoBackend

from mm import video_decode
from mm.acc.wrapper.image_wrapper import Image as mmImage


def _load_video(self, file_path: Path) -> tuple[np.typing.NDArray, dict[str, Any]]:
    if not file_path.is_file():
        raise ValueError(f"Not a file: {file_path}")
    num_frames = self.num_frames
    extras = dict(self.kwargs)  # copy: pop must not mutate self.kwargs
    fps = extras.pop("fps", -1)
    max_duration = extras.pop("max_duration", -1)
    target = VideoTargetMetadata(num_frames=num_frames, fps=fps, max_duration=max_duration)

    try:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file {file_path}")
        source = VideoBackend.get_video_metadata(cap)
    finally:
        cap.release()

    frame_idx = VideoBackend.compute_frames_index_to_sample(source=source, target=target, **extras)
    frames = video_decode(file_path.as_posix(), "cpu", frame_idx)
    np_frames = [frame.numpy() for frame in frames]
    return np_frames, VideoBackend.create_hf_metadata(
        source=source, video_backend="multimodal_sdk", valid_frame_indices=frame_idx
    )


def _load_image(self, file_path: Path) -> MediaWithBytes[Image.Image]:
    image = mmImage.open(file_path.as_posix(), b"cpu")
    pillow_image = image.pillow()
    return MediaWithBytes(pillow_image, file_path.read_bytes())


VideoMediaIO.load_file = _load_video
ImageMediaIO.load_file = _load_image
