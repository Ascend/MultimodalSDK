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
from pathlib import Path
import numpy as np
from vllm.multimodal import video
from ...comm.log import _Logger as log
from ...acc.wrapper.video_wrapper import video_decode


class VideoMediaIOPatcher(video.VideoMediaIO):
    """
    Patch the VLLM VideoMediaIO class to use custom video decoding via
    Multimodal SDK.

    - Overrides load_file() to decode videos using video_decode.
    - Returns a numpy array of shape [N, H, W, C] for N frames.
    """
    def load_file(self, filepath: Path) -> np.ndarray:
        log.info("Multimodal SDK Video Patcher Enabled!")
        file_path = str(filepath)
        # Decode video frames on CPU, limit to self.num_frames which __init__ from vllm media io
        mm_images = video_decode(file_path, 'cpu', None, self.num_frames)  # [[H, W, C], ...]
        frames = [img.numpy() for img in mm_images]  # [[H, W, C], ...]
        # Stack frames into a single array [N, H, W, C]
        frames = np.stack(frames, axis=0)
        return frames

# Override the original VideoMediaIO with the patched version
# current support VLLM-Ascend 0.8.5.rc1
video.VideoMediaIO = VideoMediaIOPatcher
