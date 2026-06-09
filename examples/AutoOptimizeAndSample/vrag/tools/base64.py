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
# Base64 encoding for video frames.
# This module provides utilities for encoding video frames as base64 strings for transmission to API endpoints.

import asyncio
import base64
from io import BytesIO
from typing import List

import numpy as np
from PIL import Image

from vrag.shared import into_u8_frames


def encode_image_to_base64(image: np.ndarray) -> str:
    pil_image = Image.fromarray(into_u8_frames(image)).convert("RGB")
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


async def encode_frames_async(frames: List[np.ndarray]) -> List[str]:
    loop = asyncio.get_running_loop()
    tasks = [loop.run_in_executor(None, encode_image_to_base64, frame) for frame in frames]
    return await asyncio.gather(*tasks)
