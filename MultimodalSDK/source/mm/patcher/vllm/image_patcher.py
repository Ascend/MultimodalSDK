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
from vllm.multimodal import image
from PIL import Image
from ...comm.log import _Logger as log
from ...acc.wrapper.image_wrapper import Image as mmImage

DEVICE_CPU = "cpu"
MAX_IMAGE_SIZE = 4096
MIN_IMAGE_SIZE = 10


class ImageMediaIOPatcher(image.ImageMediaIO):
    """
    Patch the VLLM ImageMediaIO class to use custom image decoding via
    Multimodal SDK.

    - Overrides load_file() to decode image using video_decode.
    - Returns a PIL image.
    """

    def load_file(self, filepath: Path) -> Image.Image:
        log.info("Multimodal SDK Image Patcher Enabled!")
        file_path = str(filepath)
        mm_images = mmImage.open(file_path, DEVICE_CPU)
        img_width = mm_images.width
        img_height = mm_images.height
        if (img_width > MAX_IMAGE_SIZE or img_height > MAX_IMAGE_SIZE or
                img_width < MIN_IMAGE_SIZE or img_height < MIN_IMAGE_SIZE):
            raise ValueError("The input image size must in range [10, 4096].")
        pillow_image = mm_images.pillow()
        return pillow_image


# Override the original ImageMediaIO with the patched version
# current support VLLM-Ascend 0.8.5.rc1
from vllm.multimodal import image
image.ImageMediaIO = ImageMediaIOPatcher
