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
from vllm.model_executor.models import internvl
import torch
from typing import Union, List, Dict, Tuple, Optional
import PIL.Image as Image
from ...comm.log import _Logger as log
from ...adapter.internvl2_preprocessor import InternVL2PreProcessor


class InternVLProcessorPatcher(internvl.InternVLProcessor):
    """
    Patch the VLLM InternVLProcessor class to use custom image process via
    Multimodal SDK.

    - Overrides _images_to_pixel_values_lst() to preprocess image using Multimodal SDK api.
    - Returns a list of torch tensor.
    """
    def _images_to_pixel_values_lst(
            self,
            images: list[Image.Image],
            min_dynamic_patch: Optional[int] = None,
            max_dynamic_patch: Optional[int] = None,
            dynamic_image_size: Optional[bool] = None,
    ) -> list[torch.Tensor]:
        min_num, max_num = self.resolve_min_max_num(
            min_dynamic_patch=min_dynamic_patch,
            max_dynamic_patch=max_dynamic_patch,
            dynamic_image_size=dynamic_image_size,
            use_thumbnail=False,
        )
        log.info("Multimodal SDK InternVL2 Image Patcher Enabled!")
        input_size = self.image_size
        use_thumbnail = self.use_thumbnail
        internVL2PreProcessor = InternVL2PreProcessor()
        result = []
        for image in images:
            result.append(internVL2PreProcessor.preprocess_image(image, input_size, min_num, max_num, use_thumbnail))
        return result


internvl.InternVLProcessor = InternVLProcessorPatcher