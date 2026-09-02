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
# Adapted from
# `https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/video_processing_qwen3_vl.py`
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -------------------------------------------------------------------------
# pylint: disable=duplicate-code
"""
HuggingFace Transformers processor monkey-patches for Qwen3-VL.

This file replaces ``Qwen3VLVideoProcessor._preprocess`` so that the
SDK's accelerated ``mm.core.processor.resize_and_normalize`` is
invoked instead of the default HuggingFace path. Function signatures
intentionally mirror the upstream ``_preprocess`` API so this is a
drop-in replacement.
"""

import torch

from transformers.models.qwen3_vl.video_processing_qwen3_vl import smart_resize
from transformers import BatchFeature
from transformers.image_utils import SizeDict, PILImageResampling
from transformers.utils.generic import TensorType
from transformers import Qwen3VLVideoProcessor

from mm.core.processor import resize_and_normalize


def _preprocess_video(
    self,
    videos: list[torch.Tensor],
    do_convert_rgb: bool = True,
    do_resize: bool = True,
    size: SizeDict | None = None,
    resample: "PILImageResampling | int | None" = PILImageResampling.BICUBIC,
    do_rescale: bool = True,
    rescale_factor: float = 1 / 255.0,
    do_normalize: bool = True,
    image_mean: float | list[float] | None = None,
    image_std: float | list[float] | None = None,
    patch_size: int | None = None,
    temporal_patch_size: int | None = None,
    merge_size: int | None = None,
    return_tensors: str | TensorType | None = None,
    **kwargs,
):
    all_patches = []
    all_grids = []
    for video in videos:
        if len(video.shape) != 4:
            raise ValueError("video shape is not permitted")
        T, _, height, width = video.shape

        h, w = smart_resize(
            num_frames=T,
            height=height,
            width=width,
            temporal_factor=temporal_patch_size,
            factor=patch_size * merge_size,
            min_pixels=size.shortest_edge,
            max_pixels=size.longest_edge,
        )

        patches = resize_and_normalize(video, h, w, image_mean, image_std)

        T = patches.shape[0]
        if pad := -T % temporal_patch_size:
            repeats = patches[-1:].expand(pad, -1, -1, -1)
            patches = torch.cat((patches, repeats), dim=0)

        grid_t, channel = patches.shape[:2]
        grid_t = grid_t // temporal_patch_size
        grid_h, grid_w = h // patch_size, w // patch_size
        patches = patches.reshape(
            grid_t,
            temporal_patch_size,
            channel,
            grid_h // merge_size,
            merge_size,
            patch_size,
            grid_w // merge_size,
            merge_size,
            patch_size,
        )
        patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
        flatten_patches = patches.reshape(
            grid_t * grid_h * grid_w,
            channel * temporal_patch_size * patch_size * patch_size,
        )
        all_patches.append(flatten_patches)
        all_grids.append(torch.tensor([[grid_t, grid_h, grid_w]]))

    pixel_values_videos = torch.cat(all_patches, dim=0)
    video_grid_thw = torch.cat(all_grids, dim=0)
    return BatchFeature(
        data={"pixel_values_videos": pixel_values_videos, "video_grid_thw": video_grid_thw},
        tensor_type=return_tensors,
    )


Qwen3VLVideoProcessor._preprocess = _preprocess_video
