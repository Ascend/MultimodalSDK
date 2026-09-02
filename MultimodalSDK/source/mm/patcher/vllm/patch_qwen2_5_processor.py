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
# `https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py`
# `https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_vl/video_processing_qwen2_vl.py`
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
vLLM processor monkey-patches for Qwen2.5-VL.

This file replaces ``Qwen2VLImageProcessor._preprocess`` and
``Qwen2VLVideoProcessor._preprocess`` so that the SDK's accelerated
``mm.core.processor.resize_and_normalize`` is invoked instead of the
default HuggingFace path.

The function signatures below intentionally mirror the upstream vLLM
``_preprocess`` API so the monkey-patch is a drop-in replacement.
"""

import torch

from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
from transformers import BatchFeature
from transformers.image_utils import SizeDict, PILImageResampling
from transformers.utils.generic import TensorType
from transformers import Qwen2VLImageProcessor, Qwen2VLVideoProcessor


from mm.core.processor import resize_and_normalize


def _preprocess_image(
    self,
    images: list["torch.Tensor"],
    do_resize: bool,
    size: SizeDict,
    resample: "PILImageResampling | int | None",
    do_rescale: bool,
    rescale_factor: float,
    do_normalize: bool,
    image_mean: float | list[float] | None,
    image_std: float | list[float] | None,
    patch_size: int,
    temporal_patch_size: int,
    merge_size: int,
    disable_grouping: bool | None,
    return_tensors: str | TensorType | None,
    **kwargs,
):
    all_patches = []
    all_grids = []
    for image in images:
        if len(image.shape) != 3:
            raise ValueError("image shape is not permitted")
        image = image.unsqueeze(0)
        height, width = image.shape[-2:]
        h, w = smart_resize(height, width, patch_size * merge_size, size.shortest_edge, size.longest_edge)

        patches = resize_and_normalize(image, h, w, image_mean, image_std)

        if len(patches.shape) == 4:
            patches = patches.unsqueeze(1)
        T = patches.shape[1]
        if pad := -T % temporal_patch_size:
            repeats = patches[:, -1:].expand(-1, pad, -1, -1, -1)
            patches = torch.cat((patches, repeats), dim=1)
        batch_size, grid_t, channel = patches.shape[:3]
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
        all_grids.append(torch.tensor([[grid_t, grid_h, grid_w]] * batch_size))

    pixel_values_videos = torch.cat(all_patches, dim=0)
    video_grid_thw = torch.cat(all_grids, dim=0)
    return BatchFeature(
        data={"pixel_values": pixel_values_videos, "image_grid_thw": video_grid_thw},
        tensor_type=return_tensors,
    )


def _preprocess_video(
    self,
    videos: list["torch.Tensor"],
    do_resize: bool,
    size: SizeDict,
    resample: "PILImageResampling | int | None",
    do_rescale: bool,
    rescale_factor: float,
    do_normalize: bool,
    image_mean: float | list[float] | None,
    image_std: float | list[float] | None,
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
        height, width = video.shape[-2:]
        h, w = smart_resize(height, width, patch_size * merge_size, size.shortest_edge, size.longest_edge)

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


Qwen2VLImageProcessor._preprocess = _preprocess_image
Qwen2VLVideoProcessor._preprocess = _preprocess_video
