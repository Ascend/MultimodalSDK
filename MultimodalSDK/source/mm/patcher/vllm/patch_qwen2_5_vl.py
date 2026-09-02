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
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/huggingface/transformers/blob/19e6e80e10118f855137b90740936c0b11ac397f/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py
# Copyright 2024 The Qwen team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
vLLM model monkey-patches for Qwen2.5-VL.

This file replaces the following upstream vLLM entry points on
``Qwen2_5_VLForConditionalGeneration`` so that the SDK's SCC token
compression can plug in transparently:

* ``get_mrope_input_positions`` -> align M-RoPE positions with SCC lengths
* ``_process_image_input``      -> SCC-compress ViT image output
* ``_process_video_input``      -> SCC-compress ViT video output
* ``iter_mm_grid_thw``          -> iterate compressed grid for M-RoPE
* ``_get_prompt_updates``       -> shorten placeholders for compressed tokens
"""

import math
from collections.abc import Iterator, Mapping, Sequence
from functools import partial
from typing import Any

import numpy as np
import torch
from vllm.model_executor.models.qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLImageInputs,
    Qwen2_5_VLVideoInputs,
    Qwen2_5_VLMultiModalProcessor,
)
from vllm.multimodal.inputs import MultiModalFeatureSpec, MultiModalKwargsItems
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import PromptReplacement, PromptUpdate
from vllm.model_executor.models.vision import run_dp_sharded_mrope_vision_model

from .constants import MM_SCC_RATE, MM_SCC_TAU, MM_SCC_EPSILON, MM_SCC_MAX_TOKENS_PER_ITEM
from mm.core.scc import scc_shrink, scc_should_run, scc_compress_to_target, set_uniform_true


def _qwen2_5_get_prompt_updates(
    self,
    mm_items: MultiModalDataItems,
    hf_processor_mm_kwargs: Mapping[str, Any],
    out_mm_kwargs: MultiModalKwargsItems,
) -> Sequence[PromptUpdate]:
    hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
    image_processor = self.info.get_image_processor(**hf_processor_mm_kwargs)
    tokenizer = self.info.get_tokenizer()
    vocab = tokenizer.get_vocab()

    placeholder = {
        "image": vocab[hf_processor.image_token],
        "video": vocab[hf_processor.video_token],
    }

    merge_length = image_processor.merge_size**2

    def get_replacement_qwen2vl(item_idx: int, modality: str):
        out_item = out_mm_kwargs[modality][item_idx]
        grid_thw = out_item[f"{modality}_grid_thw"].data
        assert isinstance(grid_thw, torch.Tensor)

        num_tokens = int(grid_thw.prod()) // merge_length

        if modality == "video":
            T, H, W = map(int, grid_thw)
            tokens_per_frame = (H // image_processor.merge_size) * (W // image_processor.merge_size)
            if scc_should_run(tokens_per_frame, MM_SCC_MAX_TOKENS_PER_ITEM):
                num_tokens = scc_shrink(tokens_per_frame, MM_SCC_RATE) * T
        else:
            if scc_should_run(num_tokens, MM_SCC_MAX_TOKENS_PER_ITEM):
                num_tokens = scc_shrink(num_tokens, MM_SCC_RATE)

        return [placeholder[modality]] * math.ceil(num_tokens)

    return [
        PromptReplacement(
            modality=modality,
            target=[placeholder[modality]],
            replacement=partial(get_replacement_qwen2vl, modality=modality),
        )
        for modality in ("image", "video")
    ]


def _qwen2_5_iter_mm_grid_thw(
    self, mm_features: list[MultiModalFeatureSpec]
) -> Iterator[tuple[int, int, int, int, float, int]]:
    """
    Iterate over multimodal features and yield grid information.

    Args:
        mm_features: List of multimodal feature specifications

    Yields:
        Tuple of (offset, grid_t, grid_h, grid_w, t_factor, position_length) for each frame/image
    """
    spatial_merge_size = self.config.vision_config.spatial_merge_size
    tokens_per_second = getattr(self.config.vision_config, "tokens_per_second", 1.0)
    for mm_feature in sorted(mm_features, key=lambda f: f.mm_position.offset):
        offset = mm_feature.mm_position.offset
        if mm_feature.modality == "image":
            t, h, w = mm_feature.data["image_grid_thw"].data.tolist()
            assert t == 1, f"Image must have 1 frame, got {t}"
            yield offset, 1, h // spatial_merge_size, w // spatial_merge_size, 1.0, mm_feature.mm_position.length
        elif mm_feature.modality == "video":
            t, h, w = mm_feature.data["video_grid_thw"].data.tolist()
            second_per_grid_ts = 1.0
            if mm_feature.data.get("second_per_grid_ts", None):
                second_per_grid_ts = mm_feature.data["second_per_grid_ts"].data.item()
            t_factor = second_per_grid_ts * tokens_per_second
            yield (offset, t, h // spatial_merge_size, w // spatial_merge_size, t_factor, mm_feature.mm_position.length)
        else:
            raise ValueError(f"Unsupported modality: {mm_feature.modality}")


def _qwen2_5_get_mrope_input_positions(
    self,
    input_tokens: list[int],
    mm_features: list[MultiModalFeatureSpec],
) -> tuple[torch.Tensor, int]:
    llm_pos_ids_list: list = []
    st = 0

    for (
        offset,
        llm_grid_t,
        llm_grid_h,
        llm_grid_w,
        t_factor,
        pos_length,
    ) in self.iter_mm_grid_thw(mm_features):
        text_len = offset - st
        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
        llm_pos_ids_list.append(np.broadcast_to(np.arange(text_len), (3, text_len)) + st_idx)

        grid_indices = np.indices((llm_grid_t, llm_grid_h, llm_grid_w))
        if t_factor != 1.0:
            grid_indices[0] = (grid_indices[0] * t_factor).astype(np.int64)

        media_len = int(pos_length)
        media_positions = grid_indices.reshape(3, -1)

        if media_len != llm_grid_t * llm_grid_h * llm_grid_w:
            media_positions = media_positions[:, set_uniform_true(llm_grid_t * llm_grid_h * llm_grid_w, media_len)]

        llm_pos_ids_list.append(media_positions.reshape(3, -1) + text_len + st_idx)
        st = offset + media_len

    if st < len(input_tokens):
        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
        text_len = len(input_tokens) - st
        llm_pos_ids_list.append(np.broadcast_to(np.arange(text_len), (3, text_len)) + st_idx)

    llm_positions = np.concatenate(llm_pos_ids_list, axis=1).reshape(3, -1)
    mrope_position_delta = (llm_positions.max() + 1 - len(input_tokens)).item()

    return torch.from_numpy(llm_positions), mrope_position_delta


def _qwen2_5_process_image_input(self, image_input: Qwen2_5_VLImageInputs) -> tuple[torch.Tensor, ...]:
    grid_thw = image_input["image_grid_thw"]
    assert grid_thw.ndim == 2
    grid_thw_list = grid_thw.tolist()

    if image_input["type"] == "image_embeds":
        image_embeds = image_input["image_embeds"].type(self.visual.dtype)
    else:
        pixel_values = image_input["pixel_values"]
        if self.use_data_parallel:
            return run_dp_sharded_mrope_vision_model(self.visual, pixel_values, grid_thw_list, rope_type="rope_3d")
        else:
            image_embeds = self.visual(pixel_values, grid_thw=grid_thw_list)

    # Split concatenated embeddings for each image item.
    merge_size = self.visual.spatial_merge_size
    sizes = (grid_thw.prod(-1) // merge_size // merge_size).tolist()
    image_embeds_split = image_embeds.split(sizes)

    out = []

    for i, feat in enumerate(image_embeds_split):
        n = feat.shape[0]

        if scc_should_run(n, MM_SCC_MAX_TOKENS_PER_ITEM):
            t, h, w = (int(x) for x in grid_thw[i].tolist())

            k_target = scc_shrink(n, MM_SCC_RATE)

            result = scc_compress_to_target(feat, k_target, t, 0, MM_SCC_TAU, MM_SCC_EPSILON)
            out.append(result)
        else:
            out.append(feat)

    return tuple(out)


def _qwen2_5_process_video_input(self, video_input: Qwen2_5_VLVideoInputs) -> tuple[torch.Tensor, ...]:
    grid_thw = video_input["video_grid_thw"]
    assert grid_thw.ndim == 2
    grid_thw_list = grid_thw.tolist()

    if video_input["type"] == "video_embeds":
        video_embeds = video_input["video_embeds"].type(self.visual.dtype)
    else:
        pixel_values_videos = video_input["pixel_values_videos"]
        if self.use_data_parallel:
            return run_dp_sharded_mrope_vision_model(
                self.visual,
                pixel_values_videos,
                grid_thw_list,
                rope_type="rope_3d",
            )
        else:
            video_embeds = self.visual(pixel_values_videos, grid_thw=grid_thw_list)

    # Split concatenated embeddings for each video item.
    merge_size = self.visual.spatial_merge_size
    sizes = (grid_thw.prod(-1) // merge_size // merge_size).tolist()
    video_embeds_split = video_embeds.split(sizes)

    out = []

    for i, feat in enumerate(video_embeds_split):
        t, h, w = (int(x) for x in grid_thw[i].tolist())

        token_per_frame = h * w // merge_size // merge_size

        if scc_should_run(token_per_frame, MM_SCC_MAX_TOKENS_PER_ITEM):
            k_target = scc_shrink(token_per_frame, MM_SCC_RATE) * t
            compressed_feat = scc_compress_to_target(
                feat, k_target, t, h * w // merge_size // merge_size, MM_SCC_TAU, MM_SCC_EPSILON
            )
            out.append(compressed_feat)
        else:
            out.append(feat)

    return tuple(out)


Qwen2_5_VLForConditionalGeneration.get_mrope_input_positions = _qwen2_5_get_mrope_input_positions
Qwen2_5_VLForConditionalGeneration._process_image_input = _qwen2_5_process_image_input
Qwen2_5_VLForConditionalGeneration._process_video_input = _qwen2_5_process_video_input
Qwen2_5_VLForConditionalGeneration.iter_mm_grid_thw = _qwen2_5_iter_mm_grid_thw
Qwen2_5_VLMultiModalProcessor._get_prompt_updates = _qwen2_5_get_prompt_updates
