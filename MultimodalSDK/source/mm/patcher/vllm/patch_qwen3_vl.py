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

# Copyright 2025 The vLLM team.
# Copyright 2025 The Qwen Team.
# Copyright 2025 The HuggingFace Inc. team.
# All rights reserved.
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
vLLM model + processor monkey-patches for Qwen3-VL.

This file replaces the following upstream vLLM entry points on
``Qwen3VLForConditionalGeneration`` / ``Qwen3VLMultiModalProcessor`` so that
the SDK's SCC token compression can plug in transparently:

* ``_get_prompt_updates``     -> shorten placeholders for compressed tokens
* ``_process_image_input``    -> SCC-compress ViT image output
* ``_process_video_input``    -> SCC-compress ViT video output
* ``_iter_mm_grid_hw``        -> return compressed grid for M-RoPE
* ``_get_mrope_input_positions`` -> align M-RoPE positions with SCC lengths

The function signatures intentionally mirror the upstream vLLM API so
each monkey-patch is a drop-in replacement.
"""

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import numpy as np
import torch

# pylint: disable=ungrouped-imports
# vLLM imports are interleaved by intent: qwen2_5_vl / qwen3_vl share types
# but they live in different vllm sub-modules, so alphabetizing them within
# a single group would obscure the cross-mention. Keep them grouped by
# source package (vllm vs transformers vs local) instead.
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.inputs import MultiModalKwargsItems, MultiModalFeatureSpec
from vllm.multimodal.processing import PromptUpdate
from vllm.multimodal.processing import PromptReplacement
from vllm.utils.collection_utils import is_list_of
from vllm.model_executor.models.qwen2_5_vl import Qwen2_5_VLImageInputs, Qwen2_5_VLVideoInputs
from vllm.model_executor.models.qwen3_vl import Qwen3VLMultiModalProcessor, Qwen3VLForConditionalGeneration
from vllm.model_executor.models.vision import run_dp_sharded_mrope_vision_model
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig

from .constants import MM_SCC_RATE, MM_SCC_TAU, MM_SCC_EPSILON
from mm.core.scc import scc_shrink, scc_should_run, scc_compress_to_target, set_uniform_true


def _qwen3_get_prompt_updates(
    self,
    mm_items: MultiModalDataItems,
    hf_processor_mm_kwargs: Mapping[str, Any],
    out_mm_kwargs: MultiModalKwargsItems,
) -> Sequence[PromptUpdate]:
    hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
    image_processor = self.info.get_image_processor(**hf_processor_mm_kwargs)
    tokenizer = self.info.get_tokenizer()
    hf_config = self.info.get_hf_config()

    video_token_id = hf_config.video_token_id
    vision_start_token_id = hf_config.vision_start_token_id
    vision_end_token_id = hf_config.vision_end_token_id

    merge_length = image_processor.merge_size**2

    def get_image_replacement_qwen3vl(item_idx: int):
        out_item = out_mm_kwargs["image"][item_idx]
        grid_thw = out_item["image_grid_thw"].data
        assert isinstance(grid_thw, torch.Tensor)

        num_tokens = int(grid_thw.prod()) // merge_length

        if scc_should_run(num_tokens):
            r = [hf_processor.image_token_id] * scc_shrink(num_tokens, MM_SCC_RATE)
        else:
            r = [hf_processor.image_token_id] * num_tokens

        return r

    def get_video_replacement_qwen3vl(item_idx: int):
        out_item = out_mm_kwargs["video"][item_idx]
        grid_thw = out_item["video_grid_thw"].data
        assert isinstance(grid_thw, torch.Tensor)

        sampled_fps = hf_processor_mm_kwargs.get("fps")
        if is_list_of(sampled_fps, float):
            sampled_fps = sampled_fps[item_idx]

        timestamps = out_item["timestamps"].data
        assert len(timestamps) == grid_thw[0], (
            f"The timestamps length({len(timestamps)}) should be equal video length ({grid_thw[0]})."
        )

        # Compute tokens per frame, with EVS support
        num_frames = int(grid_thw[0])
        tokens_per_frame_base = int(grid_thw[1:].prod()) // merge_length

        if scc_should_run(tokens_per_frame_base):
            tokens_per_frame = [scc_shrink(tokens_per_frame_base, MM_SCC_RATE)] * num_frames
        else:
            tokens_per_frame = [tokens_per_frame_base] * num_frames

        select_token_id = True

        return Qwen3VLMultiModalProcessor.get_video_repl(
            tokens_per_frame=tokens_per_frame,
            timestamps=timestamps,
            tokenizer=tokenizer,
            vision_start_token_id=vision_start_token_id,
            vision_end_token_id=vision_end_token_id,
            video_token_id=video_token_id,
            select_token_id=select_token_id,
        )

    result = [
        PromptReplacement(
            modality="image",
            target=hf_processor.image_token,
            replacement=get_image_replacement_qwen3vl,
        ),
        # NOTE: We match string on purpose since searching sequence of
        # token ids takes more time.
        PromptReplacement(
            modality="video",
            target="<|vision_start|><|video_pad|><|vision_end|>",
            replacement=get_video_replacement_qwen3vl,
        ),
    ]

    return result


def _qwen3_process_image_input(self, image_input: Qwen2_5_VLImageInputs):
    grid_thw = image_input["image_grid_thw"]
    assert grid_thw.ndim == 2

    if image_input["type"] == "image_embeds":
        image_embeds = image_input["image_embeds"].type(self.visual.dtype)
    else:
        pixel_values = image_input["pixel_values"].type(self.visual.dtype)
        if self.use_data_parallel:
            return run_dp_sharded_mrope_vision_model(self.visual, pixel_values, grid_thw.tolist(), rope_type="rope_3d")
        else:
            image_embeds = self.visual(pixel_values, grid_thw=grid_thw)

    # Split concatenated embeddings for each image item.
    merge_size = self.visual.spatial_merge_size
    sizes = (grid_thw.prod(-1) // merge_size // merge_size).tolist()

    image_embeds_split = image_embeds.split(sizes)

    image_embeds_out = []

    for emb, size in zip(image_embeds_split, grid_thw):
        if scc_should_run(emb.shape[0]):
            r = scc_compress_to_target(emb, scc_shrink(emb.shape[0], MM_SCC_RATE), 1, 0, MM_SCC_TAU, MM_SCC_EPSILON)

            emb = r
            image_embeds_out.append(emb)
        else:
            image_embeds_out.append(emb)

    return tuple(image_embeds_out)


def _qwen3_process_video_input(self, video_input: Qwen2_5_VLVideoInputs) -> tuple[torch.Tensor, ...]:
    grid_thw = video_input["video_grid_thw"]
    assert grid_thw.ndim == 2

    if video_input["type"] == "video_embeds":
        video_embeds = video_input["video_embeds"].type(self.visual.dtype)
    else:
        pixel_values_videos = video_input["pixel_values_videos"].type(self.visual.dtype)
        if self.use_data_parallel:
            grid_thw_list = grid_thw.tolist()
            return run_dp_sharded_mrope_vision_model(
                self.visual, pixel_values_videos, grid_thw_list, rope_type="rope_3d"
            )
        else:
            video_embeds = self.visual(pixel_values_videos, grid_thw=grid_thw)

    # Split concatenated embeddings for each video item.
    merge_size = self.visual.spatial_merge_size
    sizes = (grid_thw.prod(-1) // merge_size // merge_size).tolist()
    video_embeds_split = video_embeds.split(sizes)

    video_embeds_out = []

    for video_idx, (emb, grid_size) in enumerate(zip(video_embeds_split, grid_thw.tolist())):
        # Compute positions.
        timestamps = video_input.timestamps[video_idx]
        num_frames = len(timestamps)

        t, h, w = grid_size

        token_per_frame = h * w // merge_size // merge_size

        if scc_should_run(token_per_frame):
            emb = scc_compress_to_target(
                emb,
                scc_shrink(token_per_frame, MM_SCC_RATE) * num_frames,
                t,
                token_per_frame,
                MM_SCC_TAU,
                MM_SCC_EPSILON,
            )

        video_embeds_out.append(emb)

    return tuple(video_embeds_out)


@staticmethod
def _qwen3_iter_mm_grid_hw(
    input_tokens: list[int],
    mm_features: list[MultiModalFeatureSpec],
    video_token_id: int,
    vision_start_token_id: int,
    vision_end_token_id: int,
    spatial_merge_size: int,
) -> Iterator[tuple[int, int, int, int]]:
    """Iterate over multimodal features and yield position info.

    Args:
        input_tokens: List of token IDs in the input sequence.
        mm_features: List of multimodal feature specifications containing
            image/video data and position information.
        video_token_id: Token ID used for video tokens.
        vision_start_token_id: Token ID marking the start of a vision sequence.
        vision_end_token_id: Token ID marking the end of a vision sequence.
        spatial_merge_size: Size of the spatial merge operation used to
            compute logical grid dimensions from the original feature grid.

    Yields:
        offset: Position of the first video/image token in the sequence.
        llm_grid_h: Logical grid height (may not match actual token count with EVS).
        llm_grid_w: Logical grid width (may not match actual token count with EVS).
        actual_num_tokens: Actual number of video/image tokens in the placeholder.
    """

    for mm_feature in sorted(mm_features, key=lambda f: f.mm_position.offset):
        offset = mm_feature.mm_position.offset
        if mm_feature.modality == "image":
            t, h, w = mm_feature.data["image_grid_thw"].data.tolist()
            assert t == 1, f"Image must have 1 frame, got {t}"
            llm_grid_h = h // spatial_merge_size
            llm_grid_w = w // spatial_merge_size
            if scc_should_run(llm_grid_h * llm_grid_w):
                yield offset, llm_grid_h, llm_grid_w, scc_shrink(llm_grid_h * llm_grid_w, MM_SCC_RATE)
            else:
                yield offset, llm_grid_h, llm_grid_w, llm_grid_h * llm_grid_w

        elif mm_feature.modality == "video":
            t, h, w = mm_feature.data["video_grid_thw"].data.tolist()
            llm_grid_h = h // spatial_merge_size
            llm_grid_w = w // spatial_merge_size

            for _ in range(t):
                # When EVS is enabled, some frames may have 0 video tokens in the
                # placeholder. We use `vision_start_token_id` to locate each frame
                # since it is always present for every frame.
                # We then look for the first `video_token_id` after
                # `vision_start_token_id` and before `vision_end_token_id`.
                offset = input_tokens.index(vision_start_token_id, offset)
                vision_end_offset = input_tokens.index(vision_end_token_id, offset)

                try:
                    actual_num_tokens = 0
                    video_offset = input_tokens.index(video_token_id, offset, vision_end_offset)
                    # NOTE: looking at the
                    # `Qwen3VLMultiModalProcessor.get_video_repl` code, we can
                    # see that we can use the below formula to get the token
                    # count, since everything in between `video_offset` and
                    # `vision_end_offset` is populated as `video_token_id`.
                    # This saves us from manually counting the number tokens
                    # that match `video_token_id` in between.
                    actual_num_tokens += vision_end_offset - video_offset
                except ValueError:
                    # No `video_token_id` in this frame (EVS with 0 tokens for
                    # this frame) -> use `offset + 1`` to move past
                    # `vision_start_token_id`.
                    video_offset = offset + 1

                yield video_offset, llm_grid_h, llm_grid_w, actual_num_tokens
                # Move offset past this frame for next iteration.
                offset = vision_end_offset + 1
        else:
            raise ValueError(f"Unsupported modality: {mm_feature.modality}")


@staticmethod
def _qwen3_get_mrope_input_positions(
    input_tokens: list[int],
    mm_features: list[MultiModalFeatureSpec],
    config: Qwen3VLConfig,
):
    llm_pos_ids_list = []
    st = 0
    for (
        offset,
        llm_grid_h,
        llm_grid_w,
        actual_num_tokens,
    ) in _qwen3_iter_mm_grid_hw(
        input_tokens,
        mm_features,
        video_token_id=config.video_token_id,
        vision_start_token_id=config.vision_start_token_id,
        vision_end_token_id=config.vision_end_token_id,
        spatial_merge_size=config.vision_config.spatial_merge_size,
    ):
        # Skip frames with 0 tokens (EVS placeholder with tokens lumped elsewhere)
        if actual_num_tokens == 0:
            continue

        text_len = offset - st
        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
        llm_pos_ids_list.append(np.broadcast_to(np.arange(text_len), (3, text_len)) + st_idx)

        # Check if this is a "lumped placeholder" (all tokens from multiple frames
        # assigned to the 0-th frame - see
        # `Qwen3VLMultiModalProcessor.get_video_repl`.
        expected_tokens_per_frame = llm_grid_h * llm_grid_w

        grid_indices = np.indices((1, llm_grid_h, llm_grid_w)).reshape(3, -1)[
            :, set_uniform_true(expected_tokens_per_frame, actual_num_tokens).numpy()
        ]
        llm_pos_ids_list.append(grid_indices + text_len + st_idx)

        st = offset + actual_num_tokens

    if st < len(input_tokens):
        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
        text_len = len(input_tokens) - st
        llm_pos_ids_list.append(np.broadcast_to(np.arange(text_len), (3, text_len)) + st_idx)

    llm_positions = np.concatenate(llm_pos_ids_list, axis=1).reshape(3, -1)
    mrope_position_delta = (llm_positions.max() + 1 - len(input_tokens)).item()
    return torch.from_numpy(llm_positions), mrope_position_delta


Qwen3VLMultiModalProcessor._get_prompt_updates = _qwen3_get_prompt_updates
Qwen3VLForConditionalGeneration._process_image_input = _qwen3_process_image_input
Qwen3VLForConditionalGeneration._process_video_input = _qwen3_process_video_input
Qwen3VLForConditionalGeneration._iter_mm_grid_hw = _qwen3_iter_mm_grid_hw
Qwen3VLForConditionalGeneration._get_mrope_input_positions = _qwen3_get_mrope_input_positions
