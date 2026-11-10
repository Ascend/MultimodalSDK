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

import torch

from vllm.model_executor.layers.attention import MMEncoderAttention
from vllm.model_executor.models.interfaces import MultiModalEmbeddings
# pylint: disable=ungrouped-imports
# vLLM imports are interleaved by intent: qwen2_5_vl / qwen3_vl share types
# but they live in different vllm sub-modules, so alphabetizing them within
# a single group would obscure the cross-mention. Keep them grouped by

from vllm.model_executor.models.qwen2_5_vl import Qwen2_5_VLVideoInputs
from vllm.model_executor.models.qwen3_vl import (
    Qwen3VLForConditionalGeneration,
    Qwen3_VisionBlock,
    Qwen3_VisionTransformer,
)
from vllm.model_executor.models.vision import run_dp_sharded_mrope_vision_model
from vllm.config import VllmConfig

from .method import flashvid_compression, get_prune_config
from .token_selection import prune_embeds, restore_hidden_states

from vllm.logger import init_logger

logger = init_logger("vllm.flashvid")


def _qwen3_vision_block_forward(
    self,
    x: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb_cos: torch.Tensor,
    rotary_pos_emb_sin: torch.Tensor,
    max_seqlen: torch.Tensor,  # Only used for Flash Attention
    sequence_lengths: torch.Tensor,  # Only used for FlashInfer CuDNN backend
    return_attn_score: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    attn_output, frame_scores = self.attn(
        self.norm1(x),
        cu_seqlens=cu_seqlens,
        rotary_pos_emb_cos=rotary_pos_emb_cos,
        rotary_pos_emb_sin=rotary_pos_emb_sin,
        max_seqlen=max_seqlen,
        sequence_lengths=sequence_lengths,
        return_attn_score=return_attn_score,
    )
    x = x + attn_output
    x = x + self.mlp(self.norm2(x))
    return x, frame_scores


def _qwen3_vision_transformer_forward(
    self,
    x: torch.Tensor,
    grid_thw: torch.Tensor | list[list[int]],
    *,
    encoder_metadata: dict[str, torch.Tensor] | None = None,
    return_attn_score: bool = False,
    prune_encoder: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    hidden_states = x.to(device=self.device, dtype=self.dtype, non_blocking=True)
    hidden_states = self.patch_embed(hidden_states)

    if encoder_metadata is None:
        if isinstance(grid_thw, list):
            grid_thw_list = grid_thw
        else:
            grid_thw_list = grid_thw.tolist()
        encoder_metadata = self.prepare_encoder_metadata(grid_thw_list)

    pos_embeds = encoder_metadata["pos_embeds"]
    seq_len = hidden_states.shape[0]
    cu_seqlens = encoder_metadata["cu_seqlens"]
    rotary_pos_emb_cos = encoder_metadata["rotary_pos_emb_cos"]
    rotary_pos_emb_sin = encoder_metadata["rotary_pos_emb_sin"]

    is_pruned = False
    keep_ids_tensor = None
    remove_ids_tensor = None
    if prune_encoder:
        boundaries = cu_seqlens.detach().cpu().tolist()
        new_cu_seqlens = [torch.zeros((), dtype=torch.int64, device=hidden_states.device)]
        keep_ids = []
        remove_ids = []
        for i, (s, e) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            keep_indices, remove_indices = prune_embeds(hidden_states[s:e])
            keep_ids.append(keep_indices + s)
            remove_ids.append(remove_indices + s)
            new_cu_seqlens.append(new_cu_seqlens[-1] + len(keep_indices))

        keep_ids_tensor = torch.cat(keep_ids).to(hidden_states.device)
        remove_ids_tensor = torch.cat(remove_ids).to(hidden_states.device)
        if keep_ids_tensor.shape[0] < seq_len:
            logger.debug("[FlashVID] encoder pruned %d tokens", seq_len - keep_ids_tensor.shape[0])
            is_pruned = True
            hidden_states = hidden_states[keep_ids_tensor]
            pos_embeds = pos_embeds[keep_ids_tensor]
            rotary_pos_emb_cos = rotary_pos_emb_cos[keep_ids_tensor]
            rotary_pos_emb_sin = rotary_pos_emb_sin[keep_ids_tensor]
            cu_seqlens = torch.stack(new_cu_seqlens).to(dtype=cu_seqlens.dtype)

            # 更新 max_seqlen 和 sequence_lengths
            max_seqlen_val = MMEncoderAttention.compute_max_seqlen(self.attn_backend, cu_seqlens.cpu().numpy())
            encoder_metadata["max_seqlen"] = torch.tensor(max_seqlen_val, dtype=torch.int32)
            encoder_metadata["sequence_lengths"] = MMEncoderAttention.maybe_compute_seq_lens(
                self.attn_backend, cu_seqlens.cpu().numpy(), hidden_states.device
            )
            cu_seqlens = cu_seqlens.to(device=hidden_states.device)

    hidden_states = hidden_states + pos_embeds
    hidden_states = hidden_states.unsqueeze(1)

    deepstack_feature_lists = []
    attn_score: torch.Tensor | None = None
    for layer_num, blk in enumerate(self.blocks):
        collect_scores = return_attn_score and (layer_num == len(self.blocks) - 1)
        hidden_states, attn_score = blk(
            hidden_states,
            cu_seqlens=cu_seqlens,
            rotary_pos_emb_cos=rotary_pos_emb_cos,
            rotary_pos_emb_sin=rotary_pos_emb_sin,
            max_seqlen=encoder_metadata["max_seqlen"],
            sequence_lengths=encoder_metadata.get("sequence_lengths"),
            return_attn_score=collect_scores,
        )
        if layer_num in self.deepstack_visual_indexes:
            deepstack_merger_idx = self.deepstack_visual_indexes.index(layer_num)
            deepstack_hidden_states = hidden_states
            if is_pruned:
                deepstack_hidden_states, _ = restore_hidden_states(
                    hidden_states,
                    seq_len,
                    keep_ids_tensor,
                    remove_ids_tensor,
                    None,
                )
            deepstack_feature = self.deepstack_merger_list[deepstack_merger_idx](deepstack_hidden_states)
            deepstack_feature_lists.append(deepstack_feature)

    if is_pruned:
        hidden_states, attn_score = restore_hidden_states(
            hidden_states,
            seq_len,
            keep_ids_tensor,
            remove_ids_tensor,
            attn_score,
        )

    hidden_states = self.merger(hidden_states)
    hidden_states = torch.cat(
        [hidden_states] + deepstack_feature_lists, dim=1
    )  # [seq_len, hidden_size * (1 + depth_of_deepstack)]
    if not return_attn_score:
        return hidden_states

    attn_score = attn_score.view(-1, self.spatial_merge_unit).mean(-1)

    return hidden_states, attn_score


def _process_video_input(
    self, video_input: Qwen2_5_VLVideoInputs
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...] | None]:
    grid_thw = video_input["video_grid_thw"]
    if video_input["type"] == "video_embeds":
        if self.prune_config.enable_flashvid:
            raise ValueError(
                "FlashVID requires pixel_values_videos to compute attention "
                "scores; precomputed video_embeds are not supported"
            )
        video_embeds = video_input["video_embeds"].type(self.visual.dtype)
        attn_scores = None
    else:
        pixel_values_videos = video_input["pixel_values_videos"].type(self.visual.dtype)
        if self.use_data_parallel:
            if self.prune_config.enable_flashvid:
                raise ValueError("FlashVID does not support data-parallel vision execution without attention scores")
            grid_thw_list = grid_thw.tolist()
            video_embeds_split = run_dp_sharded_mrope_vision_model(
                self.visual,
                pixel_values_videos,
                grid_thw_list,
                rope_type="rope_3d",
            )
            return tuple(video_embeds_split), None
        else:
            visual_output = self.visual(
                pixel_values_videos,
                grid_thw=grid_thw,
                return_attn_score=self.prune_config.enable_flashvid,
                prune_encoder=self.prune_config.enable_encode_prune,
            )
            if self.prune_config.enable_flashvid:
                video_embeds, attn_scores = visual_output
            else:
                video_embeds, attn_scores = visual_output, None

    # Split concatenated embeddings for each video item.
    merge_size = self.visual.spatial_merge_size
    sizes = (grid_thw.prod(-1) // merge_size // merge_size).tolist()
    if attn_scores is None:
        return video_embeds.split(sizes), None

    return video_embeds.split(sizes), attn_scores.split(sizes)


def _postprocess_video_embeds_flashvid(
    self,
    video_embeds_split: tuple[torch.Tensor, ...],
    attn_score_split: tuple[torch.Tensor, ...] | None,
    video_input: Qwen2_5_VLVideoInputs,
) -> tuple[torch.Tensor, ...]:
    grid_thw = video_input["video_grid_thw"]
    assert grid_thw.ndim == 2
    grid_thw_list = grid_thw.tolist()
    merge_size = self.visual.spatial_merge_size
    if attn_score_split is None:
        raise ValueError("FlashVID requires per-video attention scores")
    timestamps_per_video = video_input.get("timestamps")
    if timestamps_per_video is None:
        raise ValueError("FlashVID requires timestamps for each video")
    if not (len(video_embeds_split) == len(grid_thw_list) == len(attn_score_split) == len(timestamps_per_video)):
        raise ValueError("FlashVID embeddings, grids, attention scores and timestamps must have matching counts")

    # Apply EVS to each video.
    video_embeds_out = []
    initial_token_count = 0
    compressed_token_count = 0
    for video_idx, (emb, size, attn_score) in enumerate(zip(video_embeds_split, grid_thw_list, attn_score_split)):
        timestamps = timestamps_per_video[video_idx]
        num_frames, grid_h, grid_w = map(int, size)
        tokens_per_frame = (grid_h // merge_size) * (grid_w // merge_size)
        if len(timestamps) != num_frames:
            raise ValueError("FlashVID timestamps must match the number of video frames")

        video_features = emb.view(num_frames, tokens_per_frame, -1)
        attn_score = attn_score.view(num_frames, tokens_per_frame)
        compressed_video_tokens, keep_visual_global_indices = flashvid_compression(
            video_features=video_features, cls_attention=attn_score, prune_config=self.prune_config
        )

        initial_token_count += len(emb)
        compressed_token_count += len(compressed_video_tokens)
        retention_mask = torch.zeros(
            num_frames * tokens_per_frame,
            dtype=torch.bool,
            device=keep_visual_global_indices.device,
        )
        retention_mask[keep_visual_global_indices] = True
        num_tokens_per_frame = retention_mask.view(num_frames, tokens_per_frame).sum(dim=1).long().tolist()

        emb = self._create_final_video_embeddings(
            video_embeddings=compressed_video_tokens,
            num_tokens_per_frame=num_tokens_per_frame,
            timestamps=timestamps,
            video_grid_thw=size,
            retention_mask=retention_mask,
        )

        video_embeds_out.append(emb)

    actual_retention_ratio = compressed_token_count / initial_token_count if initial_token_count else 0.0
    logger.info(
        "[FlashVID] compression enabled: yes; "
        "configured retention ratio: "
        f"{self.prune_config.retention_ratio:.2%}; "
        f"visual tokens: {initial_token_count} -> {compressed_token_count}; "
        f"actual retention ratio: {actual_retention_ratio:.2%}"
    )
    return tuple(video_embeds_out)


def _embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings | None:
    mm_input_by_modality = self._parse_and_validate_multimodal_inputs(**kwargs)
    if not mm_input_by_modality:
        return None

    # The result multimodal_embeddings is tuple of tensors, with each
    # tensor corresponding to a multimodal data item (image or video).
    multimodal_embeddings: list[torch.Tensor] = []

    # NOTE: It is important to iterate over the keys in this dictionary
    # to preserve the order of the modalities.
    for modality in mm_input_by_modality:
        multimodal_input = mm_input_by_modality[modality]
        if modality == "image":
            image_embeddings = self._process_image_input(multimodal_input)
            image_embeddings = self._postprocess_image_embeds_evs(image_embeddings, multimodal_input)
            multimodal_embeddings.extend(image_embeddings)
        if modality == "video":
            video_embeddings, attn_scores = self._process_video_input(multimodal_input)
            if self.prune_config.enable_flashvid:
                video_embeddings = self._postprocess_video_embeds_flashvid(
                    video_embeddings,
                    attn_scores,
                    multimodal_input,
                )
            else:
                logger.debug("[FlashVID] compression enabled: no")
                if self.is_multimodal_pruning_enabled:
                    video_embeddings = self._postprocess_video_embeds_evs(video_embeddings, multimodal_input)
            multimodal_embeddings.extend(video_embeddings)

    embeddings_tuple = tuple(multimodal_embeddings)
    return embeddings_tuple


old_init = Qwen3VLForConditionalGeneration.__init__


def _qwen3_vl_for_condition_generation_init(self, *, vllm_config: VllmConfig, prefix: str = ""):
    old_init(self, vllm_config=vllm_config, prefix=prefix)
    self.prune_config = get_prune_config(self.video_pruning_rate)


Qwen3_VisionBlock.forward = _qwen3_vision_block_forward
Qwen3_VisionTransformer.forward = _qwen3_vision_transformer_forward
Qwen3VLForConditionalGeneration.__init__ = _qwen3_vl_for_condition_generation_init
Qwen3VLForConditionalGeneration._process_video_input = _process_video_input
Qwen3VLForConditionalGeneration.embed_multimodal = _embed_multimodal


setattr(Qwen3VLForConditionalGeneration, "_postprocess_video_embeds_flashvid", _postprocess_video_embeds_flashvid)
