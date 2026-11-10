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

Patch vision attention/block/transformer forward methods, model initialization,
video processing and multimodal embedding dispatch. Image processing keeps its
upstream Tensor return contract. The processor's ``_get_prompt_updates`` uses
the same fixed FlashVid budget as compression; selected M-RoPE coordinates are
appended to the compressed embeddings for upstream position reconstruction.

The compression and token-selection modules must be installed before loading
this patch. FlashVid requires pixel videos and tensor-parallel vision execution;
precomputed video embeddings and data-parallel vision execution are rejected
when FlashVid is enabled because they do not provide attention scores.
"""

from functools import partial

import torch
from torch import nn
import einops
from vllm.config import VllmConfig
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.model_executor.layers.attention import MMEncoderAttention

from vllm.model_executor.models.interfaces import MultiModalEmbeddings
from vllm.model_executor.models.qwen2_5_vl import (
    Qwen2_5_VisionAttention,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLVideoInputs,
    Qwen2_5_VisionBlock,
    Qwen2_5_VisionTransformer,
    Qwen2_5_VLMultiModalProcessor,
)
from vllm.model_executor.models.utils import cast_overflow_tensors
from vllm.multimodal.evs import compute_mrope_for_media
from vllm.multimodal.processing import PromptReplacement
from vllm.model_executor.models.vision import run_dp_sharded_mrope_vision_model

from .method import compute_flashvid_token_budget, flashvid_compression, get_prune_config
from .token_selection import prune_embeds, restore_hidden_states

from vllm.logger import init_logger

logger = init_logger("vllm.flashvid")


def _qwen2_5_vision_attention_forward(
    self,
    x: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb_cos: torch.Tensor,
    rotary_pos_emb_sin: torch.Tensor,
    max_seqlen: torch.Tensor,  # Only used for Flash Attention
    # Only used for FlashInfer CuDNN backend.
    sequence_lengths: torch.Tensor | None,
    return_attn_score: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    # [s, b, c] --> [s, b, head * 3 * head_dim]
    x, _ = self.qkv(x)
    seq_len, batch_size, _ = x.shape

    qkv = einops.rearrange(
        x,
        "s b (three head head_dim) -> b s three head head_dim",
        three=3,
        head=self.num_attention_heads_per_partition,
    )

    if rotary_pos_emb_cos is not None and rotary_pos_emb_sin is not None:
        qk, v = qkv[:, :, :2], qkv[:, :, 2]

        qk_reshaped = einops.rearrange(qk, "b s two head head_dim -> (two b) s head head_dim", two=2)
        qk_reshaped = qk_reshaped.contiguous()
        qk_rotated = self.apply_rotary_emb(
            qk_reshaped,
            rotary_pos_emb_cos,
            rotary_pos_emb_sin,
        )
        qk_rotated = qk_rotated.view(
            2,
            batch_size,
            seq_len,
            self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head,
        )
        q, k = qk_rotated.unbind(dim=0)
    else:
        q, k, v = qkv.unbind(dim=2)

    context_layer = self.attn(
        query=q,
        key=k,
        value=v,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        sequence_lengths=sequence_lengths,
    )
    head_dim = self.qkv.hidden_size / self.qkv.num_heads
    attn_score = None
    if return_attn_score:
        # Calculate attention weights manually, frame by frame to save memory.
        num_frames = cu_seqlens.shape[0] - 1
        q, k = q.squeeze(0), k.squeeze(0)
        attn_score_list = []
        # 根据 cu_seqlens 逐帧处理
        for i in range(num_frames):
            start_idx = cu_seqlens[i]
            end_idx = cu_seqlens[i + 1]
            # 提取当前帧的 q 和 k
            qi_frame = q[start_idx:end_idx]  # (seq_len_per_frame, num_heads, head_dim)
            ki_frame = k[start_idx:end_idx]  # (seq_len_per_frame, num_heads, head_dim)
            # 转换维度为 (num_heads, seq_len_per_frame, head_dim)
            qi = qi_frame.transpose(0, 1).contiguous()
            ki = ki_frame.transpose(0, 1).contiguous()
            # 计算注意力权重
            attn_i = torch.matmul(qi, ki.transpose(-1, -2)) / head_dim**0.5
            attn_i = nn.functional.softmax(attn_i, dim=-1, dtype=torch.float32).to(qi.dtype)
            attn_i = attn_i.mean(0).mean(0)  # (seq_len_per_frame,)
            attn_score_list.append(attn_i)
        # 直接连接所有帧的注意力权重
        attn_score = torch.cat(attn_score_list, dim=0)  # (total_seq_len,)
        if self.tp_size > 1:
            # Average scores across all TP-sharded attention heads.
            attn_score = tensor_model_parallel_all_reduce(attn_score.float().contiguous()) / self.tp_size
            attn_score = attn_score.to(q.dtype)

    context_layer = einops.rearrange(context_layer, "b s h d -> s b (h d)", b=batch_size).contiguous()

    output, _ = self.proj(context_layer)
    return output, attn_score


def _qwen2_5_vision_block_forward(
    self,
    x: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb_cos: torch.Tensor,
    rotary_pos_emb_sin: torch.Tensor,
    max_seqlen: torch.Tensor,  # Only used for Flash Attention
    # Only used for FlashInfer CuDNN backend.
    sequence_lengths: torch.Tensor | None = None,
    return_attn_score: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    x_attn, attn_score = self.attn(
        self.norm1(x),
        cu_seqlens=cu_seqlens,
        rotary_pos_emb_cos=rotary_pos_emb_cos,
        rotary_pos_emb_sin=rotary_pos_emb_sin,
        max_seqlen=max_seqlen,
        sequence_lengths=sequence_lengths,
        return_attn_score=return_attn_score,
    )
    x_fused_norm, residual = self.norm2(x, residual=x_attn)
    x = residual + self.mlp(x_fused_norm)
    return x, attn_score


def _qwen2_5_vision_transformer_forward(
    self,
    x: torch.Tensor,
    grid_thw: list[list[int]] | None,
    *,
    encoder_metadata: dict[str, torch.Tensor] | None = None,
    return_attn_score: bool = False,
    prune_encoder: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    hidden_states = x.to(device=self.device, dtype=self.dtype)
    hidden_states = self.patch_embed(hidden_states)

    seq_len = hidden_states.shape[0]
    if encoder_metadata is None:
        encoder_metadata = self.prepare_encoder_metadata(grid_thw)

    rotary_pos_emb_cos = encoder_metadata["rotary_pos_emb_cos"]
    rotary_pos_emb_sin = encoder_metadata["rotary_pos_emb_sin"]
    window_index = encoder_metadata["window_index"]
    reverse_indices = encoder_metadata["reverse_indices"]
    cu_seqlens = encoder_metadata["cu_seqlens"]
    cu_window_seqlens = encoder_metadata["cu_window_seqlens"]
    max_seqlen_full = encoder_metadata["max_seqlen_full"]
    max_seqlen_window = encoder_metadata["max_seqlen_window"]
    sequence_lengths_full = encoder_metadata.get("sequence_lengths_full")
    sequence_lengths_window = encoder_metadata.get("sequence_lengths_window")

    hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
    hidden_states = hidden_states[window_index, :, :]
    hidden_states = hidden_states.reshape(seq_len, -1)

    is_pruned = False
    keep_ids_tensor = None
    remove_ids_tensor = None

    if prune_encoder:
        # 正常模式下的修剪逻辑
        boundaries = cu_seqlens.detach().cpu().tolist()
        new_cu_seqlens = [torch.zeros((), dtype=torch.int64, device=hidden_states.device)]
        keep_ids = []
        remove_ids = []

        for s, e in zip(boundaries[:-1], boundaries[1:]):
            keep_indices, remove_indices = prune_embeds(hidden_states[s:e])
            keep_ids.append(keep_indices + s)
            remove_ids.append(remove_indices + s)
            new_cu_seqlens.append(new_cu_seqlens[-1] + len(keep_indices))

        keep_ids_tensor = torch.cat(keep_ids).to(hidden_states.device)
        remove_ids_tensor = torch.cat(remove_ids).to(hidden_states.device)

        # hidden_states.shape: [seq_len, hidden_size]
        if keep_ids_tensor.shape[0] < seq_len:
            logger.debug("FlashVid encoder pruning: %d -> %d tokens", seq_len, len(keep_ids_tensor))
            is_pruned = True
            hidden_states = hidden_states[keep_ids_tensor]
            rotary_pos_emb_cos = rotary_pos_emb_cos[keep_ids_tensor]
            rotary_pos_emb_sin = rotary_pos_emb_sin[keep_ids_tensor]
            cu_seqlens = torch.stack(new_cu_seqlens).to(cu_seqlens.dtype)

            # Window attention blocks read cu_window_seqlens, whose last
            # element must equal the pruned token count.
            # 根据 keep_ids 计算 keep_cumsum
            keep_cumsum_arr = torch.zeros(seq_len, dtype=torch.int64, device=hidden_states.device)
            keep_cumsum_arr[keep_ids_tensor] = 1
            keep_cumsum_arr = keep_cumsum_arr.cumsum(0)
            keep_cumsum = torch.cat(
                [
                    torch.zeros(
                        1,
                        dtype=torch.int64,
                        device=hidden_states.device,
                    ),
                    keep_cumsum_arr,
                ]
            )
            cu_window_seqlens = keep_cumsum[cu_window_seqlens.clamp(0, seq_len).to(torch.int64)].to(
                cu_window_seqlens.dtype
            )
            full_offsets = cu_seqlens.cpu().numpy()
            window_offsets = cu_window_seqlens.cpu().numpy()
            max_seqlen_full = torch.tensor(
                MMEncoderAttention.compute_max_seqlen(self.attn_backend, full_offsets), dtype=torch.int32
            )
            max_seqlen_window = torch.tensor(
                MMEncoderAttention.compute_max_seqlen(self.attn_backend, window_offsets), dtype=torch.int32
            )
            sequence_lengths_full = MMEncoderAttention.maybe_compute_seq_lens(
                self.attn_backend, full_offsets, hidden_states.device
            )
            sequence_lengths_window = MMEncoderAttention.maybe_compute_seq_lens(
                self.attn_backend, window_offsets, hidden_states.device
            )

    hidden_states = hidden_states.unsqueeze(1)
    attn_score: torch.Tensor | None = None
    for layer_num, blk in enumerate(self.blocks):
        if layer_num in self.fullatt_block_indexes:
            cu_seqlens_now = cu_seqlens
            max_seqlen_now = max_seqlen_full
            sequence_lengths_now = sequence_lengths_full
        else:
            cu_seqlens_now = cu_window_seqlens
            max_seqlen_now = max_seqlen_window
            sequence_lengths_now = sequence_lengths_window

        collect_scores = return_attn_score and layer_num == len(self.blocks) - 1
        hidden_states, attn_score = blk(
            hidden_states,
            cu_seqlens=cu_seqlens_now,
            rotary_pos_emb_cos=rotary_pos_emb_cos,
            rotary_pos_emb_sin=rotary_pos_emb_sin,
            max_seqlen=max_seqlen_now,
            sequence_lengths=sequence_lengths_now,
            return_attn_score=collect_scores,
        )

    # For Qwen2.5-VL-3B, float16 will overflow at last block
    # for long visual tokens sequences.
    if hidden_states.dtype == torch.float16:
        hidden_states = cast_overflow_tensors(hidden_states)

    if is_pruned:
        hidden_states, attn_score = restore_hidden_states(
            hidden_states,
            seq_len,
            keep_ids_tensor,
            remove_ids_tensor,
            attn_score,
        )

    # adapter
    hidden_states = self.merger(hidden_states)
    hidden_states = hidden_states[reverse_indices, :]

    if not return_attn_score:
        return hidden_states

    attn_score = attn_score.reshape(-1, self.spatial_merge_unit).mean(-1)[reverse_indices]

    return hidden_states, attn_score


def _qwen2_5_process_video_input(
    self, video_input: Qwen2_5_VLVideoInputs
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...] | None]:
    grid_thw = video_input["video_grid_thw"]
    assert grid_thw.ndim == 2
    grid_thw_list = grid_thw.tolist()
    attn_scores = None
    if video_input["type"] == "video_embeds":
        if self.prune_config.enable_flashvid:
            raise ValueError("FlashVid requires pixel_values_videos; precomputed video_embeds lack attention scores")
        video_embeds = video_input["video_embeds"].type(self.visual.dtype)
    else:
        pixel_values_videos = video_input["pixel_values_videos"]
        if self.use_data_parallel:
            if self.prune_config.enable_flashvid:
                raise ValueError(
                    "FlashVid requires tensor-parallel vision execution; data-parallel scores are unavailable"
                )
            return run_dp_sharded_mrope_vision_model(
                self.visual,
                pixel_values_videos,
                grid_thw_list,
                rope_type="rope_3d",
            ), None
        else:
            visual_output = self.visual(
                pixel_values_videos,
                grid_thw=grid_thw_list,
                return_attn_score=self.prune_config.enable_flashvid,
                prune_encoder=self.prune_config.enable_encode_prune,
            )
            if self.prune_config.enable_flashvid:
                video_embeds, attn_scores = visual_output
            else:
                video_embeds = visual_output

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
    """Compress video embeddings with FlashVID and append M-RoPE data."""
    grid_thw = video_input["video_grid_thw"]
    assert grid_thw.ndim == 2
    grid_thw_list = grid_thw.tolist()
    merge_size = self.visual.spatial_merge_size
    second_per_grid_ts = video_input.get("second_per_grid_ts")
    if second_per_grid_ts is None:
        raise ValueError("FlashVid requires second_per_grid_ts for video M-RoPE positions")
    if attn_score_split is None:
        raise ValueError("FlashVid requires per-video attention scores")
    if not (len(video_embeds_split) == len(grid_thw_list) == len(second_per_grid_ts) == len(attn_score_split)):
        raise ValueError("FlashVid video embeddings, grids, timestamps and attention scores must have matching counts")
    tokens_per_second = self.config.vision_config.tokens_per_second

    video_embeds_out = []
    initial_token_count = 0
    compressed_token_count = 0
    for emb, size, video_second_per_grid_t, attn_score in zip(
        video_embeds_split, grid_thw_list, second_per_grid_ts, attn_score_split
    ):
        num_frames, grid_h, grid_w = map(int, size)
        tokens_per_frame = (grid_h // merge_size) * (grid_w // merge_size)
        if len(emb) != num_frames * tokens_per_frame or attn_score.numel() != len(emb):
            raise ValueError("FlashVid embeddings and attention scores must match the video grid token count")

        video_features = emb.view(num_frames, tokens_per_frame, -1)
        attn_score = attn_score.view(num_frames, tokens_per_frame)
        budget = compute_flashvid_token_budget(
            tokens_per_frame, num_frames, self.prune_config.retention_ratio, self.prune_config.alpha
        )
        if budget.total_tokens == len(emb):
            compressed_video_tokens = emb
            keep_visual_global_indices = torch.arange(len(emb), device=emb.device)
        else:
            compressed_video_tokens, keep_visual_global_indices = flashvid_compression(
                video_features=video_features, cls_attention=attn_score, prune_config=self.prune_config
            )
        initial_token_count += len(emb)
        compressed_token_count += len(compressed_video_tokens)

        positions = compute_mrope_for_media(
            size,
            merge_size,
            tokens_per_second=tokens_per_second,
            video_second_per_grid=video_second_per_grid_t.item(),
        ).to(compressed_video_tokens.device, non_blocking=True)
        positions = positions.index_select(0, keep_visual_global_indices.long())

        emb = compressed_video_tokens
        emb = torch.cat([emb, positions], dim=1)
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


def _embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
    mm_input_by_modality = self._parse_and_validate_multimodal_inputs(**kwargs)
    if not mm_input_by_modality:
        return []

    # The result multimodal_embeddings is tuple of tensors, with each
    # tensor correspoending to a multimodal data item (image or video).
    multimodal_embeddings: tuple[torch.Tensor, ...] = ()

    # NOTE: It is important to iterate over the keys in this dictionary
    # to preserve the order of the modalities.
    for modality in mm_input_by_modality:
        multimodal_input = mm_input_by_modality[modality]
        if modality == "image":
            image_embeddings = self._process_image_input(multimodal_input)
            if self.is_multimodal_pruning_enabled:
                image_embeddings = self._postprocess_image_embeds_evs(image_embeddings, multimodal_input)
            multimodal_embeddings += tuple(image_embeddings)
        if modality == "video":
            video_embeddings, attn_scores = self._process_video_input(multimodal_input)
            if self.prune_config.enable_flashvid:
                video_embeddings = self._postprocess_video_embeds_flashvid(
                    video_embeddings, attn_scores, multimodal_input
                )
            elif self.is_multimodal_pruning_enabled:
                video_embeddings = self._postprocess_video_embeds_evs(video_embeddings, multimodal_input)
            multimodal_embeddings += tuple(video_embeddings)
    return multimodal_embeddings


def _get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs):
    config = get_prune_config(self.info.ctx.get_mm_config().video_pruning_rate)
    if not config.enable_flashvid:
        return old_get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs)
    processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
    merge_size = self.info.get_image_processor(**hf_processor_mm_kwargs).merge_size
    vocab = self.info.get_tokenizer().get_vocab()
    placeholders = {"image": vocab[processor.image_token], "video": vocab[processor.video_token]}

    def replacement(item_idx, modality):
        grid = out_mm_kwargs[modality][item_idx][f"{modality}_grid_thw"].data
        frames, height, width = map(int, grid)
        tokens_per_frame = (height // merge_size) * (width // merge_size)
        count = frames * tokens_per_frame
        if modality == "video":
            count = compute_flashvid_token_budget(
                tokens_per_frame, frames, config.retention_ratio, config.alpha
            ).total_tokens
        return [placeholders[modality]] * count

    return [
        PromptReplacement(
            modality=modality, target=[placeholders[modality]], replacement=partial(replacement, modality=modality)
        )
        for modality in ("image", "video")
    ]


old_init = Qwen2_5_VLForConditionalGeneration.__init__
old_get_prompt_updates = Qwen2_5_VLMultiModalProcessor._get_prompt_updates


def _qwen2_5_vl_for_condition_generation_init(self, *, vllm_config: VllmConfig, prefix: str = ""):
    old_init(self, vllm_config=vllm_config, prefix=prefix)
    self.prune_config = get_prune_config(self.video_pruning_rate)


Qwen2_5_VisionAttention.forward = _qwen2_5_vision_attention_forward
Qwen2_5_VisionBlock.forward = _qwen2_5_vision_block_forward
Qwen2_5_VisionTransformer.forward = _qwen2_5_vision_transformer_forward
Qwen2_5_VLForConditionalGeneration.__init__ = _qwen2_5_vl_for_condition_generation_init
Qwen2_5_VLForConditionalGeneration._process_video_input = _qwen2_5_process_video_input
Qwen2_5_VLForConditionalGeneration.embed_multimodal = _embed_multimodal
Qwen2_5_VLMultiModalProcessor._get_prompt_updates = _get_prompt_updates

setattr(Qwen2_5_VLForConditionalGeneration, "_postprocess_video_embeds_flashvid", _postprocess_video_embeds_flashvid)
