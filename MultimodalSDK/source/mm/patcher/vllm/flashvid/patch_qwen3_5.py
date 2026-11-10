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
#
# Adapted from vLLM v0.23.0:
# https://github.com/vllm-project/vllm/blob/v0.23.0/vllm/model_executor/models/qwen3_5.py
# Modifications Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Modified to initialize FlashVid configuration and reuse Qwen3-VL pruning support.
"""FlashVid initialization and position reconstruction patches for Qwen3.5."""

from .method import get_prune_config
from vllm.config import VllmConfig
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration, Qwen3_5MoeForConditionalGeneration
from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration
from vllm.tokenizers.registry import cached_tokenizer_from_config

old_cls = Qwen3_5ForConditionalGeneration
old_cls.supports_multimodal_pruning = True
old_qwen3_5_for_conditional_generation_init = Qwen3_5ForConditionalGeneration.__init__


def _qwen3_5_for_condition_generation_init(self, *, vllm_config: VllmConfig, prefix: str = "model"):
    old_qwen3_5_for_conditional_generation_init(self, vllm_config=vllm_config, prefix=prefix)
    self.is_multimodal_pruning_enabled = self.multimodal_config.is_multimodal_pruning_enabled()
    self.prune_config = get_prune_config(self.multimodal_config.video_pruning_rate)
    self.video_pruning_rate = self.multimodal_config.video_pruning_rate
    self._tokenizer = cached_tokenizer_from_config(vllm_config.model_config)

    # attributes needed by EVS-related functions inherited from Qwen3-VL
    self.use_deepstack = hasattr(self.config.vision_config, "deepstack_visual_indexes")
    self.deepstack_num_level = len(self.config.vision_config.deepstack_visual_indexes) if self.use_deepstack else 0
    self.visual_dim = self.config.vision_config.out_hidden_size
    self.multiscale_dim = self.visual_dim * self.deepstack_num_level


old_qwen3_5_moe_for_conditional_generation_init = Qwen3_5MoeForConditionalGeneration.__init__


def _qwen3_5_moe_for_condition_generation_init(self, *, vllm_config: VllmConfig, prefix: str = "model"):
    old_qwen3_5_moe_for_conditional_generation_init(self, vllm_config=vllm_config, prefix=prefix)
    self.is_multimodal_pruning_enabled = self.multimodal_config.is_multimodal_pruning_enabled()
    self.prune_config = get_prune_config(self.multimodal_config.video_pruning_rate)
    self.video_pruning_rate = self.multimodal_config.video_pruning_rate
    self._tokenizer = cached_tokenizer_from_config(vllm_config.model_config)

    # attributes needed by EVS-related functions inherited from Qwen3-VL
    self.use_deepstack = hasattr(self.config.vision_config, "deepstack_visual_indexes")
    self.deepstack_num_level = len(self.config.vision_config.deepstack_visual_indexes) if self.use_deepstack else 0
    self.visual_dim = self.config.vision_config.out_hidden_size
    self.multiscale_dim = self.visual_dim * self.deepstack_num_level


def _recompute_mrope_positions(self, *args, **kwargs):
    return Qwen3VLForConditionalGeneration.recompute_mrope_positions(self, *args, **kwargs)


Qwen3_5ForConditionalGeneration.__init__ = _qwen3_5_for_condition_generation_init
Qwen3_5ForConditionalGeneration.recompute_mrope_positions = _recompute_mrope_positions
Qwen3_5MoeForConditionalGeneration.__init__ = _qwen3_5_moe_for_condition_generation_init
