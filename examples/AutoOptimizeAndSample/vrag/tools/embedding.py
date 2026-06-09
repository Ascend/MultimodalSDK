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
# Text Embedding utilities.

from typing import Optional, Tuple

import numpy as np
import torch


def mean_pooling(model_output: Tuple[torch.Tensor, ...], attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Apply mean pooling to the token embeddings using the attention mask.

    Args:
        model_output: Output from a HuggingFace transformer model.
            The first element (model_output[0]) is expected to be the sequence of hidden states
            with shape (batch_size, seq_len, hidden_size).
        attention_mask: Tensor of shape (batch_size, hidden_size).
    """
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, dim=1) / torch.clamp(
        input_mask_expanded.sum(dim=1), min=1e-9
    )


def normalize_vectors(vectors: np.ndarray) -> Optional[np.ndarray]:
    """
    L2 normalizes a 2D numpy array of vectors.
    """
    if vectors is None or len(vectors) == 0:
        return None

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0, 1, norms)
    return vectors / safe_norms


def normalize_average_vector(vectors: np.ndarray) -> Optional[np.ndarray]:
    """
    Compute the mean of input vectors and return the L2-normalized result.
    """
    if vectors is None or len(vectors) == 0:
        return None

    avg_vec = np.mean(vectors, axis=0)
    norm_val = np.linalg.norm(avg_vec)
    if norm_val < 1e-9:
        return None
    return avg_vec / norm_val
