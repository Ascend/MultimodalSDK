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
# Qwen embedding service for computing text embeddings.

from typing import List, Optional

import bentoml
import numpy as np
import torch
from pydantic import Field
from transformers import AutoTokenizer, AutoModel

from vrag.logger import logger
from vrag.shared import ArgsBase, first_available, vrag_service
from vrag.tools.embedding import mean_pooling, normalize_vectors
from vrag.tools.path_validator import validate_dir_exists


class QwenEmbeddingArgs(ArgsBase):
    """Qwen embedding configuration."""

    embedding_model_path: str = ""
    """Local path to the Qwen embedding model directory."""
    embedding_device: str = "npu:3"
    """Device for Qwen embedding model inference, e.g. 'npu:3' or 'cpu'."""
    embedding_batch_size: int = Field(8, ge=1)
    """Batch size for text embedding inference."""
    embedding_cache_size: int = Field(4096, ge=0)
    """LRU cache capacity for embedding results."""
    default_normalize: bool = True
    """Whether to L2-normalize embedding vectors by default."""


args = bentoml.use_arguments(QwenEmbeddingArgs).override()


@vrag_service(args)
class QwenEmbeddingService:
    def __init__(self) -> None:
        self.model_path = validate_dir_exists(args.embedding_model_path, "Qwen embedding model")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        self.model = AutoModel.from_pretrained(self.model_path, device_map=args.embedding_device, local_files_only=True)
        self.model.eval()

        msg = f"QwenEmbeddingService initialized from {self.model_path}."
        logger.info(msg)

    @bentoml.api
    async def embed(self, texts: List[str], normalize: Optional[bool] = None) -> np.ndarray:
        norm = first_available(normalize, args.default_normalize)

        return self._embed_batch(texts, args.embedding_batch_size, norm)

    def _embed_batch(self, texts: List[str], batch_size: int, normalize: bool = True) -> np.ndarray:
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            cur_texts = texts[i : i + batch_size]

            msg = f"Embedding process texts: [{i}:{i + len(cur_texts)}]."
            logger.info(msg)

            inputs = self.tokenizer(cur_texts, return_tensors="pt", truncation=True, padding=True)

            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            embeddings = mean_pooling(outputs, inputs["attention_mask"]).cpu().numpy()

            all_embeddings.append(normalize_vectors(embeddings) if normalize else embeddings)

        if all_embeddings:
            return np.concatenate(all_embeddings, axis=0)

        return np.array([])
