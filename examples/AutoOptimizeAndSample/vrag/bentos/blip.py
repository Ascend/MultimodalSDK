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
# BLIP service for computing image-text similarity score.

from typing import List

import bentoml
import numpy as np
import torch
from pydantic import Field
from transformers import Blip2ForImageTextRetrieval, Blip2Processor

from vrag.tools.path_validator import validate_dir_exists
from vrag.logger import logger
from vrag.shared import ArgsBase, into_u8_frames, vrag_service
from vrag.tools.np_cacher import get_cacher


class BlipArgs(ArgsBase):
    """BLIP model configuration arguments."""

    blip_model_path: str = ""
    """Local path to the BLIP2 model directory."""
    blip_device: str = "npu:0"
    """Device for BLIP2 model inference, e.g. 'npu:0' or 'cpu'."""
    blip_batch_size: int = Field(24, ge=1)
    """Batch size for BLIP2 inference."""
    blip_cache_size: int = Field(4096, ge=0)
    """LRU cache capacity for BLIP2 score computation results."""


args = bentoml.use_arguments(BlipArgs).override()


@vrag_service(args)
class BlipService:
    def __init__(self):
        self.model_path = validate_dir_exists(args.blip_model_path, "BLIP model")

        self.model = Blip2ForImageTextRetrieval.from_pretrained(
            self.model_path,
            dtype=torch.float16,
            device_map=args.blip_device,
            local_files_only=True,
        )
        self.processor = Blip2Processor.from_pretrained(self.model_path, local_files_only=True)
        self.model.eval()

        self._cacher = get_cacher(args.blip_cache_size)

        logger.info("BlipService initialized.")

    @bentoml.api
    async def compute_scores(self, frames: np.ndarray, queries: List[str]) -> List[float]:
        """
        Compute frames score related to queries.

        Args:
            frames: Image array in (N, H, W, C).
            queries: Topics about image.
        """
        key = _get_cache_key(queries)

        @self._cacher.cached_sync_with(lambda *_, **__: key)
        def _compute(frames: np.ndarray) -> List[float]:
            return self._compute_scores(frames, queries, args.blip_batch_size)

        return _compute(frames)

    def _compute_scores(self, frames: np.ndarray, queries: List[str], batch_size) -> List[float]:
        query_string = ", ".join(queries)

        item_scores: List[torch.Tensor] = []

        for i in range(0, len(frames), batch_size):
            batch_images = into_u8_frames(frames[i : i + batch_size])
            current_size = len(batch_images)
            msg = f"Blip processing frame [{i}-{i + current_size}]"
            logger.info(msg)
            inputs = self.processor(batch_images, [query_string] * current_size, return_tensors="pt")
            inputs = inputs.to(self.model.device, torch.float16)

            with torch.no_grad():
                outputs = self.model(**inputs, return_dict=True, use_image_text_matching_head=True)

            query_similarities = outputs.logits_per_image
            item_score = torch.nn.functional.softmax(query_similarities, dim=1)
            item_score = item_score[:, 1]
            item_scores.append(item_score)

        return torch.cat(item_scores).tolist()


def _get_cache_key(queries: List[str], *_, **__) -> str:
    return "".join(sorted(queries))
