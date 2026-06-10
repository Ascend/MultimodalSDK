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
# MMDINO service for object detection in video frames.


from typing import List, Optional

import bentoml
import numpy as np
import torch
from PIL import Image
from pydantic import Field
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from vrag.logger import logger
from vrag.shared import ArgsBase, ConfigBase, first_available, into_u8_frames, vrag_service
from vrag.tools.np_cacher import get_cacher
from vrag.types import MMDINODetectionBatchResult, MMDINODetectionItem, MMDINODetectionResult
from vrag.tools.path_validator import validate_dir_exists


class MMDinoArgs(ArgsBase):
    """MMDINO object detection configuration"""

    mmdino_model_path: str = ""
    """Local path to the MMDINO model directory."""
    mmdino_device: str = "npu:2"
    """Device for MMDINO model inference, e.g. 'npu:2' or 'cpu'."""
    mmdino_batch_size: int = Field(8, ge=1)
    """Batch size for MMDINO object detection inference."""
    mmdino_cache_size: int = Field(4096, ge=0)
    """LRU cache capacity for MMDINO detection results."""
    default_mmdino_threshold: float = Field(0.43, ge=0.0, le=1.0)
    """Default confidence threshold for object detection."""


class MMDinoConfig(ConfigBase):
    """MMDINO object detection configuration for request-level"""

    mmdino_threshold: Optional[float] = None

    @staticmethod
    def merge_config(config: Optional["MMDinoConfig"]) -> "MMDinoConfig":
        if config is None:
            return MMDinoConfig(mmdino_threshold=args.default_mmdino_threshold)

        return MMDinoConfig(mmdino_threshold=first_available(config.mmdino_threshold, args.default_mmdino_threshold))


args = bentoml.use_arguments(MMDinoArgs).override()


def _get_cache_key(prompt: List[str], threshold: float, *_, **__) -> str:
    return "".join(sorted(prompt)) + str(threshold)


def _format_detection_prompt(prompts: List[str]) -> str:
    return ". ".join(prompt.lower().strip() for prompt in prompts) + "."


@vrag_service(args)
class MMDINOService:
    def __init__(self):
        self.model_path = validate_dir_exists(args.mmdino_model_path, "MMDINO model")

        self.processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=True)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.model_path, device_map=args.mmdino_device, local_files_only=True
        )
        self.model.eval()

        self._cacher = get_cacher(args.mmdino_cache_size)

        msg = f"MMDINOService initialized from {self.model_path}."
        logger.info(msg)

    @bentoml.api
    async def ov_detect(
        self, frames: List[np.ndarray], prompts: List[str], config: Optional[MMDinoConfig] = None
    ) -> MMDINODetectionBatchResult:
        merged_config = MMDinoConfig.merge_config(config)
        key = _get_cache_key(prompts, merged_config.mmdino_threshold)

        @self._cacher.cached_sync_with(lambda *_, **__: key)
        def _detect(frames: List[np.ndarray]) -> List[MMDINODetectionResult]:
            return self._detect_raw(frames, prompts, args.mmdino_batch_size, merged_config.mmdino_threshold)

        if not frames:
            return MMDINODetectionBatchResult()

        return MMDINODetectionBatchResult(results=_detect(frames))

    def _detect_raw(
        self, frames: List[np.ndarray], prompts: List[str], batch_size: int, threshold: float = 0.35
    ) -> List[MMDINODetectionResult]:
        pil_images = [Image.fromarray(into_u8_frames(frame)) for frame in frames]
        all_res: List[MMDINODetectionResult] = []
        frames_num = len(frames)

        text_prompt = _format_detection_prompt(prompts)

        for i in range(0, frames_num, batch_size):
            batch_pil_images = pil_images[i : i + batch_size]
            current_batch_size = len(batch_pil_images)
            msg = f"MMDINO detect batch [{i}:{i + current_batch_size}]"
            logger.debug(msg)

            inputs = self.processor(
                images=batch_pil_images, text=[text_prompt] * current_batch_size, return_tensors="pt"
            ).to(self.model.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            target_sizes = [(img.height, img.width) for img in batch_pil_images]

            result_list = self.processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, threshold=threshold, text_threshold=threshold, target_sizes=target_sizes
            )

            for result in result_list:
                boxes = result["boxes"]
                labels = result["labels"]
                detection_result = MMDINODetectionResult(
                    items=[
                        MMDINODetectionItem(class_name=label, bbox=[int(c) for c in box.tolist()])
                        for box, label in zip(boxes, labels, strict=True)
                    ]
                )
                all_res.append(detection_result)
        return all_res
