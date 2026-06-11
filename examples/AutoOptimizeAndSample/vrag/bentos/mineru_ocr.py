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
# MinerU OCR service for extraction text from images.


from dataclasses import dataclass
from typing import List, Optional
import os

import bentoml
import numpy as np
from PIL import Image
from mineru.backend.pipeline.batch_analyze import BatchAnalyze
from mineru.backend.pipeline.pipeline_analyze import ModelSingleton
from pydantic import Field

from vrag.logger import logger
from vrag.shared import ArgsBase, first_available, into_u8_frames, vrag_service
from vrag.tools.np_cacher import get_cacher


class MineruArgs(ArgsBase):
    """MinerU OCR configuration"""

    mineru_device: str = "npu:0"
    """Device for MinerU OCR inference, e.g. 'npu:0' or 'cpu'."""
    mineru_batch_ratio: int = Field(12, ge=1)
    """Batch ratio for MinerU OCR processing."""
    mineru_cache_size: int = Field(4096, ge=0)
    """LRU cache capacity for MinerU OCR results."""
    default_formula_enable: bool = False
    """Whether to enable formula detection in OCR by default."""
    default_table_enable: bool = False
    """Whether to enable table detection in OCR by default."""
    default_lang: str = "ch_lite"
    """Default language for OCR recognition."""
    default_line_threshold_ratio: float = Field(0.6, ge=0.0)
    """Default threshold ratio for clustering OCR lines into the same text line."""


args = bentoml.use_arguments(MineruArgs).override()


def _get_cache_key(
    lang: Optional[str],
    line_threshold_ratio: Optional[float],
    formula_enable: Optional[bool],
    table_enable: Optional[bool],
    *_,
    **__,
) -> str:
    return f"{lang}:{line_threshold_ratio}:{formula_enable}:{table_enable}"


@dataclass
class _GeometryItem:
    item: dict
    x_min: float = 0.0
    y_min: float = 0.0
    x_max: float = 0.0
    y_max: float = 0.0
    height: float = 0.0

    @staticmethod
    def cluster_lines(items: List["_GeometryItem"], threshold_ratio: float) -> List[List["_GeometryItem"]]:
        if not items:
            return []

        items.sort(key=lambda x: (x.y_min, x.x_min))

        lines = []
        current_line = [items[0]]
        avg_height = sum(i.height for i in items) / len(items)
        threshold = avg_height * threshold_ratio

        for item in items[1:]:
            curr_center = (item.y_min + item.y_max) / 2
            prev_center = (current_line[0].y_min + current_line[0].y_max) / 2

            if abs(curr_center - prev_center) < threshold:
                current_line.append(item)
            else:
                lines.append(current_line)
                current_line = [item]

        if current_line:
            lines.append(current_line)

        return lines

    @classmethod
    def from_dict(cls, item: dict) -> "_GeometryItem":
        poly = item.get("poly", [])
        if len(poly) >= 4:
            if len(poly) == 8:
                xs = [poly[i] for i in range(0, 8, 2)]
                ys = [poly[i] for i in range(1, 8, 2)]
            else:
                xs = [poly[0], poly[2]]
                ys = [poly[1], poly[3]]
            x_min, y_min, x_max, y_max = min(xs), min(ys), max(xs), max(ys)
            return cls(item=item, x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max, height=y_max - y_min)
        return cls(item=item)


@vrag_service(args)
class MineruService:
    def __init__(self) -> None:
        # use environment to set device for mineru
        os.environ["MINERU_DEVICE_MODE"] = args.mineru_device
        os.environ["MINERU_MODEL_SOURCE"] = "local"

        model_manager = ModelSingleton()
        self.model = BatchAnalyze(
            model_manager, args.mineru_batch_ratio, args.default_formula_enable, args.default_table_enable, True
        )

        self._cacher = get_cacher(args.mineru_cache_size)

        logger.info("MineruService initialized.")

    @bentoml.api
    async def extract_text(
        self, frames: List[np.ndarray], lang: Optional[str] = None, line_threshold_ratio: Optional[float] = None
    ) -> List[str]:
        key = _get_cache_key(lang, line_threshold_ratio, args.default_formula_enable, args.default_table_enable)

        @self._cacher.cached_sync_with(lambda *_, **__: key)
        def _process(frames_list: List[np.ndarray]) -> List[str]:
            image_list = [(Image.fromarray(into_u8_frames(f)).convert("RGB")) for f in frames_list]

            return self._extract_text_from_images(
                image_list,
                lang=first_available(lang, args.default_lang),
                line_threshold_ratio=first_available(line_threshold_ratio, args.default_line_threshold_ratio),
            )

        return _process(frames)

    def _extract_text_from_images(
        self, images: List["Image.Image"], lang: str = "ch_lite", line_threshold_ratio: float = 0.6
    ) -> List[str]:
        images_with_extra_info = [(img, True, lang) for img in images]

        msg = f"MinerU ocr extracting text from {len(images_with_extra_info)} images."
        logger.info(msg)

        ocr_result = self.model(images_with_extra_info)
        return _extract_text_from_ocr_results(ocr_result, line_threshold_ratio=line_threshold_ratio)


def _extract_text_from_ocr_results(ocr_results: List[List[dict]], line_threshold_ratio: float = 0.6) -> List[str]:
    def _process_page(page_items: List[dict], threshold_ratio: float) -> str:
        valid_items = [i for i in page_items if i.get("text")]
        if not valid_items:
            return ""

        lines = _GeometryItem.cluster_lines([_GeometryItem.from_dict(item) for item in valid_items], threshold_ratio)

        page_text = []

        for line in lines:
            line.sort(key=lambda x: x.x_min)
            line_str = "".join(g.item["text"].strip() for g in line if g.item["label"] != "seal")
            if line_str:
                page_text.append(line_str)

        return "\n".join(page_text)

    return [_process_page(page, line_threshold_ratio) for page in ocr_results]
