#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the MultimodalSDK project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
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
# Detection service combining keyframe selection and object detection.


from typing import List, Optional

import bentoml
import numpy as np
from pydantic import Field

from vrag.bentos.aks import AksBlipConfig, AksBlipArgs, AksBlipService
from vrag.bentos.mmdino import MMDinoArgs, MMDinoConfig, MMDINOService
from vrag.logger import logger
from vrag.shared import first_available, ConfigBase, retry_async_request, vrag_service
from vrag.tools.imagehash import ImageHasher
from vrag.tools.np_cacher import get_cacher
from vrag.tools.query import Query
from vrag.tools.scene import SceneDescriber
from vrag.tools.selecters import indexed
from vrag.types import DETDoc, DetectionResult, MMDINODetectionBatchResult, FrameExtraction


class DetectionArgs(AksBlipArgs, MMDinoArgs):
    detection_cache_size: int = Field(4096, ge=0)
    """LRU cache capacity for detection service results."""
    default_use_det: bool = True
    """Whether to use object detection by default."""
    default_det_dedup_frames: bool = Field(True)
    """Whether to deduplicate frames before detection by default."""
    default_det_dedup_threshold: int = Field(2, ge=0)
    """Default Hamming distance threshold for frame deduplication before detection."""
    default_det_dedup_block_size: int = Field(12, ge=8)
    """Default block size for perceptual hashing in frame deduplication."""
    default_det_location: bool = Field(True)
    """Whether to include location descriptions in detection results by default."""
    default_det_relation: bool = Field(True)
    """Whether to include spatial relation descriptions in detection results by default."""
    default_det_number: bool = Field(True)
    """Whether to include object count descriptions in detection results by default."""
    default_retrieve_frame_only: bool = Field(True)
    """Whether to only retrieve keyframes without generating scene descriptions by default."""


class DetectionServiceConfig(ConfigBase):
    aks: Optional[AksBlipConfig] = None
    mmdino: Optional[MMDinoConfig] = None
    use_det: Optional[bool] = None
    det_dedup_frames: Optional[bool] = None
    det_dedup_threshold: Optional[int] = None
    det_dedup_block_size: Optional[int] = None
    det_location: Optional[bool] = None
    det_relation: Optional[bool] = None
    det_number: Optional[bool] = None
    retrieve_frame_only: Optional[bool] = None

    @staticmethod
    def merge_config(config: Optional["DetectionServiceConfig"]) -> "DetectionServiceConfig":
        if config is None:
            return DetectionServiceConfig(
                aks=AksBlipConfig.merge_config(None),
                mmdino=MMDinoConfig.merge_config(None),
                use_det=args.default_use_det,
                det_dedup_frames=args.default_det_dedup_frames,
                det_dedup_threshold=args.default_det_dedup_threshold,
                det_dedup_block_size=args.default_det_dedup_block_size,
                det_location=args.default_det_location,
                det_relation=args.default_det_relation,
                det_number=args.default_det_number,
                retrieve_frame_only=args.default_retrieve_frame_only,
            )
        return DetectionServiceConfig(
            aks=AksBlipConfig.merge_config(config.aks),
            mmdino=MMDinoConfig.merge_config(config.mmdino),
            use_det=first_available(config.use_det, args.default_use_det),
            det_dedup_frames=first_available(config.det_dedup_frames, args.default_det_dedup_frames),
            det_dedup_threshold=first_available(config.det_dedup_threshold, args.default_det_dedup_threshold),
            det_dedup_block_size=first_available(config.det_dedup_block_size, args.default_det_dedup_block_size),
            det_location=first_available(config.det_location, args.default_det_location),
            det_relation=first_available(config.det_relation, args.default_det_relation),
            det_number=first_available(config.det_number, args.default_det_number),
            retrieve_frame_only=first_available(config.retrieve_frame_only, args.default_retrieve_frame_only),
        )


args = bentoml.use_arguments(DetectionArgs).override()


@vrag_service(args)
class DetectionService:
    aks = bentoml.depends(AksBlipService)
    mmdino = bentoml.depends(MMDINOService)

    def __init__(self) -> None:
        self._cacher = get_cacher(args.detection_cache_size)
        self._image_hasher = ImageHasher.with_cacher(self._cacher)
        logger.info("DetectionService initialized.")

    @bentoml.api
    async def detect(
        self, query: Query, frame_extraction: FrameExtraction, config: Optional[DetectionServiceConfig] = None
    ) -> DetectionResult:
        merged_config = DetectionServiceConfig.merge_config(config)

        scene_desc = query.access_scene_desc or query.access_filtered_targets

        if not merged_config.use_det and (scene_desc is None or len(scene_desc) == 0):
            return DetectionResult()

        if query.det and query.det.scene_occurrence_count:
            msg = f"Only select {query.det.scene_occurrence_count} key frames as the query specified."
            logger.info(msg)
            merged_config.aks.target_frame_count = query.det.scene_occurrence_count
            merged_config.det_number = query.det.num
            merged_config.det_relation = query.det.rel
            merged_config.det_location = query.det.loc

        dedup_indices = list(range(len(frame_extraction.frame_timestamps)))
        if merged_config.det_dedup_frames:
            msg = (
                f"Applying deduplication: threshold={merged_config.det_dedup_threshold}, "
                f"block_size={merged_config.det_dedup_block_size}."
            )
            logger.debug(msg)
            dedup_indices = await self._image_hasher.get_unique_frame_indices_async(
                frame_extraction.frames,
                merged_config.det_dedup_threshold,
                merged_config.det_dedup_block_size,
            )
            frame_extraction = frame_extraction.slice(dedup_indices)

        det_top_idx = (
            await retry_async_request(
                lambda: self.aks.select_keyframes(
                    frames=frame_extraction.frames, queries=scene_desc, config=merged_config.aks
                ),
                "detection_aks_sample",
            )
            if scene_desc
            else []
        )

        msg = f"Selected {len(det_top_idx)} keyframes: {det_top_idx}"
        logger.debug(msg)
        if not det_top_idx:
            logger.warning("Not select any keyframes, returning empty result.")
            return DetectionResult()

        selected_frames = [frame_extraction.frames[i] for i in det_top_idx]
        timestamps = [frame_extraction.frame_timestamps[i] for i in det_top_idx]

        det_docs: List[DETDoc] = []

        if not merged_config.retrieve_frame_only and query.access_filtered_targets:
            results: List[str] = await self._describe_scenes_inner(
                frames=selected_frames, prompt=query.access_filtered_targets, config=merged_config
            )
            det_docs = list(zip(results, timestamps, strict=True))

        return DetectionResult(det_docs=det_docs, det_top_idx=indexed(dedup_indices, det_top_idx))

    async def _describe_scenes_inner(
        self, frames: List[np.ndarray], prompt: List[str], config: Optional[DetectionServiceConfig] = None
    ) -> List[str]:
        if len(frames) == 0:
            return []

        key = _get_cache_key(
            prompt, config.det_location, config.det_relation, config.det_number, config.mmdino.mmdino_threshold
        )

        @self._cacher.cached_with(lambda *_, **__: key)
        async def _describe(frames: List[np.ndarray], location: bool, relation: bool, number: bool) -> List[str]:
            detection_results: MMDINODetectionBatchResult = await retry_async_request(
                lambda: self.mmdino.ov_detect(frames, prompt, config.mmdino), "detection_mmdino_detect"
            )

            frame_height = frames[0].shape[0]
            frame_width = frames[0].shape[1]

            return [
                SceneDescriber.from_detection_result(
                    det_result, frame_width, frame_height
                ).generate_scene_graph_description(location_desc=location, relation_desc=relation, number_desc=number)
                for det_result in detection_results.results
            ]

        return await _describe(frames, config.det_location, config.det_relation, config.det_number)


def _get_cache_key(prompts: List[str], location: bool, relation: bool, number: bool, threshold: float, *_, **__):
    return f"{''.join(sorted(prompts))}{location}{relation}{number}{str(threshold)}"
