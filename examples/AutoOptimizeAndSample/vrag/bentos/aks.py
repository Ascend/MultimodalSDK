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
# Adaptive keyframe Selection service using BLIP for keyframe selection.


from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Tuple

import bentoml
import numpy as np
from pydantic import Field

from vrag.bentos.blip import BlipArgs, BlipService
from vrag.logger import logger
from vrag.shared import flatten, ConfigBase, first_available, retry_async_request, vrag_service


class AksBlipArgs(BlipArgs):
    """AKS Blip keyframe selection configuration."""

    default_target_frame_count: int = Field(24, ge=1)
    """Target number of keyframes to select."""
    default_max_recursion_depth: int = Field(5, ge=1)
    """Maximum recursion depth for adaptive segment splitting."""
    default_mean_diff_threshold: float = Field(0.05, ge=0.0, lt=1.0)
    """Mean difference threshold below which a segment is considered uniform and no longer split."""
    default_std_dev_threshold: float = Field(0, ge=0.0, lt=1.0)
    """Standard deviation threshold below which a segment is considered uniform and no longer split."""


class AksBlipConfig(ConfigBase):
    """AKS Blip keyframe selection configuration for request-level."""

    target_frame_count: Optional[int] = Field(None, ge=1)
    max_recursion_depth: Optional[int] = Field(None, ge=1)
    mean_diff_threshold: Optional[float] = Field(None, ge=0.0, lt=1.0)
    std_dev_threshold: Optional[float] = Field(None, ge=0.0, lt=1.0)

    @staticmethod
    def merge_config(config: Optional["AksBlipConfig"]) -> "AksBlipConfig":
        if config is None:
            return AksBlipConfig(
                target_frame_count=args.default_target_frame_count,
                max_recursion_depth=args.default_max_recursion_depth,
                mean_diff_threshold=args.default_mean_diff_threshold,
                std_dev_threshold=args.default_std_dev_threshold,
            )

        return AksBlipConfig(
            target_frame_count=first_available(config.target_frame_count, args.default_target_frame_count),
            max_recursion_depth=first_available(config.max_recursion_depth, args.default_max_recursion_depth),
            mean_diff_threshold=first_available(config.mean_diff_threshold, args.default_mean_diff_threshold),
            std_dev_threshold=first_available(config.std_dev_threshold, args.default_std_dev_threshold),
        )


args = bentoml.use_arguments(AksBlipArgs).override()


@vrag_service(args)
class AksBlipService:
    """Adaptive Keyframe Selection service"""

    blip = bentoml.depends(BlipService)

    def __init__(self):
        logger.info("AksBlipService initialized.")

    @bentoml.api
    async def select_keyframes(
        self, frames: np.ndarray, queries: List[str], config: Optional[AksBlipConfig] = None
    ) -> List[int]:
        merged_config = AksBlipConfig.merge_config(config)

        scores = await retry_async_request(lambda: self.blip.compute_scores(frames, queries), "aks_compute_scores")

        return _Segment.new(scores=scores, quota=merged_config.target_frame_count).split_into_indices(
            merged_config.mean_diff_threshold, merged_config.std_dev_threshold, merged_config.max_recursion_depth
        )


@dataclass
class _Segment:
    """Represents a segment with its scores, depth, and corresponding frame indices."""

    scores: List[float]
    quota: int
    frame_indices: List[int]
    depth: int

    @classmethod
    def new(cls, scores: List[float], quota: int) -> "_Segment":
        return cls(scores=scores, quota=quota, frame_indices=list(range(len(scores))), depth=0)

    @property
    def mean(self) -> float:
        return np.mean(self.scores).item()

    @property
    def std(self) -> float:
        return np.std(self.scores).item()

    def clone(self) -> "_Segment":
        return deepcopy(self)

    def can_split(self) -> bool:
        return len(self.frame_indices) > 1 and self.quota // 2 > 0 and self.quota < self.length()

    def length(self) -> int:
        return len(self.scores)

    def top(self, k: int) -> List[float]:
        return sorted(self.scores, reverse=True)[:k]

    def mean_diff_top(self, k) -> float:
        return np.mean(self.top(k)).item() - self.mean

    def empty(self) -> bool:
        return self.length() == 0

    def frames_indices_top(self, k: int) -> List[int]:
        if k <= 0:
            return []

        top_indices: List[int] = np.argsort(self.scores)[-k:].tolist()
        return [self.frame_indices[i] for i in top_indices]

    def quota_frames_indices(self) -> List[int]:
        return self.frames_indices_top(self.quota)

    def split_into_seg(
        self, mean_diff_threshold: float, std_dev_threshold: float, max_recursion_depth: int
    ) -> List["_Segment"]:
        if not self.can_split():
            return [self]

        logger.debug("Aks start recursive split segments.")

        return _recursive_split_segments([self.clone()], mean_diff_threshold, std_dev_threshold, max_recursion_depth)

    def binary_split(self) -> Tuple["_Segment", "_Segment"]:
        if not self.can_split():
            raise RuntimeError("Aks can not split the segment.")

        mid = self.length() // 2
        quota = round(self.quota / 2)

        return (
            _Segment(
                scores=self.scores[:mid], depth=self.depth + 1, frame_indices=self.frame_indices[:mid], quota=quota
            ),
            _Segment(
                scores=self.scores[mid:],
                depth=self.depth + 1,
                frame_indices=self.frame_indices[mid:],
                quota=self.quota - quota,
            ),
        )

    def split_into_indices(
        self, mean_diff_threshold: float, std_dev_threshold: float, max_recursion_depth: int
    ) -> List[int]:
        msg = f"Aks split {self.length()} candidate frames."
        logger.info(msg)
        segments = self.split_into_seg(
            mean_diff_threshold=mean_diff_threshold,
            std_dev_threshold=std_dev_threshold,
            max_recursion_depth=max_recursion_depth,
        )
        msg = f"Aks split out {len(segments)} segments."
        logger.info(msg)
        iter_segments = (seg.quota_frames_indices() for seg in segments)
        selected = sorted(flatten(iter_segments))

        return selected


def _recursive_split_segments(
    segments: List["_Segment"], mean_diff_threshold: float, std_dev_threshold: float, max_recursion_depth: int
) -> List["_Segment"]:
    if not segments:
        return []

    split_segments: List[_Segment] = []
    no_split_segments: List[_Segment] = []

    for seg in (s for s in segments if not s.empty()):

        def _should_split_by_variation(_seg: _Segment) -> bool:
            mean_diff = _seg.mean_diff_top(_seg.quota)
            std = _seg.std
            msg = (
                f"depth[{_seg.depth}]|alloc[{_seg.quota}]|Range {_seg.frame_indices[0]}~{_seg.frame_indices[-1]}, "
                f"{len(_seg.frame_indices)} in all|{mean_diff=:.2f}|{std:.2f}"
            )
            logger.debug(msg)
            return mean_diff <= mean_diff_threshold or std <= std_dev_threshold

        should_split = seg.depth < max_recursion_depth and seg.can_split() and _should_split_by_variation(seg)

        if should_split:
            split_segments.extend(seg.binary_split())
        else:
            no_split_segments.append(seg)

    return no_split_segments + _recursive_split_segments(
        split_segments, mean_diff_threshold, std_dev_threshold, max_recursion_depth
    )
