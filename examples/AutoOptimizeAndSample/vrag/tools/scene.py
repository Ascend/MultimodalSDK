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
# Object detection and scene graph generation utilities.


from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Tuple

from vrag.shared import flatten
from vrag.types import MMDINODetectionResult, MMDINODetectionItem


@dataclass
class SceneDescriber:
    nodes: List["_SceneGraphNode"] = field(default_factory=list)

    @classmethod
    def from_detection_result(
        cls, detection_result: MMDINODetectionResult, frame_width: int = 1920, frame_height: int = 1080
    ) -> "SceneDescriber":
        if not detection_result.items:
            return cls()

        objects = [
            _SceneGraphNode.from_detection_item(idx, item, frame_width, frame_height)
            for idx, item in enumerate(detection_result.items)
        ]

        return cls(nodes=objects)

    def generate_scene_graph_description(self, location_desc: bool, relation_desc: bool, number_desc: bool) -> str:
        return " | ".join(
            flatten(
                (
                    self._location_desc() if location_desc else [],
                    self._relation_desc() if relation_desc else [],
                    self._number_desc() if number_desc else [],
                )
            )
        )

    def _location_desc(self) -> List[str]:
        return [f"{obj.unique_id}@{obj.region.name}" for obj in self.nodes]

    def _relation_desc(self) -> List[str]:
        description = []
        n = len(self.nodes)
        for i in range(n):
            for j in range(i + 1, n):
                obj1 = self.nodes[i]
                obj2 = self.nodes[j]

                description.append(
                    f"{obj1.unique_id} @{obj1.calculate_spatial_relation(obj2).name}_of {obj2.unique_id}"
                )
        return description

    def _number_desc(self) -> List[str]:
        count: Dict[str, int] = {}
        for obj in self.nodes:
            count[obj.label] = count.get(obj.label, 0) + 1
        object_count = count

        if not object_count:
            return []
        count_str = ", ".join(f"{k}:{v}" for k, v in object_count.items())
        return [f"Count:[{count_str}]"]


class _StrEnum(str, Enum):
    """String enum base class"""


class _SpatialRelation(_StrEnum):
    """Representing spatial relationships between objects."""

    overlap = auto()
    above = auto()
    below = auto()
    left = auto()
    right = auto()
    top_left = auto()
    top_right = auto()
    bottom_left = auto()
    bottom_right = auto()


class _FrameRegion(_StrEnum):
    """Representing 9-region spatial division of a frame."""

    middle = auto()
    left = auto()
    right = auto()
    top = auto()
    bottom = auto()
    top_left = auto()
    top_right = auto()
    bottom_left = auto()
    bottom_right = auto()


_REGION_MAP = {
    (_FrameRegion.middle, _FrameRegion.middle): _FrameRegion.middle,
    (_FrameRegion.top, _FrameRegion.middle): _FrameRegion.top,
    (_FrameRegion.bottom, _FrameRegion.middle): _FrameRegion.bottom,
    (_FrameRegion.middle, _FrameRegion.left): _FrameRegion.left,
    (_FrameRegion.middle, _FrameRegion.right): _FrameRegion.right,
    (_FrameRegion.top, _FrameRegion.left): _FrameRegion.top_left,
    (_FrameRegion.top, _FrameRegion.right): _FrameRegion.top_right,
    (_FrameRegion.bottom, _FrameRegion.left): _FrameRegion.bottom_left,
    (_FrameRegion.bottom, _FrameRegion.right): _FrameRegion.bottom_right,
}


@dataclass
class _SceneGraphNode:
    id: int
    """Object identifier"""
    label: str
    """Object class name"""
    center: Tuple[int, int]
    """Center coordinates (x, y)"""
    region: _FrameRegion
    """Spatial region"""
    bbox: Tuple[int, int, int, int]
    """Bounding box in [x_min, y_min, width, height] format"""

    @property
    def unique_id(self) -> str:
        return f"{self.label}[{self.id}]"

    @staticmethod
    def get_region(x: int, y: int, w: int, h: int) -> _FrameRegion:
        lw, rw = w / 3, 2 * w / 3
        th, bh = h / 3, 2 * h / 3

        x_region = _FrameRegion.left if x < lw else (_FrameRegion.right if x > rw else _FrameRegion.middle)
        y_region = _FrameRegion.top if y < th else (_FrameRegion.bottom if y > bh else _FrameRegion.middle)

        return _REGION_MAP.get((y_region, x_region), _FrameRegion.middle)

    @classmethod
    def from_detection_item(
        cls, obj_id: int, item: MMDINODetectionItem, frame_width: int, frame_height: int
    ) -> "_SceneGraphNode":
        x1, y1, x2, y2 = item.bbox

        x_min, y_min = min(x1, x2), min(y1, y2)
        width, height = abs(x2 - x1), abs(y2 - y1)

        center_x = x_min + width // 2
        center_y = y_min + height // 2
        region = cls.get_region(center_x, center_y, frame_width, frame_height)

        return cls(
            id=obj_id,
            label=item.class_name,
            center=(center_x, center_y),
            region=region,
            bbox=(x_min, y_min, width, height),
        )

    def calculate_spatial_relation(self, obj: "_SceneGraphNode") -> _SpatialRelation:
        x1, y1, w1, h1 = self.bbox
        x2, y2, w2, h2 = obj.bbox

        if not (x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1):
            return _SpatialRelation.overlap

        cx1, cy1 = self.center
        cx2, cy2 = obj.center

        dx = cx2 - cx1
        dy = cy2 - cy1
        abs_dx = abs(dx)
        abs_dy = abs(dy)

        if abs_dx == 0 and abs_dy == 0:
            return _SpatialRelation.overlap

        ratio = abs_dx / (abs_dy + 1e-6)

        if ratio > 2.0:
            return _SpatialRelation.right if dx > 0 else _SpatialRelation.left
        if ratio < 0.5:
            return _SpatialRelation.below if dy > 0 else _SpatialRelation.above
        if dx > 0 and dy > 0:
            return _SpatialRelation.bottom_right
        if dx > 0 > dy:
            return _SpatialRelation.top_right
        if dx < 0 < dy:
            return _SpatialRelation.bottom_left
        return _SpatialRelation.top_left
