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
"""SOP（标准作业流程）加载与校验模块。
SOP 文件为 JSON 格式，结构如下::
    {
        "name": "SOP 名称",
        "description": "SOP 描述",
        "steps": [
            {
                "id": "S1",                       # 步骤唯一标识，必选
                "name": "步骤名称",                # 必选
                "description": "步骤动作详细描述",  # 必选，供 VLM 判定
                "visual_query": "画面视觉描述",     # 必选，供 CLIP 关键帧检索
                "critical": true,                 # 可选，关键步骤缺失告 ERROR，默认 false
                "min_duration_s": 5,              # 可选，步骤最短耗时（秒），低于此值视为无效
                "max_duration_s": 60,             # 可选，步骤最长耗时（秒），超过此值截断
                "requires_sound": true,           # 可选，该步骤需伴随声音活动，默认 false
            }
        ]
    }
"""

import json
import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SopStep:
    """单个 SOP 步骤定义。"""

    step_id: str
    name: str
    description: str
    visual_query: str
    critical: bool = False
    min_duration_s: Optional[float] = None
    max_duration_s: Optional[float] = None
    requires_sound: bool = False


@dataclass
class Sop:
    """一份完整的标准作业流程。"""

    name: str
    description: str
    steps: List[SopStep] = field(default_factory=list)

    def step_index(self, step_id: str) -> int:
        """返回步骤在 SOP 中的顺序下标，不存在返回 -1。"""
        for i, step in enumerate(self.steps):
            if step.step_id == step_id:
                return i
        return -1

    def get_step(self, step_id: str) -> Optional[SopStep]:
        idx = self.step_index(step_id)
        return self.steps[idx] if idx >= 0 else None


def _parse_bool(value, field_name: str, position: int) -> bool:
    """将布尔值或字符串安全地解析为 bool。

    接受 True/False 以及大小写不敏感的 "true"/"false"；
    其他类型或无法识别的字符串将抛出 ValueError。
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        raise ValueError(
            f"SOP step[{position}] field '{field_name}' cannot interpret {value!r} as bool (expected true/false)"
        )
    raise TypeError(f"SOP step[{position}] field '{field_name}' must be bool or string, got {type(value).__name__}")


def _validate_step(raw: dict, position: int) -> SopStep:
    for key in ("id", "name", "description", "visual_query"):
        if not raw.get(key) or not isinstance(raw.get(key), str):
            raise ValueError(f"SOP step[{position}] missing or invalid field: {key}")
    min_d = raw.get("min_duration_s")
    max_d = raw.get("max_duration_s")
    for val, key in ((min_d, "min_duration_s"), (max_d, "max_duration_s")):
        if val is not None and (not isinstance(val, (int, float)) or val < 0):
            raise ValueError(f"SOP step[{position}] invalid {key}: {val}")
    if min_d is not None and max_d is not None and min_d > max_d:
        raise ValueError(f"SOP step[{position}] min_duration_s > max_duration_s")
    return SopStep(
        step_id=raw["id"],
        name=raw["name"],
        description=raw["description"],
        visual_query=raw["visual_query"],
        critical=_parse_bool(raw.get("critical", False), "critical", position),
        min_duration_s=min_d,
        max_duration_s=max_d,
        requires_sound=_parse_bool(raw.get("requires_sound", False), "requires_sound", position),
    )


def load_sop(sop_path: str) -> Sop:
    """加载并校验 SOP 文件。

    :param sop_path: SOP JSON 文件路径
    :return: Sop 对象
    """
    if not sop_path or not isinstance(sop_path, str):
        raise ValueError(f"invalid sop path: {sop_path}")
    if not os.path.isfile(sop_path):
        raise ValueError(f"sop file not found: {sop_path}")

    with open(sop_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    raw_steps = raw.get("steps")
    if not raw_steps or not isinstance(raw_steps, list):
        raise ValueError("SOP must contain a non-empty 'steps' list")

    steps = [_validate_step(item, i) for i, item in enumerate(raw_steps)]
    step_ids = [s.step_id for s in steps]
    if len(step_ids) != len(set(step_ids)):
        raise ValueError("SOP step ids must be unique")

    return Sop(
        name=raw.get("name", "unnamed_sop"),
        description=raw.get("description", ""),
        steps=steps,
    )
