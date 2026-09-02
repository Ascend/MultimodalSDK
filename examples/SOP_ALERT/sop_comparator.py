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
"""比对分析层：实际步骤观测序列 vs SOP，生成违规告警。
支持的告警类型：
- STEP_MISSING          步骤丢失（关键步骤为 ERROR，普通步骤为 WARNING）
- STEP_ORDER_VIOLATION  步骤顺序异常（基于最长上升子序列定位乱序步骤）
- STEP_DURATION_ANOMALY 步骤耗时异常（超出 SOP 规定的时长范围）
"""

from dataclasses import dataclass, field
from typing import List, Optional

from common import format_ts
from sop_loader import Sop
from step_extractor import ExtractionResult, StepObservation

LEVEL_ERROR = "ERROR"
LEVEL_WARNING = "WARNING"
LEVEL_INFO = "INFO"

ALERT_STEP_MISSING = "STEP_MISSING"
ALERT_ORDER_VIOLATION = "STEP_ORDER_VIOLATION"
ALERT_DURATION_ANOMALY = "STEP_DURATION_ANOMALY"
ALERT_AUDIO_SILENCE = "AUDIO_SILENCE_ANOMALY"
ALERT_UNKNOWN_OPERATION = "UNKNOWN_OPERATION"

# 各类告警的扣分权重，用于计算合规评分
_ALERT_PENALTY = {
    (ALERT_STEP_MISSING, LEVEL_ERROR): 25,
    (ALERT_STEP_MISSING, LEVEL_WARNING): 15,
    (ALERT_ORDER_VIOLATION, LEVEL_WARNING): 10,
    (ALERT_DURATION_ANOMALY, LEVEL_WARNING): 5,
    (ALERT_AUDIO_SILENCE, LEVEL_WARNING): 5,
    (ALERT_UNKNOWN_OPERATION, LEVEL_INFO): 2,
}


@dataclass
class Alert:
    """一条违规告警。"""

    alert_type: str
    level: str
    step_id: Optional[str]
    step_name: str
    message: str
    time_range: Optional[str] = None  # "mm:ss-mm:ss"，无法定位时间时为 None

    def to_dict(self) -> dict:
        return {
            "alert_type": self.alert_type,
            "level": self.level,
            "step_id": self.step_id,
            "step_name": self.step_name,
            "message": self.message,
            "time_range": self.time_range,
        }


@dataclass
class ComplianceResult:
    """比对分析结果。"""

    sop_name: str
    total_steps: int
    matched_step_ids: List[str] = field(default_factory=list)
    alerts: List[Alert] = field(default_factory=list)
    compliance_score: int = 100

    @property
    def is_compliant(self) -> bool:
        return not any(a.level in (LEVEL_ERROR, LEVEL_WARNING) for a in self.alerts)


def compare_with_sop(sop: Sop, extraction: ExtractionResult) -> ComplianceResult:
    """将实际步骤观测序列与 SOP 比对，输出合规结果与告警列表。

    :param sop: SOP 对象
    :param extraction: 步骤理解层输出
    :return: ComplianceResult
    """
    observations = extraction.observations
    matched = [o for o in observations if o.step_id and sop.step_index(o.step_id) >= 0]
    unknown = [o for o in observations if not o.step_id or sop.step_index(o.step_id) < 0]

    alerts = []
    alerts.extend(_check_missing_steps(sop, matched))
    alerts.extend(_check_step_order(sop, matched))
    alerts.extend(_check_step_duration(sop, matched))
    if extraction.audio_enabled:
        alerts.extend(_check_audio_silence(sop, matched))
    alerts.extend(_check_unknown_operations(unknown))

    score = 100
    for alert in alerts:
        score -= _ALERT_PENALTY.get((alert.alert_type, alert.level), 5)
    score = max(0, score)

    # 按视频时间排序，无时间的告警（如步骤丢失）排在最后
    alerts.sort(key=lambda a: (a.time_range is None, a.time_range or "", a.step_id or ""))

    return ComplianceResult(
        sop_name=sop.name,
        total_steps=len(sop.steps),
        matched_step_ids=[o.step_id for o in matched],
        alerts=alerts,
        compliance_score=score,
    )


def _time_range(obs: StepObservation) -> str:
    return f"{format_ts(obs.start_s)}-{format_ts(obs.end_s)}"


def _check_missing_steps(sop: Sop, matched: List[StepObservation]) -> List[Alert]:
    """步骤丢失检测：SOP 中定义但未观测到的步骤。"""
    observed_ids = {o.step_id for o in matched}
    alerts = []
    for step in sop.steps:
        if step.step_id in observed_ids:
            continue
        level = LEVEL_ERROR if step.critical else LEVEL_WARNING
        tag = "关键步骤" if step.critical else "步骤"
        alerts.append(
            Alert(
                alert_type=ALERT_STEP_MISSING,
                level=level,
                step_id=step.step_id,
                step_name=step.name,
                message=f"{tag}“{step.name}”({step.step_id}) 未在视频中检测到，疑似漏做。",
            )
        )
    return alerts


def _check_step_order(sop: Sop, matched: List[StepObservation]) -> List[Alert]:
    """顺序异常检测。

    将按时间排序的观测步骤映射为 SOP 下标序列，求最长非降子序列（LIS），
    不在 LIS 中的步骤即为破坏全局顺序的乱序步骤。
    """
    if len(matched) < 2:
        return []
    order_indices = [sop.step_index(o.step_id) for o in matched]
    keep = _longest_non_decreasing_subsequence(order_indices)

    alerts = []
    for pos, obs in enumerate(matched):
        if pos in keep:
            continue
        expected_rank = sop.step_index(obs.step_id) + 1
        alerts.append(
            Alert(
                alert_type=ALERT_ORDER_VIOLATION,
                level=LEVEL_WARNING,
                step_id=obs.step_id,
                step_name=obs.name,
                message=(
                    f"步骤“{obs.name}”({obs.step_id}) 执行顺序异常："
                    f"SOP 规定为第 {expected_rank} 步，实际执行时序与规程不符。"
                ),
                time_range=_time_range(obs),
            )
        )
    return alerts


def _check_step_duration(sop: Sop, matched: List[StepObservation]) -> List[Alert]:
    """耗时异常检测：观测耗时超出 SOP 规定的 [min, max] 范围。"""
    alerts = []
    for obs in matched:
        step = sop.get_step(obs.step_id)
        duration = obs.duration_s
        if step.min_duration_s is not None and duration < step.min_duration_s:
            alerts.append(
                Alert(
                    alert_type=ALERT_DURATION_ANOMALY,
                    level=LEVEL_WARNING,
                    step_id=obs.step_id,
                    step_name=obs.name,
                    message=(
                        f"步骤“{obs.name}”({obs.step_id}) 耗时 {duration:.1f}s，"
                        f"低于规程要求的最短 {step.min_duration_s:.1f}s，疑似操作过快或未按规程执行。"
                    ),
                    time_range=_time_range(obs),
                )
            )
        elif step.max_duration_s is not None and duration > step.max_duration_s:
            alerts.append(
                Alert(
                    alert_type=ALERT_DURATION_ANOMALY,
                    level=LEVEL_WARNING,
                    step_id=obs.step_id,
                    step_name=obs.name,
                    message=(
                        f"步骤“{obs.name}”({obs.step_id}) 耗时 {duration:.1f}s，"
                        f"超过规程允许的最长 {step.max_duration_s:.1f}s，疑似操作卡滞或异常。"
                    ),
                    time_range=_time_range(obs),
                )
            )
    return alerts


def _check_audio_silence(sop: Sop, matched: List[StepObservation]) -> List[Alert]:
    """声音缺失检测：SOP 要求伴随设备声音的步骤未检测到声音活动。"""
    alerts = []
    for obs in matched:
        step = sop.get_step(obs.step_id)
        if step.requires_sound and obs.sound_detected is False:
            alerts.append(
                Alert(
                    alert_type=ALERT_AUDIO_SILENCE,
                    level=LEVEL_WARNING,
                    step_id=obs.step_id,
                    step_name=obs.name,
                    message=(
                        f"步骤“{obs.name}”({obs.step_id}) 规程要求伴随设备运行声音，"
                        f"但音频检测未发现声音活动，疑似设备未启动或空操作。"
                    ),
                    time_range=_time_range(obs),
                )
            )
    return alerts


def _check_unknown_operations(unknown: List[StepObservation]) -> List[Alert]:
    """计划外操作检测：观测到但无法对应任何 SOP 步骤的操作。"""
    return [
        Alert(
            alert_type=ALERT_UNKNOWN_OPERATION,
            level=LEVEL_INFO,
            step_id=None,
            step_name=obs.name,
            message=f"检测到计划外操作“{obs.name}”，不属于 SOP 规定步骤，建议人工复核。",
            time_range=_time_range(obs),
        )
        for obs in unknown
    ]


def _longest_non_decreasing_subsequence(seq: List[int]) -> set:
    """返回最长非降子序列对应的位置下标集合（O(n^2) 动态规划）。"""
    n = len(seq)
    if n == 0:
        return set()
    dp = [1] * n
    prev = [-1] * n
    for i in range(1, n):
        for j in range(i):
            if seq[j] <= seq[i] and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
                prev[i] = j
    best_end = max(range(n), key=lambda i: dp[i])
    keep = set()
    cur = best_end
    while cur != -1:
        keep.add(cur)
        cur = prev[cur]
    return keep
