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
"""步骤理解层：从视频中提取实际操作步骤序列。
流程：
1. 对每个 SOP 步骤，用 ``StepRangeLocator``（mm.KRangFrameSelector）定位候选帧区间；
2. 将区间关键帧连同步骤描述送入 VLM，确认该步骤是否真实发生并给出证据描述；
3. 结合音频活动检测结果（mm.load_audio）补充声音证据；
4. 输出按时间排序的实际步骤观测序列（StepObservation 列表）。
"""

import json
import os
from dataclasses import asdict, dataclass, field
from typing import List, Optional
from urllib.parse import urlparse

from common import create_messages, extract_json_block, format_ts, imgs_to_base64_list, send_messages
from sop_loader import Sop

# 单次 VLM 确认最多送入的关键帧数。部分推理服务（如 vllm-ascend v0.21.0rc1）在
# 单请求图片数 >=5 时存在多图特征错位问题，模型只能看到最后一张图；
# 同时区间首尾帧可能混入相邻场景，因此取居中的至多 4 张关键帧送审。
_MAX_VERIFY_IMAGES = 4


def _safe_float(v, default: float = 0.0) -> float:
    """安全转换为 float，None/异常时返回默认值。"""
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _normalize_sound_detected(v) -> Optional[bool]:
    """规范化 sound_detected 字段，支持 bool/None/字符串 'true'/'false'。"""
    if v is None or isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s == "true":
            return True
        if s == "false":
            return False
        raise ValueError(f"sound_detected must be bool/None/'true'/'false', got: {v!r}")
    raise ValueError(f"sound_detected must be bool/None/str, got: {type(v).__name__}")


_VERIFY_SYSTEM_MESSAGE = {
    "role": "system",
    "content": (
        "你是工业生产合规检查助手。你将看到从工人操作视频中抽取的若干关键帧，"
        "需要判断指定的操作步骤是否在这些画面中真实发生。"
        "声音无法通过图片判断，声音要求由音频检测环节另行校验，"
        "请忽略步骤描述中所有关于声音的要求，只依据画面中的动作与场景作答。"
        "只要画面证据支持该步骤发生，performed 必须为 true；严格输出 JSON。"
    ),
}

_VERIFY_PROMPT_TEMPLATE = (
    "待确认的操作步骤：{step_name}。\n"
    "步骤动作描述：{step_description}\n"
    "本消息后附的图片是从视频 {start_ts} 至 {end_ts} 时间段按时间顺序抽取的关键帧。\n"
    "判定规则：\n"
    "1. 区间边界处可能混入相邻场景画面，请逐张查看所有关键帧；"
    "只要步骤在其中一部分关键帧中发生即可。\n"
    "2. 只要步骤的核心动作在画面中发生即判定为真，"
    "不要求描述中的每个细节（如背景物品、声音、颜色）都能在画面中验证。\n"
    "3. evidence 与 performed 必须自洽：若 evidence 描述了核心动作发生，则 performed 必须为 true。\n"
    "请按以下顺序输出 JSON（不要输出其他内容）：\n"
    '{{"evidence": "一句话画面证据描述", "performed": true/false, "confidence": 0到1的小数}}'
)


@dataclass
class StepObservation:
    """一次实际操作步骤的观测结果。"""

    step_id: Optional[str]  # 匹配到的 SOP 步骤 ID；无法对应 SOP 时为 None
    name: str  # 观测到的操作名称
    start_s: float  # 开始时刻（秒）
    end_s: float  # 结束时刻（秒）
    confidence: float = 1.0  # 置信度 [0,1]
    evidence: str = ""  # 画面/声音证据描述
    sound_detected: Optional[bool] = None  # 区间内是否检测到声音活动，未启用音频时为 None

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["duration_s"] = round(self.duration_s, 3)
        return data


@dataclass
class ExtractionResult:
    """步骤理解层输出。"""

    video_duration_s: float
    observations: List[StepObservation] = field(default_factory=list)
    audio_enabled: bool = False


class StepExtractor:
    """基于关键帧检索 + VLM 确认的实际步骤提取器。"""

    def __init__(
        self, locator, vlm_client, vlm_model_name: str, sample_num: int = 8, confidence_threshold: float = 0.5
    ):
        """
        :param locator: 步骤区间定位器（内部使用 mm.KRangFrameSelector）
        :param vlm_client: OpenAI 兼容客户端
        :param vlm_model_name: 推理服务模型名称
        :param sample_num: 每个步骤最大关键帧数量
        :param confidence_threshold: VLM 确认置信度阈值，低于该值视为步骤未发生
        """
        self._locator = locator
        self._vlm_client = vlm_client
        self._vlm_model_name = vlm_model_name
        self._sample_num = sample_num
        self._confidence_threshold = confidence_threshold

        # VLM 端点信任边界校验：工人操作视频关键帧含个人信息，
        # 推理服务必须部署在信任区域内（本地回环或 HTTPS）
        # 使用 urlparse 精确匹配主机名，防止 startswith 前缀匹配被恶意子域名绕过
        parsed = urlparse(str(getattr(vlm_client, "base_url", "")))
        if not (
            parsed.scheme == "https"
            or (parsed.scheme == "http" and parsed.hostname in ("127.0.0.1", "localhost", "::1"))
        ):
            raise ValueError(
                f"VLM endpoint must be https or loopback, got: {parsed.geturl()}. "
                "工人操作视频关键帧含个人信息，禁止明文 HTTP 跨信任区域传输。"
            )

    def extract(self, sop: Sop, video, audio=None) -> ExtractionResult:
        """从视频帧序列中提取与 SOP 相关的实际步骤观测序列。

        :param sop: SOP 对象
        :param video: media_processor.VideoFrames
        :param audio: media_processor.AudioActivity，可为 None
        """
        observations = []
        for step in sop.steps:
            step_range = self._locator.locate(step.step_id, step.visual_query, video, self._sample_num)
            if step_range is None:
                continue

            # 声音要求由音频层（mm.load_audio）单独校验，这里只用纯视觉描述
            # （visual_query）让 VLM 确认画面，避免图片无法体现声音导致误判。
            verdict = self._verify_with_vlm(step.name, step.visual_query, step_range)
            # VLM 可能返回字符串 "true"/"false" 而非布尔值，统一转换为布尔值
            performed = verdict.get("performed")
            if isinstance(performed, str):
                performed = performed.strip().lower() == "true"
            # confidence 可能为 null/None，安全转换防止 float(None) 抛 TypeError
            confidence = _safe_float(verdict.get("confidence"), 0.0)
            if not performed or confidence < self._confidence_threshold:
                continue

            sound_detected = None
            if audio is not None:
                sound_detected = audio.has_sound(step_range.start_s, step_range.end_s)

            observations.append(
                StepObservation(
                    step_id=step.step_id,
                    name=step.name,
                    start_s=step_range.start_s,
                    end_s=step_range.end_s,
                    confidence=round(_safe_float(verdict.get("confidence"), 0.0), 3),
                    evidence=str(verdict.get("evidence") or ""),
                    sound_detected=sound_detected,
                )
            )

        observations.sort(key=lambda o: (o.start_s, o.end_s))
        return ExtractionResult(
            video_duration_s=video.duration_s,
            observations=observations,
            audio_enabled=audio is not None,
        )

    def _verify_with_vlm(self, step_name: str, step_description: str, step_range) -> dict:
        """将区间关键帧送入 VLM，确认步骤是否发生。"""
        from PIL import Image

        key_frames = step_range.key_frames
        if len(key_frames) > _MAX_VERIFY_IMAGES:
            # 去掉首尾各一帧（边界帧可能混入相邻场景）后均匀采样，
            # 兼顾代表性与区间时序覆盖。
            pool = key_frames[1:-1] if len(key_frames) > _MAX_VERIFY_IMAGES + 1 else key_frames
            picks = [round(i * (len(pool) - 1) / (_MAX_VERIFY_IMAGES - 1)) for i in range(_MAX_VERIFY_IMAGES)]
            key_frames = [pool[i] for i in sorted(set(picks))]
        pil_frames = [Image.fromarray(frame) for frame in key_frames]
        prompt = _VERIFY_PROMPT_TEMPLATE.format(
            step_name=step_name,
            step_description=step_description,
            start_ts=format_ts(step_range.start_s),
            end_ts=format_ts(step_range.end_s),
        )
        user_message = create_messages(prompt, imgs_to_base64_list(pil_frames))
        try:
            answer = send_messages(self._vlm_client, self._vlm_model_name, [_VERIFY_SYSTEM_MESSAGE, user_message])
        except Exception as e:
            # 单个步骤 VLM 请求失败不阻断整体流程，降级为"未检测到"
            return {"performed": False, "confidence": 0.0, "evidence": f"VLM请求失败: {e}"}
        try:
            verdict = extract_json_block(answer)
        except (ValueError, json.JSONDecodeError):
            return {"performed": False, "confidence": 0.0, "evidence": f"VLM响应解析失败: {answer[:100]}"}
        if not isinstance(verdict, dict):
            return {"performed": False, "confidence": 0.0, "evidence": "VLM响应格式异常"}
        return verdict


def load_observations(observations_path: str) -> ExtractionResult:
    """从 JSON 文件加载步骤观测序列（离线验证模式）。

    文件结构::

        {
            "video_duration_s": 120.0,
            "audio_enabled": true,
            "observations": [
                {"step_id": "S1", "name": "...", "start_s": 1.0, "end_s": 9.0,
                 "confidence": 0.95, "evidence": "...", "sound_detected": false}
            ]
        }
    """
    if not observations_path or not os.path.isfile(observations_path):
        raise ValueError(f"observations file not found: {observations_path}")
    with open(observations_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    observations = []
    for i, item in enumerate(raw.get("observations", [])):
        try:
            observations.append(
                StepObservation(
                    step_id=item.get("step_id"),
                    name=item["name"],
                    start_s=float(item["start_s"]),
                    end_s=float(item["end_s"]),
                    confidence=_safe_float(item.get("confidence"), 1.0),
                    evidence=item.get("evidence") or "",
                    sound_detected=_normalize_sound_detected(item.get("sound_detected")),
                )
            )
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"invalid observation[{i}]: {item}") from e

    observations.sort(key=lambda o: (o.start_s, o.end_s))
    return ExtractionResult(
        video_duration_s=float(raw.get("video_duration_s", 0.0)),
        observations=observations,
        audio_enabled=bool(raw.get("audio_enabled", False)),
    )
