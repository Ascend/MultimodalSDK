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

# pylint: skip-file

import json

from PIL import Image
import cv2
import numpy as np
from openai import OpenAI

from mm.core.frame_selector.frame_selector import KRangFrameSelector
from common import imgs_to_base64_list, create_messages, validate_qa_inputs, extract_frames_from_video, send_messages


REWRITE_SYSTEM_PROMPT = """<no think>. 你是一个视频处理助手。你的任务是分析用户查询并选择适当的视频采样策略，以严格的JSON格式输出结果。
## 采样策略决策：
- 当查询中提到特定对象、动作或内容时：使用基于关键词的采样，函数名为"query_keyframe"
- 当查询指定时间范围或均匀覆盖时：使用固定步长采样，函数名为"uniform_sample"
- 如果查询中同时包含对象和时间范围，优先使用带有时间参数的"query_keyframe"
## 查询解析指南：
- 对于"function_name"，根据上述决策进行设置
- 在"params"中：
- "query"：对于"query_keyframe"，应包含一个重新分析的查询字符串。对于"uniform_sample"，应设置为空字符串
- "start_time"：根据你的查询，你认为答案可能出现在的时间区间的开始时间，解析为整数（秒）；否则为null
- "end_time"：根据你的查询，你认为答案可能出现在的时间区间的结束时间，解析为整数（秒）；否则为null
- "sample_nums"：设置采样帧数。对于"query_keyframe"使用16；对于"uniform_sample"使用24
## 输出格式：
严格返回以下JSON对象，不包含任何额外内容：
{
"function": {
"function_name": "字符串，'均匀采样'或'查询关键帧'",
"params": {
"query": "字符串，重新分析的查询字符串，用于query_keyframe，均匀采样时为空，使用中文输出。",
"start_time": "整数或null",
"end_time": "整数或null",
"sample_nums": "整数"
}
}
}
## 示例：
用户："红色小轿车是什么时候出现和消失的？"
→ {
"function": {
"function_name": "query_keyframe",
"params": {
"query": "红色小轿车",
"start_time": null,
"end_time": null,
"sample_nums": 16
}
}
}
用户："在30秒到120秒之间均匀采样"
→ {
"function": {
"function_name": "uniform_sample",
"params": {
"query": "",
"start_time": 30,
"end_time": 120,
"sample_nums": 24
}
}
}
"""

QA_SYSTEM_PROMPT = """你是一个严格遵守输出格式的视频分析工具。请根据从视频里选择的和问题相关的帧，严格按照以下要求来回答用户的问题。

## 核心原则
1. 时间来源唯一性：所有时间信息必须且只能基于"帧索引-时间映射表"。禁止使用视频内显示的任何时间戳、时钟或字幕时间。
2. 基于实际内容：仅分析提供的视频帧描述内容，不推测未显示的内容。
3. 严格匹配用户问题：只分析与用户问题直接相关的事件或物体。如果视频中存在相似但不是问题所指的物体，不应进行分析。
4. 全面分析：仔细分析所有帧，找到所有和问题相关的帧。

## 帧索引-时间映射表（唯一时间源）
映射表格式：第X帧: mm:ss

映射表内容：
{frame_indices_times}

## 时间处理规范
- 格式要求：所有时间必须格式化为"mm:ss"（分钟和秒不足两位时补零）
- 起止确定：
  - 开始时间：物体/事件首次可辨识（即使模糊）的帧对应时间
  - 结束时间：物体/事件最后一次明确可见的帧对应时间

## 输出格式
- 必须输出有效的JSON数组，无任何额外文本
- 每个对象包含且仅包含：
{{
  "start_time": "mm:ss",
  "end_time": "mm:ss"
}}
- 找到匹配事件：按时间顺序返回所有时间片段
- 未找到事件：返回空数组 []
"""


class RangeDetectionQaDemo:
    def __init__(
        self,
        model_path: str,
        device_list: list,
        similar_threshold: float = 0.03,
        similar_threshold_image: float = 0.015,
        vlm_url: str = None,
        api_key: str = "NONE",
        vlm_model_name: str = None,
        model_type: str = 'cn_clip',
    ):
        self.frame_selector = KRangFrameSelector(
            model_path, device_list[0], model_type, similar_threshold, similar_threshold_image
        )
        self.vlm_client = OpenAI(base_url=vlm_url, api_key=api_key)
        self.vlm_model_name = vlm_model_name
        self._frame_cache = {}
        self._supported_functions = {
            'uniform_sample': self._execute_uniform_sample,
            'query_keyframe': self._execute_query_keyframe,
        }

    @staticmethod
    def _get_frame_times(frame_indices):
        return [f"第{i}帧：{idx // 60:02d}:{idx % 60:02d}" for i, idx in enumerate(frame_indices, 1)]

    @staticmethod
    def _make_time_valid(start_time, end_time, frame_len):
        start_time = start_time if start_time is not None else 0
        end_time = min(end_time if end_time is not None else frame_len - 1, frame_len - 1)
        if start_time == end_time:
            start_time, end_time = 0, frame_len - 1
        return int(start_time), int(end_time)

    @staticmethod
    def _uniform_sample(start_time, end_time, frames, sample_num):
        frame_indices = np.linspace(start_time, end_time, sample_num, dtype=int)
        return frame_indices, [frames[i] for i in frame_indices]

    def qa(self, query: str, video_path: str, sample_num: int):
        validate_qa_inputs(video_path, query, sample_num)

        frames, fps, total_frames = extract_frames_from_video(video_path, cache=self._frame_cache)
        sorted_frame_indices, selected_frames = self._select_frames(query, frames, fps, total_frames, sample_num)

        frame_indices_times = self._get_frame_times(sorted_frame_indices)
        if len(selected_frames) >= 10:
            selected_frames = [cv2.resize(frame, (384, 384)) for frame in selected_frames]

        selected_frames = [Image.fromarray(frame) for frame in selected_frames]

        system_prompt = {
            "role": "system",
            "content": QA_SYSTEM_PROMPT.format(
                frame_indices_times="\n".join(f"- {item}" for item in frame_indices_times)
            ),
        }

        q = (
            f"仔细分析视频内容，然后找到与问题相关的视频帧。在用户没有明确要求分析视频中出现的时间戳的情况下，"
            f"禁止使用视频中出现的时间戳来作为问题的答案, 必须根据时间戳映射表来进行回答。你需要回答的问题如下：{query}"
        )

        base64_frames = imgs_to_base64_list(selected_frames)
        user_messages = create_messages(q, base64_frames)

        return send_messages(self.vlm_client, self.vlm_model_name, [system_prompt, user_messages])

    def _select_frames(self, query, frames, fps, total_frames, sample_num):
        rewrite_messages = [
            {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"根据视频的文字记录，提取与查询项相关的关键词和短语, 你需要提取的查询如下：{query}。"
                f"考虑视频的上下文，包括其时长为{total_frames}和帧率为{fps}，"
                f"以识别出最显著的术语、实体和概念。优先考虑与查询项的相关性，并提供一份简洁的列表。",
            },
        ]
        new_query = send_messages(self.vlm_client, self.vlm_model_name, rewrite_messages, max_tokens=512)
        return self._execute(new_query, frames, sample_num)

    def _execute(self, model_response, frames, sample_num):
        try:
            rewrite_query = json.loads(model_response)
            function_name = rewrite_query["function"]["function_name"]
            params = rewrite_query["function"]["params"]
        except Exception:
            function_name, params = "uniform_sample", {}

        handler = self._supported_functions.get(function_name, self._execute_uniform_sample)
        return handler(frames, params, sample_num)

    @staticmethod
    def _execute_uniform_sample(frames, params, sample_num):
        start_time = params.get("start_time", 0)
        end_time = params.get("end_time", len(frames) - 1)
        start_time, end_time = RangeDetectionQaDemo._make_time_valid(start_time, end_time, len(frames))
        return RangeDetectionQaDemo._uniform_sample(start_time, end_time, frames, sample_num)

    def _execute_query_keyframe(self, frames, params, sample_num):
        query = params["query"]
        start_time = params.get("start_time", 0)
        end_time = params.get("end_time", len(frames) - 1)
        start_time, end_time = self._make_time_valid(start_time, end_time, len(frames))
        return self.frame_selector.select_keyframes(query, frames[int(start_time) : int(end_time)], sample_num)


def k_range_frame_selector_example():
    demo = RangeDetectionQaDemo(
        model_path="/path/to/chinese-clip-vit-large-patch14-336px",
        device_list=[0],
        vlm_url="http://127.0.0.1:8111/v1",
        api_key="None",
        vlm_model_name="Qwen2.5-VL-32B-Instruct",
        model_type='cn_clip',
    )

    video_path = "/path/to/test_video.mp4"
    query = "红色小轿车是什么时候出现和消失的"
    sample_num = 16

    answer = demo.qa(query, video_path, sample_num)

    print(f"查询: {query}")
    print(f"回答: {answer}")


if __name__ == "__main__":
    k_range_frame_selector_example()
