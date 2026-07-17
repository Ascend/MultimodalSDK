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


from PIL import Image
from openai import OpenAI

from mm.core.frame_selector.frame_selector import KFrameSelector
from common import imgs_to_base64_list, create_messages, validate_qa_inputs, extract_frames_from_video, send_messages


class SimpleQaDemo:
    _SYSTEM_MESSAGE = {"role": "system", "content": "你是个视频问答助手，请根据提供的视频内容直接回答用户的问题。"}

    def __init__(
        self,
        model_path: str,
        device_list: list,
        similar_threshold: float = 0.06,
        image_similar_threshold: float = 0.015,
        vlm_url: str = None,
        api_key: str = "NONE",
        vlm_model_name: str = None,
        model_type: str = 'cn_clip',
    ):
        self.frame_selector = KFrameSelector(
            model_path, device_list[0], model_type, similar_threshold, image_similar_threshold
        )
        self.vlm_client = OpenAI(base_url=vlm_url, api_key=api_key)
        self.vlm_model_name = vlm_model_name
        self._frame_cache = {}

    def qa(self, query, video_path, sample_num):
        validate_qa_inputs(video_path, query, sample_num)

        frames, _, _ = extract_frames_from_video(video_path, cache=self._frame_cache)
        _, selected_frames = self.frame_selector.select_keyframes(query, frames, sample_num)

        selected_frames = [Image.fromarray(frame) for frame in selected_frames]
        q = (
            f"请根据上下文内容简短问答，不要进行任何解释或者对概念的说明，给出答案即可，"
            f"当无法从视频和上下文里面获取答案时，请回答不清楚。你需要回答的问题是：{query}。"
        )
        base64_frames = imgs_to_base64_list(selected_frames)
        user_messages = create_messages(q, base64_frames)

        return send_messages(self.vlm_client, self.vlm_model_name, [self._SYSTEM_MESSAGE, user_messages])


def kframe_selector_example():
    demo = SimpleQaDemo(
        model_path="/path/to/chinese-clip-vit-large-patch14-336px",
        device_list=[0],
        vlm_url="http://127.0.0.1:8111/v1",
        api_key="None",
        vlm_model_name="Qwen2.5-VL-32B-Instruct",
        model_type='cn_clip',
    )

    video_path = "/path/to/test_video.mp4"
    query = "视频中出现了哪些交通标志"
    sample_num = 4

    answer = demo.qa(query, video_path, sample_num)

    print(f"查询: {query}")
    print(f"回答: {answer}")


if __name__ == "__main__":
    kframe_selector_example()
