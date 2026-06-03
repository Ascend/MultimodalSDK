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

from typing import List, Tuple
from io import BytesIO
import base64

from PIL import Image
import cv2
from openai import OpenAI


def imgs_to_base64_list(imgs: List[Image.Image], img_format="JPEG") -> List[str]:
    base64_list = []
    for img in imgs:
        img_rgb = img.convert('RGB') if img.mode != 'RGB' else img
        with BytesIO() as buffer:
            img_rgb.save(buffer, format=img_format)
            base64_list.append(base64.b64encode(buffer.getvalue()).decode("utf-8"))
    return base64_list


def create_messages(prompt: str, base64_images: List[str]) -> dict:
    content = [{"type": "text", "text": prompt}]
    content.extend(
        [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}} for img in base64_images]
    )
    return {"role": "user", "content": content}


def validate_qa_inputs(video_path, query, sample_num):
    if not video_path or not query:
        raise ValueError(f"invalid input params: {video_path=}, {query=}")
    if not isinstance(video_path, str) or not isinstance(query, str):
        raise ValueError(f"invalid input type, please use string type. {video_path=}, {query=}")
    if not isinstance(sample_num, int):
        raise ValueError(f"invalid input type, please use int type. {sample_num=}")


def extract_frames_from_video(video_path: str, resize: Tuple[int, int] = (1280, 720), cache: dict = None):
    if cache is not None and video_path in cache:
        return cache[video_path]

    try:
        import decord
    except ImportError as e:
        raise ValueError("请安装decord: pip install decord") from e
    try:
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    except Exception as e:
        raise ValueError(f"无法读取视频 {video_path}") from e

    fps = vr.get_avg_fps()
    total_frames = len(vr)
    total_seconds = int(total_frames / fps)
    frame_indices = [int(s * fps) for s in range(total_seconds)]

    if not frame_indices:
        raise ValueError("failed to get frame indices")
    try:
        frames = vr.get_batch(frame_indices).asnumpy()
    except Exception as e:
        raise ValueError("failed to get frames") from e

    frames = [cv2.resize(f, resize) for f in frames]
    result = (frames, fps, total_frames)
    if cache is not None:
        cache[video_path] = result
    return result


def send_messages(client: OpenAI, model_name: str, messages: list, max_tokens: int = 1024, seed: int = 0) -> str:
    try:
        chat_completion = client.chat.completions.create(
            messages=messages, model=model_name, max_completion_tokens=max_tokens, temperature=0, seed=seed
        )
    except Exception as e:
        raise ValueError(f"failed to get llm response: {e}") from e
    return chat_completion.choices[0].message.content
