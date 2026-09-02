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
"""公共工具模块：图像编码、VLM 消息构建与调用、输入校验等。"""

import base64
import json
import re
from io import BytesIO
from typing import List


def imgs_to_base64_list(imgs: List, img_format: str = "JPEG") -> List[str]:
    """将 PIL 图像列表编码为 base64 字符串列表。"""
    base64_list = []
    for img in imgs:
        img_rgb = img.convert('RGB') if img.mode != 'RGB' else img
        with BytesIO() as buffer:
            img_rgb.save(buffer, format=img_format)
            base64_list.append(base64.b64encode(buffer.getvalue()).decode("utf-8"))
    return base64_list


def create_messages(prompt: str, base64_images: List[str], mime_type: str = "image/jpeg") -> dict:
    """构建带图像的 OpenAI 兼容 user 消息。

    :param mime_type: 图像 MIME 类型，需与 imgs_to_base64_list 的 img_format 参数对应，
                      如 img_format="PNG" 时传 "image/png"，默认 "image/jpeg"。
    """
    content = [{"type": "text", "text": prompt}]
    content.extend(
        [{"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{img}"}} for img in base64_images]
    )
    return {"role": "user", "content": content}


def send_messages(client, model_name: str, messages: list, max_tokens: int = 1024, seed: int = 0) -> str:
    """调用 OpenAI 兼容推理服务并返回文本回答。"""
    try:
        chat_completion = client.chat.completions.create(
            messages=messages, model=model_name, max_completion_tokens=max_tokens, temperature=0, seed=seed
        )
    except Exception as e:
        raise ValueError(f"failed to get llm response: {e}") from e
    if (
        not chat_completion.choices
        or not chat_completion.choices[0].message
        or not chat_completion.choices[0].message.content
    ):
        raise ValueError(f"empty response from model: {chat_completion.model_dump_json()[:200]}")
    return chat_completion.choices[0].message.content


def extract_json_block(text: str):
    """从 VLM 回答中提取第一个 JSON 对象/数组，容忍 markdown 代码块包裹。

    使用 json.JSONDecoder().raw_decode 逐位置扫描，正确处理嵌套 JSON
    及 evidence 文本中包含方括号/花括号的场景。
    """
    if not text:
        raise ValueError("empty response text")
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    candidate = fence.group(1).strip() if fence else text.strip()
    decoder = json.JSONDecoder()
    for i, ch in enumerate(candidate):
        if ch in ('{', '['):
            try:
                obj, _ = decoder.raw_decode(candidate, i)
                return obj
            except json.JSONDecodeError:
                continue
    raise ValueError(f"no json found in response: {text[:200]}")


def format_ts(seconds: float) -> str:
    """秒 -> mm:ss 字符串。"""
    seconds = max(0, int(round(seconds)))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"
