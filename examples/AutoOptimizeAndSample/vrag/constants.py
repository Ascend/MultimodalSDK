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
# Constants and templates used through the VRAG.

import asyncio
from typing import Set, Tuple, Type

import aiohttp
from jinja2 import Template

AUDIO_SAMPLE_RATE = 16_000

DET_RETRIEVE_PROMPT_TEMPLATE = Template("""
{% if det_docs %}
Below is the detected objects specified in {{ targets }}:
{% for info, timestamp in det_docs %}
- [{{ timestamp }}]: {{ info }}
{%- endfor %}
{% endif %}
""")

ASR_RETRIEVE_PROMPT_TEMPLATE = Template("""
{% if asr_docs %}
Video Automatic Speech Recognition information:
{% for text, (start_timestamp, end_timestamp) in asr_docs %}
- [{{ start_timestamp }} - {{ end_timestamp }}]: {{ text }}
{%- endfor %}
{% endif %}
""")

OCR_RETRIEVE_PROMPT_TEMPLATE = Template("""
{% if ocr_docs %}
Video OCR information:
{% for text, frame_time in ocr_docs %}
- [{{ frame_time }}]: {{ text }}
{%- endfor %}
{% endif %}
""")

FINAL_PROMPT_TEMPLATE = Template("""
{% if frame_timestamps %}
Above are frames sampled from the video, their timestamp in the video are:
{% for frame_time in frame_timestamps %}
- [{{ frame_time }}]
{%- endfor %}
{% endif %}
{% if det_instruction %}
{{ det_instruction }}
{% endif %}
{% if asr_instruction %}
{{ asr_instruction }}
{% endif %}
{% if ocr_instruction %}
{{ ocr_instruction }}
{% endif %}
Video playback duration: {{ video_duration }}
Video Frame height: {{ frame_height }} | Video Frame width: {{ frame_width }}
--- Start of the Question ---
{{ question }}
--- End of the Question ---
According to the provided relative info and data extracted from the video, give the extremely brief answer directly!
No parentheses, no source attribution, no explanations.
Return bare answer only.
{% if additional_instruction %}
{{ additional_instruction }}
{% endif %}
{{ lang_msg }}
""")

RETRIEVE_PROMPT_TEMPLATE = Template("""
```jsonschema
{{ schema }}
```
I know the answer to the question below, but I have to search the database to verify my answer, You need to write the query for me. I have provided the jsonschema of the Query object for you.
--- Start of the Question ---
{{ question }}
--- End of the Question ---
You SHALL NOT output the jsonschema or examples to me AND SHALL NOT give any explanation, ONLY output the required query in formated JSON to me.
""")

NOTHING: str = "<NO TARGET(S) DETECTED>"

HEVC: Set[str] = {"hevc", "h265", "hvc1", "hev1"}

AVC: Set[str] = {"h264", "avc1", "avc3"}

DEFAULT_LANG = "en"

SUPPORT_VIDEO_EXTENSIONS = [".mp4", ".mkv", ".avi"]

DEFAULT_RETRY_EXCEPTIONS: Tuple[Type[Exception], ...] = (
    aiohttp.ClientOSError,
    aiohttp.ServerDisconnectedError,
    ConnectionResetError,
    BrokenPipeError,
    asyncio.TimeoutError,
)
