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
# Ffmpeg utilities for video and audio probe.

from pathlib import Path
from typing import Union

import ffmpeg

from vrag.logger import logger
from vrag.types import VideoProbeResult


def probe_video(video_path: Union[str, Path]) -> VideoProbeResult:
    """
    Probe a video file and return key metadata.
    """
    video_path = Path(video_path)

    msg = f"Probe video at {video_path.resolve().as_posix()}"
    logger.debug(msg)

    try:
        probe = ffmpeg.probe(video_path.resolve().as_posix())
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to probe video at {video_path.resolve().as_posix()}") from e

    video_stream = next((s for s in probe["streams"] if s["codec_type"] == "video"), None)
    audio_stream = next((s for s in probe["streams"] if s["codec_type"] == "audio"), None)

    duration = None
    if "duration" in probe["format"]:
        duration = float(probe["format"]["duration"])
    elif video_stream and "duration" in video_stream:
        duration = float(video_stream["duration"])
    elif audio_stream and "duration" in audio_stream:
        duration = float(audio_stream["duration"])

    video_codec = video_stream["codec_name"] if video_stream else None
    fps = None
    if video_stream and "r_frame_rate" in video_stream:
        try:
            num, den = map(int, video_stream["r_frame_rate"].split("/"))
            if den != 0:
                fps = num / den
        except ValueError:
            fps = None

    audio_codec = audio_stream["codec_name"] if audio_stream else None
    audio_sample_rate = int(audio_stream["sample_rate"]) if audio_stream and "sample_rate" in audio_stream else None

    return VideoProbeResult(
        duration=duration,
        video_codec=video_codec,
        audio_codec=audio_codec,
        audio_sample_rate=audio_sample_rate,
        fps=fps,
        has_video=video_stream is not None,
        has_audio=audio_stream is not None,
    )
