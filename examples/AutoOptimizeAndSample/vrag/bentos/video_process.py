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
# Video processing service for frame and audio extraction.


import asyncio
from pathlib import Path
from typing import Optional, Tuple

import bentoml
from pydantic import Field

from vrag.logger import logger
from vrag.shared import ArgsBase, first_available, ConfigBase, vrag_service
from vrag.tools.audio import AudioChunkExtraction
from vrag.tools.decord import FrameExtraction
from vrag.tools.ffmpeg import probe_video
from vrag.tools.path_validator import validate_path_exists
from vrag.tools.video_extraction_cache import VideoExtractionCache


class VideoProcessArgs(ArgsBase):
    video_process_cache_dir: Optional[Path] = Field(default_factory=lambda: Path("cache_store/video_process"))
    """Directory for caching video extraction results."""
    video_process_cache_dir_lock_timeout: Optional[int] = Field(300, ge=1)
    """Timeout in seconds for acquiring file locks on the cache directory."""
    decord_workers: int = Field(0, ge=0)
    """Number of decord worker threads; 0 means auto-detect."""
    default_extract_frames: bool = True
    """Whether to extract video frames by default."""
    default_extract_audio: bool = True
    """Whether to extract audio from video by default."""
    default_max_frames_num: int = Field(720, ge=1)
    """Default maximum number of frames to sample from a video."""
    default_fps: float = Field(0.2, gt=0.0)
    """Default frames-per-second rate for sampling."""
    default_force_sample: bool = True
    """Whether to force uniform sampling to max_frames_num regardless of video FPS."""
    default_resolution: int = Field(720, ge=180)
    """Default target height in pixels for resizing extracted frames."""
    default_audio_chunk_length: int = Field(30, ge=1)
    """Default length in seconds for each audio chunk."""
    default_min_chunk_threshold: float = Field(1.0, ge=0.1)
    """Default minimum duration in seconds for the trailing audio chunk to be kept."""
    default_audio_sample_rate: int = Field(16000, ge=8000)
    """Default audio sample rate in Hz for resampling."""


class VideoProcessConfig(ConfigBase):
    extract_frames: Optional[bool] = None
    extract_audio: Optional[bool] = None
    max_frames_num: Optional[int] = None
    fps: Optional[float] = None
    force_sample: Optional[bool] = None
    resolution: Optional[int] = None
    audio_chunk_length: Optional[int] = None
    min_chunk_threshold: Optional[float] = None
    audio_sample_rate: Optional[int] = None

    @staticmethod
    def merge_config(config: Optional["VideoProcessConfig"]) -> "VideoProcessConfig":
        if config is None:
            return VideoProcessConfig(
                extract_frames=args.default_extract_frames,
                extract_audio=args.default_extract_audio,
                max_frames_num=args.default_max_frames_num,
                fps=args.default_fps,
                force_sample=args.default_force_sample,
                resolution=args.default_resolution,
                audio_chunk_length=args.default_audio_chunk_length,
                min_chunk_threshold=args.default_min_chunk_threshold,
                audio_sample_rate=args.default_audio_sample_rate,
            )
        return VideoProcessConfig(
            extract_frames=first_available(config.extract_frames, args.default_extract_frames),
            extract_audio=first_available(config.extract_audio, args.default_extract_audio),
            max_frames_num=first_available(config.max_frames_num, args.default_max_frames_num),
            fps=first_available(config.fps, args.default_fps),
            force_sample=first_available(config.force_sample, args.default_force_sample),
            resolution=first_available(config.resolution, args.default_resolution),
            audio_chunk_length=first_available(config.audio_chunk_length, args.default_audio_chunk_length),
            min_chunk_threshold=first_available(config.min_chunk_threshold, args.default_min_chunk_threshold),
            audio_sample_rate=first_available(config.audio_sample_rate, args.default_audio_sample_rate),
        )


args = bentoml.use_arguments(VideoProcessArgs).override()


@vrag_service(args)
class VideoProcessService:
    def __init__(self) -> None:
        logger.info("VideoProcessService initialized.")

    @staticmethod
    async def _extract_cached(
        video_path: Path, config: VideoProcessConfig
    ) -> Tuple[Optional[FrameExtraction], Optional[AudioChunkExtraction]]:
        v_cache = VideoExtractionCache.from_cache_key(
            video_path, args.video_process_cache_dir, args.video_process_cache_dir_lock_timeout
        )
        probe_res = await asyncio.to_thread(probe_video, video_path)

        extract_frames = config.extract_frames and probe_res.has_video
        extract_audio = config.extract_audio and probe_res.has_audio

        if extract_frames and extract_audio:
            return await v_cache.get_all(
                video_path,
                fps=config.fps,
                max_sample_num=config.max_frames_num,
                force_sample=config.force_sample,
                resolution=config.resolution,
                decord_workers=args.decord_workers,
                chunk_length=config.audio_chunk_length,
                min_chunk_threshold=config.min_chunk_threshold,
            )
        if extract_frames:
            return (
                await v_cache.get_with_frames_from_video(
                    video_path,
                    fps=config.fps,
                    max_sample_num=config.max_frames_num,
                    force_sample=config.force_sample,
                    resolution=config.resolution,
                    decord_workers=args.decord_workers,
                ),
                None,
            )
        if extract_audio:
            return (
                None,
                await v_cache.get_with_audio_from_video(
                    video_path, chunk_length=config.audio_chunk_length, min_chunk_threshold=config.min_chunk_threshold
                ),
            )
        raise NotImplementedError("VideoProcessService unable to extract video or audio.")

    @bentoml.api
    async def extract(
        self, video_path: str, config: Optional[VideoProcessConfig]
    ) -> Tuple[Optional[FrameExtraction], Optional[AudioChunkExtraction]]:
        validate_path_exists(video_path, "video file")
        merged_config = VideoProcessConfig.merge_config(config)

        if not merged_config.extract_frames and not merged_config.extract_audio:
            return None, None

        return await self._extract_cached(Path(video_path), merged_config)
