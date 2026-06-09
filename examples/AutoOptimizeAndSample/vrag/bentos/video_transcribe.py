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
# Video transcribe service for extraction OCR, ASR, Frames from video.

import asyncio
from pathlib import Path
from typing import List, Optional

import bentoml
from pydantic import Field

from vrag.bentos.mineru_ocr import MineruArgs, MineruService
from vrag.bentos.video_process import VideoProcessArgs, VideoProcessConfig, VideoProcessService
from vrag.bentos.whisper import WhisperArgs, WhisperService
from vrag.logger import logger
from vrag.shared import first_available, ConfigBase, retry_async_request, vrag_service
from vrag.tools.decord import FrameExtraction
from vrag.tools.audio import AudioChunkExtraction
from vrag.tools.imagehash import ImageHasher
from vrag.tools.np_cacher import get_cacher
from vrag.types import ASRDoc, OCRDoc, ExtractionResult


class VideoTranscribeArgs(VideoProcessArgs, WhisperArgs, MineruArgs):
    video_transcribe_cache_size: int = Field(4096, ge=0)
    """LRU cache capacity for video transcription results."""
    default_use_ocr: bool = False
    """Whether to enable OCR extraction by default."""
    default_use_asr: bool = False
    """Whether to enable ASR (speech recognition) extraction by default."""
    default_ocr_dedup: bool = True
    """Whether to deduplicate frames before OCR by default."""
    default_ocr_dedup_threshold: int = Field(2, ge=0)
    """Default Hamming distance threshold for frame deduplication before OCR."""
    default_ocr_dedup_block_size: int = Field(12, ge=8)
    """Default block size for perceptual hashing in OCR frame deduplication."""


class VideoTranscribeConfig(ConfigBase):
    video_process: Optional[VideoProcessConfig] = None
    use_ocr: Optional[bool] = None
    use_asr: Optional[bool] = None
    ocr_dedup: Optional[bool] = None
    ocr_dedup_threshold: Optional[int] = None
    ocr_dedup_block_size: Optional[int] = None

    @staticmethod
    def merge_config(config: Optional["VideoTranscribeConfig"]) -> "VideoTranscribeConfig":
        if config is None:
            return VideoTranscribeConfig(
                video_process=VideoProcessConfig.merge_config(None),
                use_ocr=args.default_use_ocr,
                use_asr=args.default_use_asr,
                ocr_dedup=args.default_ocr_dedup,
                ocr_dedup_threshold=args.default_ocr_dedup_threshold,
                ocr_dedup_block_size=args.default_ocr_dedup_block_size,
            )

        return VideoTranscribeConfig(
            video_process=VideoProcessConfig.merge_config(config.video_process),
            use_ocr=first_available(config.use_ocr, args.default_use_ocr),
            use_asr=first_available(config.use_asr, args.default_use_asr),
            ocr_dedup=first_available(config.ocr_dedup, args.default_ocr_dedup),
            ocr_dedup_threshold=first_available(config.ocr_dedup_threshold, args.default_ocr_dedup_threshold),
            ocr_dedup_block_size=first_available(config.ocr_dedup_block_size, args.default_ocr_dedup_block_size),
        )


args = bentoml.use_arguments(VideoTranscribeArgs).override()


@vrag_service(args)
class VideoTranscribeService:
    video_process: VideoProcessService = bentoml.depends(VideoProcessService)
    asr: WhisperService = bentoml.depends(WhisperService)
    ocr: MineruService = bentoml.depends(MineruService)

    def __init__(self) -> None:
        logger.info("VideoTranscribeService initialized.")

        self._cacher = get_cacher(args.video_transcribe_cache_size)
        self._image_hasher = ImageHasher.with_cacher(self._cacher)

    @bentoml.api
    async def extract_all(self, video_path: str, config: Optional[VideoTranscribeConfig]) -> ExtractionResult:
        merged_config = VideoTranscribeConfig.merge_config(config)
        return await self._extract_all(Path(video_path), merged_config)

    async def _process_ocr(self, frame_extraction: FrameExtraction, config: VideoTranscribeConfig) -> List[OCRDoc]:
        ocr_frames_list = frame_extraction.frames_list
        ocr_timestamps = frame_extraction.frame_timestamps

        ocr_dedup = config.ocr_dedup
        ocr_dedup_threshold = config.ocr_dedup_threshold
        ocr_dedup_block_size = config.ocr_dedup_block_size

        if ocr_dedup and ocr_dedup_threshold > 0 and len(ocr_frames_list) > 1:
            dedup_indices = await self._image_hasher.get_unique_frame_indices_async(
                frame_extraction.frames,
                ocr_dedup_threshold,
                ocr_dedup_block_size,
            )
            dedup_frames = frame_extraction.slice(dedup_indices)

            msg = f"VideoTranscribe dedup OCR: {len(ocr_frames_list)} -> {len(dedup_frames.frames_list)} frames."
            logger.info(msg)

            ocr_frames_list = dedup_frames.frames_list
            ocr_timestamps = dedup_frames.frame_timestamps

        ocr_texts = await retry_async_request(lambda: self.ocr.extract_text(ocr_frames_list), "video_transcribe_ocr")
        ocr_res = list(zip(ocr_texts, ocr_timestamps, strict=True))
        msg = f"VideoTranscribe generate {len(ocr_res)} ocr documents."
        logger.info(msg)
        return ocr_res

    async def _process_asr(self, audio_extraction: AudioChunkExtraction) -> List[ASRDoc]:
        asr_texts = await retry_async_request(
            lambda: self.asr.transcribe(audio_extraction.audio_chunks), "video_transcribe_asr"
        )
        asr_res = list(zip(asr_texts, audio_extraction.durations, strict=True))
        msg = f"VideoTranscribe generate {len(asr_res)} asr documents."
        logger.info(msg)
        return asr_res

    async def _extract_all(self, video_path: Path, config: VideoTranscribeConfig) -> ExtractionResult:
        logger.info("VideoTranscribe extracting query-agnostic data from video.")

        frame_extraction, audio_extraction = await retry_async_request(
            lambda: self.video_process.extract(video_path.as_posix(), config=config.video_process),
            "video_transcribe_process",
        )

        if not frame_extraction:
            logger.warning("No frames extracted from video.")
            return ExtractionResult()

        msg = f"VideoTranscribe get {len(frame_extraction.frames)} frames."
        logger.info(msg)

        use_ocr = config.use_ocr
        use_asr = config.use_asr

        async def _do_nothing():
            return []

        tasks = [
            self._process_ocr(frame_extraction, config) if use_ocr else _do_nothing(),
            self._process_asr(audio_extraction) if use_asr else _do_nothing(),
        ]

        results = await asyncio.gather(*tasks)

        return ExtractionResult(
            frame_extraction=frame_extraction,
            audio_extraction=audio_extraction,
            ocr_docs_total=results[0],
            asr_docs_total=results[1],
        )
