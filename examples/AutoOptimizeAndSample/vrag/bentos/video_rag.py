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
# Video rag service for end-to-end video question answering.

import time
from pathlib import Path
from typing import Optional

import bentoml

from vrag.bentos.qwenvl import QwenVLArgs, QwenVLConfig, QwenVLService
from vrag.bentos.video_retrieval import (
    VideoRetrievalArgs,
    VideoRetrievalConfig,
    VideoRetrievalResult,
    VideoRetrievalService,
)
from vrag.logger import logger
from vrag.shared import first_available, ConfigBase, retry_async_request, vrag_service
from vrag.tools.base64 import encode_frames_async
from vrag.tools.query import Query
from vrag.tools.render import (
    generate_ocr_instruction,
    generate_asr_instruction,
    generate_detection_instruction,
    generate_final_prompt,
)


class VideoRagArgs(VideoRetrievalArgs, QwenVLArgs):  # pylint: disable=too-many-ancestors
    default_det_retrieval_frames_only: bool = True
    """Whether to only use detection-retrieved frames (no OCR/ASR instructions) in the final prompt by default."""
    default_rag_discard_empty_detection: bool = True
    """Whether to discard empty detection results when generating the final prompt by default."""
    default_return_retrieval_result: bool = False
    """Whether to include the retrieval result in the inference response by default."""


class VideoRagConfig(ConfigBase):
    retrieval: Optional[VideoRetrievalConfig] = None
    qwenvl: Optional[QwenVLConfig] = None
    det_retrieval_frames_only: Optional[bool] = None
    rag_discard_empty_detection: Optional[bool] = None
    return_retrieval_result: Optional[bool] = None

    @staticmethod
    def merge_config(config: Optional["VideoRagConfig"] = None) -> "VideoRagConfig":
        if config is None:
            return VideoRagConfig(
                retrieval=VideoRetrievalConfig.merge_config(None),
                qwenvl=QwenVLConfig.merge_config(None),
                det_retrieval_frames_only=args.default_det_retrieval_frames_only,
                rag_discard_empty_detection=args.default_rag_discard_empty_detection,
                return_retrieval_result=args.default_return_retrieval_result,
            )
        return VideoRagConfig(
            retrieval=VideoRetrievalConfig.merge_config(config.retrieval),
            qwenvl=QwenVLConfig.merge_config(config.qwenvl),
            det_retrieval_frames_only=first_available(
                config.det_retrieval_frames_only, args.default_det_retrieval_frames_only
            ),
            rag_discard_empty_detection=first_available(
                config.rag_discard_empty_detection, args.default_rag_discard_empty_detection
            ),
            return_retrieval_result=first_available(
                config.return_retrieval_result, args.default_return_retrieval_result
            ),
        )


class VideoRagInferenceResult(bentoml.IODescriptor):
    question: str = ""
    answer: str = ""
    digested_info: str = ""
    processing_time: float = 0.0
    retrieval_result: Optional[VideoRetrievalResult] = None


args = bentoml.use_arguments(VideoRagArgs).override()


@vrag_service(args)
class VideoRagService:
    retrieval: VideoRetrievalService = bentoml.depends(VideoRetrievalService)
    qwenvl: QwenVLService = bentoml.depends(QwenVLService)

    def __init__(self) -> None:
        logger.info("VideoRagService initialized.")

    @bentoml.api
    async def ask(
        self, video_path: str, question: str, config: Optional[VideoRagConfig] = None
    ) -> VideoRagInferenceResult:
        merged_config = VideoRagConfig.merge_config(config)
        return await self._ask(Path(video_path), question, merged_config)

    async def _ask(self, video_path: Path, question: str, config: VideoRagConfig) -> VideoRagInferenceResult:
        start = time.time()
        msg = (
            f"VideoRag start video RAG inference for question:\n{question}\n"
            f"on video:\n{video_path.resolve().as_posix()}"
        )
        logger.info(msg)

        query: Query = await retry_async_request(
            lambda: self.qwenvl.generate_query(question, config=config.qwenvl), "rag_generate_query"
        )

        retrieval_result: VideoRetrievalResult = await retry_async_request(
            lambda: self.retrieval.retrieve_with_related_frames(
                video_path.as_posix(), query, question, config.retrieval
            ),
            "rag_retrieval_frames",
        )

        related_frame_extraction = retrieval_result.frame_extraction

        final_prompt = generate_final_prompt(
            question=question,
            frame_extraction=related_frame_extraction,
            det_instruction=generate_detection_instruction(
                det_docs=retrieval_result.det_docs,
                targets=query.access_filtered_targets,
                discard_empty=config.rag_discard_empty_detection,
            )
            if not config.det_retrieval_frames_only
            else None,
            asr_instruction=generate_asr_instruction(retrieval_result.asr_docs)
            if not config.det_retrieval_frames_only
            else None,
            ocr_instruction=generate_ocr_instruction(retrieval_result.ocr_docs)
            if not config.det_retrieval_frames_only
            else None,
        )

        if related_frame_extraction:
            frames_b64 = await encode_frames_async(related_frame_extraction.frames_list)
        else:
            frames_b64 = None

        answer = await retry_async_request(
            lambda: self.qwenvl.generate(final_prompt, frames_b64, config.qwenvl), "rag_generate_final_answer"
        )

        processing_time = time.time() - start
        msg = f"VideoRag apply video RAG inference completed in {processing_time:.2f}s."
        logger.info(msg)

        return VideoRagInferenceResult(
            question=question,
            answer=answer,
            digested_info=final_prompt,
            processing_time=processing_time,
            retrieval_result=retrieval_result if config.return_retrieval_result else None,
        )
