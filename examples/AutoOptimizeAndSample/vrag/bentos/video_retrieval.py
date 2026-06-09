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
# Video retrieval service for document retrieval and frame selection.


import asyncio
from pathlib import Path
from typing import List, Optional, Tuple, TypeVar

import bentoml
from pydantic import Field

from vrag.bentos.detection_service import DetectionArgs, DetectionServiceConfig, DetectionService, DetectionResult
from vrag.bentos.faiss_search import FaissSearchArgs, FaissSearchConfig, FaissService
from vrag.bentos.qwen_reranker import QwenRerankerArgs, QwenRerankerService
from vrag.bentos.video_transcribe import VideoTranscribeArgs, VideoTranscribeConfig, VideoTranscribeService
from vrag.logger import logger
from vrag.shared import (
    first_available,
    downsample,
    ConfigBase,
    flatten,
    unique_everseen,
    retry_async_request,
    vrag_service,
)
from vrag.tools.decord import FrameExtraction
from vrag.tools.imagehash import ImageHasher
from vrag.tools.np_cacher import get_cacher
from vrag.tools.query import Query
from vrag.tools.selecters import (
    indexed,
    sample_k_uniformly,
    select_indices_for_subset,
    select_related_asr_docs,
    select_spans,
)
from vrag.types import OCRDoc, ASRDoc, DETDoc, ExtractionResult


class VideoRetrievalArgs(VideoTranscribeArgs, DetectionArgs, FaissSearchArgs, QwenRerankerArgs):  # pylint: disable=too-many-ancestors
    retrieval_cache_size: int = Field(4096, ge=0)
    """LRU cache capacity for video retrieval results."""
    default_ocr_discard_min_length: int = Field(5, ge=1)
    """Default minimum text length for OCR documents to be considered in retrieval."""
    default_asr_discard_min_length: int = Field(5, ge=1)
    """Default minimum text length for ASR documents to be considered in retrieval."""

    default_retrieval_enable_fallback: bool = True
    """Whether to enable fallback uniform sampling when retrieval returns no results."""
    default_retrieval_fallback_uniform_samples_k: int = Field(32, ge=1)
    """Default number of frames to uniformly sample as fallback."""

    default_retrieval_infer_always_use_frames: bool = False
    """Whether to always include frames in inference regardless of retrieval results."""
    default_retrieval_dedup_related_frames: bool = True
    """Whether to deduplicate related frames after retrieval."""
    default_retrieval_dedup_related_frames_threshold: int = Field(2, ge=0)
    """Default Hamming distance threshold for deduplicating related frames."""
    default_retrieval_dedup_related_frames_block_size: int = Field(12, ge=8)
    """Default block size for perceptual hashing in related frame deduplication."""

    default_retrieval_ocr_top_k: int = Field(3, ge=1)
    """Default top-k OCR documents to retrieve after reranking."""
    default_retrieval_ocr_retrieve_span_length: int = 0
    """Default context span length to expand around retrieved OCR documents; 0 means no expansion."""
    default_retrieval_ocr_related_frames: bool = True
    """Whether to attach related video frames to retrieved OCR documents."""
    default_retrieval_ocr_related_frames_top_k: int = Field(1, ge=1)
    """Default top-k related frames to attach per OCR document."""

    default_retrieval_always_related_asr_docs: bool = False
    """Whether to always attach ASR documents related to detection frames."""
    default_retrieval_asr_top_k: int = Field(34, ge=1)
    """Default top-k ASR documents to retrieve after reranking."""
    default_retrieval_asr_retrieve_span_length: int = 0
    """Default context span length to expand around retrieved ASR documents; 0 means no expansion."""
    default_retrieval_asr_related_frames: bool = True
    """Whether to attach related video frames to retrieved ASR documents."""
    default_retrieval_asr_related_frames_top_k: int = Field(10, ge=1)
    """Default top-k related frames to attach per ASR document."""
    default_retrieval_asr_max_related_frames: int = Field(1, ge=0)
    """Default maximum number of related frames per ASR document; 0 means no limit."""


class VideoRetrievalConfig(ConfigBase):
    transcribe: Optional[VideoTranscribeConfig] = None
    detect: Optional[DetectionServiceConfig] = None
    faiss: Optional[FaissSearchConfig] = None

    ocr_discard_min_length: Optional[int] = None
    asr_discard_min_length: Optional[int] = None

    retrieval_enable_fallback: Optional[bool] = None
    retrieval_fallback_uniform_samples_k: Optional[int] = None

    retrieval_infer_always_use_frames: Optional[bool] = None
    retrieval_dedup_related_frames: Optional[bool] = None
    retrieval_dedup_related_frames_threshold: Optional[int] = None
    retrieval_dedup_related_frames_block_size: Optional[int] = None

    retrieval_ocr_top_k: Optional[int] = None
    retrieval_ocr_retrieve_span_length: Optional[int] = None
    retrieval_ocr_related_frames: Optional[bool] = None
    retrieval_ocr_related_frames_top_k: Optional[int] = None

    retrieval_always_related_asr_docs: Optional[bool] = None
    retrieval_asr_top_k: Optional[int] = None
    retrieval_asr_retrieve_span_length: Optional[int] = None
    retrieval_asr_related_frames: Optional[bool] = None
    retrieval_asr_related_frames_top_k: Optional[int] = None
    retrieval_asr_max_related_frames: Optional[int] = None

    @staticmethod
    def merge_config(config: Optional["VideoRetrievalConfig"]) -> "VideoRetrievalConfig":
        if config is None:
            return VideoRetrievalConfig(
                transcribe=VideoTranscribeConfig.merge_config(None),
                detect=DetectionServiceConfig.merge_config(None),
                faiss=FaissSearchConfig.merge_config(None),
                ocr_discard_min_length=args.default_ocr_discard_min_length,
                asr_discard_min_length=args.default_asr_discard_min_length,
                retrieval_enable_fallback=args.default_retrieval_enable_fallback,
                retrieval_fallback_uniform_samples_k=args.default_retrieval_fallback_uniform_samples_k,
                retrieval_infer_always_use_frames=args.default_retrieval_infer_always_use_frames,
                retrieval_dedup_related_frames=args.default_retrieval_dedup_related_frames,
                retrieval_dedup_related_frames_threshold=args.default_retrieval_dedup_related_frames_threshold,
                retrieval_dedup_related_frames_block_size=args.default_retrieval_dedup_related_frames_block_size,
                retrieval_ocr_top_k=args.default_retrieval_ocr_top_k,
                retrieval_ocr_retrieve_span_length=args.default_retrieval_ocr_retrieve_span_length,
                retrieval_ocr_related_frames=args.default_retrieval_ocr_related_frames,
                retrieval_ocr_related_frames_top_k=args.default_retrieval_ocr_related_frames_top_k,
                retrieval_always_related_asr_docs=args.default_retrieval_always_related_asr_docs,
                retrieval_asr_top_k=args.default_retrieval_asr_top_k,
                retrieval_asr_retrieve_span_length=args.default_retrieval_asr_retrieve_span_length,
                retrieval_asr_related_frames=args.default_retrieval_asr_related_frames,
                retrieval_asr_related_frames_top_k=args.default_retrieval_asr_related_frames_top_k,
                retrieval_asr_max_related_frames=args.default_retrieval_asr_max_related_frames,
            )
        return VideoRetrievalConfig(
            transcribe=VideoTranscribeConfig.merge_config(config.transcribe),
            detect=DetectionServiceConfig.merge_config(config.detect),
            faiss=FaissSearchConfig.merge_config(config.faiss),
            ocr_discard_min_length=first_available(config.ocr_discard_min_length, args.default_ocr_discard_min_length),
            asr_discard_min_length=first_available(config.asr_discard_min_length, args.default_asr_discard_min_length),
            retrieval_enable_fallback=first_available(
                config.retrieval_enable_fallback, args.default_retrieval_enable_fallback
            ),
            retrieval_fallback_uniform_samples_k=first_available(
                config.retrieval_fallback_uniform_samples_k, args.default_retrieval_fallback_uniform_samples_k
            ),
            retrieval_infer_always_use_frames=first_available(
                config.retrieval_infer_always_use_frames, args.default_retrieval_infer_always_use_frames
            ),
            retrieval_dedup_related_frames=first_available(
                config.retrieval_dedup_related_frames, args.default_retrieval_dedup_related_frames
            ),
            retrieval_dedup_related_frames_threshold=first_available(
                config.retrieval_dedup_related_frames_threshold, args.default_retrieval_dedup_related_frames_threshold
            ),
            retrieval_dedup_related_frames_block_size=first_available(
                config.retrieval_dedup_related_frames_block_size, args.default_retrieval_dedup_related_frames_block_size
            ),
            retrieval_ocr_top_k=first_available(config.retrieval_ocr_top_k, args.default_retrieval_ocr_top_k),
            retrieval_ocr_retrieve_span_length=first_available(
                config.retrieval_ocr_retrieve_span_length, args.default_retrieval_ocr_retrieve_span_length
            ),
            retrieval_ocr_related_frames=first_available(
                config.retrieval_ocr_related_frames, args.default_retrieval_ocr_related_frames
            ),
            retrieval_ocr_related_frames_top_k=first_available(
                config.retrieval_ocr_related_frames_top_k, args.default_retrieval_ocr_related_frames_top_k
            ),
            retrieval_always_related_asr_docs=first_available(
                config.retrieval_always_related_asr_docs, args.default_retrieval_always_related_asr_docs
            ),
            retrieval_asr_top_k=first_available(config.retrieval_asr_top_k, args.default_retrieval_asr_top_k),
            retrieval_asr_retrieve_span_length=first_available(
                config.retrieval_asr_retrieve_span_length, args.default_retrieval_asr_retrieve_span_length
            ),
            retrieval_asr_related_frames=first_available(
                config.retrieval_asr_related_frames, args.default_retrieval_asr_related_frames
            ),
            retrieval_asr_related_frames_top_k=first_available(
                config.retrieval_asr_related_frames_top_k, args.default_retrieval_asr_related_frames_top_k
            ),
            retrieval_asr_max_related_frames=first_available(
                config.retrieval_asr_max_related_frames, args.default_retrieval_asr_max_related_frames
            ),
        )


args = bentoml.use_arguments(VideoRetrievalArgs).override()

_T = TypeVar("_T")


class VideoRetrievalResult(bentoml.IODescriptor):
    ocr_docs: List[OCRDoc] = Field(default_factory=list)
    asr_docs: List[ASRDoc] = Field(default_factory=list)
    det_docs: List[DETDoc] = Field(default_factory=list)
    frame_extraction: Optional[FrameExtraction] = Field(default=None)


@vrag_service(args)
class VideoRetrievalService:
    transcribe: VideoTranscribeService = bentoml.depends(VideoTranscribeService)
    detection: DetectionService = bentoml.depends(DetectionService)
    faiss: FaissService = bentoml.depends(FaissService)
    reranker: QwenRerankerService = bentoml.depends(QwenRerankerService)

    def __init__(self) -> None:
        logger.info("VideoRetrievalService initialized.")

        self._cacher = get_cacher(args.retrieval_cache_size)

        self._image_hasher = ImageHasher.with_cacher(self._cacher)

    @staticmethod
    def _select_span(
        result: VideoRetrievalResult,
        ocr_docs_idx: List[int],
        asr_docs_idx: List[int],
        extraction_result: ExtractionResult,
        config: VideoRetrievalConfig,
    ) -> None:
        ocr_span_length = config.retrieval_ocr_retrieve_span_length
        asr_span_length = config.retrieval_asr_retrieve_span_length

        if ocr_span_length:
            spanned_ocr_indices = select_spans(
                context_span=ocr_span_length, initial_indices=ocr_docs_idx, n_docs=len(extraction_result.ocr_docs_total)
            )
            result.ocr_docs = indexed(extraction_result.ocr_docs_total, spanned_ocr_indices)

        if asr_span_length:
            spanned_asr_indices = select_spans(
                context_span=asr_span_length, initial_indices=asr_docs_idx, n_docs=len(extraction_result.asr_docs_total)
            )
            result.asr_docs = indexed(extraction_result.asr_docs_total, spanned_asr_indices)

    @staticmethod
    def _discard_docs_with_indices(
        docs: List[Tuple[str, _T]], min_length: Optional[int], label: str = ""
    ) -> List[Tuple[int, Tuple[str, _T]]]:
        if min_length is None or min_length < 0:
            ret = list(enumerate(docs))
        else:
            ret = [(i, d) for i, d in enumerate(docs) if len(d[0]) >= min_length]

        discarded = len(docs) - len(ret)

        if discarded > 0:
            msg = f"VideoRetrieval discard {discarded} {label} docs with length < {min_length}."
            logger.debug(msg)

        return ret

    @staticmethod
    def _gather_related_frames_indices(
        ocr_docs: List[OCRDoc],
        asr_docs: List[ASRDoc],
        det_top_idx: List[int],
        frame_extraction: FrameExtraction,
        config: VideoRetrievalConfig,
    ) -> List[int]:
        ocr_related_frames_idx = (
            select_indices_for_subset(frame_extraction.frame_timestamps, [x[1] for x in ocr_docs])
            if config.retrieval_ocr_related_frames
            else []
        )
        msg = f"VideoRetrieval gather {len(ocr_related_frames_idx)} raw related frames for OCR."
        logger.debug(msg)

        asr_related_frames_idx = (
            [
                sample_k_uniformly(
                    frame_extraction.frame_timestamps, a[1][0], a[1][1], config.retrieval_asr_max_related_frames
                )
                for a in asr_docs
            ]
            if config.retrieval_asr_related_frames
            else []
        )

        msg = f"VideoRetrieval gather {len(asr_related_frames_idx)} raw related frames for ASR."
        logger.debug(msg)

        if det_top_idx:
            msg = f"VideoRetrieval combine frames with DET: {len(det_top_idx)}."
            logger.debug(msg)

        selected_frames_ids = sorted({*det_top_idx, *list(flatten(asr_related_frames_idx)), *ocr_related_frames_idx})
        msg = f"VideoRetrieval gather {len(selected_frames_ids)} frames:\n{selected_frames_ids}."
        logger.debug(msg)
        return selected_frames_ids

    @bentoml.api
    async def retrieve_with_related_frames(
        self, video_path: str, query: Query, question: str, config: Optional[VideoRetrievalConfig]
    ) -> VideoRetrievalResult:
        merged_config = VideoRetrievalConfig.merge_config(config)

        return await self._retrieve_with_related_frames(Path(video_path), query, question, merged_config)

    async def _retrieve_and_rerank_docs(
        self,
        filtered_docs_packs: List[Tuple[int, Tuple[str, _T]]],
        all_docs: List[Tuple[str, _T]],
        queries: List[str],
        question: str,
        config: VideoRetrievalConfig,
        retrieval_top_k: int,
    ) -> List[int]:
        candidate_local_indices: List[int] = await retry_async_request(
            lambda: self.faiss.retrieve_document_indices(
                documents=[doc[0] for _, doc in filtered_docs_packs], queries=[*queries, question], config=config.faiss
            ),
            "retrieval_faiss_retrieval_document",
        )

        if candidate_local_indices:
            candidate_global_indices = [filtered_docs_packs[i][0] for i in candidate_local_indices]
            raw_docs = indexed(all_docs, candidate_global_indices)
            rerank_local_indices = await retry_async_request(
                lambda: self.reranker.rerank(query=question, documents=[d[0] for d in raw_docs], top_k=retrieval_top_k),
                "retrieval_rerank_documents",
            )
            docs_idx = indexed(candidate_global_indices, rerank_local_indices)
            return docs_idx

        return []

    async def _retrieve_ocr_docs(
        self, extraction_result: ExtractionResult, query: Query, question: str, config: VideoRetrievalConfig
    ) -> List[int]:
        filtered_ocr_docs_packs = self._discard_docs_with_indices(
            extraction_result.ocr_docs_total, config.ocr_discard_min_length, "OCR"
        )

        ocr_docs_idx: List[int] = []

        if filtered_ocr_docs_packs and query.access_related_subtitles:
            ocr_docs_idx = await self._retrieve_and_rerank_docs(
                filtered_docs_packs=filtered_ocr_docs_packs,
                all_docs=extraction_result.ocr_docs_total,
                queries=query.access_related_subtitles,
                question=question,
                config=config,
                retrieval_top_k=config.retrieval_ocr_top_k,
            )

        return ocr_docs_idx

    async def _retrieve_asr_docs(
        self, extraction_result: ExtractionResult, query: Query, question: str, config: VideoRetrievalConfig
    ) -> List[int]:
        filtered_asr_docs_packs = self._discard_docs_with_indices(
            extraction_result.asr_docs_total, config.asr_discard_min_length, "ASR"
        )

        asr_docs_idx: List[int] = []
        if filtered_asr_docs_packs and query.access_related_subtitles:
            asr_docs_idx = await self._retrieve_and_rerank_docs(
                filtered_docs_packs=filtered_asr_docs_packs,
                all_docs=extraction_result.asr_docs_total,
                queries=query.access_related_subtitles,
                question=question,
                config=config,
                retrieval_top_k=config.retrieval_asr_top_k,
            )
            msg = f"VideoRetrieval filtered subtitles: [{len(asr_docs_idx)}/{len(filtered_asr_docs_packs)}]."
            logger.debug(msg)
        elif query.retrieve_all_subtitles:
            asr_docs_idx = [i for i, _ in filtered_asr_docs_packs]
            msg = f"VideoRetrieval all subtitles: {len(asr_docs_idx)}."
            logger.debug(msg)
        else:
            logger.debug("VideoRetrieval not retrieve any subtitles.")
        return asr_docs_idx

    async def _retrieve_documents_with_index_mapping(
        self,
        result: VideoRetrievalResult,
        extraction_result: ExtractionResult,
        query: Query,
        question: str,
        config: VideoRetrievalConfig,
        detection_result: Optional[DetectionResult],
    ) -> Tuple[List[int], List[int]]:
        rerank_ocr_docs_idx, rerank_asr_docs_idx = await asyncio.gather(
            self._retrieve_ocr_docs(extraction_result, query, question, config),
            self._retrieve_asr_docs(extraction_result, query, question, config),
        )

        rel_asr_docs_idx = []
        if (
            not query.retrieve_all_subtitles
            and any((query.retrieve_related_docs_for_det_frames, config.retrieval_always_related_asr_docs))
            and detection_result
            and detection_result.det_docs
        ):
            asr_total = extraction_result.asr_docs_total
            iter_seq = [select_related_asr_docs(asr_total, det_time) for _, det_time in detection_result.det_docs]
            rel_asr_docs_idx = list(unique_everseen(filter(bool, iter_seq)))
            msg = f"VideoRetrieval gathered {len(rel_asr_docs_idx)} unique docs for frames."
            logger.debug(msg)

        result.ocr_docs = sorted(
            indexed(extraction_result.ocr_docs_total, rerank_ocr_docs_idx), key=lambda ocr_doc: ocr_doc[1]
        )
        combined_asr_indices = list(unique_everseen(rerank_asr_docs_idx + rel_asr_docs_idx))
        result.asr_docs = sorted(
            indexed(extraction_result.asr_docs_total, combined_asr_indices), key=lambda asr_doc: asr_doc[1][0]
        )

        if detection_result:
            result.det_docs = detection_result.det_docs

        return rerank_ocr_docs_idx, combined_asr_indices

    async def _create_related_frame_extraction(
        self,
        config: VideoRetrievalConfig,
        ocr_docs: List[OCRDoc],
        asr_docs: List[ASRDoc],
        det_top_idx: List[int],
        frame_extraction: FrameExtraction,
    ) -> FrameExtraction:
        gather_frames_idx = self._gather_related_frames_indices(
            ocr_docs, asr_docs, det_top_idx, frame_extraction, config
        )
        fi = frame_extraction.slice(gather_frames_idx)
        if config.retrieval_dedup_related_frames:
            msg = f"VideoRetrieval deduplicate related frames:\n{fi.frame_timestamps}"
            logger.debug(msg)
            dedup_indices = await self._image_hasher.get_unique_frame_indices_async(
                fi.frames,
                config.retrieval_dedup_related_frames_threshold,
                config.retrieval_dedup_related_frames_block_size,
            )
            fi = fi.slice(dedup_indices)
        msg = f"VideoRetrieval final resolved related frames timestamps:\n{fi.frame_timestamps}."
        logger.debug(msg)
        return fi

    async def _select_related_frames(
        self,
        extraction_result: ExtractionResult,
        query: Query,
        detection_result: DetectionResult,
        config: VideoRetrievalConfig,
        ocr_top_idx: List[int],
        asr_top_idx: List[int],
    ) -> Optional[FrameExtraction]:
        if (
            extraction_result.frame_extraction
            and config.retrieval_enable_fallback
            and not any((ocr_top_idx, asr_top_idx, detection_result.det_docs))
        ):
            logger.debug("VideoRetrieval fallback uniform sample enabled.")
            unik = extraction_result.frame_extraction.uniform_k(config.retrieval_fallback_uniform_samples_k)

            if config.retrieval_dedup_related_frames:
                dedup_indices = await self._image_hasher.get_unique_frame_indices_async(
                    unik.frames,
                    config.retrieval_dedup_related_frames_threshold,
                    config.retrieval_dedup_related_frames_block_size,
                )
                related_frame_extraction = unik.slice(dedup_indices)
            else:
                related_frame_extraction = unik
        else:
            need_rel_frame = query.retrieve_related_frames or config.retrieval_infer_always_use_frames

            if extraction_result.frame_extraction is None:
                related_frame_extraction = None
            elif need_rel_frame:
                if query.retrieve_all_subtitles:
                    ocr_docs_idx = downsample(ocr_top_idx, config.retrieval_ocr_related_frames_top_k)
                    asr_docs_idx = downsample(asr_top_idx, config.retrieval_asr_related_frames_top_k)
                else:
                    ocr_docs_idx = ocr_top_idx[: config.retrieval_ocr_related_frames_top_k]
                    asr_docs_idx = asr_top_idx[: config.retrieval_asr_related_frames_top_k]
                ocr_docs = indexed(extraction_result.ocr_docs_total, ocr_docs_idx)
                asr_docs = indexed(extraction_result.asr_docs_total, asr_docs_idx)

                related_frame_extraction = await self._create_related_frame_extraction(
                    config, ocr_docs, asr_docs, detection_result.det_top_idx, extraction_result.frame_extraction
                )
            else:
                related_frame_extraction = await self._create_related_frame_extraction(
                    config, [], [], detection_result.det_top_idx, extraction_result.frame_extraction
                )

        return related_frame_extraction

    async def _retrieve_from_extraction(
        self, extraction_result: ExtractionResult, query: Query, question: str, config: VideoRetrievalConfig
    ) -> VideoRetrievalResult:
        logger.info("VideoRetrieval light weight retrieval from pre-extracted data.")

        if not extraction_result.frame_extraction:
            logger.warning("VideoRetrieval get no frames extracted from video.")
            return VideoRetrievalResult()

        detection_result: DetectionResult = await retry_async_request(
            lambda: self.detection.detect(
                query=query, frame_extraction=extraction_result.frame_extraction, config=config.detect
            ),
            "retrieval_detect_scenes",
        )

        result = VideoRetrievalResult()
        (ocr_docs_idx, asr_docs_idx) = await self._retrieve_documents_with_index_mapping(
            result=result,
            extraction_result=extraction_result,
            query=query,
            question=question,
            config=config,
            detection_result=detection_result,
        )
        msg = (
            f"VideoRetrieval retrieved {len(result.ocr_docs)} OCR docs, "
            f"{len(result.asr_docs)} ASR docs, "
            f"{len(result.det_docs)} DET docs."
        )
        logger.info(msg)

        related_frame_extraction = await self._select_related_frames(
            extraction_result, query, detection_result, config, ocr_docs_idx, asr_docs_idx
        )

        self._select_span(result, ocr_docs_idx, asr_docs_idx, extraction_result, config)

        result.frame_extraction = (
            related_frame_extraction
            if (related_frame_extraction and related_frame_extraction.frame_timestamps)
            else None
        )

        return result

    async def _retrieve_with_related_frames(
        self, video_path: Path, query: Query, question: str, config: VideoRetrievalConfig
    ) -> VideoRetrievalResult:
        logger.info("VideoRetrieval full pipeline retrieve the related frames.")

        extraction_result = await retry_async_request(
            lambda: self.transcribe.extract_all(video_path.as_posix(), config=config.transcribe),
            "retrieval_transcribe_video",
        )

        return await self._retrieve_from_extraction(extraction_result, query, question, config)
