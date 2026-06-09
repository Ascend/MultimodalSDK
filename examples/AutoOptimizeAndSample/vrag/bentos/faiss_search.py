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
# FAISS vector search service for semantic document retrieval.


from typing import List, Optional, Tuple

import bentoml
import faiss
import numpy as np
from pydantic import Field

from vrag.bentos.qwen_embedding import QwenEmbeddingArgs, QwenEmbeddingService
from vrag.logger import logger
from vrag.shared import ConfigBase, first_available, retry_async_request, vrag_service
from vrag.tools.embedding import normalize_vectors, normalize_average_vector


class FaissSearchArgs(QwenEmbeddingArgs):
    """Faiss vector search service configuration arguments."""

    default_faiss_threshold: float = Field(0.25, ge=0.0, le=1.0)
    """Default cosine similarity threshold for FAISS document retrieval."""
    default_faiss_enable_dedup: bool = True
    """Whether to enable deduplication of FAISS search results by default."""
    default_faiss_dedup_threshold: float = Field(0.97, ge=0.0, le=1.0)
    """Default cosine similarity threshold for considering two documents as duplicates."""
    default_faiss_dedup_lap: int = Field(2, ge=1)
    """Default number of deduplication passes to apply."""
    default_faiss_sparse_search: bool = False
    """Whether to use sparse search (per-query independent search) by default instead of average vector search."""


class FaissSearchConfig(ConfigBase):
    faiss_threshold: Optional[float] = None
    faiss_enable_dedup: Optional[bool] = None
    faiss_dedup_threshold: Optional[float] = None
    faiss_dedup_lap: Optional[int] = None
    faiss_sparse_search: Optional[bool] = None

    @staticmethod
    def merge_config(config: Optional["FaissSearchConfig"]) -> "FaissSearchConfig":
        if config is None:
            return FaissSearchConfig(
                faiss_threshold=args.default_faiss_threshold,
                faiss_enable_dedup=args.default_faiss_enable_dedup,
                faiss_dedup_threshold=args.default_faiss_dedup_threshold,
                faiss_dedup_lap=args.default_faiss_dedup_lap,
                faiss_sparse_search=args.default_faiss_sparse_search,
            )
        return FaissSearchConfig(
            faiss_threshold=first_available(config.faiss_threshold, args.default_faiss_threshold),
            faiss_enable_dedup=first_available(config.faiss_enable_dedup, args.default_faiss_enable_dedup),
            faiss_dedup_threshold=first_available(config.faiss_dedup_threshold, args.default_faiss_dedup_threshold),
            faiss_dedup_lap=first_available(config.faiss_dedup_lap, args.default_faiss_dedup_lap),
            faiss_sparse_search=first_available(config.faiss_sparse_search, args.default_faiss_sparse_search),
        )


args = bentoml.use_arguments(FaissSearchArgs).override()


@vrag_service(args)
class FaissService:
    embedding = bentoml.depends(QwenEmbeddingService)

    def __init__(self):
        logger.info("FaissService initialized.")

    @staticmethod
    def _search_faiss(
        doc_embeddings: np.ndarray, query_vector: np.ndarray, threshold: float
    ) -> Tuple[List[int], List[float]]:
        """
        Faiss search and return candidate indices and their scores.

        Args:
            doc_embeddings: Normalized document with shape (N, D).
            query_vector: Normalized query vector with shape (1, D).
            threshold: Minimum cosine similarity threshold.

        Returns:
            Tuple of (indices, scores).
        """

        dim = doc_embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        embedding_float32 = doc_embeddings.astype(np.float32)

        index.add(embedding_float32)

        query_vec = query_vector.astype(np.float32).reshape(1, -1)

        lims, distances, indices = index.range_search(query_vec, threshold)

        if lims[1] <= lims[0]:
            return [], []

        start, end = int(lims[0]), int(lims[1])
        candidate_indices = indices[start:end].tolist()
        candidate_scores = distances[start:end].tolist()
        msg = f"Faiss search candidate scores: {candidate_scores}."
        logger.debug(msg)
        return candidate_indices, candidate_scores

    @staticmethod
    def _deduplicate_candidates(
        candidate_indices: List[int], doc_embedding_norm: np.ndarray, dedup_threshold: float
    ) -> List[int]:
        n = len(candidate_indices)

        if n <= 1:
            return candidate_indices

        indices_arr = np.asarray(candidate_indices, dtype=np.intp)

        prev_embs = doc_embedding_norm[indices_arr[:-1]]
        curr_embs = doc_embedding_norm[indices_arr[1:]]

        similarities = np.sum(prev_embs * curr_embs, axis=1)
        keep_mask = similarities < dedup_threshold

        kept_followers = indices_arr[1:][keep_mask]

        if kept_followers.size > 0:
            final_indices = np.concatenate(([indices_arr[0]], kept_followers))
            return final_indices.tolist()

        return [indices_arr[0]]

    @bentoml.api
    async def retrieve_document_indices(
        self, documents: List[str], queries: List[str], config: Optional[FaissSearchConfig]
    ) -> List[int]:
        return await self._retrieve_documents_indices_inner(documents, queries, config)

    def _search_and_merge_candidates(
        self, query_embeddings: np.ndarray, doc_embedding_norm: np.ndarray, threshold: float
    ) -> List[int]:
        """
        Normalize queries, search FAISS for each independently, and merge results.

        Args:
            query_embeddings: Raw query vectors with shape (N_q, D).
            doc_embedding_norm: Normalized document vectors with shape (N_q, D).
            threshold: Minimum cosine similarity required.

        Returns:
            Sorted list of unique document indices.
        """
        if query_embeddings.ndim != 2 or doc_embedding_norm.ndim != 2:
            msg = (
                f"Faiss search needs input with shape (N_q, D), but get queries in {query_embeddings.shape}, "
                f"docs in {doc_embedding_norm.shape}"
            )
            logger.error(msg)
            return []

        msg = f"Faiss search for {len(query_embeddings)} queries in {len(doc_embedding_norm)} docs."
        logger.debug(msg)

        query_embeddings_norm = normalize_vectors(query_embeddings)

        all_candidate_indices = []

        for q_vec in query_embeddings_norm:
            q_vec_reshaped = q_vec.reshape(1, -1) if q_vec.ndim == 1 else q_vec
            indices, _ = self._search_faiss(doc_embedding_norm, q_vec_reshaped, threshold)
            if indices:
                all_candidate_indices.extend(indices)

        if not all_candidate_indices:
            return []

        return sorted(set(all_candidate_indices))

    def _search_with_average_vector(
        self, query_embeddings: np.ndarray, doc_embedding_norm: np.ndarray, threshold: float
    ) -> List[int]:
        msg = f"Faiss average search for {len(query_embeddings)} queries in {len(doc_embedding_norm)} docs."
        logger.debug(msg)

        query_embeddings_avg_norm = normalize_average_vector(query_embeddings)
        if query_embeddings_avg_norm is None:
            msg = f"Faiss average search get bad normalized average vector for query: {query_embeddings}."
            logger.warning(msg)
            return []

        candidate_indices, _ = self._search_faiss(doc_embedding_norm, query_embeddings_avg_norm, threshold)
        return candidate_indices

    def _deduplicate_laps(
        self, dedup_lap: int, dedup_threshold: float, doc_embedding_norm: np.ndarray, final_indices: List[int]
    ) -> List[int]:
        original_length = len(final_indices)
        for _ in range(dedup_lap):
            reduced = self._deduplicate_candidates(final_indices, doc_embedding_norm, dedup_threshold)
            if len(reduced) == len(final_indices):
                logger.debug("No further deduplicated candidates, ending lap.")
                break
            final_indices = reduced
        msg = f"Faiss deduplicate {original_length} docs to {len(final_indices)} docs."
        logger.debug(msg)
        return final_indices

    async def _retrieve_documents_indices_inner(
        self, documents: List[str], queries: List[str], config: Optional[FaissSearchConfig]
    ) -> List[int]:
        if not documents or not queries:
            return []

        merged_config = FaissSearchConfig.merge_config(config)

        threshold = merged_config.faiss_threshold

        all_texts = queries + documents
        all_embeddings = await retry_async_request(lambda: self.embedding.embed(all_texts), "faiss_embed_text")

        if all_embeddings.size == 0:
            return []

        query_embeddings = all_embeddings[: len(queries)]
        doc_embeddings = all_embeddings[len(queries) :]

        doc_embedding_norm = normalize_vectors(doc_embeddings)

        if merged_config.faiss_sparse_search:
            final_indices = self._search_and_merge_candidates(query_embeddings, doc_embedding_norm, threshold)
        else:
            final_indices = self._search_with_average_vector(query_embeddings, doc_embedding_norm, threshold)

        if not merged_config.faiss_enable_dedup:
            return final_indices

        dedup_threshold = merged_config.faiss_dedup_threshold
        dedup_lap = merged_config.faiss_dedup_lap
        final_indices = self._deduplicate_laps(dedup_lap, dedup_threshold, doc_embedding_norm, final_indices)
        msg = f"Faiss search return {len(final_indices)} docs."
        logger.info(msg)
        return final_indices
