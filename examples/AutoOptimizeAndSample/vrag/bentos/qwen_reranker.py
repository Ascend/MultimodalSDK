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
# Qwen reranker service for computing text reranking.

from typing import List, Optional, Tuple

import bentoml
import torch
from pydantic import Field
from transformers import AutoTokenizer, AutoModelForCausalLM

from vrag.logger import logger
from vrag.shared import ArgsBase, vrag_service
from vrag.tools.path_validator import validate_dir_exists

DEFAULT_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query."

PROMPT_PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
    "Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n"
    "<|im_start|>user\n"
)

PROMPT_SUFFIX = "<|im_end|>\n<|im_start|>assistant"


class QwenRerankerArgs(ArgsBase):
    """Qwen reranker configuration."""

    reranker_model_path: str = ""
    """Local path to the Qwen reranker model directory."""
    reranker_device: str = "npu:3"
    """Device for Qwen reranker model inference, e.g. 'npu:3' or 'cpu'."""
    reranker_batch_size: int = Field(4, ge=1)
    """Batch size for reranking inference."""
    default_max_length: int = Field(8192, gt=0)
    """Maximum token length for reranker input sequences."""
    default_top_k: int = Field(5, ge=1)
    """Default number of top documents to return after reranking."""


args = bentoml.use_arguments(QwenRerankerArgs).override()


@vrag_service(args)
class QwenRerankerService:
    def __init__(self):
        self.model_path = validate_dir_exists(args.reranker_model_path, "Qwen reranker model")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, padding_side="left", local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, device_map=args.reranker_device, local_files_only=True
        )
        self.model.eval()

        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")

        self.prefix_tokens = self.tokenizer.encode(PROMPT_PREFIX, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(PROMPT_SUFFIX, add_special_tokens=False)

        msg = f"QwenRerankerService initialized from {self.model_path}."
        logger.info(msg)

    @staticmethod
    def _format_instruction_pairs(
        query: str, docs: List[str], instruction: str = DEFAULT_INSTRUCTION
    ) -> List[Tuple[str, str]]:
        return [(query, f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}") for doc in docs]

    @bentoml.api
    async def rerank(self, query: str, documents: List[str], top_k: Optional[int] = None) -> List[int]:
        msg = f"Rerank process indices of {len(documents)} docs."
        logger.info(msg)

        docs_num = len(documents)
        top_k = top_k if top_k is not None else args.default_top_k

        if docs_num <= top_k:
            return list(range(docs_num))

        all_scores = self._rerank_scores_inner(query, documents)

        indices = list(range(docs_num))
        scored_pairs = sorted(zip(all_scores, indices, strict=True), key=lambda x: x[0], reverse=True)

        return [idx for score, idx in scored_pairs[:top_k]]

    def _process_input(self, prompt_pairs: List[Tuple[str, str]]) -> dict[str, torch.Tensor]:
        max_content_length = args.default_max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        if max_content_length < 0:
            raise ValueError("QwenRerankerService should set a bigger default_max_length")

        inputs = self.tokenizer(
            prompt_pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=max_content_length,
        )

        for i, element in enumerate(inputs["input_ids"]):
            inputs["input_ids"][i] = self.prefix_tokens + element + self.suffix_tokens

        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=args.default_max_length)

        return {k: v.to(self.model.device) for k, v in inputs.items()}

    def _compute_logits(self, inputs: dict[str, torch.Tensor]) -> List[float]:
        with torch.no_grad():
            batch_scores = self.model(**inputs).logits[:, -1, :]
            true_vector = batch_scores[:, self.token_true_id]
            false_vector = batch_scores[:, self.token_false_id]
            batch_scores = torch.stack([false_vector, true_vector], dim=1)
            batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
            return batch_scores[:, 1].exp().tolist()

    def _process_batch(self, query: str, batch_docs: List[str]) -> List[float]:
        prompt_pairs = self._format_instruction_pairs(query, batch_docs)
        inputs = self._process_input(prompt_pairs)
        return self._compute_logits(inputs)

    def _rerank_scores_inner(self, query: str, documents: List[str]) -> List[float]:
        docs_num = len(documents)

        all_score = []

        for i in range(0, docs_num, args.reranker_batch_size):
            batch_docs = documents[i : i + args.reranker_batch_size]
            all_score.extend(self._process_batch(query, batch_docs))

        return all_score
