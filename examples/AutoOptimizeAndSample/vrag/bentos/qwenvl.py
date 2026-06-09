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
# Qwen VL model for multimodal generation.
import asyncio
from copy import deepcopy
from typing import List, Optional

import bentoml
from openai import AsyncOpenAI, APIConnectionError, RateLimitError, APITimeoutError
from pydantic import Field

from vrag.logger import logger
from vrag.shared import ArgsBase, first_available, ConfigBase, vrag_service
from vrag.tools.query import Query
from vrag.tools.render import generate_retrieval_prompt


class QwenVLArgs(ArgsBase):
    qwenvl_api_base: str = "http://localhost:8000/v1"
    """Base URL for the QwenVL vLLM API endpoint."""
    qwenvl_api_key: str = "EMPTY"
    """API key for the QwenVL vLLM service."""
    qwenvl_model_name: str = "qwen2.5-vl-32b"
    """Model name identifier used in the vLLM API."""
    default_max_completion_tokens: int = Field(512, ge=1)
    """Default maximum number of tokens in the generated completion."""
    default_temperature: float = Field(0.0, ge=0.0, le=2.0)
    """Default sampling temperature for generation."""
    default_top_p: float = Field(1.0, ge=0.0, le=1.0)
    """Default top-p (nucleus) sampling parameter."""
    default_seed: Optional[int] = Field(114514, gt=0)
    """Default random seed for reproducible generation."""
    default_timeout: float = Field(3600, gt=0)
    """Default request timeout in seconds for the vLLM API call."""


class QwenVLConfig(ConfigBase):
    max_completion_tokens: Optional[int] = Field(None, ge=1)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    seed: Optional[int] = None
    timeout: Optional[float] = Field(None, gt=0)

    @staticmethod
    def merge_config(config: Optional["QwenVLConfig"]) -> "QwenVLConfig":
        if config is None:
            return QwenVLConfig(
                max_completion_tokens=args.default_max_completion_tokens,
                temperature=args.default_temperature,
                top_p=args.default_top_p,
                seed=args.default_seed,
                timeout=args.default_timeout,
            )

        return QwenVLConfig(
            max_completion_tokens=first_available(config.max_completion_tokens, args.default_max_completion_tokens),
            temperature=first_available(config.temperature, args.default_temperature),
            top_p=first_available(config.top_p, args.default_top_p),
            seed=first_available(config.seed, args.default_seed),
            timeout=first_available(config.timeout, args.default_timeout),
        )


args = bentoml.use_arguments(QwenVLArgs).override()


@vrag_service(args)
class QwenVLService:
    def __init__(self) -> None:
        self.client = AsyncOpenAI(api_key=args.qwenvl_api_key, base_url=args.qwenvl_api_base)

        msg = f"QwenVLService initialized connected to {args.qwenvl_api_base}."
        logger.info(msg)

    @staticmethod
    def _build_content_payload(query: str, frames_b64: Optional[List[str]]) -> List[dict]:
        return [
            *[
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_str}"}}
                for b64_str in frames_b64 or []
            ],
            {"type": "text", "text": query},
        ]

    @bentoml.api
    async def generate(
        self, query: str, frames_b64: Optional[List[str]] = None, config: Optional[QwenVLConfig] = None
    ) -> str:
        return await self._generate(query, frames_b64, config)

    @bentoml.api
    async def generate_query(
        self,
        question: str,
        query_max_tokens: int = 512,
        frames_b64: Optional[List[str]] = None,
        config: Optional[QwenVLConfig] = None,
    ) -> "Query":
        if config:
            config = deepcopy(config)
        else:
            config = QwenVLConfig()

        logger.info(f"QwenVLService set max_completion_tokens to {query_max_tokens} to generate Query.")
        config.max_completion_tokens = query_max_tokens

        raw_string = await self._generate(
            generate_retrieval_prompt(question, schema=Query.schema_string()), frames_b64=frames_b64, config=config
        )

        msg = f"Question:\n{question}\nGenerated with qwenvl search query json: {raw_string}."
        logger.debug(msg)

        return Query.from_raw_json(raw_string)

    async def _generate(
        self, query: str, frames_b64: Optional[List[str]] = None, config: Optional[QwenVLConfig] = None
    ) -> str:
        merged_config = QwenVLConfig.merge_config(config)

        content_payload = self._build_content_payload(query, frames_b64)

        try:
            response = await self.client.chat.completions.create(
                model=args.qwenvl_model_name,
                messages=[{"role": "user", "content": content_payload}],
                max_completion_tokens=merged_config.max_completion_tokens,
                temperature=merged_config.temperature,
                top_p=merged_config.top_p,
                timeout=merged_config.timeout,
                seed=merged_config.seed,
            )

            if response.choices and response.choices[0].message and response.choices[0].message.content:
                return response.choices[0].message.content.strip()
        except APITimeoutError as e:
            msg = f"QwenVL call vLLM failed with timeout: {e}"
            logger.error(msg)
            raise
        except RateLimitError as e:
            msg = f"QwenVL call vLLM failed with rate limit: {e}"
            logger.error(msg)
            await asyncio.sleep(1)
            raise
        except APIConnectionError as e:
            msg = f"QwenVL call vLLM failed with connection error: {e}"
            logger.error(msg)
            raise
        except Exception as e:
            msg = f"QwenVL call vLLM failed: {e}"
            logger.error(msg)
            raise RuntimeError("QwenVL call vLLM failed") from e

        logger.warning("QwenVL call vLLM and return nothing.")
        return ""
