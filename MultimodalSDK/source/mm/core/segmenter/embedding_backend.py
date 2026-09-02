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
"""Embedding backends for video semantic segmentation.

Provides:
    - ``EmbeddingBackend``: abstract base class with ``encode_images`` /
      ``encode_text`` / ``close``.
    - ``OpenAIEmbeddingBackend``: built-in remote backend talking to an
      OpenAI-compatible embedding service (Qwen3-VL-Embedding-2B served by
      vLLM). Image requests use the vLLM Chat Embeddings wire format (one
      request per image, concurrent, order-preserving); text requests use the
      standard ``/embeddings`` endpoint, matching the retrieval evaluation
      baseline. The chat-style text protocol is kept as ``encode_text_chat``
      for protocol A/B verification.
"""

import base64
import os
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Optional, Sequence

import numpy as np

from ...comm.log import _Logger

_API_KEY_PLACEHOLDER = "EMPTY"
_IMAGE_DATA_URL_TEMPLATE = "data:image/jpeg;base64,{b64}"
_SYSTEM_INSTRUCTION = "Represent the user's input."


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    """L2-normalizes a 1-D vector; zero vectors are returned unchanged."""
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        return (vec / norm).astype(np.float32)
    return vec


def _l2_normalize_rows(mat: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization; zero rows are returned unchanged."""
    mat = np.asarray(mat, dtype=np.float32)
    if mat.ndim != 2:
        raise RuntimeError(f"embedding matrix must be 2-D, got shape {mat.shape}")
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype(np.float32)


class EmbeddingBackend(ABC):
    """Abstract embedding backend: maps images / text to L2-normalized vectors."""

    @abstractmethod
    def encode_images(self, image_paths: List[str]) -> np.ndarray:
        """Encodes a batch of frame images.

        Args:
            image_paths: JPEG frame file paths.

        Returns:
            np.ndarray of shape (N, dim), each row L2-normalized, row order
            aligned with ``image_paths``.
        """

    @abstractmethod
    def encode_text(self, text: str) -> np.ndarray:
        """Encodes a query text.

        Args:
            text: Non-empty query text.

        Returns:
            np.ndarray of shape (dim,), L2-normalized.
        """

    @abstractmethod
    def close(self) -> None:
        """Releases backend resources. Idempotent."""


class OpenAIEmbeddingBackend(EmbeddingBackend):
    """Remote embedding backend for an OpenAI-compatible service.

    Wire formats (must stay byte-identical to the validated baseline):
        - Image: vLLM Chat Embeddings API — ``POST /embeddings`` with a
          three-segment ``messages`` body (system instruction, user
          image_url + empty text, empty assistant), ``encoding_format=float``,
          ``continue_final_message=true``, ``add_special_tokens=true``.
          One request per image, ``max_concurrent`` in flight (default 32),
          order preserved.
        - Text: standard ``/embeddings`` with ``input=[text]`` (primary path of
          the retrieval evaluation baseline). The chat-style text protocol is
          available via ``encode_text_chat`` for A/B verification.
    """

    # NOTE: the validated client baseline used 32 concurrent requests; this
    # value is now configurable per deployment via ``max_concurrent``.
    def __init__(
        self,
        base_url: str,
        model_name: str = "Qwen3-VL-Embedding-2B",
        timeout: float = 60.0,
        max_concurrent: int = 32,
        api_key: Optional[str] = None,
        protocol: str = "openai",
        custom_encode_images: Optional[Callable] = None,
        custom_encode_text: Optional[Callable] = None,
    ):
        """Initializes the backend.

        Args:
            base_url: OpenAI-compatible service address, ``http(s)://ip:port/v1``.
            model_name: Served model name.
            timeout: Per-request timeout in seconds.
            max_concurrent: Max concurrent image requests (positive integer,
                default 32, matching the validated client baseline).
            api_key: API key for the remote service; when ``None`` (default)
                the validated baseline placeholder ``"EMPTY"`` is used (vLLM
                ignores the key by default). Set it explicitly when the
                service requires authentication.
            protocol: ``"openai"`` for the built-in implementation, ``"custom"``
                to inject encoding functions (e.g. an internal client wrapper).
            custom_encode_images: Required when ``protocol="custom"``:
                ``List[frame path] -> np.ndarray (N x dim)`` (raw vectors; this
                class applies L2 normalization).
            custom_encode_text: Required when ``protocol="custom"``:
                ``str -> np.ndarray (dim,)`` (raw vector; normalized here).
        """
        if not isinstance(base_url, str):
            raise TypeError("base_url must be str")
        if "\x00" in base_url:
            raise ValueError("base_url must not contain null characters")
        if not (base_url.startswith("http://") or base_url.startswith("https://")):
            raise ValueError(f"base_url must start with http:// or https://: {base_url}")
        if not base_url.endswith("/v1"):
            raise ValueError(f"base_url must end with /v1: {base_url}")
        if not isinstance(model_name, str) or not model_name:
            raise ValueError("model_name must be a non-empty string")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("timeout must be a positive number")
        if isinstance(max_concurrent, bool) or not isinstance(max_concurrent, int) or max_concurrent <= 0:
            raise ValueError("max_concurrent must be a positive integer")
        if api_key is not None and (not isinstance(api_key, str) or not api_key):
            raise ValueError("api_key must be a non-empty string when provided")
        if protocol not in ("openai", "custom"):
            raise ValueError("protocol must be 'openai' or 'custom'")
        if protocol == "custom":
            if custom_encode_images is None or custom_encode_text is None:
                raise ValueError("custom_encode_images and custom_encode_text are required when protocol='custom'")
            if not callable(custom_encode_images) or not callable(custom_encode_text):
                raise TypeError("custom_encode_images and custom_encode_text must be callable")

        self._base_url = base_url
        self._model_name = model_name
        self._timeout = float(timeout)
        self._max_concurrent = max_concurrent
        self._api_key = api_key if api_key is not None else _API_KEY_PLACEHOLDER
        self._protocol = protocol
        self._custom_encode_images = custom_encode_images
        self._custom_encode_text = custom_encode_text
        self._closed = False
        self._client = None
        self._resp_type = None

        if protocol == "openai":
            self._init_client()

        _Logger.info(
            f"OpenAIEmbeddingBackend initialized: url={base_url}, "
            f"model={model_name}, protocol={protocol}, timeout={self._timeout}s, "
            f"max_concurrent={self._max_concurrent}"
        )

    # ------------------------------------------------------------------ #
    # construction / lifecycle
    # ------------------------------------------------------------------ #
    def _init_client(self):
        """Lazily imports the openai package and builds the HTTP client."""
        try:
            from openai import OpenAI

            try:
                # canonical path (same import as the validated mx_rag baseline)
                from openai.types.create_embedding_response import CreateEmbeddingResponse
            except ImportError:
                from openai.types import CreateEmbeddingResponse  # version fallback
        except ImportError as e:
            raise ImportError(
                "the 'openai' package (>=1.30) is required by OpenAIEmbeddingBackend; "
                "install it or use protocol='custom' with injected functions"
            ) from e
        # api key is resolved at construction (placeholder by default); the
        # key value itself is intentionally never logged.
        self._client = OpenAI(base_url=self._base_url, api_key=self._api_key, timeout=self._timeout)
        self._resp_type = CreateEmbeddingResponse

    @property
    def backend_id(self) -> str:
        """Cache identity tag, e.g. ``openai:Qwen3-VL-Embedding-2B``."""
        if self._protocol == "custom":
            return f"custom:{id(self)}"
        return f"openai:{self._model_name}"

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError("OpenAIEmbeddingBackend is closed")

    def close(self) -> None:
        """Closes the HTTP client. Idempotent; safe after close."""
        if self._closed:
            return
        self._closed = True
        if self._client is not None:
            try:
                self._client.close()
            except Exception as e:  # noqa: BLE001  closing must never raise
                _Logger.warn(f"OpenAIEmbeddingBackend close warning: {e}")
            self._client = None
        _Logger.info("OpenAIEmbeddingBackend closed")

    # ------------------------------------------------------------------ #
    # image encoding (vLLM Chat Embeddings wire format)
    # ------------------------------------------------------------------ #
    def encode_images(self, image_paths: List[str]) -> np.ndarray:
        """Encodes frame JPEG files into (N, dim) L2-normalized vectors.

        One HTTP request per image with up to ``max_concurrent`` in flight;
        output row order matches the input path order.

        Fail-fast contract: if any single frame request fails, a
        ``RuntimeError`` is raised and no partial result is returned.
        Requests already submitted to the worker threads are not cancelled
        (they keep running until they finish or time out). Callers that
        need partial success (e.g. skip a bad frame) should call
        ``encode_images`` per frame and handle exceptions at the call site.
        """
        self._ensure_open()
        self._validate_image_paths(image_paths)

        if self._protocol == "custom":
            raw = self._custom_encode_images(list(image_paths))
            mat = np.asarray(raw, dtype=np.float32)
            if mat.ndim != 2 or mat.shape[0] != len(image_paths):
                raise RuntimeError(
                    f"custom_encode_images returned shape {mat.shape}, expected ({len(image_paths)}, dim)"
                )
            return _l2_normalize_rows(mat)

        with ThreadPoolExecutor(max_workers=self._max_concurrent) as ex:
            vecs = list(ex.map(self._encode_single_image, image_paths))
        return np.stack(vecs, axis=0)

    def _validate_image_paths(self, image_paths):
        if isinstance(image_paths, (str, bytes)) or not isinstance(image_paths, Sequence):
            raise TypeError("image_paths must be a list of str")
        if len(image_paths) == 0:
            raise ValueError("image_paths must not be empty")
        for p in image_paths:
            if not isinstance(p, str) or not p:
                raise TypeError("each image path must be a non-empty str")
        for p in image_paths:
            if not os.path.isfile(p):
                raise FileNotFoundError(f"frame file not found: {p}")

    def _build_image_body(self, image_path: str) -> dict:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        return {
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": _SYSTEM_INSTRUCTION}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": _IMAGE_DATA_URL_TEMPLATE.format(b64=b64)}},
                        {"type": "text", "text": ""},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": ""}]},
            ],
            "model": self._model_name,
            "encoding_format": "float",
            "continue_final_message": True,
            "add_special_tokens": True,
        }

    def _encode_single_image(self, image_path: str) -> np.ndarray:
        """Sends one image request and returns the L2-normalized vector."""
        body = self._build_image_body(image_path)
        try:
            resp = self._client.post("/embeddings", body=body, cast_to=self._resp_type)
        except Exception as e:  # noqa: BLE001  surface server/network causes
            raise RuntimeError(f"image embedding request failed ({image_path}): {e}") from e
        return self._parse_embedding_response(resp, context=image_path)

    # ------------------------------------------------------------------ #
    # text encoding
    # ------------------------------------------------------------------ #
    def encode_text(self, text: str) -> np.ndarray:
        """Encodes text via the standard ``/embeddings`` endpoint (``input=[text]``).

        This is the primary path of the retrieval evaluation baseline
        (``embed_query`` tries the standard endpoint first and only falls back
        to the chat protocol on failure).
        """
        self._ensure_open()
        self._validate_text(text)

        if self._protocol == "custom":
            raw = self._custom_encode_text(text)
            return _l2_normalize(np.asarray(raw, dtype=np.float32))

        try:
            resp = self._client.embeddings.create(model=self._model_name, input=[text])
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"text embedding request failed: {e}") from e
        return self._parse_embedding_response(resp, context="text")

    def encode_text_chat(self, text: str) -> np.ndarray:
        """Encodes text via the chat-style protocol (system + user + assistant).

        Kept for protocol A/B verification against the legacy chat-based text
        path; not used by ``KtsSegmenter`` unless verification concludes the
        chat protocol was the effective baseline path.
        """
        self._ensure_open()
        self._validate_text(text)
        if self._protocol == "custom":
            raise RuntimeError("encode_text_chat is not available with protocol='custom'")
        body = {
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": _SYSTEM_INSTRUCTION}]},
                {"role": "user", "content": [{"type": "text", "text": text}]},
                {"role": "assistant", "content": [{"type": "text", "text": ""}]},
            ],
            "model": self._model_name,
            "encoding_format": "float",
            "continue_final_message": True,
            "add_special_tokens": True,
        }
        try:
            resp = self._client.post("/embeddings", body=body, cast_to=self._resp_type)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"text embedding (chat protocol) request failed: {e}") from e
        return self._parse_embedding_response(resp, context="text-chat")

    @staticmethod
    def _validate_text(text):
        if not isinstance(text, str):
            raise TypeError("text must be str")
        if not text:
            raise ValueError("text must not be empty")

    def _parse_embedding_response(self, resp, context: str) -> np.ndarray:
        """Extracts ``resp.data[0].embedding`` as an L2-normalized vector."""
        try:
            embedding = resp.data[0].embedding
        except (AttributeError, IndexError, TypeError) as e:
            raise RuntimeError(f"invalid embedding response ({context}): {e}") from e
        if embedding is None:
            raise RuntimeError(f"invalid embedding response ({context}): empty embedding")
        vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vec.size == 0:
            raise RuntimeError(f"invalid embedding response ({context}): zero-length vector")
        return _l2_normalize(vec)
