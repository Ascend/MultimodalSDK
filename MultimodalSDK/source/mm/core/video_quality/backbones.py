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

"""Shared model backbones — process-level model pool, load-once-reuse-forever.

Lifecycle design (per the spec §3.3.2 "权重一次加载、常驻复用"):

    1.  The first scorer calls ``get_shared_clip_backbone()`` → the backbone is
        lazily loaded and stored in the **process-level** ``_cache``.
    2.  Later scorers / Pipeline instances with the same configuration get the
        cached instance — **no re-loading**.
    3.  ``scorer.close()`` and ``Pipeline.close()`` drop their own references
        but do **NOT** evict from the cache.
    4.  ``clear_cache()`` (or process exit, via ``atexit``) is the only
        mechanism that frees model memory.
"""

from __future__ import annotations

import atexit
import gc
import threading
from contextlib import suppress
from typing import Any

from .device import DeviceResolver


_cache: dict[str, Any] = {}
_lock = threading.Lock()
_atexit_registered = False


def _cache_key(
    backbone_type: str,
    model_path: str,
    device: str,
    dtype: str = "fp16",
    batch_size: int | None = None,
) -> str:
    """Build the cache key; all behaviour-changing options are part of it."""
    return "|".join([backbone_type, model_path, device, dtype, str(batch_size)])


def _empty_device_cache(device: str) -> None:
    """Return the framework's cached device memory to the system (best effort)."""
    # Best effort: unavailable framework / unsupported device must not raise.
    with suppress(Exception):
        import torch

        if device.startswith("npu"):
            torch.npu.empty_cache()
        elif device.startswith("cuda"):
            torch.cuda.empty_cache()


def _atexit_cleanup() -> None:
    """Safety net: release all model memory at process exit."""
    clear_cache()


def _ensure_atexit() -> None:
    """Register the atexit handler once (idempotent).  Caller must hold ``_lock``."""
    global _atexit_registered
    if not _atexit_registered:
        atexit.register(_atexit_cleanup)
        _atexit_registered = True


def clear_cache() -> None:
    """Explicitly release all cached backbone instances and free model memory.

    Call this when you are done with ALL videos and want to return NPU/CPU
    memory to the system; the next invocation re-loads from disk.
    """
    with _lock:
        instances = list(_cache.values())
        _cache.clear()

    # ``close()`` runs outside the lock — it may block on device teardown.
    for instance in instances:
        close = getattr(instance, "close", None)
        if callable(close):
            with suppress(Exception):
                close()
    gc.collect()
    for device in {getattr(inst, "_device", "") for inst in instances}:
        _empty_device_cache(device)


def get_cache_info() -> dict[str, dict[str, Any]]:
    """Return cache status for diagnostics.

    The shared dict is read through a snapshot taken under ``_lock``, so a
    concurrent ``clear_cache()`` cannot mutate it mid-iteration.

    Returns:
        Dict mapping cache keys to ``{"backbone_type": ..., "model_path": ...,
        "device": ..., "dtype": ..., "batch_size": ..., "loaded": bool}``.
    """
    with _lock:
        snapshot = list(_cache.items())

    info = {}
    for key, instance in snapshot:
        parts = key.split("|")
        info[key] = {
            "backbone_type": parts[0] if len(parts) > 0 else "",
            "model_path": parts[1] if len(parts) > 1 else "",
            "device": parts[2] if len(parts) > 2 else "",
            "dtype": parts[3] if len(parts) > 3 else "",
            "batch_size": parts[4] if len(parts) > 4 else "",
            "loaded": getattr(instance, "_model", None) is not None,
        }
    return info


class SharedClipBackbone:
    """Lazily-loaded CLIP model shared across the consistency family.

    Loaded on the first ``get_image_features`` / ``get_text_features`` call and
    kept in the process-level cache for all subsequent calls.

    Args:
        model_path: Path to the CLIP model directory.
        device: Resolved device string (``"npu"`` or ``"cpu"``).
        dtype: ``"fp16"`` or ``"fp32"``.
        batch_size: Inference batch size; ``None`` → the device default from
            ``DeviceResolver.default_frame_batch_size``.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        dtype: str = "fp16",
        batch_size: int | None = None,
    ):
        self._model_path = model_path
        self._device = device
        self._dtype = dtype
        self._batch_size = (
            DeviceResolver.default_frame_batch_size(device) if batch_size is None else max(1, int(batch_size))
        )
        self._model = None
        self._processor = None
        self._torch_device = None

    def _load(self) -> None:
        """Load the CLIP model and processor (called on first use).

        Weights are read from ``model.safetensors`` only: safetensors cannot
        execute code while loading, so pickle deserialisation (CWE-502) is
        ruled out.  A checkpoint dir that only ships ``pytorch_model.bin`` is
        rejected instead of being loaded.
        """
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self._torch_device = torch.device(self._device)
        use_fp16 = self._dtype == "fp16" and self._device.startswith(("npu", "cuda"))
        torch_dtype = torch.float16 if use_fp16 else torch.float32

        try:
            model = CLIPModel.from_pretrained(
                self._model_path,
                torch_dtype=torch_dtype,
                use_safetensors=True,
                local_files_only=True,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load {self._model_path} as safetensors: {exc}. Only safetensors "
                "weights are accepted; pickle-based pytorch_model.bin is rejected."
            ) from exc
        self._model = model.to(self._torch_device).eval()
        self._processor = CLIPProcessor.from_pretrained(self._model_path, local_files_only=True)

    @property
    def model(self) -> Any:
        if self._model is None:
            self._load()
        return self._model

    @property
    def processor(self) -> Any:
        if self._processor is None:
            self._load()
        return self._processor

    def get_image_features(self, frames_bgr: list) -> Any:
        """Extract L2-normalised image features from BGR frames (HxWx3 uint8) → [N, D]."""
        import cv2 as cv
        import torch
        import torch.nn.functional as F
        from PIL import Image

        batch_size = self._batch_size
        all_feats = []
        # Frames are converted to PIL one batch at a time: materialising every
        # frame up front would double peak memory for long videos.
        for i in range(0, len(frames_bgr), batch_size):
            chunk = [Image.fromarray(cv.cvtColor(f, cv.COLOR_BGR2RGB)) for f in frames_bgr[i : i + batch_size]]
            inputs = self.processor(  # pylint: disable=not-callable
                images=chunk, return_tensors="pt"
            )
            pixel_values = inputs["pixel_values"].to(self._torch_device)
            with torch.inference_mode():
                out = self.model.get_image_features(pixel_values=pixel_values)
                if not isinstance(out, torch.Tensor):
                    pooled = getattr(out, "pooler_output", None)
                    out = pooled if pooled is not None else out.last_hidden_state[:, 0]
            all_feats.append(out)
            del inputs, pixel_values, chunk

        feats = torch.cat(all_feats, dim=0)
        feats = F.normalize(feats.float(), p=2, dim=1)
        del all_feats
        return feats

    def get_text_features(self, texts: list[str]) -> Any:
        """Extract L2-normalised text features from prompts → [N, D]."""
        import torch
        import torch.nn.functional as F

        batch_size = self._batch_size
        all_feats = []
        for i in range(0, len(texts), batch_size):
            inputs = self.processor(  # pylint: disable=not-callable
                text=texts[i : i + batch_size], return_tensors="pt", truncation=True, padding=True
            )
            input_ids = inputs["input_ids"].to(self._torch_device)
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self._torch_device)
            with torch.inference_mode():
                out = self.model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
                if not isinstance(out, torch.Tensor):
                    pooled = getattr(out, "pooler_output", None)
                    out = pooled if pooled is not None else out.last_hidden_state[:, 0]
            all_feats.append(out)
            del inputs, input_ids, attention_mask

        feats = F.normalize(torch.cat(all_feats, dim=0).float(), p=2, dim=1)
        del all_feats
        return feats

    def close(self) -> None:
        """Release model weights, processor, and device memory.  Idempotent."""
        was_loaded = self._model is not None
        self._model = None
        self._processor = None
        self._torch_device = None
        if was_loaded:
            gc.collect()
            # Dropping the references is not enough: the framework keeps the
            # freed blocks in its own allocator cache (notably on NPU).
            _empty_device_cache(self._device)

    def __del__(self):
        """Safety net: release model if close() was not called."""
        with suppress(Exception):
            self.close()


def get_shared_clip_backbone(
    model_path: str,
    device: str = "cpu",
    dtype: str = "fp16",
    batch_size: int | None = None,
) -> SharedClipBackbone:
    """Return a cached ``SharedClipBackbone`` instance for the given config.

    Load once, reuse forever: neither ``scorer.close()`` nor
    ``Pipeline.close()`` evicts the cache — call ``clear_cache()`` for that.
    """
    if batch_size is None:
        batch_size = DeviceResolver.default_frame_batch_size(device)
    key = _cache_key("clip", model_path, device, dtype, batch_size)
    with _lock:
        # Registration shares process-global state with the cache itself.
        _ensure_atexit()
        if key not in _cache:
            _cache[key] = SharedClipBackbone(model_path, device, dtype, batch_size)
        return _cache[key]
