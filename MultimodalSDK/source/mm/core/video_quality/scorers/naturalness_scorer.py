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

"""Naturalness scorer — CLIP ViT-L/14 + linear head (sigmoid).

Dual-weight scorer: CLIP model (role ``clip``) + linear regression head JSON
(role ``head``).  Per-frame CLIP image features → L2-normalise → linear head
→ sigmoid → naturalness probability ∈ [0, 1].
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import numpy as np

from ..base import (
    BaseQualityScorer,
    LoadedFrames,
    MetricResult,
    QualityScorerRegistry,
    ScorerConfig,
)
from ..weights import WeightResolver

logger = logging.getLogger(__name__)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def _feature_tensor(model_output):
    """Handle transformers 5.x ModelOutput vs raw Tensor."""
    import torch

    if isinstance(model_output, torch.Tensor):
        return model_output
    pooled = getattr(model_output, "pooler_output", None)
    if isinstance(pooled, torch.Tensor):
        return pooled
    last = getattr(model_output, "last_hidden_state", None)
    if isinstance(last, torch.Tensor):
        return last[:, 0]
    raise TypeError(f"Unsupported CLIP feature output: {type(model_output).__name__}")


@QualityScorerRegistry.register
class NaturalnessScorer(BaseQualityScorer):
    """Naturalness scorer (CLIP ViT-L/14 + logistic regression head + sigmoid).

    Dual-weight: ``weights={"clip": ..., "head": ...}`` or resolved via env
    vars ``NATURAL_CLIP_MODEL_ID`` / ``NATURAL_WEIGHTS_JSON``.
    """

    name: str = "naturalness"

    def __init__(self, config: ScorerConfig | None = None, **kwargs: Any):
        super().__init__(config, **kwargs)
        self._model = None
        self._processor = None
        self._torch_device = None
        self._dtype = None
        self._coef = None
        self._intercept = None
        self._image_size = 224
        self._image_mean = None
        self._image_std = None
        self._paths: dict[str, str] | None = None

    def _load_model(self) -> None:
        import torch
        from transformers import CLIPImageProcessor, CLIPModel

        resolver = WeightResolver(self.name)
        self._paths = resolver.resolve_multi(self._config.weights)

        self._torch_device = torch.device(self.device)
        use_fp16 = self._config.dtype == "fp16" and self.device.startswith(("npu", "cuda"))
        self._dtype = torch.float16 if use_fp16 else torch.float32

        clip_dir = self._paths["clip"]
        self._model = (
            CLIPModel.from_pretrained(
                clip_dir,
                torch_dtype=self._dtype,
                local_files_only=True,
                use_safetensors=True,
            )
            .to(self._torch_device)
            .eval()
        )
        processor = CLIPImageProcessor.from_pretrained(clip_dir, local_files_only=True)
        crop_size = processor.crop_size
        if hasattr(crop_size, "get"):
            edge = crop_size.get("height") or crop_size.get("width")
            self._image_size = int(edge or 224)
        elif crop_size:
            self._image_size = int(crop_size)
        else:
            self._image_size = 224
        self._image_mean = torch.tensor(list(processor.image_mean), dtype=torch.float32).view(1, 3, 1, 1)
        self._image_std = torch.tensor(list(processor.image_std), dtype=torch.float32).view(1, 3, 1, 1)

        head_path = os.path.abspath(self._paths["head"])
        if not os.path.isfile(head_path):
            raise FileNotFoundError(f"Naturalness head weights not found: {head_path}")
        with open(head_path, encoding="utf-8") as f:
            meta = json.load(f)
        self._coef = np.array(meta["coef"], dtype=np.float32)
        self._intercept = np.float32(meta["intercept"])

    def score_frames(self, frames: LoadedFrames) -> MetricResult:
        t0 = time.perf_counter()

        if len(frames) == 0:
            return MetricResult(
                name=self.name,
                scores={
                    "score_natural_list": [],
                    "score_natural_mean": -1.0,
                    "score_natural_std": 0.0,
                    "score_natural_min": -1.0,
                    "score_natural_max": -1.0,
                },
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )

        try:
            self._ensure_model()
            import cv2 as cv
            import torch
            import torch.nn.functional as F

            frames_rgb = [cv.cvtColor(f, cv.COLOR_BGR2RGB) for f in frames.frames]
            resized = [
                cv.resize(f, (self._image_size, self._image_size), interpolation=cv.INTER_CUBIC)
                if f.shape[0] != self._image_size or f.shape[1] != self._image_size
                else f
                for f in frames_rgb
            ]
            arr = np.stack(resized, axis=0)
            tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).float().div_(255.0)
            std = self._image_std.clamp_min(1e-6)
            tensor = (tensor - self._image_mean) / std

            with torch.inference_mode():
                feats = _feature_tensor(
                    self._model.get_image_features(pixel_values=tensor.to(self._torch_device, dtype=self._dtype))
                )
                feats = F.normalize(feats.float(), p=2, dim=1).cpu().numpy()

            logits = feats @ self._coef.reshape(-1) + float(self._intercept)
            scores = _sigmoid(logits)

            return MetricResult(
                name=self.name,
                scores={
                    "score_natural_list": [round(float(s), 6) for s in scores],
                    "score_natural_mean": float(np.mean(scores)),
                    "score_natural_std": float(np.std(scores)),
                    "score_natural_min": float(np.min(scores)),
                    "score_natural_max": float(np.max(scores)),
                },
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )
        except Exception:
            logger.error("naturalness scoring failed", exc_info=True)
            return MetricResult(
                name=self.name,
                scores={
                    "score_natural_list": [],
                    "score_natural_mean": None,
                    "score_natural_std": None,
                    "score_natural_min": None,
                    "score_natural_max": None,
                },
                error="naturalness scoring failed",
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )

    def close(self) -> None:
        self._model = None
        self._processor = None
        self._model_loaded = False
