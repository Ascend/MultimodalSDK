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

"""Aesthetics scorer — SigLIP so400m + AestheticPredictor v2.5.

Dual-weight scorer: SigLIP encoder (role ``encoder``) + v2.5 predictor
(role ``predictor``).  One inference yields two output scales:
    - score_aes_*: raw aesthetic score ∈ [1, 10] (data-filter view)
    - aesthetic_quality: normalised ∈ [0, 1] (eval view)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..device import DeviceResolver
from ..base import (
    BaseQualityScorer,
    LoadedFrames,
    MetricResult,
    QualityScorerRegistry,
    ScorerConfig,
)
from ..weights import WeightResolver

logger = logging.getLogger(__name__)


@QualityScorerRegistry.register
class AestheticsScorer(BaseQualityScorer):
    """Aesthetics scorer (SigLIP so400m-patch14-384 + AestheticPredictor v2.5).

    Dual-weight: ``weights={"encoder": ..., "predictor": ...}`` or resolved
    via env vars ``AESTHETIC_SIGLIP_MODEL_DIR`` / ``AESTHETIC_PREDICTOR_WEIGHTS_PATH``.
    Remote code execution is disabled by default; set ``extra={\"trust_remote_code\": True}``
    only when the configured local model artifacts are explicitly trusted.
    """

    name: str = "aesthetics"

    def __init__(self, config: ScorerConfig | None = None, **kwargs: Any):
        super().__init__(config, **kwargs)
        self._model = None
        self._processor = None
        self._head = None
        self._torch_device = None
        self._paths: dict[str, str] | None = None

    def _load_model(self) -> None:
        import torch

        resolver = WeightResolver(self.name)
        self._paths = resolver.resolve_multi(self._config.weights)
        encoder_path = Path(self._paths["encoder"]).expanduser()
        predictor_path = Path(self._paths["predictor"]).expanduser()
        if not encoder_path.is_dir():
            logger.error("aesthetic encoder directory is unavailable: %s", encoder_path)
            raise FileNotFoundError("Aesthetic encoder model directory not found")
        if not predictor_path.exists():
            logger.error("aesthetic predictor path is unavailable: %s", predictor_path)
            raise FileNotFoundError("Aesthetic predictor weights not found")

        self._torch_device = torch.device(self.device)
        use_fp16 = self._config.dtype == "fp16" and self.device.startswith(("npu", "cuda"))

        try:
            from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip
            from transformers import AutoProcessor

            trust_remote_code = self._config.extra.get("trust_remote_code", False)
            if not isinstance(trust_remote_code, bool):
                raise ValueError("aesthetics extra.trust_remote_code must be a boolean")
            model, processor = convert_v2_5_from_siglip(
                predictor_name_or_path=str(predictor_path),
                encoder_model_name=str(encoder_path),
                trust_remote_code=trust_remote_code,
            )
            self._model = model.to(self._torch_device).eval()
            self._processor = processor
            self._head = None
        except ImportError:
            from transformers import AutoModel, AutoProcessor

            self._model = (
                AutoModel.from_pretrained(
                    self._paths["encoder"],
                    local_files_only=True,
                    use_safetensors=True,
                )
                .to(self._torch_device)
                .eval()
            )
            self._processor = AutoProcessor.from_pretrained(self._paths["encoder"], local_files_only=True)
            head_file = Path(self._paths["predictor"])
            if not head_file.is_file():
                candidates = [
                    head_file / "aesthetic_predictor_v2_5.pth",
                    head_file / "model.pth",
                    head_file / "pytorch_model.bin",
                ]
                head_file = next((c for c in candidates if c.is_file()), None)
            if head_file is None:
                logger.error("aesthetic predictor weights not found: %s", self._paths["predictor"])
                raise FileNotFoundError("Aesthetic predictor weights not found")
            loaded = torch.load(head_file, map_location=self._torch_device, weights_only=True)
            self._head = self._to_module(loaded, self._torch_device)

        if use_fp16:
            self._model = self._model.half()
            if self._head is not None:
                self._head = self._head.half()

    @staticmethod
    def _to_module(loaded, device):
        from torch import nn

        if isinstance(loaded, nn.Module):
            return loaded.to(device).eval()

        class _LinearHead(nn.Module):
            def __init__(self, state):
                super().__init__()
                w = state.get("linear.weight", state.get("weight"))
                b = state.get("linear.bias", state.get("bias"))
                in_f = w.shape[1] if w is not None else 1152
                out_f = w.shape[0] if w is not None else 1
                self.linear = nn.Linear(in_f, out_f)
                if w is not None:
                    self.linear.weight.data = w
                if b is not None:
                    self.linear.bias.data = b

            def forward(self, x):
                return self.linear(x)

        return _LinearHead(loaded).to(device).eval()

    def score_frames(self, frames: LoadedFrames) -> MetricResult:
        t0 = time.perf_counter()

        if len(frames) == 0:
            return MetricResult(
                name=self.name,
                scores={
                    "score_aes_list": [],
                    "score_aes_mean": 0.0,
                    "score_aes_std": 0.0,
                    "score_aes_min": 0.0,
                    "score_aes_max": 0.0,
                    "aesthetic_quality": 0.0,
                    "aesthetic_quality_frames": [],
                },
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )

        try:
            self._ensure_model()
            import cv2 as cv
            import torch
            from PIL import Image

            pil = [Image.fromarray(cv.cvtColor(f, cv.COLOR_BGR2RGB)) for f in frames.frames]
            batch_size = max(
                1,
                int(self._config.frame_batch_size or DeviceResolver.default_frame_batch_size(self.device)),
            )
            scores = []
            for batch_start in range(0, len(pil), batch_size):
                pil_batch = pil[batch_start : batch_start + batch_size]
                inputs = self._processor(images=pil_batch, return_tensors="pt").to(self._torch_device)
                if self._config.dtype == "fp16" and self.device.startswith(("npu", "cuda")):
                    inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}

                with torch.no_grad():
                    out = self._model(**inputs)
                    raw = getattr(out, "logits", None)
                    if raw is None:
                        pooler_output = getattr(out, "pooler_output", None)
                        if pooler_output is None:
                            raise ValueError("Aesthetics model returned neither logits nor pooler_output")
                        if self._head is None:
                            raise ValueError("Aesthetics fallback model requires a scalar prediction head")
                        raw = self._head(pooler_output)

                    if not isinstance(raw, torch.Tensor):
                        raise TypeError("Aesthetics model output must be a tensor")
                    batch_scores = raw.detach().float().reshape(-1)
                    if batch_scores.numel() != len(pil_batch):
                        raise ValueError("Aesthetics model must return exactly one scalar score per input frame")
                    scores.extend(batch_scores.cpu().tolist())

            scores_arr = np.array(scores, dtype=np.float32)
            # Normalise to [0, 1] for the eval-view metric
            aes_min, aes_max = 1.0, 10.0
            norm_scores = (scores_arr - aes_min) / (aes_max - aes_min)
            norm_scores = np.clip(norm_scores, 0.0, 1.0)

            return MetricResult(
                name=self.name,
                scores={
                    "score_aes_list": [round(float(s), 6) for s in scores],
                    "score_aes_mean": float(np.mean(scores_arr)),
                    "score_aes_std": float(np.std(scores_arr)),
                    "score_aes_min": float(np.min(scores_arr)),
                    "score_aes_max": float(np.max(scores_arr)),
                    "aesthetic_quality": round(float(np.mean(norm_scores)), 6),
                    "aesthetic_quality_frames": [round(float(s), 6) for s in norm_scores],
                },
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )
        except Exception:
            logger.exception("aesthetics scoring failed")
            return MetricResult(
                name=self.name,
                scores={
                    "score_aes_list": [],
                    "score_aes_mean": None,
                    "score_aes_std": None,
                    "score_aes_min": None,
                    "score_aes_max": None,
                    "aesthetic_quality": None,
                    "aesthetic_quality_frames": [],
                },
                error="aesthetics_scoring_failed",
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )

    def close(self) -> None:
        self._model = None
        self._head = None
        self._processor = None
        self._model_loaded = False
