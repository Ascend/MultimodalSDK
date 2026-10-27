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

"""Image quality scorer — LIQE + NIQE/BRISQUE.

Weight resolution (all paths resolved from the user-specified ``weights`` or
``LIQE_WEIGHTS_PATH`` env var — no network downloads, no hardcoded cache dirs):

    1. LIQE weights (``liqe_mix.pt``):
       - ``weights`` param / ``LIQE_WEIGHTS_PATH`` env var → direct path
    2. LIQE text features (``liqe_text_feat_mix.pt``):
       - Same directory as the LIQE weights file
       - Override via ``PYIQA_LIQE_TEXT_FEAT`` env var
    3. CLIP ViT-B/32 backbone (``ViT-B-32.pt``):
       - ``{liqe_dir}/../clip/ViT-B-32.pt`` (sibling ``clip/`` subdir)
       - Override via ``PYIQA_CLIP_VITB32`` env var

pyiqa's LIQE class hardcodes cache paths for items 2 and 3 internally.
This scorer bypasses those by passing the local file path as the ``backbone``
parameter (which ``load()`` recognises as a file path) and by manually
injecting text features after model construction.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, ClassVar

import numpy as np

from ..base import (
    BaseQualityScorer,
    LoadedFrames,
    QualityScorerRegistry,
    ScorerConfig,
)
from ..weights import WeightResolver
from ._common import TimedScorerMixin

logger = logging.getLogger(__name__)
_LIQE_LOAD_LOCK = threading.Lock()

LIQE_DIST_LABELS = [
    "jpeg2000 compression",
    "jpeg compression",
    "noise",
    "blur",
    "color",
    "contrast",
    "overexposure",
    "underexposure",
    "spatial",
    "quantization",
    "other",
]


def _stats(values: list[float]) -> dict[str, Any]:
    a = np.array(values)
    return {
        "list": [round(float(p), 6) for p in values],
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
    }


@QualityScorerRegistry.register
class ImageQualityScorer(TimedScorerMixin, BaseQualityScorer):
    """Image quality scorer (LIQE + NIQE/BRISQUE).

    Weight resolution: user passes ``weights="/path/to/liqe_mix.pt"`` (or sets
    ``LIQE_WEIGHTS_PATH``).  The scorer automatically locates the sibling
    ``liqe_text_feat_mix.pt`` and ``../clip/ViT-B-32.pt`` files; environment
    variables ``PYIQA_LIQE_TEXT_FEAT`` and ``PYIQA_CLIP_VITB32`` override
    those auto-discovered paths.

    Metrics are configurable via ``metrics`` tuple:
    ``("liqe", "niqe", "brisque")`` (default: all).
    """

    name: ClassVar[str] = "image_quality"
    score_key: ClassVar[str] = "image_quality"
    empty_scores_when_insufficient: ClassVar[bool] = True
    empty_scores_on_error: ClassVar[bool] = True

    def __init__(self, config: ScorerConfig | None = None, **kwargs: Any):
        super().__init__(config, **kwargs)
        self._liqe_model = None
        self._niqe_metric = None
        self._brisque_metric = None
        self._torch_device = None
        self._liqe_path: str | None = None

    def _load_model(self) -> None:
        import torch

        metrics = tuple(self._config.extra.get("metrics", ("liqe", "niqe", "brisque")))
        self._torch_device = torch.device(self.device)

        if "liqe" in metrics:
            resolver = WeightResolver(self.name)
            self._liqe_path = resolver.resolve_single(self._config.weights, role="liqe")
            try:
                import pyiqa
            except ImportError as exc:
                raise ImportError("ImageQualityScorer requires pyiqa. Install with: pip install mm[quality]") from exc

            liqe_dir = os.path.dirname(self._liqe_path)

            # --- Resolve CLIP ViT-B/32 backbone path ---
            clip_path = os.environ.get("PYIQA_CLIP_VITB32", "")
            if not (clip_path and os.path.isfile(clip_path)):
                # Try sibling clip/ subdirectory
                candidate = os.path.join(liqe_dir, "..", "clip", "ViT-B-32.pt")
                candidate = os.path.normpath(candidate)
                if os.path.isfile(candidate):
                    clip_path = candidate
                else:
                    clip_path = None  # fall back to pyiqa default (may download)

            # --- Resolve text features path ---
            text_feat_path = os.environ.get("PYIQA_LIQE_TEXT_FEAT", "")
            if not (text_feat_path and os.path.isfile(text_feat_path)):
                candidate = os.path.join(liqe_dir, "liqe_text_feat_mix.pt")
                if os.path.isfile(candidate):
                    text_feat_path = candidate
                else:
                    text_feat_path = None

            # --- Construct LIQE model ---
            # pyiqa's LIQE.__init__ calls `load(backbone, 'cpu')` where
            # backbone='ViT-B/32'. The load() function accepts file paths
            # (clip_model.py line 111), but LIQE asserts backbone == 'ViT-B/32'.
            # We monkey-patch the load function in liqe_arch's namespace to
            # intercept 'ViT-B/32' and redirect to the local CLIP file.
            import pyiqa.archs.liqe_arch as _liqe_arch

            _orig_load = _liqe_arch.load
            with _LIQE_LOAD_LOCK:
                if clip_path and os.path.isfile(clip_path):

                    def _patched_load(name, *args, **kwargs):
                        if name == "ViT-B/32":
                            return _orig_load(clip_path, *args, **kwargs)
                        return _orig_load(name, *args, **kwargs)

                    _liqe_arch.load = _patched_load

                try:
                    self._liqe_model = pyiqa.create_metric(
                        "liqe_mix",
                        device=self.device,
                        pretrained="mix",
                        pretrained_model_path=self._liqe_path,
                    )
                finally:
                    _liqe_arch.load = _orig_load

            # --- Inject text features from local file ---
            # pyiqa.create_metric returns an InferenceModel wrapper; the actual
            # LIQE model is in .net. pyiqa's LIQE hardcodes text_feat_cache_path
            # to DEFAULT_CACHE_DIR; if that file doesn't exist there, it generates
            # from scratch (slow). We override .net.text_features directly.
            liqe_net = getattr(self._liqe_model, "net", self._liqe_model)
            if text_feat_path and os.path.isfile(text_feat_path):
                liqe_net.text_features = torch.load(text_feat_path, map_location="cpu", weights_only=False)

        if "niqe" in metrics:
            self._niqe_metric = self._create_optional_metric("niqe")

        if "brisque" in metrics:
            self._brisque_metric = self._create_optional_metric("brisque")

    def _create_optional_metric(self, metric_name: str) -> Any | None:
        """Create an optional pyiqa metric, skipping it only when pyiqa is absent."""
        try:
            import pyiqa

            return pyiqa.create_metric(metric_name, device=self.device)
        except ImportError as exc:
            logger.warning(
                "%s: pyiqa is unavailable, skipping %s metric: %s",
                self.name,
                metric_name,
                exc,
            )
            return None

    def _liqe_dist_probs(self, pil_img) -> np.ndarray:
        """Compute LIQE distortion probabilities from a PIL image.

        pyiqa's LIQE.forward() returns a 1-5 quality score, not distortion
        probabilities.  This method replicates the internal forward pass up
        to the softmax stage, returning the 11-class distortion probability
        vector (blur, dark, overexposure, etc.).
        """
        import torch
        import torch.nn.functional as F

        # Access the inner LIQE model (InferenceModel wraps it in .net)
        model = getattr(self._liqe_model, "net", self._liqe_model)
        # Preprocess: PIL → tensor [1, 3, H, W] ∈ [0, 1]
        img_tensor = (
            torch.from_numpy(np.array(pil_img).transpose(2, 0, 1))
            .unsqueeze(0)
            .float()
            .div_(255.0)
            .to(self._torch_device)
        )

        mean = model.default_mean.to(img_tensor)
        std = model.default_std.to(img_tensor)
        x = (img_tensor - mean) / std

        bs = x.size(0)
        x = (
            x.unfold(2, 224, model.step)
            .unfold(3, 224, model.step)
            .permute(0, 2, 3, 1, 4, 5)
            .reshape(bs, -1, 3, 224, 224)
        )
        num_patch = min(x.size(1), model.num_patch)
        sel_step = max(1, x.size(1) // num_patch)
        sel = torch.arange(num_patch, device=x.device) * sel_step
        x = x[:, sel.long(), ...]

        x_flat = x.reshape(bs * num_patch, 3, 224, 224)
        image_features = model.clip_model.encode_image(x_flat, pos_embedding=True)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)

        logit_scale = model.clip_model.logit_scale.exp()
        text_features = model.text_features.to(x_flat)
        logits = logit_scale * (image_features @ text_features.t())
        logits = logits.view(bs, num_patch, -1).mean(1)
        logits = F.softmax(logits, dim=1)

        num_distortions = len(LIQE_DIST_LABELS)
        logits = logits.view(bs, -1, num_distortions)
        return logits.sum(1).squeeze(0).detach().cpu().numpy()

    def _mean_metric(self, metric, pil_frames) -> float:
        values = []
        for pil_img in pil_frames:
            try:
                values.append(float(metric(pil_img)))
            except Exception as exc:
                logger.warning("%s: metric failed on a frame; recording 0.0: %s", self.name, exc)
                values.append(0.0)
        return round(float(np.mean(values)), 6)

    def _compute_scores(self, frames: LoadedFrames) -> dict[str, Any]:
        self._ensure_model()
        import cv2 as cv
        from PIL import Image

        metrics = tuple(self._config.extra.get("metrics", ("liqe", "niqe", "brisque")))
        pil_frames = [Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB)) for frame in frames.frames]

        scores: dict[str, Any] = {}

        if "liqe" in metrics and self._liqe_model is not None:
            blur_scores, dark_scores, over_scores = [], [], []
            blur_idx = LIQE_DIST_LABELS.index("blur")
            over_idx = LIQE_DIST_LABELS.index("overexposure")
            under_idx = LIQE_DIST_LABELS.index("underexposure")

            for pil_img in pil_frames:
                try:
                    probs = self._liqe_dist_probs(pil_img)
                except Exception as exc:
                    logger.warning(
                        "%s: LIQE failed on a frame; recording zeros: %s",
                        self.name,
                        exc,
                    )
                    blur_scores.append(0.0)
                    dark_scores.append(0.0)
                    over_scores.append(0.0)
                    continue
                blur_scores.append(float(probs[blur_idx]))
                dark_scores.append(float(probs[under_idx]))
                over_scores.append(float(probs[over_idx]))

            blur = _stats(blur_scores)
            dark = _stats(dark_scores)
            over = _stats(over_scores)
            scores.update(
                {
                    "score_blur_mean": blur["mean"],
                    "score_blur_list": blur["list"],
                    "score_blur_std": blur["std"],
                    "score_blur_min": blur["min"],
                    "score_blur_max": blur["max"],
                    "score_dark_mean": dark["mean"],
                    "score_dark_list": dark["list"],
                    "score_dark_std": dark["std"],
                    "score_dark_min": dark["min"],
                    "score_dark_max": dark["max"],
                    "score_overexposure_mean": over["mean"],
                    "score_overexposure_list": over["list"],
                    "score_overexposure_std": over["std"],
                    "score_overexposure_min": over["min"],
                    "score_overexposure_max": over["max"],
                }
            )

        if "niqe" in metrics and self._niqe_metric is not None:
            scores["niqe"] = self._mean_metric(self._niqe_metric, pil_frames)

        if "brisque" in metrics and self._brisque_metric is not None:
            scores["brisque"] = self._mean_metric(self._brisque_metric, pil_frames)

        return scores

    def close(self) -> None:
        self._liqe_model = None
        self._niqe_metric = None
        self._brisque_metric = None
        self._model_loaded = False
