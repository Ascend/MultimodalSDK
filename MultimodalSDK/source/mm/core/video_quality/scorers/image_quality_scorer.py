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

Weight resolution is local-only. LIQE weights come from the explicit
``weights`` argument or ``LIQE_WEIGHTS_PATH``; pickle checkpoints require the
matching ``LIQE_WEIGHTS_SHA256`` digest and are loaded with ``weights_only=True``.
Text features come from the sibling ``liqe_text_feat_mix.pt`` or
``PYIQA_LIQE_TEXT_FEAT`` and are also loaded with ``weights_only=True``. The
CLIP ViT-B/32 checkpoint comes from the sibling ``../clip/ViT-B-32.pt`` or
``PYIQA_CLIP_VITB32``; it must match the official SHA-256 digest below, or a
trusted custom digest supplied via ``PYIQA_CLIP_VITB32_SHA256``. No model
checkpoint or cache fallback is downloaded from the network.

pyiqa's LIQE class hardcodes cache paths and uses unsafe defaults in its
checkpoint loaders. This scorer temporarily redirects its cache to a private
scratch directory and replaces the LIQE/CLIP loaders with local, verified,
weights-only implementations while the model is constructed.
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
import threading
import time
from typing import Any, ClassVar

import numpy as np

from ..base import (
    BaseQualityScorer,
    LoadedFrames,
    MetricResult,
    QualityScorerRegistry,
    ScorerConfig,
)
from ..weights import WeightResolver
from ..device import DeviceResolver
from ._common import TimedScorerMixin

logger = logging.getLogger(__name__)
_LIQE_LOAD_LOCK = threading.Lock()
_CLIP_VIT_B32_SHA256 = "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"

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


class _WeightsOnlyTorchProxy:
    """Module proxy that forces pyiqa's internal torch.load calls to be safe."""

    def __init__(self, torch_module):
        self._torch_module = torch_module

    def __getattr__(self, name):
        if name == "load":
            return self.load
        return getattr(self._torch_module, name)

    def load(self, *args, **kwargs):
        kwargs["weights_only"] = True
        try:
            return self._torch_module.load(*args, **kwargs)
        except TypeError as exc:
            raise RuntimeError("ImageQualityScorer requires PyTorch torch.load(weights_only=True) support") from exc


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as checkpoint:
        for chunk in iter(lambda: checkpoint.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_state_dict_weights_only(net, model_path, strict=True, weight_keys=None):
    """Load a local LIQE state dict without executing checkpoint pickle code."""
    import torch

    if not isinstance(model_path, str) or model_path.startswith(("http://", "https://")):
        raise ValueError("Remote LIQE checkpoint URLs are disabled")
    if not os.path.isfile(model_path):
        raise FileNotFoundError("Local LIQE checkpoint does not exist")

    if model_path.endswith(".safetensors"):
        from safetensors.torch import load_file

        state_dict = load_file(model_path, device="cpu")
        if weight_keys and weight_keys in state_dict:
            state_dict = state_dict[weight_keys]
    else:
        expected_sha256 = os.environ.get("LIQE_WEIGHTS_SHA256", "").strip().lower()
        if len(expected_sha256) != 64 or any(c not in "0123456789abcdef" for c in expected_sha256):
            raise ValueError("LIQE_WEIGHTS_SHA256 must contain the trusted SHA-256 digest for the local checkpoint")
        if _sha256_file(model_path) != expected_sha256:
            raise ValueError("LIQE checkpoint SHA-256 mismatch")
        state_dict = torch.load(model_path, map_location=torch.device("cpu"), weights_only=True)
        if weight_keys is not None:
            state_dict = state_dict[weight_keys]

    if not isinstance(state_dict, dict):
        raise TypeError("LIQE checkpoint must contain a state-dict mapping")
    cleaned_state_dict = {key[7:] if key.startswith("module.") else key: value for key, value in state_dict.items()}
    net.load_state_dict(cleaned_state_dict, strict=strict)


def _load_local_clip_model(clip_model_module, checkpoint_path: str, device):
    """Load the hash-verified official CLIP TorchScript checkpoint locally."""
    import torch

    try:
        scripted_model = torch.jit.load(checkpoint_path, map_location=device).eval()
    except Exception as exc:
        raise RuntimeError("CLIP ViT-B/32 checkpoint must be a verified local TorchScript archive") from exc
    model = clip_model_module.build_model(scripted_model.state_dict()).to(device)
    if str(device) == "cpu":
        model.float()
    return model


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

            if not os.path.isfile(self._liqe_path):
                raise FileNotFoundError("Local LIQE checkpoint does not exist; provide a local weights file")
            if not self._liqe_path.endswith(".safetensors"):
                expected_liqe_sha256 = os.environ.get("LIQE_WEIGHTS_SHA256", "").strip().lower()
                if len(expected_liqe_sha256) != 64 or any(c not in "0123456789abcdef" for c in expected_liqe_sha256):
                    raise ValueError(
                        "LIQE_WEIGHTS_SHA256 must contain the trusted SHA-256 digest for the local checkpoint"
                    )
                if _sha256_file(self._liqe_path) != expected_liqe_sha256:
                    raise ValueError("LIQE checkpoint SHA-256 mismatch")

            liqe_dir = os.path.dirname(self._liqe_path)

            # Resolve and verify a local CLIP checkpoint. The upstream pyiqa
            # loader downloads by model name, so never allow its default path.
            configured_clip_path = os.environ.get("PYIQA_CLIP_VITB32", "").strip()
            if configured_clip_path:
                clip_path = os.path.abspath(os.path.expanduser(configured_clip_path))
                if not os.path.isfile(clip_path):
                    raise FileNotFoundError("Configured local CLIP ViT-B/32 checkpoint does not exist")
            else:
                candidate = os.path.normpath(os.path.join(liqe_dir, "..", "clip", "ViT-B-32.pt"))
                clip_path = candidate if os.path.isfile(candidate) else None
            if clip_path is None:
                raise FileNotFoundError("Local CLIP ViT-B/32 checkpoint is required; set PYIQA_CLIP_VITB32")

            expected_clip_sha256 = os.environ.get("PYIQA_CLIP_VITB32_SHA256", _CLIP_VIT_B32_SHA256).strip().lower()
            if len(expected_clip_sha256) != 64 or any(c not in "0123456789abcdef" for c in expected_clip_sha256):
                raise ValueError("PYIQA_CLIP_VITB32_SHA256 must be a 64-character SHA-256 digest")
            if _sha256_file(clip_path) != expected_clip_sha256:
                raise ValueError("CLIP ViT-B/32 checkpoint SHA-256 mismatch")

            # Resolve text features locally. If absent, pyiqa may generate them
            # in the private scratch cache below; it must never read a shared
            # default cache containing an unverified pickle.
            configured_text_path = os.environ.get("PYIQA_LIQE_TEXT_FEAT", "").strip()
            if configured_text_path:
                text_feat_path = os.path.abspath(os.path.expanduser(configured_text_path))
                if not os.path.isfile(text_feat_path):
                    raise FileNotFoundError("Configured local LIQE text-feature file does not exist")
            else:
                candidate = os.path.join(liqe_dir, "liqe_text_feat_mix.pt")
                text_feat_path = candidate if os.path.isfile(candidate) else None
            text_feat_tensor = (
                torch.load(text_feat_path, map_location="cpu", weights_only=True) if text_feat_path else None
            )

            # Patch only pyiqa's LIQE module namespace. The proxy forces its
            # internal text-cache torch.load(..., weights_only=False) call to
            # weights_only=True without mutating the process-global torch API.
            import pyiqa.archs.liqe_arch as _liqe_arch
            import pyiqa.archs.clip_model as _clip_model

            with tempfile.TemporaryDirectory(prefix="mmsdk-liqe-cache-") as cache_dir:
                if text_feat_tensor is not None:
                    torch.save(
                        text_feat_tensor,
                        os.path.join(cache_dir, "liqe_text_feat_mix.pt"),
                    )

                with _LIQE_LOAD_LOCK:
                    original_load = _liqe_arch.load
                    original_loader = _liqe_arch.load_pretrained_network
                    original_cache_dir = _liqe_arch.DEFAULT_CACHE_DIR
                    original_torch = _liqe_arch.torch

                    def _safe_clip_load(name, *args, device="cpu", **kwargs):
                        if name != "ViT-B/32":
                            raise ValueError("Only the configured local CLIP ViT-B/32 checkpoint is allowed")
                        return _load_local_clip_model(_clip_model, clip_path, device)

                    _liqe_arch.load = _safe_clip_load
                    _liqe_arch.load_pretrained_network = _load_state_dict_weights_only
                    _liqe_arch.DEFAULT_CACHE_DIR = cache_dir
                    _liqe_arch.torch = _WeightsOnlyTorchProxy(original_torch)
                    try:
                        self._liqe_model = pyiqa.create_metric(
                            "liqe_mix",
                            device=self.device,
                            pretrained="mix",
                            pretrained_model_path=self._liqe_path,
                        )
                    finally:
                        _liqe_arch.load = original_load
                        _liqe_arch.load_pretrained_network = original_loader
                        _liqe_arch.DEFAULT_CACHE_DIR = original_cache_dir
                        _liqe_arch.torch = original_torch

            if text_feat_tensor is not None:
                liqe_net = getattr(self._liqe_model, "net", self._liqe_model)
                liqe_net.text_features = text_feat_tensor

        if "niqe" in metrics or "brisque" in metrics:
            try:
                import pyiqa
            except ImportError as exc:
                raise ImportError("pyiqa is required when NIQE or BRISQUE metrics are enabled") from exc

        if "niqe" in metrics:
            try:
                self._niqe_metric = pyiqa.create_metric("niqe", device=self.device)
            except Exception as exc:
                raise RuntimeError("Failed to initialize the NIQE metric") from exc

        if "brisque" in metrics:
            try:
                self._brisque_metric = pyiqa.create_metric("brisque", device=self.device)
            except Exception as exc:
                raise RuntimeError("Failed to initialize the BRISQUE metric") from exc

    def _liqe_dist_probs_batch(self, pil_imgs: list) -> np.ndarray:
        """Compute LIQE distortion probabilities for a batch of PIL images."""
        import torch
        import torch.nn.functional as F

        if not pil_imgs:
            return np.empty((0, len(LIQE_DIST_LABELS)), dtype=np.float32)

        model = getattr(self._liqe_model, "net", self._liqe_model)
        img_tensor = (
            torch.stack(
                [torch.from_numpy(np.array(pil_img, copy=True).transpose(2, 0, 1)) for pil_img in pil_imgs],
                dim=0,
            )
            .float()
            .div_(255.0)
            .to(self._torch_device)
        )

        mean = model.default_mean.to(img_tensor)
        std = model.default_std.to(img_tensor)
        x = (img_tensor - mean) / std

        batch_size = x.size(0)
        x = (
            x.unfold(2, 224, model.step)
            .unfold(3, 224, model.step)
            .permute(0, 2, 3, 1, 4, 5)
            .reshape(batch_size, -1, 3, 224, 224)
        )
        num_patch = min(x.size(1), model.num_patch)
        sel_step = max(1, x.size(1) // num_patch)
        sel = torch.arange(num_patch, device=x.device) * sel_step
        x = x[:, sel.long(), ...]

        x_flat = x.reshape(batch_size * num_patch, 3, 224, 224)
        image_features = model.clip_model.encode_image(x_flat, pos_embedding=True)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)

        logit_scale = model.clip_model.logit_scale.exp()
        text_features = model.text_features.to(x_flat)
        logits = logit_scale * (image_features @ text_features.t())
        logits = logits.view(batch_size, num_patch, -1).mean(1)
        logits = F.softmax(logits, dim=1)

        num_distortions = len(LIQE_DIST_LABELS)
        return logits.view(batch_size, -1, num_distortions).sum(1).detach().cpu().numpy()

    def _liqe_dist_probs(self, pil_img) -> np.ndarray:
        """Backward-compatible single-image wrapper around batched LIQE inference."""
        return self._liqe_dist_probs_batch([pil_img])[0]

    def score_frames(self, frames: LoadedFrames) -> MetricResult:
        t0 = time.perf_counter()

        if len(frames) == 0:
            return MetricResult(
                name=self.name,
                scores={},
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )

        try:
            self._ensure_model()
            import cv2 as cv
            from PIL import Image

            metrics = tuple(self._config.extra.get("metrics", ("liqe", "niqe", "brisque")))
            pil_frames = [Image.fromarray(cv.cvtColor(f, cv.COLOR_BGR2RGB)) for f in frames.frames]

            scores: dict[str, Any] = {}

            if "liqe" in metrics and self._liqe_model is not None:
                blur_scores, dark_scores, over_scores = [], [], []
                blur_idx = LIQE_DIST_LABELS.index("blur")
                over_idx = LIQE_DIST_LABELS.index("overexposure")
                under_idx = LIQE_DIST_LABELS.index("underexposure")
                batch_size = max(
                    1,
                    int(self._config.frame_batch_size or DeviceResolver.default_frame_batch_size(self.device)),
                )
                failed_frames = 0
                for batch_start in range(0, len(pil_frames), batch_size):
                    batch = pil_frames[batch_start : batch_start + batch_size]
                    try:
                        probs_batch = self._liqe_dist_probs_batch(batch)
                        if len(probs_batch) != len(batch):
                            raise ValueError("LIQE batch result count does not match input count")
                        for probs in probs_batch:
                            blur_scores.append(float(probs[blur_idx]))
                            dark_scores.append(float(probs[under_idx]))
                            over_scores.append(float(probs[over_idx]))
                    except Exception as exc:
                        failed_frames += len(batch)
                        logger.warning(
                            "LIQE inference failed for frame batch starting at %d (%s)",
                            batch_start,
                            type(exc).__name__,
                            exc_info=True,
                        )
                if failed_frames:
                    scores["liqe_failed_frames"] = failed_frames
                if not blur_scores:
                    raise RuntimeError("LIQE failed for all sampled frames")

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
                niqe_scores = []
                failed_frames = 0
                for frame_idx, pil_img in enumerate(pil_frames):
                    try:
                        niqe_scores.append(float(self._niqe_metric(pil_img)))
                    except Exception as exc:
                        failed_frames += 1
                        logger.warning(
                            "NIQE inference failed for frame %d (%s)",
                            frame_idx,
                            type(exc).__name__,
                            exc_info=True,
                        )
                if failed_frames:
                    scores["niqe_failed_frames"] = failed_frames
                if not niqe_scores:
                    raise RuntimeError("NIQE failed for all sampled frames")
                scores["niqe"] = round(float(np.mean(niqe_scores)), 6)

            if "brisque" in metrics and self._brisque_metric is not None:
                brisque_scores = []
                failed_frames = 0
                for frame_idx, pil_img in enumerate(pil_frames):
                    try:
                        brisque_scores.append(float(self._brisque_metric(pil_img)))
                    except Exception as exc:
                        failed_frames += 1
                        logger.warning(
                            "BRISQUE inference failed for frame %d (%s)",
                            frame_idx,
                            type(exc).__name__,
                            exc_info=True,
                        )
                if failed_frames:
                    scores["brisque_failed_frames"] = failed_frames
                if not brisque_scores:
                    raise RuntimeError("BRISQUE failed for all sampled frames")
                scores["brisque"] = round(float(np.mean(brisque_scores)), 6)

            return MetricResult(
                name=self.name,
                scores=scores,
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )
        except Exception:
            logger.exception("image_quality scoring failed")
            return MetricResult(
                name=self.name,
                scores={},
                error="image_quality_inference_failed",
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )

    def close(self) -> None:
        self._liqe_model = None
        self._niqe_metric = None
        self._brisque_metric = None
        self._model_loaded = False
