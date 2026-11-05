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

"""Distribution consistency scorer — FVD / KVD (dataset-level metrics).

Single-weight scorer: I3D/SVD feature model directory (role ``feature``).

FVD: video-level features (I3D) → Fréchet distance between real/generated sets.
KVD: keyframe-level features (CLIP) → Fréchet distance between real/generated sets.

Dataset-level interface: input is ``real``/``generated`` video path lists (or
pre-extracted feature matrices), not single-video frames.
"""

from __future__ import annotations

import contextlib
import logging
import time
from pathlib import Path
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

# Accepted ``feature`` channels; the same value drives both the metric key and
# lazy model loading (single source of truth).
_FEATURE_CHOICES = ("i3d", "svd", "keyframe")

# Extensions scanned when a directory path is expanded to a video list.
_VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv")


@contextlib.contextmanager
def _vendored_path(path: str):
    """Temporarily expose *path* on ``sys.path`` for a vendored module import."""
    import sys

    if path in sys.path:
        yield
        return
    sys.path.insert(0, path)
    try:
        yield
    finally:
        with contextlib.suppress(ValueError):
            sys.path.remove(path)


def _frechet_distance(feats_a: np.ndarray, feats_b: np.ndarray) -> float:
    """Compute Fréchet distance between two feature matrices."""
    from scipy import linalg

    mu1, mu2 = feats_a.mean(0), feats_b.mean(0)
    s1 = np.cov(feats_a, rowvar=False)
    s2 = np.cov(feats_b, rowvar=False)

    covmean, _ = linalg.sqrtm(s1.dot(s2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    diff = mu1 - mu2
    return float(diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


def _load_videos_as_tensor(paths: list[str], frames: int, size: int):
    """Load video paths → Tensor[B, C, T, H, W] ∈ [0, 1]."""
    import cv2 as cv
    import torch

    batch = []
    for p in paths:
        cap = cv.VideoCapture(p)
        try:
            total = int(cap.get(cv.CAP_PROP_FRAME_COUNT)) or frames
            start = max(0, (total - frames) // 2)
            cap.set(cv.CAP_PROP_POS_FRAMES, start)
            clip = []
            while len(clip) < frames:
                ok, f = cap.read()
                if not ok:
                    break
                clip.append(cv.resize(cv.cvtColor(f, cv.COLOR_BGR2RGB), (size, size)))
        finally:
            # Released even when read/resize raises mid-loop.
            cap.release()
        if len(clip) < frames:
            if len(clip) < 2:
                continue
            clip = (clip * frames)[:frames]
        arr = np.stack(clip).transpose(3, 0, 1, 2) / 255.0
        batch.append(torch.from_numpy(arr).float())
    if not batch:
        raise ValueError("No valid videos found in the provided paths")
    return torch.stack(batch)


@QualityScorerRegistry.register
class DistributionConsistencyScorer(BaseQualityScorer):
    """Distribution consistency scorer (FVD/KVD, dataset-level).

    Single-weight: ``weights="/path/feature_model_dir"`` or env var
    ``FVD_FEATURE_MODEL_DIR``.

    This is a dataset-level interface — input is ``real``/``generated`` video
    path lists or pre-extracted feature matrices.  Does not support
    ``score_frames`` with single-video frames.
    """

    name: str = "distribution"
    requires_continuous_frames: bool = False

    def __init__(self, config: ScorerConfig | None = None, **kwargs: Any):
        super().__init__(config, **kwargs)
        self._model = None
        self._torch_device = None
        self._feature_path: str | None = None
        self._feature_type: str | None = None

    def _load_model(self) -> None:
        import torch

        resolver = WeightResolver(self.name)
        self._feature_path = resolver.resolve_single(self._config.weights, role="feature")

        self._torch_device = torch.device(self.device)
        # ``_feature_type`` is set by ``score``; fall back to the configured
        # default so both entry points agree on a single feature channel.
        feature_type = self._feature_type or self._config.extra.get("feature", "i3d")

        if feature_type in ("i3d", "svd"):
            vendor = str(Path(self._feature_path))
            try:
                # Vendored checkpoints ship ``pytorch_i3d`` next to the weights;
                # expose that directory only for the duration of the import so
                # the process-wide module search path is not polluted.
                with _vendored_path(vendor):
                    from pytorch_i3d import InceptionI3d  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    "DistributionConsistencyScorer requires the 'pytorch_i3d' module: "
                    f"expected it under '{vendor}', or install it with "
                    "'pip install mm[quality]'."
                ) from exc

            self._model = InceptionI3d(400, in_channels=3).eval().to(self._torch_device)
            ckpt = Path(self._feature_path)
            ckpt_file = ckpt / "i3d_pretrained_400.pt" if ckpt.is_dir() else ckpt
            if not ckpt_file.is_file():
                raise FileNotFoundError(
                    f"I3D weights not found: {ckpt_file} (HF: Xiaodong/FVD_I3D i3d_pretrained_400.pt)"
                )
            state = torch.load(str(ckpt_file), map_location=self._torch_device, weights_only=True)
            self._model.load_state_dict(state)

    def score(  # pylint: disable=arguments-renamed
        # Dataset-level scorer: the first argument is a *set* of videos
        # (``real``), so the base-class ``video_path`` name does not apply.
        self,
        real: list[str] | np.ndarray | None = None,
        generated: list[str] | np.ndarray | None = None,
        feature: str | None = None,
        sample_frames: int = 16,
        **kwargs: Any,
    ) -> MetricResult:
        """Dataset-level scoring interface.

        Args:
            real: Real video path list or pre-extracted features [N, D].
            generated: Generated video path list or pre-extracted features [N, D].
            feature: Feature extractor type: ``"i3d"``, ``"svd"`` or
                ``"keyframe"``.  Defaults to ``config.extra["feature"]``.
            sample_frames: Frames per video (default 16).

        Returns:
            MetricResult with ``scores={"fvd": f}`` or ``{"kvd": f}``.
        """
        t0 = time.perf_counter()

        feature = feature or self._config.extra.get("feature", "i3d")
        if feature not in _FEATURE_CHOICES:
            return MetricResult(
                name=self.name,
                scores={"fvd": None},
                error=f"Unsupported feature {feature!r}; expected one of {list(_FEATURE_CHOICES)}",
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )
        # Single source of truth: the channel chosen here also drives the lazy
        # model load performed by ``_extract_video_features``.
        self._feature_type = feature

        if real is None or generated is None:
            return MetricResult(
                name=self.name,
                scores={"fvd": None},
                error="real and generated must be provided",
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )

        try:
            max_videos = int(self._config.extra.get("max_videos", 32))
            size = int(self._config.extra.get("size", 224))

            # Expand directory paths to sorted video lists
            real_paths = self._expand_paths(real)[:max_videos] if isinstance(real, list) else None
            gen_paths = self._expand_paths(generated)[:max_videos] if isinstance(generated, list) else None

            if isinstance(real, np.ndarray) and isinstance(generated, np.ndarray):
                # Pre-extracted features
                if real.shape[1] != generated.shape[1]:
                    raise ValueError(
                        f"Feature dimension mismatch: real={real.shape[1]}, generated={generated.shape[1]}"
                    )
                feats_real = real
                feats_gen = generated
                metric_key = "fvd" if feature in ("i3d", "svd") else "kvd"
            else:
                if real_paths is None or gen_paths is None:
                    raise ValueError("real/generated must be path lists or feature arrays")
                if len(real_paths) < 2 or len(gen_paths) < 2:
                    return MetricResult(
                        name=self.name,
                        scores={"kvd" if feature == "keyframe" else "fvd": -1.0},
                        error="FVD/KVD requires at least 2 videos per group",
                        elapsed_ms=(time.perf_counter() - t0) * 1000,
                    )

                if feature == "keyframe":
                    metric_key = "kvd"
                    feats_real = self._extract_keyframe_features(real_paths, sample_frames)
                    feats_gen = self._extract_keyframe_features(gen_paths, sample_frames)
                else:
                    metric_key = "fvd"
                    feats_real = self._extract_video_features(real_paths, sample_frames, size)
                    feats_gen = self._extract_video_features(gen_paths, sample_frames, size)

            if len(feats_real) < 32 or len(feats_gen) < 32:
                logger.warning(
                    "Small sample for distribution distance (real=%d, generated=%d); "
                    "the Fréchet estimate may be unreliable.",
                    len(feats_real),
                    len(feats_gen),
                )

            fvd = _frechet_distance(feats_real, feats_gen)

            return MetricResult(
                name=self.name,
                scores={metric_key: round(float(fvd), 4)},
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )
        except Exception as exc:
            metric_key = "kvd" if feature == "keyframe" else "fvd"
            return MetricResult(
                name=self.name,
                scores={metric_key: None},
                error=str(exc),
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )

    def score_frames(self, frames: LoadedFrames) -> MetricResult:
        """Not supported — distribution is a dataset-level metric."""
        return MetricResult(
            name=self.name,
            scores={"fvd": None},
            error="DistributionConsistencyScorer is a dataset-level metric. "
            "Use score(real=..., generated=...) instead.",
        )

    def _expand_paths(self, paths: list[str]) -> list[str]:
        """Expand directory entries to video files; other files are skipped."""
        out = []
        for p in paths:
            path = Path(p)
            if not path.is_dir():
                out.append(p)
                continue
            files = sorted(f for f in path.iterdir() if f.is_file())
            videos = [str(f) for f in files if f.suffix.lower() in _VIDEO_EXTENSIONS]
            ignored = [f for f in files if f.suffix.lower() not in _VIDEO_EXTENSIONS]
            if ignored:
                logger.warning(
                    "Ignoring %d non-video file(s) under '%s' (e.g. %s)",
                    len(ignored),
                    p,
                    ", ".join(f.name for f in ignored[:3]),
                )
            out.extend(videos)
        return out

    def _extract_video_features(self, paths: list[str], frames: int, size: int) -> np.ndarray:
        import torch

        self._ensure_model()
        videos = _load_videos_as_tensor(paths, frames, size)
        with torch.inference_mode():
            feats = self._model(videos.to(self._torch_device))
        return feats.detach().float().cpu().numpy()

    def _extract_keyframe_features(self, paths: list[str], frames_per_video: int) -> np.ndarray:
        from ..backbones import get_shared_clip_backbone
        from ..frame_loader import FrameLoader

        resolver = WeightResolver("subject_consistency")
        clip_path = resolver.resolve_single(None, role="clip")
        backbone = get_shared_clip_backbone(clip_path, device=self.device, dtype=self._config.dtype)

        loader = FrameLoader(device=self.device)
        all_feats = []
        for p in paths:
            loaded = loader.load(p, sample_frames=frames_per_video)
            if len(loaded) < 1:
                continue
            feats = backbone.get_image_features(loaded.frames)
            all_feats.append(feats.cpu().numpy())
        if not all_feats:
            raise ValueError("No valid videos found")
        return np.concatenate(all_feats, axis=0)

    def close(self) -> None:
        self._model = None
        self._model_loaded = False
