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

"""Video quality pipeline — one-decode, shared-frame, batched inference.

Each video is decoded once (uniform sampling) and the same ``LoadedFrames``
is fed to every frame-level scorer; results are aggregated into a
``VideoQualityReport``.  Scorers with ``requires_continuous_frames`` get an
independent continuous-fps decode path.
"""

from __future__ import annotations

import time
from contextlib import suppress
from typing import Any

from .base import (
    BaseQualityScorer,
    MetricResult,
    QualityScorerRegistry,
    ScorerConfig,
    VideoQualityReport,
)
from .defaults import get_scorer_defaults
from .device import DeviceResolver
from .frame_loader import FrameLoader


class VideoQualityPipeline:
    """Combination scorer — one decode, shared frames, batched inference.

    Args:
        scorers: List of scorer registration names or scorer instances.
        device: ``"auto"`` | ``"npu"`` | ``"cpu"``.
        weights: Optional ``{scorer_name: weights_value}`` dict to pass
            specific weights to each scorer.
        decode_workers: Decode thread count (unused for now — single-threaded).
        frame_batch_size: Inference batch size (None → device default).
    """

    def __init__(
        self,
        scorers: list[str] | list[BaseQualityScorer],
        device: str = "auto",
        weights: dict[str, Any] | None = None,
        decode_workers: int = 4,
        frame_batch_size: int | None = None,
    ):
        self._device = DeviceResolver.resolve(device)
        self._weights = weights or {}
        self._decode_workers = decode_workers
        self._frame_batch_size = frame_batch_size

        self._scorers: list[BaseQualityScorer] = []
        for s in scorers:
            if isinstance(s, str):
                cfg = self._build_config(s)
                if s in self._weights:
                    cfg.weights = self._weights[s]
                scorer = QualityScorerRegistry.get(s)(config=cfg)
                self._scorers.append(scorer)
            else:
                self._scorers.append(s)

    def _build_config(self, name: str) -> ScorerConfig:
        """Merge declared scorer defaults with the pipeline's explicit args.

        Precedence: constructor argument (when given) > ``SCORER_DEFAULTS`` >
        dataclass default.  The declared ``extra`` dict is copied in so the
        scorer sees its own parameters (``target_fps``, ``prompt_set``, ...).
        """
        defaults = get_scorer_defaults(name)
        cfg = ScorerConfig(
            device=self._device,
            sample_frames=defaults.sample_frames,
            frame_batch_size=(
                self._frame_batch_size if self._frame_batch_size is not None else defaults.frame_batch_size
            ),
            dtype=defaults.dtype,
        )
        cfg.extra.update(defaults.extra)
        return cfg

    def score(self, video_path: str) -> VideoQualityReport:
        """Score a single video with all configured scorers."""
        t0 = time.perf_counter()

        loader = FrameLoader(device=self._device)
        report = VideoQualityReport(video_path=video_path, device=self._device)

        # Split scorers: continuous-frame vs shared-frame
        continuous_scorers = [s for s in self._scorers if s.requires_continuous_frames]
        shared_scorers = [s for s in self._scorers if not s.requires_continuous_frames]

        # Decode shared frames (uniform sampling) once
        if shared_scorers:
            max_sample = max(s.config.sample_frames for s in shared_scorers)
            loaded = loader.load(video_path, sample_frames=max_sample)
            report.frame_indices = loaded.indices

            for scorer in shared_scorers:
                result = scorer.score_frames(loaded)
                report.metrics[scorer.name] = result

        # Decode continuous frames separately for optical-flow scorers
        for scorer in continuous_scorers:
            target_fps = scorer.config.extra.get("target_fps", 4.0)
            loaded = loader.load_continuous(video_path, fps=target_fps)
            result = scorer.score_frames(loaded)
            report.metrics[scorer.name] = result

        # Check if all failed (no scorers at all is not a failure)
        if report.metrics and all(r.error is not None for r in report.metrics.values()):
            raise RuntimeError(f"All scorers failed for {video_path}. Errors: {report.errors}")

        report.elapsed_ms = (time.perf_counter() - t0) * 1000
        return report

    def score_batch(
        self,
        video_paths: list[str],
        max_concurrency: int = 1,
    ) -> list[VideoQualityReport]:
        """Score a batch of videos (sequential for now).

        Args:
            video_paths: List of video file paths.
            max_concurrency: Reserved for future parallel execution.

        Returns:
            One ``VideoQualityReport`` per video; failures are captured in a
            ``_pipeline`` metric instead of aborting the batch.
        """
        reports = []
        for path in video_paths:
            try:
                reports.append(self.score(path))
            except Exception as exc:
                report = VideoQualityReport(
                    video_path=path,
                    device=self._device,
                )
                report.metrics["_pipeline"] = MetricResult(
                    name="_pipeline",
                    scores={},
                    error=str(exc),
                )
                reports.append(report)
        return reports

    def close(self) -> None:
        """Release per-scorer state (does NOT evict cached backbones).

        The process-level backbone cache persists so the next Pipeline can
        reuse loaded models; call ``clear_backbone_cache()`` to free memory.
        """
        for scorer in self._scorers:
            # One failing scorer must not block the remaining ones.
            with suppress(Exception):
                scorer.close()
