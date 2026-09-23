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

"""Core base classes, data contracts, and scorer registry for video quality.

This module defines:
    - ``LoadedFrames``: dataclass holding decoded frames + metadata.
    - ``MetricResult``: dataclass for a single scorer's output.
    - ``VideoQualityReport``: aggregated report for a pipeline run.
    - ``ScorerConfig``: configuration dataclass.
    - ``BaseQualityScorer``: ABC that all 15 scorers inherit.
    - ``QualityScorerRegistry``: registration-based factory driven by ``name``.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from typing import Any, ClassVar

import numpy as np


# ---------------------------------------------------------------------------
# Data contracts (pure data structures — no storage / framework deps)
# ---------------------------------------------------------------------------


@dataclass
class LoadedFrames:
    """Container for decoded video frames.

    Attributes:
        frames: List of BGR numpy arrays (HxWxC, uint8), one per sampled frame.
        indices: Frame indices in the original video (0-based, ascending).
        meta: Video metadata dict (e.g. ``{"fps": 30.0, "width": 1280, ...}``).
        continuous: True if frames were decoded at a fixed fps (not uniformly
            sampled).  Required by the optical-flow scorer.
    """

    frames: list[np.ndarray]
    indices: list[int]
    meta: dict[str, Any] = field(default_factory=dict)
    continuous: bool = False

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx):
        return self.frames[idx]


@dataclass
class MetricResult:
    """Result of a single scorer run.

    Attributes:
        name: Scorer registration name (e.g. ``"aesthetics"``).
        scores: Dict of output fields (the scorer's output contract).
        error: Non-None means degraded — scores contain default/None values
            and the error message explains why.
        elapsed_ms: Wall-clock elapsed time in milliseconds.
    """

    name: str
    scores: dict[str, Any]
    error: str | None = None
    elapsed_ms: float = 0.0

    @property
    def is_success(self) -> bool:
        """True when the scorer succeeded (error is None)."""
        return self.error is None


@dataclass
class VideoQualityReport:
    """Aggregated report for a pipeline run on a single video.

    Attributes:
        video_path: Path of the scored video.
        metrics: ``{scorer_name: MetricResult}`` mapping.
        frame_indices: Sampled frame indices (reproducible).
        device: Actual device string used.
    """

    video_path: str
    metrics: dict[str, MetricResult] = field(default_factory=dict)
    frame_indices: list[int] = field(default_factory=list)
    device: str = ""

    @property
    def all_scores(self) -> dict[str, Any]:
        """Flatten all scorer scores into a single dict."""
        out: dict[str, Any] = {}
        for result in self.metrics.values():
            out.update(result.scores)
        return out

    @property
    def errors(self) -> dict[str, str]:
        """Return ``{scorer_name: error_msg}`` for degraded scorers only."""
        return {n: r.error for n, r in self.metrics.items() if r.error is not None}


# ---------------------------------------------------------------------------
# Configuration contract
# ---------------------------------------------------------------------------


@dataclass
class ScorerConfig:
    """Unified configuration for a quality scorer.

    Attributes:
        device: ``"auto"`` | ``"npu"`` | ``"cpu"`` (no ``"cuda"``).
        sample_frames: Number of frames to uniformly sample.
        frame_batch_size: Inference batch size (None → device default).
        weights: Weight path(s) — ``str`` for single-weight, ``dict[str, str]``
            for multi-weight scorers.
        dtype: ``"fp16"`` or ``"fp32"``.
        extra: Scorer-specific keyword args.
    """

    device: str = "auto"
    sample_frames: int = 8
    frame_batch_size: int | None = None
    weights: str | dict[str, str] | None = None
    dtype: str = "fp16"
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Base scorer ABC
# ---------------------------------------------------------------------------


class BaseQualityScorer(ABC):
    """Abstract base class for all video-quality scorers.

    Subclasses must set ``name`` (a ClassVar matching the registry key) and
    implement ``score_frames``.

    Two entry points:
        - ``score(video_path, **kwargs)``: convenience — decodes then calls
          ``score_frames``.
        - ``score_frames(frames)``: decoupled — consumes already-decoded
          ``LoadedFrames``.

    Model lifecycle: weights are lazily loaded on first use and kept resident
    for subsequent calls (no re-loading).
    """

    name: ClassVar[str] = ""
    requires_continuous_frames: ClassVar[bool] = False

    def __init__(self, config: ScorerConfig | None = None, **kwargs: Any):
        if config is None:
            config = ScorerConfig()
        self._config = config
        self._device: str | None = None
        self._model_loaded = False
        self._extra_kwargs = kwargs

    # -- public API --------------------------------------------------------

    @abstractmethod
    def score_frames(self, frames: LoadedFrames) -> MetricResult:
        """Score already-decoded frames.  Must be implemented by subclasses."""
        pass

    def score(self, video_path: str, **kwargs: Any) -> MetricResult:
        """Convenience entry: decode *video_path* then call ``score_frames``.

        Keyword args override config values for this call only (e.g.
        ``sample_frames=16``).

        The default implementation raises ``NotImplementedError`` until the
        frame_loader module is delivered.  Subclasses that need custom
        decode logic (e.g. streaming, continuous-frame-only scorers) may
        override this method.

        .. note::

           Video frame decoding relies on ``FrameLoader`` (delivered in
           a companion PR).  Until then ``score_frames`` is the primary
           entry point — callers should decode frames externally and pass
           a ``LoadedFrames`` object.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.score() requires the frame_loader module "
            f"(delivered in a companion PR).  Use {type(self).__name__}."
            f"score_frames() instead."
        )

    # -- model lifecycle helpers ------------------------------------------

    def _ensure_model(self) -> None:
        """Lazily load the model on first use (override in subclasses)."""
        if not self._model_loaded:
            self._load_model()
            self._model_loaded = True

    def _load_model(self) -> None:
        """Override to load model weights.  Default: no-op (model-free scorers)."""
        pass

    def close(self) -> None:
        """Release model resources.  Override if cleanup is needed."""
        self._model_loaded = False

    # -- config helpers ----------------------------------------------------

    def _effective_config(self, **kwargs: Any) -> ScorerConfig:
        """Merge kwargs into the stored config, returning a new ScorerConfig."""
        import copy

        cfg = copy.deepcopy(self._config)
        # ``extra`` needs special handling (dict merge, not replacement), so it
        # is popped before the generic field loop below.
        extra_override = kwargs.pop("extra", None)
        for key in _CONFIG_FIELDS:
            if key in kwargs:
                setattr(cfg, key, kwargs.pop(key))
        cfg.extra.update(self._extra_kwargs)
        cfg.extra.update(kwargs)
        if extra_override:
            cfg.extra.update(extra_override)
        return cfg

    @property
    def device(self) -> str:
        """Return the resolved device string (cached after first resolution)."""
        if self._device is None:
            from .device import DeviceResolver

            self._device = DeviceResolver.resolve(self._config.device)
        return self._device

    @property
    def config(self) -> ScorerConfig:
        return self._config

    # -- timing helper -----------------------------------------------------

    @staticmethod
    def _timed():
        """Return a context manager exposing ``elapsed_ms`` (wall-clock, in ms)."""

        class _Timer:
            def __init__(self):
                self.start = 0.0
                self.elapsed_ms = 0.0

            def __enter__(self):
                self.start = time.perf_counter()
                return self

            def __exit__(self, *exc):
                self.elapsed_ms = (time.perf_counter() - self.start) * 1000
                return False

        return _Timer()


# ---------------------------------------------------------------------------
# Scorer registry
# ---------------------------------------------------------------------------


class QualityScorerRegistry:
    """Registration-based factory for quality scorers."""

    _registry: dict[str, type[BaseQualityScorer]] = {}

    @classmethod
    def register(cls, scorer_cls: type[BaseQualityScorer]) -> type[BaseQualityScorer]:
        """Register *scorer_cls* under its ``name`` ClassVar.  Usable as decorator."""
        name = scorer_cls.name
        if not name:
            raise ValueError(f"Cannot register {scorer_cls.__name__}: 'name' ClassVar is empty.")
        cls._registry[name] = scorer_cls
        return scorer_cls

    @classmethod
    def get(cls, name: str) -> type[BaseQualityScorer]:
        """Return the registered scorer class for *name*."""
        if name not in cls._registry:
            raise KeyError(f"Unknown scorer: {name!r}. Registered: {sorted(cls._registry)}")
        return cls._registry[name]

    @classmethod
    def list_scorers(cls) -> list[str]:
        """Return sorted list of registered scorer names."""
        return sorted(cls._registry)

    @classmethod
    def create(cls, name: str, config: dict[str, Any] | None = None) -> BaseQualityScorer:
        """Instantiate a scorer by name from a flat config dict.

        Each key corresponds to a ``ScorerConfig`` field or a scorer-specific
        extra parameter (passed via ``ScorerConfig.extra``).
        """
        flat = dict(config or {})
        scorer_cls = cls.get(name)
        cfg = ScorerConfig(**_filter_config_kwargs(flat))
        return scorer_cls(config=cfg, **_filter_extra_kwargs(flat))


# Single source of truth for the ``ScorerConfig`` fields, derived from the
# dataclass itself so the two can never drift apart.
_CONFIG_FIELDS = tuple(f.name for f in fields(ScorerConfig))


def _filter_config_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    """Extract ScorerConfig-compatible kwargs from a flat dict."""
    return {k: v for k, v in config.items() if k in _CONFIG_FIELDS}


def _filter_extra_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    """Extract scorer-specific kwargs (everything not in ScorerConfig)."""
    return {k: v for k, v in config.items() if k not in _CONFIG_FIELDS}


# Convenience module-level functions
def create_scorer(name: str, config: dict[str, Any] | None = None) -> BaseQualityScorer:
    """Factory: create a scorer instance by registration name."""
    return QualityScorerRegistry.create(name, config)


def list_scorers() -> list[str]:
    """List all registered scorer names."""
    return QualityScorerRegistry.list_scorers()
