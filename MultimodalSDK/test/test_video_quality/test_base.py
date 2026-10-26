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

"""Tests for base infrastructure: data contracts, config, registry, device, weights."""

from __future__ import annotations

import os
from unittest.mock import patch

import numpy as np
import pytest

from mm.core.video_quality.base import (
    LoadedFrames,
    MetricResult,
    QualityScorerRegistry,
    ScorerConfig,
    VideoQualityReport,
)
from mm.core.video_quality.device import DeviceResolver
from mm.core.video_quality.weights import WeightResolver


# ---------------------------------------------------------------------------
# MetricResult
# ---------------------------------------------------------------------------


class TestMetricResult:
    def test_ok_when_no_error(self):
        r = MetricResult(name="test", scores={"a": 1.0})
        assert r.is_success is True
        assert r.error is None

    def test_not_ok_when_error(self):
        r = MetricResult(name="test", scores={"a": None}, error="failed")
        assert r.is_success is False
        assert r.error == "failed"

    def test_default_elapsed_ms(self):
        r = MetricResult(name="test", scores={})
        assert r.elapsed_ms == 0.0


# ---------------------------------------------------------------------------
# VideoQualityReport
# ---------------------------------------------------------------------------


class TestVideoQualityReport:
    def test_all_scores_flattens(self):
        report = VideoQualityReport(video_path="/v.mp4")
        report.metrics["a"] = MetricResult(name="a", scores={"x": 1.0, "y": 2.0})
        report.metrics["b"] = MetricResult(name="b", scores={"z": 3.0})
        assert report.all_scores == {"x": 1.0, "y": 2.0, "z": 3.0}

    def test_errors_only_degraded(self):
        report = VideoQualityReport(video_path="/v.mp4")
        report.metrics["a"] = MetricResult(name="a", scores={"x": 1.0})
        report.metrics["b"] = MetricResult(name="b", scores={"y": None}, error="fail")
        assert report.errors == {"b": "fail"}


# ---------------------------------------------------------------------------
# LoadedFrames
# ---------------------------------------------------------------------------


class TestLoadedFrames:
    def test_len(self):
        frames = [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(5)]
        lf = LoadedFrames(frames=frames, indices=[0, 1, 2, 3, 4])
        assert len(lf) == 5

    def test_getitem(self):
        frames = [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(3)]
        lf = LoadedFrames(frames=frames, indices=[0, 1, 2])
        assert np.array_equal(lf[0], frames[0])

    def test_continuous_default_false(self):
        lf = LoadedFrames(frames=[], indices=[])
        assert lf.continuous is False


# ---------------------------------------------------------------------------
# ScorerConfig
# ---------------------------------------------------------------------------


class TestScorerConfig:
    def test_defaults(self):
        cfg = ScorerConfig()
        assert cfg.device == "auto"
        assert cfg.sample_frames == 8
        assert cfg.frame_batch_size is None
        assert cfg.weights is None
        assert cfg.dtype == "fp16"

    def test_custom(self):
        cfg = ScorerConfig(device="cpu", sample_frames=16, dtype="fp32")
        assert cfg.device == "cpu"
        assert cfg.sample_frames == 16
        assert cfg.dtype == "fp32"


# ---------------------------------------------------------------------------
# QualityScorerRegistry
# ---------------------------------------------------------------------------


class TestQualityScorerRegistry:
    def test_get_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown scorer"):
            QualityScorerRegistry.get("nonexistent_scorer")


# ---------------------------------------------------------------------------
# DeviceResolver
# ---------------------------------------------------------------------------


class TestDeviceResolver:
    def test_auto_resolves_to_cpu_when_no_npu(self, force_cpu_device):
        assert DeviceResolver.resolve("auto") == "cpu"

    def test_explicit_npu(self):
        assert DeviceResolver.resolve("npu") == "npu"

    def test_explicit_cpu(self):
        assert DeviceResolver.resolve("cpu") == "cpu"

    def test_cuda_raises(self):
        with pytest.raises(ValueError, match="GPU.*not supported"):
            DeviceResolver.resolve("cuda")

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Unsupported device"):
            DeviceResolver.resolve("tpu")

    def test_default_frame_batch_size_cpu(self):
        assert DeviceResolver.default_frame_batch_size("cpu") == 1

    def test_default_frame_batch_size_npu(self):
        assert DeviceResolver.default_frame_batch_size("npu") == 2


# ---------------------------------------------------------------------------
# WeightResolver
# ---------------------------------------------------------------------------


class TestWeightResolver:
    def test_resolve_single_with_explicit_path(self):
        resolver = WeightResolver("watermark")
        result = resolver.resolve_single("/fake/best.pt")
        assert result == "/fake/best.pt"

    def test_resolve_single_from_env(self):
        resolver = WeightResolver("watermark")
        with patch.dict(os.environ, {"WATERMARK_YOLO_MODEL_PATH": "/env/best.pt"}):
            result = resolver.resolve_single(None)
            assert result == "/env/best.pt"

    def test_resolve_single_missing_raises(self):
        resolver = WeightResolver("watermark")
        env = {k: v for k, v in os.environ.items() if k != "WATERMARK_YOLO_MODEL_PATH"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError, match="Weight role 'detector'"):
                resolver.resolve_single(None)

    def test_resolve_multi_with_partial_explicit(self):
        resolver = WeightResolver("naturalness")
        with patch.dict(os.environ, {"NATURAL_WEIGHTS_JSON": "/env/head.json"}):
            result = resolver.resolve_multi({"clip": "/explicit/clip"})
            assert result["clip"] == "/explicit/clip"
            assert result["head"] == "/env/head.json"

    def test_resolve_auto_dispatch_str(self):
        resolver = WeightResolver("watermark")
        assert resolver.resolve("/explicit/path") == "/explicit/path"

    def test_resolve_auto_dispatch_dict(self):
        resolver = WeightResolver("naturalness")
        with patch.dict(os.environ, {"NATURAL_WEIGHTS_JSON": "/env/head.json"}):
            result = resolver.resolve({"clip": "/explicit/clip"})
            assert isinstance(result, dict)
            assert result["clip"] == "/explicit/clip"

    def test_no_weight_scorer_raises(self):
        resolver = WeightResolver("temporal_flickering")
        with pytest.raises(ValueError, match="does not declare any weight roles"):
            resolver.resolve_single(None)
