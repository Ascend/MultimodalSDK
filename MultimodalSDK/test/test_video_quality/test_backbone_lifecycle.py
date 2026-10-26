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

"""Tests for backbone cache lifecycle — process-level model pool semantics.

Key invariant: ``scorer.close()`` and ``Pipeline.close()`` do NOT evict
cached backbones.  Models stay resident for reuse across Pipeline instances.
Only ``clear_backbone_cache()`` or process exit releases model memory.
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from mm.core.video_quality import (
    VideoQualityPipeline,
)
from mm.core.video_quality.backbones import (
    _cache,
    _cache_key,
    get_shared_clip_backbone,
    clear_cache,
    get_cache_info,
)
import mm.core.video_quality.backbones as backbone_module


@pytest.fixture(autouse=True)
def clean_cache():
    """Ensure a clean cache before and after each test."""
    clear_cache()
    yield
    clear_cache()


@pytest.fixture(autouse=True)
def mock_backbone_class():
    """Replace SharedClipBackbone with a lightweight mock to avoid loading real CLIP.

    Each call to SharedClipBackbone(...) creates a fresh MagicMock so that
    different cache keys get different instances (matching real behavior).
    """

    def _factory(*args, **kwargs):
        inst = MagicMock()
        inst.close = MagicMock()
        inst._model = None
        return inst

    with patch.object(backbone_module, "SharedClipBackbone", side_effect=_factory):
        yield


class TestCacheBasics:
    def test_cache_starts_empty(self):
        assert len(_cache) == 0

    def test_get_creates_cache_entry(self):
        # ``batch_size=None`` resolves to the CPU default (1), which is part of
        # the cache key — pass it explicitly so the expected key is exact.
        get_shared_clip_backbone("/fake/clip", "cpu", "fp16", batch_size=1)
        key = _cache_key("clip", "/fake/clip", "cpu", "fp16", 1)
        assert key in _cache

    def test_get_returns_same_instance(self):
        bb1 = get_shared_clip_backbone("/fake/clip", "cpu", "fp16")
        bb2 = get_shared_clip_backbone("/fake/clip", "cpu", "fp16")
        assert bb1 is bb2

    def test_different_paths_different_entries(self):
        bb1 = get_shared_clip_backbone("/fake/clip_a", "cpu", "fp16")
        bb2 = get_shared_clip_backbone("/fake/clip_b", "cpu", "fp16")
        assert bb1 is not bb2
        assert len(_cache) == 2

    def test_get_cache_info(self):
        get_shared_clip_backbone("/fake/clip", "cpu", "fp16")
        info = get_cache_info()
        assert len(info) == 1
        entry = list(info.values())[0]
        assert entry["model_path"] == "/fake/clip"
        assert entry["device"] == "cpu"


class TestPipelineCloseDoesNotEvict:
    """Core invariant: Pipeline.close() does NOT evict from cache."""

    def test_pipeline_close_keeps_cache(self):
        pipeline = VideoQualityPipeline(
            scorers=["subject_consistency", "background_consistency"],
            device="cpu",
        )
        for s in pipeline._scorers:
            s._ensure_model()
        assert len(_cache) == 1

        pipeline.close()
        assert len(_cache) == 1  # ← STILL IN CACHE

    def test_new_pipeline_reuses_after_close(self):
        """Pattern B: new Pipeline per video — models NOT reloaded."""
        pipeline1 = VideoQualityPipeline(
            scorers=["subject_consistency"],
            device="cpu",
        )
        pipeline1._scorers[0]._ensure_model()
        bb1 = pipeline1._scorers[0]._backbone
        pipeline1.close()

        pipeline2 = VideoQualityPipeline(
            scorers=["subject_consistency"],
            device="cpu",
        )
        pipeline2._scorers[0]._ensure_model()
        bb2 = pipeline2._scorers[0]._backbone
        assert bb1 is bb2  # same cached instance, no re-load


class TestExplicitClear:
    def test_clear_cache_evicts_all(self):
        get_shared_clip_backbone("/fake/clip", "cpu", "fp16")
        assert len(_cache) == 1

        clear_cache()
        assert len(_cache) == 0

    def test_clear_calls_backbone_close(self):
        bb = get_shared_clip_backbone("/fake/clip", "cpu", "fp16")
        clear_cache()
        bb.close.assert_called_once()

    def test_reuse_after_clear_reloads(self):
        """After clear_cache(), next get loads a fresh instance."""
        bb1 = get_shared_clip_backbone("/fake/clip", "cpu", "fp16")
        clear_cache()
        bb2 = get_shared_clip_backbone("/fake/clip", "cpu", "fp16")
        assert bb1 is not bb2  # new instance after clear
