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
"""Mock-level unit tests for the on-disk KTS analysis cache (``KtsCache``).

Pure cache-behavior tests (design doc §8.3): lazy key / path layout, meta +
times + embeds roundtrip, corruption / interval-mismatch misses, no-cache
semantics and frame-directory housekeeping. No ``_acc`` extension, video
decoding or embedding services are required.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from mm.core.segmenter.kts_cache import CACHE_DIR_NAME, KtsCache  # noqa: E402


@pytest.fixture(name="video")
def _video(tmp_path):
    p = tmp_path / "video.mp4"
    p.write_bytes(b"fake-mp4-content")
    return p


def make_cache(video, **kwargs):
    defaults = dict(backend_id="fake:v1", sample_interval_sec=1.0)
    defaults.update(kwargs)
    return KtsCache(str(video), **defaults)


# --------------------------------------------------------------------------- #
# key / paths (lazy)
# --------------------------------------------------------------------------- #
class TestKeyAndPaths:
    def test_constructor_side_effect_free(self, tmp_path):
        # no stat / mkdir until first use (design doc §8.3)
        cache = make_cache(tmp_path / "missing.mp4")
        assert not cache.base.is_dir()

    def test_key_lazy_and_stable(self, video):
        c1 = make_cache(video)
        c2 = make_cache(video)
        assert c1._key is None
        assert c1.key == c2.key
        assert len(c1.key) == 16

    def test_key_changes_with_backend_id(self, video):
        assert make_cache(video, backend_id="fake:v1").key != make_cache(video, backend_id="other:v2").key

    def test_key_changes_with_interval(self, video):
        assert make_cache(video, sample_interval_sec=1.0).key != make_cache(video, sample_interval_sec=2.0).key

    def test_key_changes_with_file_content(self, video):
        k1 = make_cache(video).key
        video.write_bytes(b"fake-mp4-content-changed")
        assert make_cache(video).key != k1

    def test_key_invalidates_on_replacement_same_path(self, video):
        k1 = make_cache(video).key
        video.unlink()
        video.write_bytes(b"rebuilt-content")
        assert make_cache(video).key != k1

    def test_default_cache_dir_is_video_dir_subfolder(self, video):
        cache = make_cache(video)
        assert cache.base == video.parent / CACHE_DIR_NAME

    def test_explicit_cache_dir_respected(self, video, tmp_path):
        custom = tmp_path / "custom_cache"
        cache = make_cache(video, cache_dir=str(custom))
        assert cache.base == custom

    def test_path_layout_names(self, video):
        cache = make_cache(video)
        key = cache.key
        assert cache.meta_path.name == f"kts_{key}_meta.json"
        assert cache.times_path.name == f"kts_{key}_times.npy"
        assert cache.embeds_path.name == f"kts_{key}_embeds.npy"
        assert cache.frame_dir.name == f"frames_{key}"


# --------------------------------------------------------------------------- #
# read / write roundtrip and misses
# --------------------------------------------------------------------------- #
class TestSaveLoad:
    def test_roundtrip(self, video):
        cache = make_cache(video)
        times = [0.0, 1.0, 2.0]
        embeds = np.arange(6, dtype=np.float32).reshape(3, 2)
        cache.save(times, embeds, fps=25.0, total_duration=2.5)

        loaded_times, loaded = make_cache(video).load()
        assert loaded_times == times
        assert isinstance(loaded_times[0], float)
        np.testing.assert_array_equal(loaded, embeds)

    def test_meta_content(self, video):
        cache = make_cache(video)
        cache.save([0.0], np.zeros((1, 4), dtype=np.float32), fps=25.0, total_duration=1.0)
        with open(cache.meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        assert meta["sample_interval_sec"] == 1.0
        assert meta["fps"] == 25.0
        assert meta["total_duration"] == 1.0
        assert meta["n_frames"] == 1
        assert meta["embed_dim"] == 4

    def test_load_none_before_save(self, video):
        assert make_cache(video).load() is None

    def test_load_none_on_partial_files(self, video):
        cache = make_cache(video)
        cache.save([0.0], np.zeros((1, 2), dtype=np.float32), 25.0, 1.0)
        cache.times_path.unlink()
        assert make_cache(video).load() is None

    def test_load_none_on_interval_mismatch(self, video):
        cache = make_cache(video)
        cache.save([0.0], np.zeros((1, 2), dtype=np.float32), 25.0, 1.0)
        with open(cache.meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        meta["sample_interval_sec"] = 9.9
        with open(cache.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f)
        assert make_cache(video).load() is None

    def test_load_none_on_corrupted_meta(self, video):
        cache = make_cache(video)
        cache.save([0.0], np.zeros((1, 2), dtype=np.float32), 25.0, 1.0)
        cache.meta_path.write_text("{not-json", encoding="utf-8")
        assert make_cache(video).load() is None

    def test_load_none_on_length_mismatch(self, video):
        cache = make_cache(video)
        cache.save([0.0, 1.0], np.zeros((2, 2), dtype=np.float32), 25.0, 2.0)
        np.save(cache.times_path, np.asarray([0.0, 1.0, 2.0], dtype=np.float64))
        assert make_cache(video).load() is None

    def test_save_embeds_none_writes_times_only(self, video):
        cache = make_cache(video)
        cache.save([0.0], None, 25.0, 1.0)
        assert cache.meta_path.is_file()
        assert cache.times_path.is_file()
        assert not cache.embeds_path.is_file()
        assert make_cache(video).load() is None  # embeds missing -> miss

    def test_use_cache_false_skips_write_and_read(self, video):
        writer = make_cache(video)
        writer.save([0.0], np.zeros((1, 2), dtype=np.float32), 25.0, 1.0)

        disabled = make_cache(video, use_cache=False)
        disabled.save([1.0], np.ones((1, 2), dtype=np.float32), 25.0, 1.0)
        assert disabled.load() is None
        # the valid writer cache is left intact for normal readers
        assert make_cache(video).load() is not None


# --------------------------------------------------------------------------- #
# frame directory housekeeping
# --------------------------------------------------------------------------- #
class TestFrameDir:
    def test_frame_paths_naming(self, video):
        cache = make_cache(video)
        paths = cache.frame_paths(3)
        assert [Path(p).name for p in paths] == ["f000000.jpg", "f000001.jpg", "f000002.jpg"]

    def test_ensure_frame_dir_creates(self, video):
        cache = make_cache(video)
        assert not cache.frame_dir.exists()
        assert cache.ensure_frame_dir() == cache.frame_dir
        assert cache.frame_dir.is_dir()

    def test_frame_paths_if_complete(self, video):
        cache = make_cache(video)
        cache.ensure_frame_dir()
        expected = cache.frame_paths(2)
        for p in expected:
            Path(p).write_bytes(b"jpeg")
        assert cache.frame_paths_if_complete(2) == expected
        # incomplete (cleaned) directory -> []
        assert cache.frame_paths_if_complete(3) == []

    def test_clean_frames_removes_all(self, video):
        cache = make_cache(video)
        cache.ensure_frame_dir()
        for p in cache.frame_paths(2):
            Path(p).write_bytes(b"jpeg")
        cache.clean_frames()
        assert not list(cache.frame_dir.glob("*.jpg"))
