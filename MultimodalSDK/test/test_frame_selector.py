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

# pylint: skip-file

import os
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import torch

from mm.core.frame_selector.frame_selector import (
    BaseFrameSelector,
    KRangFrameSelector,
    KFrameSelector,
)

MODULE_PATH = "mm.core.frame_selector.frame_selector"

VALID_MODEL_PATH = "/fake/model/path"
VALID_DEVICE_ID = 0


def _make_processor_mock():
    mock = MagicMock()
    mock.return_value.to.return_value = mock
    return mock


def _make_model_mock(feature_dim=512):
    mock = MagicMock()
    mock.eval.return_value = mock
    text_feat = torch.randn(1, feature_dim)
    text_feat = text_feat / text_feat.norm(p=2, dim=-1, keepdim=True)
    mock.get_text_features.return_value = text_feat

    def _make_image_features(n):
        img_feat = torch.randn(n, feature_dim)
        img_feat = img_feat / img_feat.norm(p=2, dim=-1, keepdim=True)
        return img_feat

    mock._make_image_features = _make_image_features
    mock.get_image_features.return_value = _make_image_features(4)
    return mock


def _setup_model_for_frames(model_mock, num_frames, batch_size=64):
    model_mock._call_count = 0
    batch_sizes = []
    for i in range(0, num_frames, batch_size):
        batch_sizes.append(min(batch_size, num_frames - i))

    def _get_image_features(**kwargs):
        model_mock._call_count += 1
        idx = model_mock._call_count - 1
        bs = batch_sizes[idx] if idx < len(batch_sizes) else 4
        return model_mock._make_image_features(bs)

    model_mock.get_image_features.side_effect = _get_image_features
    model_mock._call_count = 0


def _make_frames(num_frames=4, height=672, width=672):
    return [np.random.randint(0, 256, (height, width, 3), dtype=np.uint8) for _ in range(num_frames)]


def _make_normalized_features(n, dim=512):
    feats = torch.randn(n, dim)
    feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
    return feats


@pytest.fixture(autouse=True)
def mock_transformers():
    with (
        patch(f"{MODULE_PATH}.CLIPProcessor") as mock_clip_proc_cls,
        patch(f"{MODULE_PATH}.CLIPModel") as mock_clip_model_cls,
        patch(f"{MODULE_PATH}.ChineseCLIPProcessor") as mock_cn_proc_cls,
        patch(f"{MODULE_PATH}.ChineseCLIPModel") as mock_cn_model_cls,
    ):
        clip_proc = _make_processor_mock()
        cn_proc = _make_processor_mock()
        clip_model = _make_model_mock()
        cn_model = _make_model_mock()

        mock_clip_proc_cls.from_pretrained.return_value = clip_proc
        mock_clip_model_cls.from_pretrained.return_value = clip_model
        mock_cn_proc_cls.from_pretrained.return_value = cn_proc
        mock_cn_model_cls.from_pretrained.return_value = cn_model

        yield {
            "clip_processor_cls": mock_clip_proc_cls,
            "clip_model_cls": mock_clip_model_cls,
            "cn_processor_cls": mock_cn_proc_cls,
            "cn_model_cls": mock_cn_model_cls,
            "clip_processor": clip_proc,
            "clip_model": clip_model,
            "cn_processor": cn_proc,
            "cn_model": cn_model,
        }


@pytest.fixture(autouse=True)
def mock_security():
    with patch.object(BaseFrameSelector, "_validate_model_file_security"):
        yield


@pytest.fixture
def clip_selector():
    return KRangFrameSelector(
        model_path=VALID_MODEL_PATH,
        device_id=VALID_DEVICE_ID,
        model_type="clip",
    )


@pytest.fixture
def cn_clip_selector():
    return KRangFrameSelector(
        model_path=VALID_MODEL_PATH,
        device_id=VALID_DEVICE_ID,
        model_type="cn_clip",
    )


@pytest.fixture
def kframe_selector():
    return KFrameSelector(
        model_path=VALID_MODEL_PATH,
        device_id=VALID_DEVICE_ID,
        model_type="clip",
    )


@pytest.fixture
def sample_frames():
    return _make_frames(10)


@pytest.fixture
def sample_image_features():
    return _make_normalized_features(10)


@pytest.fixture
def sample_query_feature():
    feat = torch.randn(1, 512)
    feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
    return feat


class TestBaseFrameSelectorInit:
    def test_init_clip_model(self, mock_transformers, clip_selector):
        mock_transformers["clip_processor_cls"].from_pretrained.assert_called_once_with(VALID_MODEL_PATH)
        mock_transformers["clip_model_cls"].from_pretrained.assert_called_once()
        assert clip_selector.model_type == "clip"
        assert clip_selector.device_id == VALID_DEVICE_ID
        assert clip_selector.batch_size == 64

    def test_init_cn_clip_model(self, mock_transformers, cn_clip_selector):
        mock_transformers["cn_processor_cls"].from_pretrained.assert_called_once_with(VALID_MODEL_PATH)
        mock_transformers["cn_model_cls"].from_pretrained.assert_called_once()
        assert cn_clip_selector.model_type == "cn_clip"

    def test_default_thresholds(self, mock_transformers, clip_selector):
        assert clip_selector.similar_threshold == 0.025
        assert clip_selector.image_similar_threshold == 0.015
        assert clip_selector.image_size == (672, 672)

    def test_custom_thresholds(self, mock_transformers):
        selector = KRangFrameSelector(
            model_path=VALID_MODEL_PATH,
            device_id=VALID_DEVICE_ID,
            model_type="clip",
            similar_threshold=0.05,
            image_similar_threshold=0.03,
            image_size=(336, 336),
        )
        assert selector.similar_threshold == 0.05
        assert selector.image_similar_threshold == 0.03
        assert selector.image_size == (336, 336)


class TestCheckInputValid:
    def test_empty_model_path(self, mock_transformers):
        with pytest.raises(ValueError, match="model_path must be a non-empty string"):
            KRangFrameSelector(model_path="", device_id=0, model_type="clip")

    def test_whitespace_model_path(self, mock_transformers):
        with pytest.raises(ValueError, match="model_path must be a non-empty string"):
            KRangFrameSelector(model_path="   ", device_id=0, model_type="clip")

    def test_non_string_model_path(self, mock_transformers):
        with pytest.raises(ValueError, match="model_path must be a non-empty string"):
            KRangFrameSelector(model_path=123, device_id=0, model_type="clip")

    def test_non_int_device_id(self, mock_transformers):
        with pytest.raises(TypeError, match="device_id must be an integer"):
            KRangFrameSelector(model_path=VALID_MODEL_PATH, device_id="0", model_type="clip")

    def test_empty_model_type(self, mock_transformers):
        with pytest.raises(ValueError, match="model_type must be a non-empty string"):
            KRangFrameSelector(model_path=VALID_MODEL_PATH, device_id=0, model_type="")

    def test_invalid_model_type(self, mock_transformers):
        with pytest.raises(ValueError, match="model_type must be in"):
            KRangFrameSelector(model_path=VALID_MODEL_PATH, device_id=0, model_type="invalid")

    def test_invalid_similar_threshold_type(self, mock_transformers):
        with pytest.raises(TypeError, match="similar_threshold must be a number"):
            KRangFrameSelector(model_path=VALID_MODEL_PATH, device_id=0, model_type="clip", similar_threshold="high")

    def test_similar_threshold_out_of_range(self, mock_transformers):
        with pytest.raises(ValueError, match="similar_threshold must be in"):
            KRangFrameSelector(model_path=VALID_MODEL_PATH, device_id=0, model_type="clip", similar_threshold=1.5)

    def test_invalid_image_similar_threshold_type(self, mock_transformers):
        with pytest.raises(TypeError, match="image_similar_threshold must be a number"):
            KRangFrameSelector(
                model_path=VALID_MODEL_PATH, device_id=0, model_type="clip", image_similar_threshold="low"
            )

    def test_image_similar_threshold_out_of_range(self, mock_transformers):
        with pytest.raises(ValueError, match="image_similar_threshold must be in"):
            KRangFrameSelector(
                model_path=VALID_MODEL_PATH, device_id=0, model_type="clip", image_similar_threshold=-0.1
            )

    def test_invalid_image_size_type(self, mock_transformers):
        with pytest.raises(ValueError, match="image_size must be a tuple"):
            KRangFrameSelector(model_path=VALID_MODEL_PATH, device_id=0, model_type="clip", image_size=[672, 672])

    def test_image_size_wrong_length(self, mock_transformers):
        with pytest.raises(ValueError, match="image_size must be a tuple"):
            KRangFrameSelector(model_path=VALID_MODEL_PATH, device_id=0, model_type="clip", image_size=(672,))

    def test_image_size_non_int_values(self, mock_transformers):
        with pytest.raises(TypeError, match="image_size values must be integers"):
            KRangFrameSelector(model_path=VALID_MODEL_PATH, device_id=0, model_type="clip", image_size=(672.0, 672.0))

    def test_image_size_out_of_range(self, mock_transformers):
        with pytest.raises(ValueError, match="image_size width and height must be in"):
            KRangFrameSelector(model_path=VALID_MODEL_PATH, device_id=0, model_type="clip", image_size=(5, 5))


class TestValidateModelFileSecurity:
    @pytest.fixture(autouse=True)
    def mock_security(self):
        yield

    @pytest.fixture
    def security_mocks(self):
        if not hasattr(os, "getuid"):
            pytest.skip("os.getuid not available on this platform")

        with (
            patch(f"{MODULE_PATH}.os.path.exists", return_value=True) as mock_exists,
            patch(f"{MODULE_PATH}.os.path.isdir", return_value=True) as mock_isdir,
            patch(f"{MODULE_PATH}.os.stat") as mock_stat,
            patch(f"{MODULE_PATH}.os.getuid", return_value=1000),
        ):
            mock_stat_result = MagicMock()
            mock_stat_result.st_uid = 1000
            mock_stat_result.st_mode = 0o40750
            mock_stat.return_value = mock_stat_result
            yield {
                "exists": mock_exists,
                "isdir": mock_isdir,
                "stat": mock_stat,
            }

    def test_path_not_exists(self, mock_transformers, security_mocks):
        security_mocks["exists"].return_value = False
        with pytest.raises(FileNotFoundError, match="model_path does not exist"):
            KRangFrameSelector(model_path="/nonexistent", device_id=0, model_type="clip")

    def test_path_not_directory(self, mock_transformers, security_mocks):
        security_mocks["isdir"].return_value = False
        with pytest.raises(ValueError, match="model_path must be a directory"):
            KRangFrameSelector(model_path=VALID_MODEL_PATH, device_id=0, model_type="clip")

    def test_owner_mismatch(self, mock_transformers, security_mocks):
        with patch(f"{MODULE_PATH}.os.getuid", return_value=9999):
            with pytest.raises(PermissionError, match="does not match current user"):
                KRangFrameSelector(model_path=VALID_MODEL_PATH, device_id=0, model_type="clip")

    def test_wrong_permissions(self, mock_transformers, security_mocks):
        mock_stat_result = MagicMock()
        mock_stat_result.st_uid = 1000
        mock_stat_result.st_mode = 0o40755
        security_mocks["stat"].return_value = mock_stat_result
        with pytest.raises(PermissionError, match="permissions must be 750"):
            KRangFrameSelector(model_path=VALID_MODEL_PATH, device_id=0, model_type="clip")

    def test_valid_security(self, mock_transformers, security_mocks):
        selector = KRangFrameSelector(model_path=VALID_MODEL_PATH, device_id=0, model_type="clip")
        assert selector is not None


class TestValidateSelectInputs:
    def test_empty_query(self, clip_selector):
        with pytest.raises(ValueError, match="query and frames must not be empty"):
            clip_selector._validate_select_inputs("", [np.zeros((672, 672, 3))], 5)

    def test_empty_frames(self, clip_selector):
        with pytest.raises(ValueError, match="query and frames must not be empty"):
            clip_selector._validate_select_inputs("query", [], 5)

    def test_non_string_query(self, clip_selector):
        with pytest.raises(TypeError, match="query must be string"):
            clip_selector._validate_select_inputs(123, [np.zeros((672, 672, 3))], 5)

    def test_non_positive_sample_num(self, clip_selector):
        with pytest.raises(ValueError, match="sample_num must be positive integer"):
            clip_selector._validate_select_inputs("query", [np.zeros((672, 672, 3))], 0)

    def test_negative_sample_num(self, clip_selector):
        with pytest.raises(ValueError, match="sample_num must be positive integer"):
            clip_selector._validate_select_inputs("query", [np.zeros((672, 672, 3))], -1)

    def test_sample_num_capped_by_frame_count(self, clip_selector):
        frames = [np.zeros((672, 672, 3))] * 3
        result = clip_selector._validate_select_inputs("query", frames, 100)
        assert result == 3

    def test_valid_inputs(self, clip_selector):
        frames = [np.zeros((672, 672, 3))] * 10
        result = clip_selector._validate_select_inputs("query", frames, 5)
        assert result == 5


class TestExtractFeatures:
    def test_extract_query_features(self, clip_selector, mock_transformers):
        clip_selector.processor = mock_transformers["clip_processor"]
        clip_selector.model = mock_transformers["clip_model"]
        clip_selector.device = "cpu"

        result = clip_selector._extract_query_features("test query")
        assert result.shape[0] == 1
        norm = result.norm(p=2, dim=-1)
        assert torch.allclose(norm, torch.ones_like(norm), atol=1e-5)

    def test_extract_image_features(self, clip_selector, mock_transformers):
        clip_selector.processor = mock_transformers["clip_processor"]
        clip_selector.model = mock_transformers["clip_model"]
        clip_selector.device = "cpu"
        frames = _make_frames(4)
        _setup_model_for_frames(mock_transformers["clip_model"], 4)

        result = clip_selector._extract_image_features(frames)
        assert result.shape[0] == 4
        norms = result.norm(p=2, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_extract_image_features_batching(self, clip_selector, mock_transformers):
        clip_selector.processor = mock_transformers["clip_processor"]
        clip_selector.model = mock_transformers["clip_model"]
        clip_selector.device = "cpu"
        clip_selector.batch_size = 2
        _setup_model_for_frames(mock_transformers["clip_model"], 6, batch_size=2)
        frames = _make_frames(6)

        result = clip_selector._extract_image_features(frames)
        assert result.shape[0] == 6

    def test_extract_features_returns_both(self, clip_selector, mock_transformers):
        clip_selector.processor = mock_transformers["clip_processor"]
        clip_selector.model = mock_transformers["clip_model"]
        clip_selector.device = "cpu"
        frames = _make_frames(4)
        _setup_model_for_frames(mock_transformers["clip_model"], 4)

        query_feat, img_feat = clip_selector._extract_features("query", frames)
        assert query_feat.shape[0] == 1
        assert img_feat.shape[0] == 4


class TestComputeSimilarities:
    def test_compute_similarities_shape(self, clip_selector, sample_frames, mock_transformers):
        clip_selector.processor = mock_transformers["clip_processor"]
        clip_selector.model = mock_transformers["clip_model"]
        clip_selector.device = "cpu"
        _setup_model_for_frames(mock_transformers["clip_model"], len(sample_frames))

        query_feat, img_feat, sims = clip_selector._compute_similarities("query", sample_frames)
        assert query_feat.shape[0] == 1
        assert img_feat.shape[0] == len(sample_frames)
        assert sims.shape[0] == len(sample_frames)

    def test_similarities_in_range(self, clip_selector, sample_frames, mock_transformers):
        clip_selector.processor = mock_transformers["clip_processor"]
        clip_selector.model = mock_transformers["clip_model"]
        clip_selector.device = "cpu"
        _setup_model_for_frames(mock_transformers["clip_model"], len(sample_frames))

        _, _, sims = clip_selector._compute_similarities("query", sample_frames)
        assert (sims >= -1.0).all() and (sims <= 1.0).all()


class TestFindBoundary:
    def test_find_left_at_start(self, clip_selector, sample_image_features):
        result = clip_selector._find_left(sample_image_features, 0)
        assert result == 0

    def test_find_right_at_end(self, clip_selector, sample_image_features):
        result = clip_selector._find_right(sample_image_features, len(sample_image_features) - 1)
        assert result == len(sample_image_features) - 1

    def test_find_left_returns_int(self, clip_selector, sample_image_features):
        result = clip_selector._find_left(sample_image_features, 5)
        assert isinstance(result, int)
        assert 0 <= result <= 5

    def test_find_right_returns_int(self, clip_selector, sample_image_features):
        result = clip_selector._find_right(sample_image_features, 5)
        assert isinstance(result, int)
        assert 5 <= result < len(sample_image_features)

    def test_find_boundary_left_returns_zero_on_no_boundary(self, clip_selector):
        uniform_feats = _make_normalized_features(20)
        clip_selector.image_similar_threshold = 999.0
        result = clip_selector._find_boundary(uniform_feats, 10, direction='left')
        assert result == 0

    def test_find_boundary_right_returns_last_on_no_boundary(self, clip_selector):
        uniform_feats = _make_normalized_features(20)
        clip_selector.image_similar_threshold = 999.0
        result = clip_selector._find_boundary(uniform_feats, 5, direction='right')
        assert result == 19

    def test_find_boundary_detects_scene_change(self, clip_selector):
        feats_part1 = _make_normalized_features(10)
        feats_part2 = _make_normalized_features(10)
        feats = torch.cat([feats_part1, feats_part2], dim=0)
        clip_selector.image_similar_threshold = 0.001

        left = clip_selector._find_boundary(feats, 15, direction='left')
        assert 0 <= left <= 15

    def test_find_boundary_max_span(self, clip_selector):
        feats = _make_normalized_features(500)
        clip_selector.image_similar_threshold = 0.0
        result = clip_selector._find_boundary(feats, 250, direction='left', max_span=50)
        assert result >= 200


class TestKRangFrameSelector:
    def test_get_sample_num_long_interval(self):
        assert KRangFrameSelector._get_sample_num(30) == 12
        assert KRangFrameSelector._get_sample_num(100) == 12

    def test_get_sample_num_short_interval(self):
        assert KRangFrameSelector._get_sample_num(29) == 6
        assert KRangFrameSelector._get_sample_num(5) == 6

    def test_adaptive_resample_empty_interval(self, clip_selector, sample_image_features, sample_query_feature):
        result = KRangFrameSelector._adaptive_resample(
            [], sample_image_features, sample_query_feature, base_sample_num=5
        )
        assert result == []

    def test_adaptive_resample_interval_exceeds_quota(self, clip_selector, sample_image_features, sample_query_feature):
        interval = list(range(10))
        result = KRangFrameSelector._adaptive_resample(
            interval, sample_image_features, sample_query_feature, base_sample_num=3
        )
        assert len(result) <= 3
        assert result == sorted(result)

    def test_adaptive_resample_within_quota(self, clip_selector, sample_image_features, sample_query_feature):
        interval = [2, 5]
        result = KRangFrameSelector._adaptive_resample(
            interval, sample_image_features, sample_query_feature, base_sample_num=6
        )
        assert len(result) <= 6
        assert result == sorted(result)

    def test_adaptive_resample_no_extra_quota(self, clip_selector, sample_image_features, sample_query_feature):
        interval = [1, 2, 3]
        result = KRangFrameSelector._adaptive_resample(
            interval, sample_image_features, sample_query_feature, base_sample_num=3
        )
        assert len(result) <= 3

    def test_select_keyframes_empty_query(self, clip_selector, sample_frames):
        with pytest.raises(ValueError, match="query and frames must not be empty"):
            clip_selector.select_keyframes("", sample_frames, 5)

    def test_select_keyframes_empty_frames(self, clip_selector):
        with pytest.raises(ValueError, match="query and frames must not be empty"):
            clip_selector.select_keyframes("query", [], 5)

    def test_select_keyframes_returns_sorted_unique(self, clip_selector, sample_frames, mock_transformers):
        clip_selector.processor = mock_transformers["clip_processor"]
        clip_selector.model = mock_transformers["clip_model"]
        clip_selector.device = "cpu"
        _setup_model_for_frames(mock_transformers["clip_model"], len(sample_frames))

        indices, frames = clip_selector.select_keyframes("query", sample_frames, 5)
        assert indices == sorted(indices)
        assert indices == sorted(set(indices))
        assert len(indices) == len(frames)

    def test_select_keyframes_respects_sample_num(self, clip_selector, sample_frames, mock_transformers):
        clip_selector.processor = mock_transformers["clip_processor"]
        clip_selector.model = mock_transformers["clip_model"]
        clip_selector.device = "cpu"
        _setup_model_for_frames(mock_transformers["clip_model"], len(sample_frames))

        indices, frames = clip_selector.select_keyframes("query", sample_frames, 3)
        assert len(indices) <= 3

    def test_select_keyframes_no_resample(self, clip_selector, sample_frames, mock_transformers):
        clip_selector.processor = mock_transformers["clip_processor"]
        clip_selector.model = mock_transformers["clip_model"]
        clip_selector.device = "cpu"
        _setup_model_for_frames(mock_transformers["clip_model"], len(sample_frames))

        indices, frames = clip_selector.select_keyframes("query", sample_frames, 5, do_resample=False)
        assert indices == sorted(indices)
        assert len(indices) == len(frames)

    def test_select_diverse_frames_returns_intervals(self, clip_selector, sample_image_features, sample_query_feature):
        query_sims = torch.rand(len(sample_image_features))
        result = clip_selector._select_diverse_frames(query_sims, sample_image_features, 3)
        assert isinstance(result, list)
        for interval in result:
            assert isinstance(interval, list)
            assert len(interval) == 3

    def test_select_diverse_frames_stops_at_threshold(self, clip_selector, sample_image_features):
        query_sims = torch.ones(len(sample_image_features)) * 0.01
        query_sims[0] = 0.9
        clip_selector.similar_threshold = 0.025

        result = clip_selector._select_diverse_frames(query_sims, sample_image_features, 10)
        assert len(result) >= 1

    def test_merge_intervals_empty(self, clip_selector):
        result = clip_selector._merge_intervals([])
        assert result == []

    def test_merge_intervals_single(self, clip_selector):
        result = clip_selector._merge_intervals([[0, 5, 10]])
        assert len(result) == 1
        assert result[0] == [0, 10]

    def test_merge_intervals_adjacent_similar(self, clip_selector, sample_image_features):
        intervals = [[0, 3, 6], [8, 10, 14]]
        query_sims = torch.ones(len(sample_image_features)) * 0.5
        clip_selector.image_similar_threshold = 1.0

        result = clip_selector._merge_intervals(intervals, threshold=30, query_similarities=query_sims)
        assert len(result) >= 1

    def test_merge_intervals_far_apart(self, clip_selector, sample_image_features):
        intervals = [[0, 2, 4], [50, 55, 60]]
        query_sims = torch.ones(100) * 0.5

        result = clip_selector._merge_intervals(intervals, threshold=5, query_similarities=query_sims)
        assert len(result) == 2

    def test_merge_intervals_semantically_different(self, clip_selector, sample_image_features):
        intervals = [[0, 2, 6], [8, 10, 14]]
        query_sims = torch.ones(len(sample_image_features)) * 0.5
        query_sims[0:7] = 0.9
        query_sims[8:15] = 0.1
        clip_selector.image_similar_threshold = 0.001

        result = clip_selector._merge_intervals(intervals, threshold=30, query_similarities=query_sims)
        assert len(result) == 2

    def test_resample_respects_budget(self, clip_selector, sample_image_features, sample_query_feature):
        intervals = [[0, 5], [8, 14]]
        result = clip_selector._resample(intervals, sample_image_features, sample_query_feature, sample_num=4)
        assert len(result) <= 4

    def test_resample_empty_intervals(self, clip_selector, sample_image_features, sample_query_feature):
        result = clip_selector._resample([], sample_image_features, sample_query_feature, sample_num=5)
        assert result == []


class TestKFrameSelector:
    def test_select_keyframes_empty_query(self, kframe_selector, sample_frames):
        with pytest.raises(ValueError, match="query and frames must not be empty"):
            kframe_selector.select_keyframes("", sample_frames, 5)

    def test_select_keyframes_empty_frames(self, kframe_selector):
        with pytest.raises(ValueError, match="query and frames must not be empty"):
            kframe_selector.select_keyframes("query", [], 5)

    def test_select_keyframes_returns_sorted_unique(self, kframe_selector, sample_frames, mock_transformers):
        kframe_selector.processor = mock_transformers["clip_processor"]
        kframe_selector.model = mock_transformers["clip_model"]
        kframe_selector.device = "cpu"
        _setup_model_for_frames(mock_transformers["clip_model"], len(sample_frames))

        indices, frames = kframe_selector.select_keyframes("query", sample_frames, 5)
        assert indices == sorted(indices)
        assert indices == sorted(set(indices))
        assert len(indices) == len(frames)

    def test_select_keyframes_respects_sample_num(self, kframe_selector, sample_frames, mock_transformers):
        kframe_selector.processor = mock_transformers["clip_processor"]
        kframe_selector.model = mock_transformers["clip_model"]
        kframe_selector.device = "cpu"
        _setup_model_for_frames(mock_transformers["clip_model"], len(sample_frames))

        indices, frames = kframe_selector.select_keyframes("query", sample_frames, 3)
        assert len(indices) <= 3

    def test_select_diverse_frames_returns_indices(self, kframe_selector, sample_image_features):
        query_sims = torch.rand(len(sample_image_features))
        result = kframe_selector._select_diverse_frames(query_sims, sample_image_features, 5)
        assert isinstance(result, list)
        for idx in result:
            assert isinstance(idx, int)
            assert 0 <= idx < len(sample_image_features)

    def test_select_diverse_frames_first_is_argmax(self, kframe_selector, sample_image_features):
        query_sims = torch.zeros(len(sample_image_features))
        query_sims[7] = 1.0

        result = kframe_selector._select_diverse_frames(query_sims, sample_image_features, 5)
        assert 7 in result

    def test_select_diverse_frames_diversity_filter(self, kframe_selector):
        feat_dim = 512
        n_frames = 20
        feats = _make_normalized_features(n_frames, feat_dim)
        query_sims = torch.linspace(0.9, 0.1, n_frames)

        result = kframe_selector._select_diverse_frames(query_sims, feats, 10)
        assert len(result) >= 1

    def test_select_keyframes_single_frame(self, kframe_selector, mock_transformers):
        kframe_selector.processor = mock_transformers["clip_processor"]
        kframe_selector.model = mock_transformers["clip_model"]
        kframe_selector.device = "cpu"
        _setup_model_for_frames(mock_transformers["clip_model"], 1)
        single_frame = [np.random.randint(0, 256, (672, 672, 3), dtype=np.uint8)]

        indices, frames = kframe_selector.select_keyframes("query", single_frame, 1)
        assert len(indices) <= 1
        if indices:
            assert indices[0] == 0


class TestInitModelDispatch:
    def test_unsupported_model_type_raises(self, mock_transformers):
        with patch.object(BaseFrameSelector, "_validate_model_file_security"):
            selector = KRangFrameSelector.__new__(KRangFrameSelector)
            selector.model_path = VALID_MODEL_PATH
            selector.device_id = 0
            selector.model_type = "clip"
            selector.batch_size = 64
            selector.similar_threshold = 0.025
            selector.image_similar_threshold = 0.015
            selector.image_size = (672, 672)
            selector.device = "npu:0"
            selector.model_initializers = {
                "clip": selector._init_clip_model,
                "cn_clip": selector._init_cn_clip_model,
            }

            selector.model_type = "unsupported"
            with pytest.raises(ValueError, match="Unsupported model_type"):
                selector._init_model()


class TestEdgeCases:
    def test_select_keyframes_with_none_query(self, clip_selector, sample_frames):
        with pytest.raises(ValueError, match="query and frames must not be empty"):
            clip_selector._validate_select_inputs(None, sample_frames, 5)

    def test_select_keyframes_with_zero_frames(self, clip_selector):
        with pytest.raises(ValueError, match="query and frames must not be empty"):
            clip_selector._validate_select_inputs("query", [], 5)

    def test_find_boundary_with_single_frame(self, clip_selector):
        feats = _make_normalized_features(1)
        left = clip_selector._find_left(feats, 0)
        right = clip_selector._find_right(feats, 0)
        assert left == 0
        assert right == 0

    def test_adaptive_resample_single_frame_interval(self, clip_selector, sample_image_features, sample_query_feature):
        interval = [5]
        result = KRangFrameSelector._adaptive_resample(
            interval, sample_image_features, sample_query_feature, base_sample_num=5
        )
        assert 5 in result

    def test_merge_intervals_preserves_order(self, clip_selector, sample_image_features):
        intervals = [[5, 8, 12], [0, 2, 4], [20, 22, 25]]
        query_sims = torch.ones(30) * 0.5
        clip_selector.image_similar_threshold = 1.0

        result = clip_selector._merge_intervals(intervals, threshold=5, query_similarities=query_sims)
        for i in range(len(result) - 1):
            assert result[i][0] < result[i + 1][0]

    def test_krang_select_keyframes_capped_sample_num(self, clip_selector, mock_transformers):
        clip_selector.processor = mock_transformers["clip_processor"]
        clip_selector.model = mock_transformers["clip_model"]
        clip_selector.device = "cpu"
        _setup_model_for_frames(mock_transformers["clip_model"], 3)
        frames = _make_frames(3)

        indices, frames_out = clip_selector.select_keyframes("query", frames, 100)
        assert len(indices) <= 3
