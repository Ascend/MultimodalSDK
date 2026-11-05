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

"""Tests for ImageQualityScorer (LIQE + NIQE/BRISQUE, mocked pyiqa)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import torch
from PIL import Image

from mm.core.video_quality import ImageQualityScorer, ScorerConfig
from mm.core.video_quality.base import LoadedFrames
from .conftest import make_loaded_frames


MODULE_PATH = "mm.core.video_quality.scorers.image_quality_scorer"


@pytest.fixture(autouse=True)
def mock_weight_resolver(tmp_path, monkeypatch):
    liqe_path = tmp_path / "liqe_mix.pt"
    torch.save({"params": {"weight": torch.tensor(1.0)}}, liqe_path)
    monkeypatch.setenv("LIQE_WEIGHTS_SHA256", hashlib.sha256(liqe_path.read_bytes()).hexdigest())

    clip_path = tmp_path / "ViT-B-32.pt"
    clip_path.write_bytes(b"mock local CLIP checkpoint")
    monkeypatch.setenv("PYIQA_CLIP_VITB32", str(clip_path))
    monkeypatch.setenv("PYIQA_CLIP_VITB32_SHA256", hashlib.sha256(clip_path.read_bytes()).hexdigest())

    text_features_path = tmp_path / "liqe_text_feat_mix.pt"
    torch.save(torch.zeros(11, 2), text_features_path)
    monkeypatch.setenv("PYIQA_LIQE_TEXT_FEAT", str(text_features_path))

    with patch(f"{MODULE_PATH}.WeightResolver") as MockResolver:
        resolver = MagicMock()
        resolver.resolve_single.return_value = str(liqe_path)
        MockResolver.return_value = resolver
        yield resolver


@pytest.fixture(autouse=True, name="mock_pyiqa")
def _mock_pyiqa_fixture(tmp_path, monkeypatch):
    """Mock pyiqa.create_metric and its LIQE loader modules."""
    mock_pyiqa = ModuleType("pyiqa")
    mock_pyiqa.__path__ = []
    mock_archs = ModuleType("pyiqa.archs")
    mock_archs.__path__ = []
    mock_liqe_arch = ModuleType("pyiqa.archs.liqe_arch")
    mock_liqe_arch.load = MagicMock()
    mock_liqe_arch.load_pretrained_network = MagicMock()
    mock_liqe_arch.DEFAULT_CACHE_DIR = str(tmp_path / "original-pyiqa-cache")
    mock_liqe_arch.torch = torch

    mock_clip_model = ModuleType("pyiqa.archs.clip_model")
    mock_clip_model.build_model = MagicMock(return_value=MagicMock())
    fake_scripted_model = MagicMock()
    fake_scripted_model.eval.return_value = fake_scripted_model
    fake_scripted_model.state_dict.return_value = {}
    monkeypatch.setattr(torch.jit, "load", MagicMock(return_value=fake_scripted_model))

    def _create_metric(name, **kwargs):
        metric = MagicMock()
        if name == "liqe_mix":

            def _liqe_call(img):
                # Return a 1D tensor of 11 distortion probabilities
                return torch.tensor([[0.15, 0.0, 0.0, 0.25, 0.0, 0.0, 0.10, 0.50, 0.0, 0.0, 0.0]])

            metric.side_effect = _liqe_call
            metric.net = SimpleNamespace(text_features=None)
        elif name == "niqe":
            metric.side_effect = lambda img: torch.tensor(5.0)
        elif name == "brisque":
            metric.side_effect = lambda img: torch.tensor(10.0)
        return metric

    mock_pyiqa.create_metric = MagicMock(side_effect=_create_metric)

    with patch.dict(
        "sys.modules",
        {
            "pyiqa": mock_pyiqa,
            "pyiqa.archs": mock_archs,
            "pyiqa.archs.liqe_arch": mock_liqe_arch,
            "pyiqa.archs.clip_model": mock_clip_model,
        },
    ):
        yield mock_pyiqa


class TestImageQualityScorerInit:
    def test_name(self):
        scorer = ImageQualityScorer(config=ScorerConfig(device="cpu"))
        assert scorer.name == "image_quality"


class TestImageQualityScoring:
    def test_score_frames_returns_metric(self, mock_pyiqa):
        scorer = ImageQualityScorer(config=ScorerConfig(device="cpu", frame_batch_size=4))
        frames = make_loaded_frames(4, 224, 224)
        fixed_probs = np.array([0.15, 0.0, 0.0, 0.25, 0.0, 0.0, 0.10, 0.50, 0.0, 0.0, 0.0])
        with patch.object(
            ImageQualityScorer,
            "_liqe_dist_probs_batch",
            return_value=np.tile(fixed_probs, (4, 1)),
        ):
            result = scorer.score_frames(frames)
        assert result.name == "image_quality"
        assert result.is_success
        assert "score_blur_mean" in result.scores
        assert "score_dark_mean" in result.scores
        assert "score_overexposure_mean" in result.scores
        assert result.scores["score_blur_mean"] == pytest.approx(0.25)
        assert result.scores["score_dark_mean"] == pytest.approx(0.50)
        assert result.scores["score_overexposure_mean"] == pytest.approx(0.10)
        assert result.scores["score_blur_std"] == pytest.approx(0.0)

    def test_empty_frames_returns_empty(self, mock_pyiqa):
        scorer = ImageQualityScorer(config=ScorerConfig(device="cpu"))
        frames = LoadedFrames(frames=[], indices=[])
        result = scorer.score_frames(frames)
        assert not result.scores

    def test_error_degrades_gracefully(self, mock_pyiqa):
        scorer = ImageQualityScorer(config=ScorerConfig(device="cpu"))
        scorer._model = None
        scorer._liqe_model = None
        scorer._model_loaded = False
        with patch.object(ImageQualityScorer, "_load_model", side_effect=RuntimeError("load error")):
            frames = make_loaded_frames(4, 224, 224)
            result = scorer.score_frames(frames)
            assert not result.is_success
            assert result.error == "image_quality_inference_failed"
            assert not result.scores

    def test_liqe_uses_configured_frame_batches(self, mock_pyiqa):
        scorer = ImageQualityScorer(config=ScorerConfig(device="cpu", frame_batch_size=2, extra={"metrics": ("liqe",)}))
        frames = make_loaded_frames(5, 224, 224)
        probs = np.array([0.15, 0.0, 0.0, 0.25, 0.0, 0.0, 0.10, 0.50, 0.0, 0.0, 0.0])
        with patch.object(
            ImageQualityScorer,
            "_liqe_dist_probs_batch",
            side_effect=lambda batch: np.tile(probs, (len(batch), 1)),
        ) as batch_infer:
            result = scorer.score_frames(frames)
        assert result.is_success
        assert [len(call.args[0]) for call in batch_infer.call_args_list] == [2, 2, 1]

    @pytest.mark.parametrize("metric_name", ["niqe", "brisque"])
    def test_metric_failed_frames_are_excluded(self, mock_pyiqa, metric_name):
        scorer = ImageQualityScorer(config=ScorerConfig(device="cpu", extra={"metrics": (metric_name,)}))
        scorer._model_loaded = True
        setattr(
            scorer,
            f"_{metric_name}_metric",
            MagicMock(side_effect=[10.0, RuntimeError("frame failure"), 20.0]),
        )
        result = scorer.score_frames(make_loaded_frames(3, 224, 224))
        assert result.is_success
        assert result.scores[metric_name] == pytest.approx(15.0)
        assert result.scores[f"{metric_name}_failed_frames"] == 1

    @pytest.mark.parametrize("metric_name", ["niqe", "brisque"])
    def test_all_metric_frames_fail_returns_error(self, mock_pyiqa, metric_name):
        scorer = ImageQualityScorer(config=ScorerConfig(device="cpu", extra={"metrics": (metric_name,)}))
        scorer._model_loaded = True
        setattr(scorer, f"_{metric_name}_metric", MagicMock(side_effect=RuntimeError("all frames failed")))
        result = scorer.score_frames(make_loaded_frames(3, 224, 224))
        assert not result.is_success
        assert result.error == "image_quality_inference_failed"
        assert not result.scores

    def test_all_liqe_batches_fail_returns_error(self, mock_pyiqa):
        scorer = ImageQualityScorer(config=ScorerConfig(device="cpu", frame_batch_size=2, extra={"metrics": ("liqe",)}))
        scorer._model_loaded = True
        scorer._liqe_model = MagicMock()
        with patch.object(ImageQualityScorer, "_liqe_dist_probs_batch", side_effect=RuntimeError("batch failed")):
            result = scorer.score_frames(make_loaded_frames(3, 224, 224))
        assert not result.is_success
        assert result.error == "image_quality_inference_failed"
        assert not result.scores

    def test_liqe_batch_method_returns_normalized_probabilities(self, mock_pyiqa):
        class FakeClip:
            logit_scale = torch.tensor(0.0)

            def encode_image(self, images, pos_embedding=True):
                return torch.tensor([[1.0, 0.0]] * images.shape[0])

        model = SimpleNamespace(
            default_mean=torch.zeros(1, 3, 1, 1),
            default_std=torch.ones(1, 3, 1, 1),
            step=224,
            num_patch=1,
            clip_model=FakeClip(),
            text_features=torch.tensor([[1.0, 0.0]] + [[0.0, 1.0]] * 10),
        )
        scorer = ImageQualityScorer(config=ScorerConfig(device="cpu"))
        scorer._liqe_model = SimpleNamespace(net=model)
        scorer._torch_device = torch.device("cpu")
        images = [Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)) for _ in range(2)]

        probs = scorer._liqe_dist_probs_batch(images)
        assert probs.shape == (2, 11)
        assert np.allclose(probs.sum(axis=1), 1.0)
        assert np.all(probs[:, 0] > probs[:, 1])

    def test_liqe_clip_loader_override_is_restored(self, mock_pyiqa, monkeypatch):
        from pyiqa.archs import liqe_arch

        original_load = liqe_arch.load
        original_loader = liqe_arch.load_pretrained_network
        original_cache_dir = liqe_arch.DEFAULT_CACHE_DIR

        def create_metric(name, **kwargs):
            assert liqe_arch.load("ViT-B/32", "cpu") is not None
            return MagicMock()

        mock_pyiqa.create_metric.side_effect = create_metric
        scorer = ImageQualityScorer(config=ScorerConfig(device="cpu", extra={"metrics": ("liqe",)}))
        scorer._load_model()

        assert liqe_arch.load is original_load
        assert liqe_arch.load_pretrained_network is original_loader
        assert liqe_arch.DEFAULT_CACHE_DIR == original_cache_dir

    def test_missing_local_clip_checkpoint_fails_closed(self, mock_pyiqa, monkeypatch):
        monkeypatch.delenv("PYIQA_CLIP_VITB32", raising=False)
        with pytest.raises(FileNotFoundError, match="Local CLIP ViT-B/32 checkpoint is required"):
            ImageQualityScorer(config=ScorerConfig(device="cpu", extra={"metrics": ("liqe",)}))._load_model()
        mock_pyiqa.create_metric.assert_not_called()

    def test_clip_checkpoint_hash_mismatch_fails_closed(self, mock_pyiqa, monkeypatch):
        monkeypatch.setenv("PYIQA_CLIP_VITB32_SHA256", "0" * 64)
        with pytest.raises(ValueError, match="CLIP ViT-B/32 checkpoint SHA-256 mismatch"):
            ImageQualityScorer(config=ScorerConfig(device="cpu", extra={"metrics": ("liqe",)}))._load_model()
        mock_pyiqa.create_metric.assert_not_called()

    def test_liqe_checkpoint_and_cache_loads_are_weights_only(self, mock_pyiqa):
        from pyiqa.archs import liqe_arch

        def create_metric(name, **kwargs):
            metric = MagicMock()
            if name == "liqe_mix":
                liqe_arch.load("ViT-B/32", "cpu")
                fake_net = SimpleNamespace(load_state_dict=MagicMock())
                liqe_arch.load_pretrained_network(fake_net, kwargs["pretrained_model_path"], True, "params")
                cache_file = Path(liqe_arch.DEFAULT_CACHE_DIR) / "liqe_text_feat_mix.pt"
                liqe_arch.torch.load(cache_file, map_location="cpu", weights_only=False)
                metric.net = SimpleNamespace(text_features=None)
            return metric

        mock_pyiqa.create_metric.side_effect = create_metric
        scorer = ImageQualityScorer(config=ScorerConfig(device="cpu", extra={"metrics": ("liqe",)}))
        with patch("torch.load", wraps=torch.load) as torch_load_mock:
            scorer._load_model()

        assert torch_load_mock.call_count >= 3
        assert all(call.kwargs.get("weights_only") is True for call in torch_load_mock.call_args_list)

    def test_text_features_use_weights_only_loading(self, mock_pyiqa, tmp_path, monkeypatch):
        text_features = tmp_path / "liqe_text_features.pt"
        text_features.write_bytes(b"mocked tensor weights")
        monkeypatch.setenv("PYIQA_LIQE_TEXT_FEAT", str(text_features))
        scorer = ImageQualityScorer(config=ScorerConfig(device="cpu", extra={"metrics": ("liqe",)}))
        with patch("torch.load", return_value=torch.zeros(11, 2)) as torch_load_mock:
            scorer._load_model()
        assert torch_load_mock.call_args.kwargs["weights_only"] is True

    def test_registered_in_registry(self):
        from mm.core.video_quality import list_scorers

        assert "image_quality" in list_scorers()

    def test_metrics_config(self, mock_pyiqa):
        scorer = ImageQualityScorer(config=ScorerConfig(device="cpu", extra={"metrics": ("liqe",)}))
        assert "liqe" in scorer.config.extra["metrics"]
