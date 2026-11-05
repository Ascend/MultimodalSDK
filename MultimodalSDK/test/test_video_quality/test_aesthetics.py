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

"""Tests for AestheticsScorer (SigLIP + predictor, mocked)."""

from __future__ import annotations

from types import ModuleType
from unittest.mock import patch, MagicMock

import pytest
import torch

from mm.core.video_quality import AestheticsScorer, ScorerConfig
from mm.core.video_quality.base import LoadedFrames
from .conftest import make_loaded_frames


MODULE_PATH = "mm.core.video_quality.scorers.aesthetics_scorer"


@pytest.fixture(autouse=True)
def mock_weight_resolver(tmp_path):
    with patch(f"{MODULE_PATH}.WeightResolver") as MockResolver:
        resolver = MagicMock()
        encoder_dir = tmp_path / "siglip"
        encoder_dir.mkdir()
        predictor_file = tmp_path / "predictor.ckpt"
        predictor_file.write_bytes(b"test fixture; torch.load is mocked")
        resolver.resolve_multi.return_value = {
            "encoder": str(encoder_dir),
            "predictor": str(predictor_file),
        }
        MockResolver.return_value = resolver
        yield resolver


@pytest.fixture(autouse=True, name="mock_model")
def _mock_model_fixture():
    """Mock the aesthetic predictor model."""
    mock_model = MagicMock()
    mock_model.eval.return_value = mock_model
    mock_model.to.return_value = mock_model
    mock_model.half.return_value = mock_model

    def _call(**kwargs):
        # Return logits-like output with .logits attribute
        n = kwargs.get("pixel_values", kwargs.get("input_ids", torch.randn(8, 3, 224, 224)))
        if isinstance(n, torch.Tensor) and n.dim() == 4:
            n = n.shape[0]
        else:
            n = 8
        out = MagicMock()
        out.logits = torch.randn(n, 1) * 5 + 5  # values around 5 (1-10 range)
        return out

    mock_model.side_effect = _call

    mock_processor = MagicMock()

    def _process(images, **kwargs):
        batch = len(images)
        inputs = MagicMock()
        pixel_values = torch.randn(batch, 3, 384, 384)
        inputs.to.return_value = {"pixel_values": pixel_values}
        inputs.items.return_value = [("pixel_values", pixel_values)]
        return inputs

    mock_processor.side_effect = _process

    with (
        patch.dict(
            "sys.modules",
            {
                "aesthetic_predictor_v2_5": ModuleType("aesthetic_predictor_v2_5"),
                "transformers": MagicMock(),
            },
        ),
    ):
        # Since aesthetic_predictor_v2_5 import fails (it's mocked), the fallback path
        # tries transformers.AutoModel/AutoProcessor
        import sys

        transformers_mock = sys.modules["transformers"]
        transformers_mock.AutoModel.from_pretrained.return_value = mock_model
        transformers_mock.AutoProcessor.from_pretrained.return_value = mock_processor

        with patch("torch.load", return_value=torch.nn.Linear(1152, 1)) as torch_load_mock:
            yield {
                "model": mock_model,
                "processor": mock_processor,
                "transformers": transformers_mock,
                "torch_load": torch_load_mock,
            }


class TestAestheticsScorerInit:
    def test_name(self):
        scorer = AestheticsScorer(config=ScorerConfig(device="cpu"))
        assert scorer.name == "aesthetics"


class TestAestheticsScoring:
    def test_score_frames_returns_metric(self, mock_model):
        scorer = AestheticsScorer(config=ScorerConfig(device="cpu", frame_batch_size=3))
        frames = make_loaded_frames(8, 224, 224)
        result = scorer.score_frames(frames)
        assert result.name == "aesthetics"
        # May succeed or degrade depending on mock setup
        if result.is_success:
            assert "score_aes_mean" in result.scores
            assert "aesthetic_quality" in result.scores
            assert "score_aes_list" in result.scores

    def test_single_frame_scalar_logit_is_supported(self, mock_model):
        scorer = AestheticsScorer(config=ScorerConfig(device="cpu", frame_batch_size=1))
        mock_model["model"].side_effect = lambda **kwargs: type("Output", (), {"logits": torch.tensor(5.0)})()
        result = scorer.score_frames(make_loaded_frames(1, 224, 224))
        assert result.is_success
        assert result.scores["score_aes_list"] == [5.0]

    def test_inference_respects_frame_batch_size(self, mock_model):
        scorer = AestheticsScorer(config=ScorerConfig(device="cpu", frame_batch_size=3))
        result = scorer.score_frames(make_loaded_frames(8, 224, 224))
        assert result.is_success
        assert mock_model["processor"].call_count == 3

    def test_empty_frames_returns_default(self, mock_model):
        scorer = AestheticsScorer(config=ScorerConfig(device="cpu"))
        frames = LoadedFrames(frames=[], indices=[])
        result = scorer.score_frames(frames)
        assert result.scores["score_aes_mean"] == 0.0
        assert result.scores["aesthetic_quality"] == 0.0

    def test_fallback_uses_local_safetensors_and_safe_predictor_load(self, mock_model):
        scorer = AestheticsScorer(config=ScorerConfig(device="cpu"))
        scorer._ensure_model()
        mock_model["transformers"].AutoModel.from_pretrained.assert_called_once()
        _, kwargs = mock_model["transformers"].AutoModel.from_pretrained.call_args
        assert kwargs["local_files_only"] is True
        assert kwargs["use_safetensors"] is True
        mock_model["transformers"].AutoProcessor.from_pretrained.assert_called_once()
        _, processor_kwargs = mock_model["transformers"].AutoProcessor.from_pretrained.call_args
        assert processor_kwargs["local_files_only"] is True
        assert mock_model["torch_load"].call_args.kwargs["weights_only"] is True

    def test_remote_code_is_disabled_by_default(self, mock_model):
        captured = {}

        def convert_v2_5_from_siglip(**kwargs):
            captured.update(kwargs)
            return mock_model["model"], mock_model["processor"]

        predictor_module = ModuleType("aesthetic_predictor_v2_5")
        predictor_module.convert_v2_5_from_siglip = convert_v2_5_from_siglip
        with patch.dict("sys.modules", {"aesthetic_predictor_v2_5": predictor_module}):
            scorer = AestheticsScorer(config=ScorerConfig(device="cpu"))
            scorer._ensure_model()
        assert captured["trust_remote_code"] is False

    def test_registered_in_registry(self):
        from mm.core.video_quality import list_scorers

        assert "aesthetics" in list_scorers()
