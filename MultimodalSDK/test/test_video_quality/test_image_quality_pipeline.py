#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the MultimodalSDK project.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

"""Tests for VideoQualityPipeline."""

from __future__ import annotations

from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mm.core.video_quality import (
    VideoQualityPipeline,
    ScorerConfig,
    ImageQualityScorer,
)
from mm.core.video_quality.base import (
    MetricResult,
)


@pytest.fixture(autouse=True)
def mock_image_quality_dependencies():
    with patch("mm.core.video_quality.scorers.image_quality_scorer.WeightResolver") as resolver_cls:
        resolver = MagicMock()
        resolver.resolve_single.return_value = "/fake/liqe.pt"
        resolver_cls.return_value = resolver

        mock_pyiqa = ModuleType("pyiqa")
        mock_pyiqa.__path__ = []
        mock_archs = ModuleType("pyiqa.archs")
        mock_archs.__path__ = []
        mock_liqe_arch = ModuleType("pyiqa.archs.liqe_arch")
        mock_liqe_arch.load = MagicMock()

        def create_metric(name, **kwargs):
            metric = MagicMock()
            if name == "niqe":
                metric.side_effect = lambda image: 5.0
            elif name == "brisque":
                metric.side_effect = lambda image: 10.0
            return metric

        mock_pyiqa.create_metric = MagicMock(side_effect=create_metric)
        with patch.dict(
            "sys.modules",
            {
                "pyiqa": mock_pyiqa,
                "pyiqa.archs": mock_archs,
                "pyiqa.archs.liqe_arch": mock_liqe_arch,
            },
        ):

            def _mock_load_model(scorer):
                # The integration tests verify pipeline wiring and aggregation;
                # model/checkpoint loading is covered by scorer unit tests.
                scorer._liqe_model = object()
                scorer._niqe_metric = lambda image: 5.0
                scorer._brisque_metric = lambda image: 10.0

            with patch.object(
                ImageQualityScorer,
                "_load_model",
                autospec=True,
                side_effect=_mock_load_model,
            ):
                with patch.object(
                    ImageQualityScorer,
                    "_liqe_dist_probs_batch",
                    side_effect=lambda batch: np.tile(
                        np.array([0.15, 0.0, 0.0, 0.25, 0.0, 0.0, 0.10, 0.50, 0.0, 0.0, 0.0]),
                        (len(batch), 1),
                    ),
                ):
                    yield


def _write_test_video(video_path, frames):
    import cv2 as cv

    writer = cv.VideoWriter(video_path, cv.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 64))
    try:
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer for {video_path}")
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


class TestPipelineInit:
    def test_init_with_scorer_names(self, force_cpu_device):
        pipeline = VideoQualityPipeline(
            scorers=["image_quality"],
            device="cpu",
        )
        assert len(pipeline._scorers) == 1
        assert pipeline._scorers[0].name == "image_quality"

    def test_init_with_scorer_instances(self, force_cpu_device):
        s1 = ImageQualityScorer(config=ScorerConfig(device="cpu"))
        pipeline = VideoQualityPipeline(scorers=[s1], device="cpu")
        assert len(pipeline._scorers) == 1

    def test_init_with_weights(self, force_cpu_device):
        pipeline = VideoQualityPipeline(
            scorers=["image_quality"],
            device="cpu",
            weights={"image_quality": None},
        )
        assert len(pipeline._scorers) == 1


class TestPipelineScore:
    def test_score_single_video(self, tmp_path, force_cpu_device):
        """Test pipeline with a scorer on a real video file."""

        video_path = str(tmp_path / "test.mp4")
        _write_test_video(video_path, [np.full((64, 64, 3), i * 5, dtype=np.uint8) for i in range(30)])

        pipeline = VideoQualityPipeline(
            scorers=["image_quality"],
            device="cpu",
        )
        report = pipeline.score(video_path)

        assert report.video_path == video_path
        assert "image_quality" in report.metrics
        assert report.metrics["image_quality"].is_success
        assert "score_blur_mean" in report.all_scores
        blur = report.metrics["image_quality"].scores["score_blur_mean"]
        assert blur is not None and 0.0 <= blur <= 1.0
        assert report.all_scores["score_blur_mean"] == blur
        assert blur == pytest.approx(0.25)
        assert report.all_scores["score_dark_mean"] == pytest.approx(0.50)
        assert report.all_scores["score_overexposure_mean"] == pytest.approx(0.10)
        assert report.all_scores["niqe"] == pytest.approx(5.0)
        assert report.all_scores["brisque"] == pytest.approx(10.0)

    def test_score_batch(self, tmp_path, force_cpu_device):
        paths = []
        for idx in range(2):
            video_path = str(tmp_path / f"test_{idx}.mp4")
            _write_test_video(
                video_path,
                [np.full((64, 64, 3), (i + idx * 10) * 5, dtype=np.uint8) for i in range(20)],
            )
            paths.append(video_path)

        pipeline = VideoQualityPipeline(
            scorers=["image_quality"],
            device="cpu",
        )
        reports = pipeline.score_batch(paths)
        assert len(reports) == 2
        for r in reports:
            assert "image_quality" in r.metrics
            assert r.metrics["image_quality"].is_success
            blur = r.metrics["image_quality"].scores["score_blur_mean"]
            assert blur is not None and 0.0 <= blur <= 1.0
            assert r.all_scores["score_blur_mean"] == blur
            assert blur == pytest.approx(0.25)
            assert r.all_scores["score_dark_mean"] == pytest.approx(0.50)
            assert r.all_scores["score_overexposure_mean"] == pytest.approx(0.10)
            assert r.all_scores["niqe"] == pytest.approx(5.0)
            assert r.all_scores["brisque"] == pytest.approx(10.0)

    def test_all_failed_raises_runtime_error(self, tmp_path, force_cpu_device):
        """When all scorers fail, pipeline should raise RuntimeError."""

        video_path = str(tmp_path / "test.mp4")
        _write_test_video(video_path, [np.full((64, 64, 3), 100, dtype=np.uint8) for _ in range(20)])

        pipeline = VideoQualityPipeline(
            scorers=["image_quality"],
            device="cpu",
        )
        with patch.object(
            pipeline._scorers[0],
            "score_frames",
            return_value=MetricResult(name="image_quality", scores={}, error="fail"),
        ):
            with pytest.raises(RuntimeError, match="All scorers failed"):
                pipeline.score(video_path)

    def test_pipeline_close(self, force_cpu_device):
        pipeline = VideoQualityPipeline(
            scorers=["image_quality"],
            device="cpu",
        )
        pipeline.close()
