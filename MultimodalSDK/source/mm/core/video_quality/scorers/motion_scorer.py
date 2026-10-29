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

"""Motion scorer — optical-flow based multi-metric motion scoring.

Uses Sea-RAFT-S (via ptlflow) to compute optical flow between consecutive
frames, then derives four metrics:
    - score_motion_*: top-k% flow magnitude (motion intensity)
    - motion_smoothness: inverse of flow magnitude variance (higher = smoother)
    - dynamic_degree: fraction of high-motion frames (0=static, 1=excessive)
    - optical_flow_consistency: stability of flow direction across frames

Requires continuous-fps decoding (``requires_continuous_frames = True``).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from ..base import (
    BaseQualityScorer,
    LoadedFrames,
    MetricResult,
    QualityScorerRegistry,
    ScorerConfig,
)
from ..weights import WeightResolver


@QualityScorerRegistry.register
class MotionScorer(BaseQualityScorer):
    """Motion scoring via optical flow (Sea-RAFT-S, ptlflow).

    One optical-flow inference produces four motion statistics covering the
    original 4 source capabilities (intensity / smoothness / dynamic / coherence).
    """

    name: str = "motion"
    requires_continuous_frames: ClassVar[bool] = True

    def __init__(self, config: ScorerConfig | None = None, **kwargs: Any):
        super().__init__(config, **kwargs)
        self._model = None
        self._torch_device = None
        self._flow_ckpt: str | None = None

    def _load_model(self) -> None:
        """Lazily load the Sea-RAFT-S optical flow model."""
        import torch

        resolver = WeightResolver(self.name)
        self._flow_ckpt = resolver.resolve_single(self._config.weights, role="flow")

        try:
            import ptlflow
        except ImportError as exc:
            raise ImportError("MotionScorer requires ptlflow. Install with: pip install mm[quality]") from exc

        ckpt_dir = Path(self._flow_ckpt)
        candidates = sorted(ckpt_dir.glob("sea_raft_s-sintel*.ckpt")) if ckpt_dir.is_dir() else []
        if not candidates:
            raise FileNotFoundError(f"Sea-RAFT-S weight not found in {self._flow_ckpt} (glob: sea_raft_s-sintel*.ckpt)")

        self._torch_device = torch.device(self.device)
        use_fp16 = self._config.dtype == "fp16" and self.device.startswith(("npu", "cuda"))
        self._model = ptlflow.get_model("sea_raft_s", str(candidates[0])).eval().to(self._torch_device)
        if use_fp16:
            self._model = self._model.half()

    def score_frames(self, frames: LoadedFrames) -> MetricResult:
        t0 = time.perf_counter()

        if len(frames) < 2:
            return MetricResult(
                name=self.name,
                scores={
                    "score_motion_list": [],
                    "score_motion_mean": 0.0,
                    "score_motion_std": 0.0,
                    "score_motion_min": 0.0,
                    "score_motion_max": 0.0,
                    "motion_smoothness": 1.0,
                    "dynamic_degree": 0.0,
                    "dynamic_degree_frames": [],
                    "optical_flow_consistency": 0.0,
                },
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )

        try:
            self._ensure_model()
            import torch

            metrics = tuple(self._config.extra.get("metrics", ("motion", "smoothness", "dynamic", "coherence")))
            agg_top_percent = float(self._config.extra.get("agg_top_percent", 0.05))
            batch_size = int(self._config.extra.get("batch_size", 8))

            frames_u8 = [np.ascontiguousarray(f).transpose(2, 0, 1) for f in frames.frames]
            batch_size = max(1, batch_size)
            t_chunks = []
            for start in range(0, len(frames_u8), batch_size):
                chunk = np.stack(frames_u8[start : start + batch_size])
                t_chunk = torch.from_numpy(chunk).to(self._torch_device)
                t_chunk = t_chunk.float().div_(255.0)
                if self._config.dtype == "fp16" and self.device.startswith(("npu", "cuda")):
                    t_chunk = t_chunk.half()
                t_chunks.append(t_chunk)
            t = torch.cat(t_chunks, dim=0)

            t1, t2 = t[:-1], t[1:]
            n_pairs = t1.shape[0]
            magnitudes = []
            flow_dirs = []

            with torch.no_grad():
                for start in range(0, n_pairs, batch_size):
                    end = min(start + batch_size, n_pairs)
                    images = torch.stack([t1[start:end], t2[start:end]], dim=1).contiguous()
                    out = self._model({"images": images})
                    flows = out["flows"].float()
                    if flows.dim() == 5:
                        flows = flows[:, 0]
                    mag = torch.norm(flows, dim=1)  # [B, H, W]
                    b = mag.shape[0]
                    flat = mag.reshape(b, -1)
                    k = max(1, int(flat.shape[1] * agg_top_percent))
                    magnitudes.append(flat.topk(k, dim=1).values.mean(dim=1))

                    if "coherence" in metrics:
                        # Flow direction (angle) per pair
                        ang = torch.atan2(flows[:, 1], flows[:, 0])  # [B, H, W]
                        mean_ang = ang.reshape(b, -1).mean(dim=1)  # [B]
                        flow_dirs.append(mean_ang)

            vals = torch.cat(magnitudes).cpu().numpy()

            scores: dict[str, Any] = {
                "score_motion_list": [round(float(v), 6) for v in vals],
                "score_motion_mean": float(np.mean(vals)),
                "score_motion_std": float(np.std(vals)),
                "score_motion_min": float(np.min(vals)),
                "score_motion_max": float(np.max(vals)),
            }

            if "smoothness" in metrics:
                # Smoothness: inverse of magnitude variance (normalised to [0,1])
                std_val = float(np.std(vals))
                mean_val = float(np.mean(vals)) + 1e-8
                smoothness = max(0.0, min(1.0, 1.0 - std_val / (mean_val + std_val)))
                scores["motion_smoothness"] = round(smoothness, 6)

            if "dynamic" in metrics:
                # Dynamic degree: fraction of frames with motion above threshold
                threshold = float(np.mean(vals)) + float(np.std(vals))
                dynamic_frames = (vals > threshold).astype(float)
                scores["dynamic_degree"] = round(float(dynamic_frames.mean()), 6)
                scores["dynamic_degree_frames"] = [round(float(d), 6) for d in dynamic_frames]

            if "coherence" in metrics and flow_dirs:
                angles = torch.cat(flow_dirs).cpu().numpy()
                # Consistency of flow direction: 1 - circular variance
                sin_sum = np.mean(np.sin(angles))
                cos_sum = np.mean(np.cos(angles))
                R = np.sqrt(sin_sum**2 + cos_sum**2)  # resultant length
                scores["optical_flow_consistency"] = round(float(R), 6)

            return MetricResult(
                name=self.name,
                scores=scores,
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )
        except Exception as exc:
            return MetricResult(
                name=self.name,
                scores={
                    "score_motion_list": [],
                    "score_motion_mean": None,
                    "score_motion_std": None,
                    "score_motion_min": None,
                    "score_motion_max": None,
                    "motion_smoothness": None,
                    "dynamic_degree": None,
                    "dynamic_degree_frames": [],
                    "optical_flow_consistency": None,
                },
                error=str(exc),
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )

    def close(self) -> None:
        self._model = None
        self._model_loaded = False
