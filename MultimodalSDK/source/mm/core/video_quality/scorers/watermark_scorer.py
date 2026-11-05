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

"""Watermark detector — YOLO11x preprocessing + NMS.

Single-weight scorer: YOLO11x best.pt (role ``detector``).  Detects
watermarks/logos in sampled frames, outputs bbox list + area ratio.
"""

from __future__ import annotations

import hashlib
import importlib
import hmac
import inspect
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

from ..base import (
    BaseQualityScorer,
    LoadedFrames,
    MetricResult,
    QualityScorerRegistry,
    ScorerConfig,
)
from ..device import DeviceResolver
from ..weights import WeightResolver

logger = logging.getLogger(__name__)
_YOLO_LOAD_LOCK = threading.Lock()


@QualityScorerRegistry.register
class WatermarkDetector(BaseQualityScorer):
    """Watermark detector (YOLO11x, conf≥threshold, NMS).

    Single-weight: ``weights="/path/best.pt"`` or env var
    ``WATERMARK_YOLO_MODEL_PATH``. Pin the local checkpoint with
    ``WATERMARK_YOLO_MODEL_SHA256``; restricted weights-only loading is required.
    ``frame_batch_size`` controls frames per YOLO inference call.
    """

    name: str = "watermark"

    def __init__(self, config: ScorerConfig | None = None, **kwargs: Any):
        super().__init__(config, **kwargs)
        self._model = None
        self._det_path: str | None = None

    def _load_model(self) -> None:
        resolver = WeightResolver(self.name)
        resolved_path = Path(resolver.resolve_single(self._config.weights, role="detector")).expanduser()
        try:
            self._det_path = str(resolved_path.resolve(strict=True))
        except OSError as exc:
            raise FileNotFoundError("Watermark detector weights must be an existing local file") from exc
        if not Path(self._det_path).is_file() or Path(self._det_path).suffix.lower() != ".pt":
            raise ValueError("Watermark detector requires a local .pt checkpoint")

        expected_sha256 = os.environ.get("WATERMARK_YOLO_MODEL_SHA256", "").strip().lower()
        if len(expected_sha256) != 64 or any(ch not in "0123456789abcdef" for ch in expected_sha256):
            raise RuntimeError("WATERMARK_YOLO_MODEL_SHA256 must contain the pinned 64-character SHA-256 digest")
        digest = hashlib.sha256()
        with open(self._det_path, "rb") as checkpoint_file:
            for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
                digest.update(chunk)
        if not hmac.compare_digest(digest.hexdigest(), expected_sha256):
            raise ValueError("Watermark detector checkpoint SHA-256 mismatch")

        # Ultralytics checkpoints use the .pt container. Require the version
        # exposing its restricted _SafeLoad implementation and force every
        # torch_safe_load call made during YOLO construction into safe_only mode.
        # The loader is module-global, so serialize the temporary patch and restore
        # it even when model construction fails.
        with _YOLO_LOAD_LOCK:
            from ultralytics import YOLO

            tasks = importlib.import_module("ultralytics.nn.tasks")

            safe_loader = getattr(tasks, "torch_safe_load", None)
            safe_load_class = getattr(tasks, "_SafeLoad", None)
            if (
                safe_loader is None
                or "safe_only" not in inspect.signature(safe_loader).parameters
                or safe_load_class is None
                or not getattr(safe_load_class, "SUPPORTED", False)
            ):
                raise RuntimeError(
                    "WatermarkDetector requires Ultralytics >= 8.4.126 and PyTorch >= 2.6 "
                    "for restricted checkpoint loading"
                )

            def _restricted_torch_safe_load(weight, *args, **kwargs):
                kwargs["safe_only"] = True
                return safe_loader(weight, *args, **kwargs)

            tasks.torch_safe_load = _restricted_torch_safe_load
            try:
                self._model = YOLO(self._det_path)
            finally:
                tasks.torch_safe_load = safe_loader

    def score_frames(self, frames: LoadedFrames) -> MetricResult:
        t0 = time.perf_counter()

        if len(frames) == 0:
            return MetricResult(
                name=self.name,
                scores={
                    "watermark_list": [],
                    "watermark_count": 0,
                    "watermark_area_ratio_sum": 0.0,
                },
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )

        try:
            self._ensure_model()
            import cv2 as cv

            conf_threshold = float(self._config.extra.get("conf_threshold", 0.35))
            iou_threshold = 0.45
            max_det = 100
            imgsz = 640

            frames_rgb = [cv.cvtColor(f, cv.COLOR_BGR2RGB) for f in frames.frames]

            all_detections: list[dict] = []
            total_area_ratio = 0.0

            frame_batch_size = max(
                1,
                int(self._config.frame_batch_size or DeviceResolver.default_frame_batch_size(self.device)),
            )
            for batch_start in range(0, len(frames_rgb), frame_batch_size):
                frame_batch = frames_rgb[batch_start : batch_start + frame_batch_size]
                results_batch = self._model(
                    frame_batch,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    imgsz=imgsz,
                    max_det=max_det,
                    verbose=False,
                )
                for batch_offset, result in enumerate(results_batch or []):
                    frame_idx = batch_start + batch_offset
                    if result is None or result.boxes is None or len(result.boxes) == 0:
                        continue

                    boxes = result.boxes
                    frame_rgb = frame_batch[batch_offset]
                    det_list = []
                    for box, conf_val, cls_id in zip(boxes.xyxy, boxes.conf, boxes.cls):
                        x1, y1, x2, y2 = box.tolist()
                        w, h = x2 - x1, y2 - y1
                        frame_area = frame_rgb.shape[0] * frame_rgb.shape[1]
                        area_ratio = (w * h) / frame_area if frame_area > 0 else 0.0
                        total_area_ratio += area_ratio

                        det_list.append(
                            {
                                "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                                "conf": round(float(conf_val), 4),
                                "cls": int(cls_id),
                            }
                        )

                    if det_list:
                        all_detections.append(
                            {
                                "frame_index": frame_idx,
                                "detections": det_list,
                            }
                        )

            return MetricResult(
                name=self.name,
                scores={
                    "watermark_list": all_detections,
                    "watermark_count": sum(len(d["detections"]) for d in all_detections),
                    "watermark_area_ratio_sum": round(min(total_area_ratio, 1.0), 6),
                },
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )
        except Exception:
            logger.exception("watermark scoring failed")
            return MetricResult(
                name=self.name,
                scores={
                    "watermark_list": [],
                    "watermark_count": 0,
                    "watermark_area_ratio_sum": None,
                },
                error="watermark_inference_failed",
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )

    def close(self) -> None:
        self._model = None
        self._model_loaded = False
