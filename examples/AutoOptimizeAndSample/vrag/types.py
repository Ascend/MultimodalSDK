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
# Type aliases and common types used through the VRAG.


from typing import Literal, Tuple, TypeAlias, List, Optional, NamedTuple

import bentoml
from pydantic import BaseModel, Field
import numpy as np

from vrag.constants import HEVC

Timestamp: TypeAlias = float

Duration: TypeAlias = Tuple[Timestamp, Timestamp]

FrameDoc: TypeAlias = Tuple[str, Timestamp]

OCRDoc: TypeAlias = Tuple[str, Timestamp]

DETDoc: TypeAlias = Tuple[str, Timestamp]

ASRDoc: TypeAlias = Tuple[str, Duration]

EmbeddingBackend: TypeAlias = Literal["qwen", "qwen-q"]

RerankerBackend: TypeAlias = Literal["qwen"]

ObjectDetectionBackend: TypeAlias = Literal["mmdino-s", "mmdino-m", "mmdino-l"]


class VideoProbeResult(NamedTuple):
    duration: Optional[float]
    video_codec: Optional[str]
    audio_codec: Optional[str]
    audio_sample_rate: Optional[int]
    fps: Optional[float]
    has_video: bool
    has_audio: bool

    @property
    def is_hevc(self) -> bool:
        return self.video_codec in HEVC


class FrameExtraction(bentoml.IODescriptor):
    frames: np.ndarray
    frame_timestamps: List[float]
    avg_fps: float
    video_duration: float
    total_frame_num: int

    @property
    def frame_shape(self) -> Tuple[int, int]:
        if self.frames.ndim < 3:
            raise ValueError("Frames dims at least 3")
        h, w = self.frames[0].shape[:2]
        return h, w

    @property
    def frame_height(self) -> int:
        return self.frame_shape[0]

    @property
    def frame_width(self) -> int:
        return self.frame_shape[1]

    @property
    def frames_count(self) -> int:
        return len(self.frames)

    @property
    def frames_list(self) -> List[np.ndarray]:
        return [self.frames[i] for i in range(self.frames_count)]

    def slice(self, indices: List[int]) -> "FrameExtraction":
        """
        Extract frames at specific indices.

        Args:
            indices: List of frame indices to extract.

        Returns:
            FrameExtraction containing target frames.
        """

        sliced_frames = self.frames[indices]
        sliced_timestamps = [self.frame_timestamps[i] for i in indices]

        return FrameExtraction(
            frames=sliced_frames,
            frame_timestamps=sliced_timestamps,
            avg_fps=self.avg_fps,
            video_duration=self.video_duration,
            total_frame_num=len(indices),
        )

    def uniform_k(self, k: int) -> "FrameExtraction":
        """
        Uniform sample k frames from the extraction.

        Args:
            k: Number of frames to sample.
            If greater than total frames, returns the original extraction.

        Returns:
            FrameExtraction containing target frames.
        """
        if k >= self.frames_count:
            return self

        indices = np.linspace(0, self.frames_count - 1, k, dtype=int).tolist()
        return self.slice(indices)


class AudioChunkExtraction(bentoml.IODescriptor):
    audio_chunks: List[np.ndarray] = Field(default_factory=list)
    durations: List[Duration] = Field(default_factory=list)


class DetectionResult(bentoml.IODescriptor):
    det_docs: List[DETDoc] = Field(default_factory=list)
    det_top_idx: List[int] = Field(default_factory=list)


class ExtractionResult(bentoml.IODescriptor):
    frame_extraction: Optional[FrameExtraction] = Field(default=None)
    audio_extraction: Optional[AudioChunkExtraction] = Field(default=None)
    ocr_docs_total: List[OCRDoc] = Field(default_factory=list)
    asr_docs_total: List[ASRDoc] = Field(default_factory=list)


class MMDINODetectionItem(BaseModel):
    """Represent a single detected object with its bounding box."""

    class_name: str
    """The name of the detected object class."""

    bbox: List[int]
    """Bounding box coordinates [x1, y1, x2, y2]"""

    def to_str(self) -> str:
        bbox_str = ", ".join(map(str, self.bbox))
        return f"{self.class_name}: [{bbox_str}]"


class MMDINODetectionResult(BaseModel):
    items: List[MMDINODetectionItem] = Field(default_factory=list)


class MMDINODetectionBatchResult(bentoml.IODescriptor):
    """Represents batch detection results for multiple frames"""

    results: List[MMDINODetectionResult] = Field(default_factory=list)
