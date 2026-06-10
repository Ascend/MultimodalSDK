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

from vrag.bentos.aks import AksBlipService, AksBlipConfig, AksBlipArgs
from vrag.bentos.blip import BlipService, BlipArgs
from vrag.bentos.detection_service import DetectionService, DetectionServiceConfig, DetectionArgs
from vrag.bentos.faiss_search import FaissService, FaissSearchConfig, FaissSearchArgs
from vrag.bentos.mineru_ocr import MineruService, MineruArgs
from vrag.bentos.mmdino import MMDINOService, MMDinoConfig, MMDinoArgs
from vrag.bentos.qwen_embedding import QwenEmbeddingService, QwenEmbeddingArgs
from vrag.bentos.qwen_reranker import QwenRerankerService, QwenRerankerArgs
from vrag.bentos.qwenvl import QwenVLService, QwenVLConfig, QwenVLArgs
from vrag.bentos.video_process import VideoProcessService, VideoProcessConfig, VideoProcessArgs
from vrag.bentos.video_rag import VideoRagService, VideoRagConfig, VideoRagArgs
from vrag.bentos.video_retrieval import VideoRetrievalService, VideoRetrievalConfig, VideoRetrievalArgs
from vrag.bentos.video_transcribe import VideoTranscribeService, VideoTranscribeConfig, VideoTranscribeArgs
from vrag.bentos.whisper import WhisperService, WhisperArgs

__all__ = [
    "AksBlipService",
    "AksBlipConfig",
    "AksBlipArgs",
    "BlipService",
    "BlipArgs",
    "DetectionService",
    "DetectionServiceConfig",
    "DetectionArgs",
    "FaissService",
    "FaissSearchConfig",
    "FaissSearchArgs",
    "MineruService",
    "MineruArgs",
    "MMDINOService",
    "MMDinoConfig",
    "MMDinoArgs",
    "QwenEmbeddingService",
    "QwenEmbeddingArgs",
    "QwenRerankerService",
    "QwenRerankerArgs",
    "QwenVLService",
    "QwenVLConfig",
    "QwenVLArgs",
    "VideoProcessService",
    "VideoProcessConfig",
    "VideoProcessArgs",
    "VideoRagService",
    "VideoRagConfig",
    "VideoRagArgs",
    "VideoRetrievalService",
    "VideoRetrievalConfig",
    "VideoRetrievalArgs",
    "VideoTranscribeService",
    "VideoTranscribeConfig",
    "VideoTranscribeArgs",
    "WhisperService",
    "WhisperArgs",
]
