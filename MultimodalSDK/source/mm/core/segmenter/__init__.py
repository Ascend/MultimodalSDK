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
"""Video segmentation package: KTS building blocks and embedding backends.

``kts_algorithm`` provides the stateless KTS algorithm functions and
``kts_cache`` the on-disk analysis cache. Both are internals of the
segmentation pipeline and are intentionally not re-exported here: they are
not part of the customer-facing API. The public entry point is the
orchestrator ``KtsSegmenter`` (kts_segmenter.py), which imports these
submodules directly.

``embedding_backend`` provides the embedding API used by the pipeline
(``EmbeddingBackend`` / ``OpenAIEmbeddingBackend``); these classes are
re-exported here for callers of this package.
"""

__all__ = [
    "EmbeddingBackend",
    "OpenAIEmbeddingBackend",
    "VideoSegment",
    "SegmentResult",
    "KtsSegmenter",
]

from .embedding_backend import EmbeddingBackend, OpenAIEmbeddingBackend
from .kts_segmenter import VideoSegment, SegmentResult, KtsSegmenter
