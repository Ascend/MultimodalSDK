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
# Whisper for automatic speech recognition (ASR).

from typing import List

import bentoml
import numpy as np
import torch
from pydantic import Field
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from vrag.constants import AUDIO_SAMPLE_RATE
from vrag.logger import logger
from vrag.shared import ArgsBase, vrag_service
from vrag.tools.np_cacher import get_cacher
from vrag.tools.path_validator import validate_dir_exists


class WhisperArgs(ArgsBase):
    whisper_model_path: str = ""
    """Local path to the Whisper model directory."""
    whisper_device: str = "npu:1"
    """Device for Whisper model inference, e.g. 'npu:1' or 'cpu'."""
    whisper_batch_size: int = Field(20, ge=1)
    """Batch size for Whisper ASR inference."""
    whisper_cache_size: int = Field(4096, ge=0)
    """LRU cache capacity for Whisper transcription results."""


args = bentoml.use_arguments(WhisperArgs).override()


@vrag_service(args)
class WhisperService:
    def __init__(self) -> None:
        self.model_path = validate_dir_exists(args.whisper_model_path, "Whisper model")

        self.processor = WhisperProcessor.from_pretrained(self.model_path, local_files_only=True)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            self.model_path, dtype=torch.float16, device_map=args.whisper_device, local_files_only=True
        )
        self.model.eval()

        self._cacher = get_cacher(args.whisper_cache_size)

        msg = f"WhisperService initialized from {self.model_path}."
        logger.info(msg)

    @bentoml.api
    async def transcribe(self, audio_chunks: List[np.ndarray]) -> List[str]:
        @self._cacher.cached_sync
        def _process(chunks: List[np.ndarray]) -> List[str]:
            return self._transcribe_batch(chunks, args.whisper_batch_size)

        return _process(audio_chunks)

    def _transcribe_batch(self, audio_chunks: List[np.ndarray], batch_size: int) -> List[str]:
        all_transcription: List[str] = []

        for i in range(0, len(audio_chunks), batch_size):
            current_batch = audio_chunks[i : i + batch_size]

            msg = f"Whisper Service processing audio chunks [{i}:{i + len(current_batch)}]."
            logger.info(msg)

            inputs = self.processor(
                current_batch,
                sampling_rate=AUDIO_SAMPLE_RATE,
                return_tensors="pt",
                padding=True,
                return_attention_mask=True,
            )

            inputs = {k: v.to(self.model.device, torch.float16) for k, v in inputs.items()}

            with torch.no_grad():
                predicate_ids = self.model.generate(**inputs)

            transcription = self.processor.batch_decode(predicate_ids, skip_special_tokens=True)

            all_transcription.extend(transcription)

        return all_transcription
