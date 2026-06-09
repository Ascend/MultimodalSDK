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
# Extract audio chunks.


from pathlib import Path
from typing import Union

import torch
import torchaudio

from vrag.constants import AUDIO_SAMPLE_RATE
from vrag.types import AudioChunkExtraction


def chunk_audio(
    audio_path: Union[str, Path],
    chunk_length: float = 30.0,
    min_chunk_threshold: float = 1.0,
    sample_rate: int = AUDIO_SAMPLE_RATE,
) -> AudioChunkExtraction:
    """
    Splits audio into fixed-length chunks.
    Discard the tail if it is shorter than min_chunk_threshold.

    Args:
        audio_path: Path to audio file.
        chunk_length: Duration of each chunk in seconds.
        min_chunk_threshold: Minimum duration in seconds of tail to be kept.
        sample_rate: The sample rate of normal.
    """
    if chunk_length < min_chunk_threshold:
        raise ValueError(
            f"Audio chunk_length: {chunk_length} must be larger than min_chunk_threshold: {min_chunk_threshold}"
        )

    audio_path = Path(audio_path)

    speech, sr = torchaudio.load(audio_path)

    # convert to 1D tensor
    speech = speech.mean(dim=0)

    if sr != sample_rate:
        speech = torchaudio.transforms.Resample(sr, sample_rate)(speech)

    total_samples_num = len(speech)
    num_samples_per_chunk = int(chunk_length * sample_rate)
    min_samples_threshold = int(min_chunk_threshold * sample_rate)

    chunks = []
    durations = []

    for i in range(0, total_samples_num, num_samples_per_chunk):
        chunk = speech[i : i + num_samples_per_chunk]
        current_length = len(chunk)

        if current_length < num_samples_per_chunk:
            if current_length < min_samples_threshold:
                break
            chunk = torch.nn.functional.pad(chunk, (0, num_samples_per_chunk - current_length))

        start_time = i / sample_rate
        end_time = (i + min(current_length, num_samples_per_chunk)) / sample_rate

        chunks.append(chunk.numpy())
        durations.append((start_time, end_time))

    return AudioChunkExtraction(audio_chunks=chunks, durations=durations)
