#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the MultimodalSDK project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
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
from typing import List, Tuple, Optional, Union, Iterable
from pathlib import Path

from .._impl import acc as _acc
from .util import _ensure_bytes
from .tensor_wrapper import Tensor


def _normalize_audio_inputs(audio_inputs: Union[str, List[str]]) -> Tuple[List[Path], bool]:
    """
    Normalize audio_inputs into a list of wav file Paths.

    Returns:
        paths: List[Path]
        is_single_file: True if input represents a single file
    """
    paths: List[Path] = []
    is_single_file: bool = False

    if isinstance(audio_inputs, str):
        path = Path(audio_inputs)

        if path.is_dir():
            wav_files = sorted(
                audio_path for audio_path in path.iterdir()
                if audio_path.is_file() and audio_path.suffix.lower() == ".wav"
            )
            if not wav_files:
                raise ValueError(f"No .wav files found in directory: {path}")
            paths = wav_files
            is_single_file = False

        elif path.is_file():
            if path.suffix.lower() != ".wav":
                raise ValueError(f"Not a .wav file: {path}")
            paths = [path]
            is_single_file = True
        else:
            raise ValueError(f"Path is neither file nor directory: {path}")

    elif isinstance(audio_inputs, list):
        if not audio_inputs:
            raise ValueError("audio_inputs list is empty")

        paths = []
        for item in audio_inputs:
            if not isinstance(item, str):
                raise TypeError("audio_inputs list must contain only str")
            file_path = Path(item)
            if not file_path.is_file():
                raise ValueError(f"Not a file: {file_path}")
            if file_path.suffix.lower() != ".wav":
                raise ValueError(f"Not a .wav file: {file_path}")
            paths.append(file_path)
        is_single_file = False

    else:
        raise TypeError(f"audio_inputs must be str or List[str], got {type(audio_inputs).__name__}")

    return paths, is_single_file


def load_audio(audio_inputs: Union[str, List[str]], sr: Optional[int] = None) \
        -> Union[Tuple[Tensor, int], List[Tuple[Tensor, int]]]:
    """
    Unified audio loading interface.

    - Single file -> calls load_audio -> returns (Tensor, sr)
    - List of files or directory -> calls load_audio_batch -> returns list of (Tensor, sr)

    Args:
        audio_inputs: str | bytes | Path | iterable of these
        sr: Optional sample rate

    Returns:
        Either:
            (Tensor, int) if single file
            List[(Tensor, int)] if batch
    """
    wav_files, is_single_file = _normalize_audio_inputs(audio_inputs)

    if sr is not None:
        if not isinstance(sr, int) or sr <= 0:
            raise ValueError(f"sr must be positive int, got {sr}")

    if is_single_file:
        audio_path = wav_files[0]
        path_bytes = _ensure_bytes(str(audio_path), "audio_path")

        dst_acc_tensor = _acc.Tensor()
        try:
            if sr is None:
                sample_rate = _acc.load_audio(path_bytes, dst_acc_tensor)
            else:
                sample_rate = _acc.load_audio(path_bytes, dst_acc_tensor, sr)
        except Exception as e:
            raise RuntimeError(f"load_audio failed: {e}") from e

        tensor_obj = object.__new__(Tensor)
        tensor_obj._inner = dst_acc_tensor
        return tensor_obj, sample_rate
    else:
        wav_files_vec = _acc.StringVector()
        for audio_path in wav_files:
            path_bytes = _ensure_bytes(str(audio_path), "audio_path")
            wav_files_vec.push_back(path_bytes)

        dst_acc_tensors = _acc.Tensorvector()
        dst_acc_tensors.resize(len(wav_files))

        try:
            if sr is None:
                srs = _acc.load_audio_batch(wav_files_vec, dst_acc_tensors)
            else:
                srs = _acc.load_audio_batch(wav_files_vec, dst_acc_tensors, sr)
        except Exception as e:
            raise RuntimeError(f"load_audio_batch failed: {e}") from e

        tensor_objs = []
        for acc_tensor in dst_acc_tensors:
            tensor_obj = object.__new__(Tensor)
            tensor_obj._inner = acc_tensor
            tensor_objs.append(tensor_obj)

        return list(zip(tensor_objs, list(srs)))
