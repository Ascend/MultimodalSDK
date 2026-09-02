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
"""媒体处理层：本模块是 MultimodalSDK 接口的唯一调用入口。
Pipeline 中 MultimodalSDK 承担的工作全部集中在这里：
1. ``mm.video_decode``      —— 对操作视频按固定采样率解码为 RGB 帧序列；
2. ``mm.load_audio``        —— 加载从视频分离出的音轨并统一重采样，
3. ``mm.KRangFrameSelector`` —— 以 SOP 步骤的视觉描述为查询文本，
"""

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class VideoFrames:
    """视频解码结果：帧序列与时间戳。"""

    frames: List[np.ndarray]  # RGB, HWC, uint8
    timestamps_s: List[float]  # 每帧对应的视频时刻（秒）
    duration_s: float  # 视频总时长（秒）


@dataclass
class AudioActivity:
    """音频活动检测结果。"""

    sample_rate: int
    duration_s: float
    # 每秒一个 [0,1] 归一化能量值，用于判断该秒是否存在设备/操作声音
    energy_per_second: List[float] = field(default_factory=list)
    activity_threshold: float = 0.1

    def has_sound(self, start_s: float, end_s: float) -> bool:
        """判断 [start_s, end_s) 区间内是否检测到明显声音活动。"""
        lo = max(0, int(start_s))
        hi = min(len(self.energy_per_second), max(lo + 1, int(round(end_s))))
        window = self.energy_per_second[lo:hi]
        if not window:
            return False
        return max(window) >= self.activity_threshold


@dataclass
class StepRange:
    """关键帧选择器为某个 SOP 步骤定位到的候选帧区间。"""

    step_id: str
    frame_indices: List[int]  # 关键帧在采样帧序列中的下标
    key_frames: List[np.ndarray]  # 关键帧图像
    start_s: float  # 区间起始时刻（秒）
    end_s: float  # 区间结束时刻（秒）


def decode_video(video_path: str, sample_fps: float = 1.0) -> VideoFrames:
    """使用 mm.video_decode 将视频按 sample_fps 采样率解码为帧序列。

    :param video_path: mp4 视频路径
    :param sample_fps: 采样帧率（帧/秒），默认每秒 1 帧
    :return: VideoFrames
    """
    if not video_path or not os.path.isfile(video_path):
        raise ValueError(f"video file not found: {video_path}")
    if sample_fps <= 0:
        raise ValueError(f"invalid sample_fps: {sample_fps}")

    # SDK C++ 层文件校验要求 other 组无任何权限（≤0640），前置检查给出可读提示
    # Windows 平台 st_mode 恒返回 0o666，跳过 POSIX 权限校验（由 SDK C++ 层自身校验兜底）
    if os.name == "posix":
        mode = os.stat(video_path).st_mode & 0o777
        if mode & ~0o640:
            raise ValueError(f"input video permission {oct(mode)} exceeds 0640, run: chmod 640 {video_path}")
        if os.stat(video_path).st_uid != os.getuid():
            raise ValueError(f"input video owner mismatch, expected uid {os.getuid()}")

    from mm import video_decode

    duration_s, total_frames, native_fps = _probe_video(video_path)
    sample_num = max(1, int(duration_s * sample_fps))
    sample_num = min(sample_num, total_frames)

    # MultimodalSDK 视频解码：等间隔抽取 sample_num 帧，返回 RGB Image 对象列表。
    # 注意：Image.numpy() 是 C++ 缓冲的零拷贝视图，Image 对象释放后视图悬空，
    # 必须深拷贝后再脱离 Image 对象生命周期使用。
    mm_images = video_decode(video_path, "cpu", set(), sample_num)
    frames = [np.array(img.numpy(), copy=True) for img in mm_images]

    # video_decode 等间隔抽帧，直接用 duration_s 计算步长，避免 total_frames/native_fps 精度损失
    step = duration_s / max(1, len(frames))
    timestamps = [round(i * step, 3) for i in range(len(frames))]
    return VideoFrames(frames=frames, timestamps_s=timestamps, duration_s=duration_s)


def load_audio_activity(
    video_path: str, sample_rate: int = 16000, activity_threshold: float = 0.1, workdir: Optional[str] = None
) -> Optional[AudioActivity]:
    """分离视频音轨并使用 mm.load_audio 加载，计算逐秒声音能量。

    视频不含音轨时返回 None。

    :param video_path: mp4 视频路径
    :param sample_rate: 重采样率，mm.load_audio 的 sr 参数
    :param activity_threshold: 声音活动判定阈值（归一化能量）
    :param workdir: 临时 wav 文件输出目录，默认使用系统临时目录
    :return: AudioActivity 或 None
    """
    from mm import load_audio

    wav_path = None
    tmp_dir = None
    try:
        tmp_dir = workdir or tempfile.mkdtemp(prefix="sop_alert_audio_")
        wav_path = os.path.join(tmp_dir, "extracted_audio.wav")
        if not _extract_wav(video_path, wav_path, sample_rate):
            return None

        # MultimodalSDK 音频加载：wav -> (Tensor, sr)，多通道自动转单通道
        waveform_tensor, sr = load_audio(wav_path, sr=sample_rate)
        waveform = waveform_tensor.numpy().astype(np.float32).reshape(-1)
    finally:
        if workdir is None:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    if waveform.size == 0:
        return None

    # 逐秒 RMS 能量，按最大值归一化
    total_seconds = max(1, int(np.ceil(waveform.size / sr)))
    energies = []
    for sec in range(total_seconds):
        chunk = waveform[sec * sr : (sec + 1) * sr]
        energies.append(float(np.sqrt(np.mean(np.square(chunk)))) if chunk.size else 0.0)
    peak = max(energies) or 1.0
    energies = [e / peak for e in energies]

    return AudioActivity(
        sample_rate=sr,
        duration_s=waveform.size / sr,
        energy_per_second=energies,
        activity_threshold=activity_threshold,
    )


class StepRangeLocator:
    """基于 mm.KRangFrameSelector 的步骤区间定位器。

    对每个 SOP 步骤，以其 visual_query 为查询文本，在采样帧序列中定位
    语义相关的连续帧区间，并在区间内自适应重采样关键帧。
    """

    def __init__(
        self,
        model_path: str,
        device_id: int,
        model_type: str = "cn_clip",
        similar_threshold: float = 0.03,
        image_similar_threshold: float = 0.015,
    ):
        try:
            # MultimodalSDK 26.1.0+（master）：顶层导出
            from mm import KRangFrameSelector
        except ImportError:
            # 26.0.0 发布版 + 手动补装 frame_selector 模块时的完整路径
            from mm.core.frame_selector.frame_selector import KRangFrameSelector

        self._selector = KRangFrameSelector(
            model_path, device_id, model_type, similar_threshold, image_similar_threshold
        )

    def locate(self, step_id: str, visual_query: str, video: VideoFrames, sample_num: int = 8) -> Optional[StepRange]:
        """定位单个步骤的候选帧区间。

        :param step_id: SOP 步骤 ID
        :param visual_query: 步骤画面视觉描述
        :param video: decode_video 的输出
        :param sample_num: 最大关键帧数量
        :return: StepRange；未定位到相关帧时返回 None
        """
        # MultimodalSDK 区间关键帧选择：文本-图像相似度定位相关场景区间
        indices, key_frames = self._selector.select_keyframes(
            query=visual_query, frames=video.frames, sample_num=sample_num, do_resample=True
        )
        if not indices:
            return None
        # 校验索引边界，防止 select_keyframes 返回越界下标导致 IndexError
        valid = [i for i in indices if 0 <= i < len(video.timestamps_s)]
        if not valid:
            return None
        return StepRange(
            step_id=step_id,
            frame_indices=list(indices),
            key_frames=list(key_frames),
            start_s=video.timestamps_s[min(valid)],
            end_s=video.timestamps_s[max(valid)],
        )


def _probe_video(video_path: str) -> Tuple[float, int, float]:
    """通过 ffprobe 获取视频时长、总帧数与帧率。"""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise ValueError("ffprobe not found, please install ffmpeg")
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_packets",
        "-show_entries",
        "stream=nb_read_packets,avg_frame_rate:format=duration",
        "-of",
        "default=noprint_wrappers=1",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)
    if result.returncode != 0:
        raise ValueError(f"ffprobe failed: {result.stderr[:200]}")

    info = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            info[key.strip()] = value.strip()
    try:
        num, den = info.get("avg_frame_rate", "25/1").split("/")
        # 分子为零（如 "0/1"）或分母为零时均回退默认帧率 25.0
        fps = float(num) / float(den) if float(den) and float(num) else 25.0
        duration = float(info["duration"])
        total_frames = int(info.get("nb_read_packets") or int(duration * fps))
    except (KeyError, ValueError, ZeroDivisionError) as e:
        raise ValueError(f"failed to parse ffprobe output: {result.stdout[:200]}") from e
    return duration, total_frames, fps


def _extract_wav(video_path: str, wav_path: str, sample_rate: int) -> bool:
    """用 ffmpeg 从视频分离单声道 wav；无音轨返回 False。"""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise ValueError("ffmpeg not found, please install ffmpeg")
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        video_path,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        wav_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=False)
    if result.returncode == 0 and os.path.isfile(wav_path):
        # SDK 文件校验要求 other 组无任何权限，否则 load_audio 拒绝读取
        os.chmod(wav_path, 0o640)
    # WAV 文件头固定字节数（RIFF header + fmt chunk），含扩展头时实际更大
    _MIN_WAV_HEADER_SIZE = 44
    return result.returncode == 0 and os.path.isfile(wav_path) and os.path.getsize(wav_path) > _MIN_WAV_HEADER_SIZE
