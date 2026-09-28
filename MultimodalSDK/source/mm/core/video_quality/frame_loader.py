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

"""Frame loader — uniform-sampling and continuous-fps video decoding.

Decoding strategy (per design spec §3.1.3 / §3.3.4):

1. **Primary: ``mm.acc.video_decode``** — the SDK's C++ accelerated decoder,
   used when the native library (``libcore.so``) is available.
2. **Fallback: ``cv2.VideoCapture``** — pure CPU, portable, slower; used when
   ``mm.acc`` is unavailable (dev environments, wheel-only installs).

Both paths return the same ``LoadedFrames`` structure (BGR uint8 arrays), so
the fallback is transparent to callers.
"""

from __future__ import annotations

from contextlib import suppress
from typing import Any

import numpy as np

from .base import LoadedFrames


def _acc_video_decode_available() -> bool:
    """Return True if ``mm.acc.video_decode`` is importable and callable."""
    try:
        from mm.acc import video_decode
    except Exception:
        return False
    return callable(video_decode)


def _validate_sample_frames(sample_frames: int) -> None:
    """Reject non-positive sample counts before touching the video."""
    if sample_frames < 1:
        raise ValueError(f"sample_frames must be >= 1, got {sample_frames}")


def _linspace_indices(total: int, n: int) -> list[int]:
    """Return *n* evenly-spaced frame indices in ``[0, total)``.

    Integer arithmetic ``i * (total - 1) // (n - 1)`` matches the accelerated
    decoder's own sampling formula, so the indices stay reproducible.
    """
    if total <= 0 or n <= 0:
        return []
    if n == 1:
        return [0]
    if n >= total:
        return list(range(total))
    return (np.arange(n) * (total - 1) // (n - 1)).tolist()


def _time_based_indices(native_fps: float, fps: float, duration: float, total: int) -> list[int]:
    """Return frame indices sampled every ``1 / fps`` seconds.

    Frame *k* is taken at ``k / fps`` seconds, i.e. index
    ``round(k * native_fps / fps)``, clamped to the last real frame.  An
    integer stride would silently change the realised rate (30 fps / 4 fps
    → stride 8 → 3.75 fps), hence the time base.
    """
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")
    if native_fps <= 0 or total <= 0:
        return []
    span = duration if duration > 0 else total / native_fps
    count = max(1, int(span * fps))
    step = native_fps / fps
    return [min(int(round(k * step)), total - 1) for k in range(count)]


def _require_frames(frames: list[np.ndarray], video_path: str) -> None:
    """Fallback guard: raise if decoding produced no frame at all."""
    if not frames:
        raise RuntimeError(f"No frames decoded from video: {video_path}")


def _realised_fps(num_frames: int, duration: float) -> float:
    """Return the sampling rate actually achieved, in frames per second.

    Consumers that scale by the frame interval must not use a rate the
    decoder did not deliver.
    """
    if num_frames <= 0 or duration <= 0:
        return 0.0
    return round(num_frames / duration, 4)


def _acc_load(video_path: str, sample_frames: int, device: str) -> LoadedFrames:
    """Uniform-sampling decode via ``mm.acc.video_decode`` (``sample_num``)."""
    from mm.acc import video_decode, video_info

    _validate_sample_frames(sample_frames)
    info = video_info(video_path)
    fps = info["fps"]
    total = info["n_frames"]
    duration = info["duration_sec"]

    if total <= 0:
        raise ValueError(f"Cannot determine frame count for video: {video_path}")

    # Requested count is clipped to the real frame count before the decoder is
    # called, so the grid is reproducible from (total, n) alone.
    target_indices = _linspace_indices(total, min(sample_frames, total))
    images = video_decode(video_path, "cpu", sample_num=len(target_indices))

    # ``Image.numpy()`` may be a zero-copy view of a decoder-owned buffer — copy
    # so the frames stay valid after the decoder moves on.
    frames = [np.array(img.numpy()) for img in images]
    _require_frames(frames, video_path)
    # The decoder may return fewer frames than requested; re-derive for the
    # actual count with the same integer formula it uses internally.
    target_indices = _linspace_indices(total, len(frames))

    return LoadedFrames(
        frames=frames,
        indices=target_indices,
        meta={
            "fps": fps,
            "width": frames[0].shape[1] if frames else 0,
            "height": frames[0].shape[0] if frames else 0,
            "total_frames": total,
            "duration_sec": duration,
            "device": device,
            "decoder": "mm.acc",
        },
        continuous=False,
    )


def _acc_load_continuous(video_path: str, fps: float, resize_short: int | None, device: str) -> LoadedFrames:
    """Continuous-fps decode via ``mm.acc.video_decode`` / ``frame_indices``."""
    import cv2 as cv
    from mm.acc import video_decode, video_info

    info = video_info(video_path)
    native_fps = info["fps"]
    total = info["n_frames"]
    duration = info["duration_sec"]

    if total <= 0:
        raise ValueError(f"Cannot determine frame count for video: {video_path}")

    target_indices = _time_based_indices(native_fps, fps, duration, total)
    if not target_indices:
        raise ValueError(f"Cannot sample {fps} fps from video: {video_path}")

    images = video_decode(video_path, "cpu", frame_indices=set(target_indices))

    frames = [np.array(img.numpy()) for img in images]
    _require_frames(frames, video_path)

    if resize_short is not None:
        for i, frame in enumerate(frames):
            h, w = frame.shape[:2]
            if min(h, w) > resize_short:
                scale = resize_short / min(h, w)
                frames[i] = cv.resize(
                    frame,
                    (int(w * scale + 0.5), int(h * scale + 0.5)),
                    interpolation=cv.INTER_AREA,
                )

    actual_indices = target_indices[: len(frames)]

    return LoadedFrames(
        frames=frames,
        indices=actual_indices,
        meta={
            "fps": native_fps,
            "target_fps": _realised_fps(len(frames), duration),
            "requested_fps": fps,
            "width": frames[0].shape[1] if frames else 0,
            "height": frames[0].shape[0] if frames else 0,
            "total_frames": total,
            "duration_sec": duration,
            "device": device,
            "decoder": "mm.acc",
            "resize_short": resize_short,
        },
        continuous=True,
    )


def _cv2_load(video_path: str, sample_frames: int, device: str) -> LoadedFrames:
    """Uniform-sampling decode via ``cv2.VideoCapture`` (CPU fallback)."""
    import cv2 as cv

    _validate_sample_frames(sample_frames)
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv.CAP_PROP_FPS) or 24.0
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    if total <= 0:
        cap.release()
        raise ValueError(f"Cannot determine frame count for video: {video_path}")

    n = min(sample_frames, total)
    target_indices = set(_linspace_indices(total, n))

    frames: list[np.ndarray] = []
    indices: list[int] = []
    idx = 0
    while len(frames) < n:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in target_indices:
            frames.append(frame)
            indices.append(idx)
        idx += 1
    cap.release()

    _require_frames(frames, video_path)

    return LoadedFrames(
        frames=frames,
        indices=indices,
        meta={
            "fps": fps,
            "width": width,
            "height": height,
            "total_frames": total,
            "device": device,
            "decoder": "cv2",
        },
        continuous=False,
    )


def _cv2_load_continuous(video_path: str, fps: float, resize_short: int | None, device: str) -> LoadedFrames:
    """Continuous-fps decode via ``cv2.VideoCapture`` (CPU fallback)."""
    import cv2 as cv

    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")

    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    native_fps = cap.get(cv.CAP_PROP_FPS) or 24.0
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    # Keep the frame nearest each ``k / fps`` second (time base rather than an
    # integer stride, which would silently change the realised rate).
    step = native_fps / fps

    frames: list[np.ndarray] = []
    indices: list[int] = []
    idx = 0
    next_target = 0.0
    while True:
        if idx >= round(next_target):
            ok, frame = cap.read()
            if not ok:
                break
            if resize_short is not None:
                h, w = frame.shape[:2]
                if min(h, w) > resize_short:
                    scale = resize_short / min(h, w)
                    frame = cv.resize(
                        frame,
                        (int(w * scale + 0.5), int(h * scale + 0.5)),
                        interpolation=cv.INTER_AREA,
                    )
            frames.append(frame)
            indices.append(idx)
            next_target += step
        else:
            if not cap.grab():
                break
        idx += 1
    cap.release()

    _require_frames(frames, video_path)

    return LoadedFrames(
        frames=frames,
        indices=indices,
        meta={
            "fps": native_fps,
            "target_fps": _realised_fps(len(frames), idx / native_fps),
            "requested_fps": fps,
            "width": width,
            "height": height,
            "total_frames": idx,
            "device": device,
            "decoder": "cv2",
            "resize_short": resize_short,
        },
        continuous=True,
    )


class FrameLoader:
    """Load video frames via uniform sampling or continuous fps decoding.

    Args:
        device: ``"auto"``/``"npu"``/``"cpu"``.  Decoding is always CPU-side;
            *device* is only recorded in ``LoadedFrames.meta`` for downstream
            inference dispatch.
    """

    def __init__(self, device: str = "auto"):
        from .device import DeviceResolver

        self._device = DeviceResolver.resolve(device)
        self._use_acc = _acc_video_decode_available()

    def load(
        self,
        video_path: str,
        sample_frames: int = 8,
        **kwargs: Any,
    ) -> LoadedFrames:
        """Uniformly sample *sample_frames* frames (``continuous=False``)."""
        if self._use_acc:
            # Unsupported codec etc. — fall back to the cv2 path.
            with suppress(Exception):
                return _acc_load(video_path, sample_frames, self._device)
        return _cv2_load(video_path, sample_frames, self._device)

    def load_continuous(
        self,
        video_path: str,
        fps: float = 4.0,
        resize_short: int | None = None,
        **kwargs: Any,
    ) -> LoadedFrames:
        """Decode at a fixed *fps* (``continuous=True``), for optical-flow scorers."""
        if self._use_acc:
            # Unsupported codec etc. — fall back to the cv2 path.
            with suppress(Exception):
                return _acc_load_continuous(video_path, fps, resize_short, self._device)
        return _cv2_load_continuous(video_path, fps, resize_short, self._device)


# Module-level convenience function
def load_frames(
    video_path: str,
    sample_frames: int = 8,
    device: str = "auto",
    **kwargs: Any,
) -> LoadedFrames:
    """Public helper: decode *video_path* once and return ``LoadedFrames``.

    The result can be fed to any scorer's ``score_frames`` — "decode once,
    feed many scorers" without going through the pipeline.
    """
    loader = FrameLoader(device=device)
    if kwargs.get("continuous", False) or kwargs.get("fps") is not None:
        return loader.load_continuous(
            video_path,
            fps=kwargs.get("fps", 4.0),
            resize_short=kwargs.get("resize_short"),
        )
    return loader.load(video_path, sample_frames=sample_frames)
