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
from typing import Union, List, Dict, Tuple, Optional
import numpy as np
import PIL.Image as PILImage
from PIL.Image import Resampling
from concurrent.futures import ThreadPoolExecutor
import math
# disable huggingface connection
# set env before import transformers
import os
os.environ['HF_DATASETS_OFFLINE'] = "1"
os.environ['HF_HUB_OFFLINE'] = "1"

from transformers import BatchFeature, Qwen2VLImageProcessor

from ..acc.wrapper import Image, ImageFormat
from ..comm.log import _Logger as log
from ..acc._impl import acc as _acc
from ..acc.wrapper.util import ObjectWrapper

ImageInput = Union[Image, PILImage.Image, np.ndarray, List[Union[Image, PILImage.Image, np.ndarray]]]
FrameInput = Union[np.ndarray, PILImage.Image, Image]
SingleVideoInput = Union[np.ndarray, List[FrameInput]]
VideoInput = Union[FrameInput, SingleVideoInput, List[SingleVideoInput]]
MEAN = [0.48145466, 0.4578275, 0.40821073]
STD = [0.26862954, 0.26130258, 0.27577711]
DEFAULT_LOWER_BOUNDARY_MIN_PIXELS = 10 * 10
DEFAULT_UPPER_BOUNDARY_MAX_PIXELS = 4096 * 4096
DEFAULT_IMAGE_MIN_SIZE = 10
DEFAULT_IMAGE_MAX_SIZE = 4096
DEFAULT_LOWER_BOUNDARY_MEAN = 0.0
DEFAULT_UPPER_BOUNDARY_MEAN = 1.0
PATCH_SIZE = 14
TEMPORAL_PATCH_SIZE = 2
MERGE_SIZE = 2
MIN_PIXELS = 56 * 56
MAX_PIXELS = 28 * 28 * 1280
RESCALE_FACTOR = 1 / 255
MEAN_STD_LENGTH = 3
GLOBAL_PARAM_WARNING = {
    "do_resize": {
        "type": "forced",
        "message": "Parameter `do_resize` is not configurable and is always enabled."
    },
    "do_rescale": {
        "type": "forced",
        "message": "Parameter `do_rescale` is not configurable and is always enabled."
    },
    "do_normalize": {
        "type": "forced",
        "message": "Parameter `do_normalize` is not configurable and is always enabled."
    },
    "do_convert_rgb": {
        "type": "unsupported",
        "message": "Parameter `do_convert_rgb` is not supported. Only RGB images are allowed."
    },
    "data_format": {
        "type": "unsupported",
        "message": "Parameter `data_format` is not supported. Default output channels will be first (C,H,W)."
    },
    "resample": {
        "type": "unsupported",
        "message": "Parameter `resample` is not supported. Only BICUBIC interpolation is used."
    },
    "rescale_factor": {
        "type": "unsupported",
        "message": "Parameter `rescale_factor` is not supported. Only 1/255 is used."
    },
    "input_data_format": {
        "type": "unsupported",
        "message": "Parameter `input_data_format` is not supported. Only RGB with HWC format supported."
    }
}


def _log_param_warnings(params: dict):
    for k, v in params.items():
        if k in GLOBAL_PARAM_WARNING:
            warning = GLOBAL_PARAM_WARNING[k]
            log.warn(f"{warning['message']} (passed value={v})")


class ImageConstraints:
    """Constraints for validating image and video inputs.

    Args:
        patch_size (int): Patch size used in preprocessing.
        merge_size (int): Merge size used in preprocessing.
        min_pixels (int): Minimum allowed pixels after resizing.
        max_pixels (int): Maximum allowed pixels after resizing.
        temporal_patch_size (int): Temporal patch size (should be 2 for Qwen).
    """

    def __init__(self, patch_size, merge_size, min_pixels, max_pixels, temporal_patch_size):
        self.patch_size = patch_size
        self.merge_size = merge_size
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.temporal_patch_size = temporal_patch_size


def _smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int) -> [int, int]:
    """Resize height and width smartly based on constraints.

    Args:
        height (int): Input height.
        width (int): Input width.
        factor (int): Resize factor, must divide both height and width.
        min_pixels (int): Minimum allowed pixels.
        max_pixels (int): Maximum allowed pixels.

    Returns:
        tuple[int, int]: New (height, width).

    Raises:
        ValueError: If aspect ratio > 200 or dimensions < factor.
    """
    if height < factor or width < factor:
        raise ValueError(f"height:{height} and width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def _check_do_flags(**kwargs):
    """Check if do_* parameters are explicitly disabled. Disabling is not supported.

    Args:
        **kwargs: do_* parameters.

    Raises:
        Warning: Logs warning if any do_* flag is set to False.
    """
    for name, value in kwargs.items():
        if value is None or value is False:
            log.debug(f"{name} is explicitly set to False or None. In this version, {name} is always enabled.")


def _warn_unused_params(**kwargs):
    """Warn about unsupported parameters which are ignored internally.

    Args:
        **kwargs: User-specified parameters.

    Raises:
        Warning: Logs warning for unsupported parameters.
    """
    for key, rule in GLOBAL_PARAM_WARNING.items():
        if key in kwargs and kwargs[key] is not None:
            log.debug(rule["message"])


def _to_mm_image(frame: Union[np.ndarray, Image, PILImage.Image]) -> Image:
    """
    Convert any single frame to mm.Image.

    Args:
        frame (np.ndarray | mm.Image | PIL.Image):
            - np.ndarray: 3D array H,W,3 (RGB)
            - Image: already internal Image
            - PIL.Image: will be converted

    Returns:
        Image: Image object

    Raises:
        TypeError: if input is not supported or ndarray shape is invalid
    """
    if isinstance(frame, Image):
        return frame
    if isinstance(frame, PILImage.Image):
        return Image.from_pillow(frame)
    if isinstance(frame, np.ndarray):
        if frame.ndim == 3 and frame.shape[2] == 3:
            frame = frame.astype(np.uint8)
            return Image.from_numpy(frame, ImageFormat.RGB)
    raise TypeError(f"Unsupported frame type: {type(frame)}, shape={getattr(frame, 'shape', None)}")


def _make_flat_list_of_images(imgs: ImageInput) -> list[Image]:
    """Convert input images into a flat list of internal Image objects.

    Args:
        imgs (ImageInput): Input images.

    Returns:
        list[Image]: Flattened list of Image objects.

    Raises:
        TypeError: If unsupported type is passed.
    """
    if not isinstance(imgs, list):
        imgs = [imgs]
    return [_to_mm_image(img) for img in imgs]


def _process_single_video(video: np.ndarray) -> List[List[Image]]:
    """
    Process a single video ndarray into the standard batch format.

    Args:
        video: np.ndarray of shape [T,H,W,3] or single frame [H,W,3]

    Returns:
        List[List[Image]]: [[frame1, frame2, ..., frameT]]

    Raises:
        ValueError: if ndarray shape is invalid
    """
    if video.ndim == 4 and video.shape[3] == 3 and video.shape[0] > 0:
        res = [[_to_mm_image(f) for f in video]]
        return res
    elif video.ndim == 3 and video.shape[2] == 3:
        res = [[_to_mm_image(video)]]
        return res
    else:
        raise ValueError(f"ndarray must be 3D (H,W,3) or 4D (T,H,W,3) and T>0, got {video.shape}")


def _process_list_of_frames(frames: List[Union[np.ndarray, Image, PILImage.Image]]) -> List[List[Image]]:
    """
    Process a list of frames into a single video batch.

    Args:
        frames: list of frames (np.ndarray 3D, mm.Image, PIL.Image)

    Returns:
        List[List[Image]]: [[frame1, frame2, ...]] — single video batch

    """
    if len(frames) == 0:
        raise ValueError(f"Video list can not be empty!")
    return [[_to_mm_image(f) for f in frames]]


def _process_list_of_videos(videos: List[Union[np.ndarray, Image, PILImage.Image, List]]) -> List[
    List[_acc.Image]]:
    """
    Process a list of videos into the standard batch format.

    Args:
        videos: list containing:
            - np.ndarray 4D [T,H,W,3] → single video
            - list of frames → single video
            - mm.Image or PIL.Image → single-frame video

    Returns:
        List[List[Image]]: list of videos, each video is a list of frames

    Raises:
        TypeError: if any element type is unsupported
    """
    batched = []
    for v in videos:
        if isinstance(v, list):
            # v is a list of frames
            batched.append(_process_list_of_frames(v)[0])
        elif isinstance(v, np.ndarray):
            # v is a single video or single frame
            batched.extend(_process_single_video(v))
        elif isinstance(v, (Image, PILImage.Image)):
            # single frame
            batched.append([_to_mm_image(v)])
        else:
            raise TypeError(f"Unsupported element in list: {type(v)}")
    return batched


def _make_batched_videos(videos: VideoInput) -> List[List[_acc.Image]]:
    """
    Convert input into a standardized List[List[Image]] format.

    Rules:
        - Single frame → [[frame]]
        - Single video ndarray T,H,W,3 → [[frame1, ..., frameT]]
        - List of frames → [[frame1, ..., frameN]]
        - List of videos → list of [[frame1,...], ...]

    Args:
        videos(VideoInput): input images/videos in various supported formats

    Returns:
        List[List[Image]]: standardized batch of videos
    """
    # Single mm.Image / PIL.Image treated as single frame
    if isinstance(videos, (Image, PILImage.Image)):
        return [[_to_mm_image(videos)]]

    # Single ndarray
    if isinstance(videos, np.ndarray):
        return _process_single_video(videos)

    # List input
    if isinstance(videos, list):
        # Check if list of frames (all 3D arrays / Image / PIL.Image)
        if all(isinstance(f, (Image, PILImage.Image, np.ndarray)) and (not isinstance(f, np.ndarray) or f.ndim == 3)
               for f in videos):
            return _process_list_of_frames(videos)
        # Otherwise treat as list of videos
        return _process_list_of_videos(videos)

    raise TypeError(f"Unsupported input type: {type(videos)}")


def _check_image(img: Image, cons: ImageConstraints):
    """Validate a single image against constraints.

    Args:
        img (Image): Input image.
        cons (ImageConstraints): Constraint object.

    Raises:
        ValueError: If image size or constraints are violated.
    """
    h, w = img.height, img.width
    patch_merge = cons.patch_size * cons.merge_size
    if img.format != ImageFormat.RGB:
        raise TypeError(f"Image must be RGB mode!")
    if not (DEFAULT_IMAGE_MIN_SIZE <= h <= DEFAULT_IMAGE_MAX_SIZE) or not (
            DEFAULT_IMAGE_MIN_SIZE <= w <= DEFAULT_IMAGE_MAX_SIZE):
        raise ValueError(
            f"Image height and width must be between "
            f"{DEFAULT_IMAGE_MIN_SIZE} and {DEFAULT_IMAGE_MAX_SIZE}, "
            f"got h={h}, w={w}"
        )
    if patch_merge > h or patch_merge > w:
        raise ValueError(f"patch_size*merge_size={patch_merge} exceeds h={h}, w={w}")
    # Ensure valid grid: scale by patch_merge² and aspect ratio (w/h)
    min_allowed_max_pixels = (patch_merge) ** 2 * w / h
    if cons.max_pixels < min_allowed_max_pixels:
        raise ValueError(f"max_pixels={cons.max_pixels} too small")
    if cons.min_pixels >= cons.max_pixels:
        raise ValueError(f"min_pixels={cons.min_pixels} must < max_pixels={cons.max_pixels}")
    if cons.temporal_patch_size != 2:
        raise ValueError(f"temporal_patch_size={cons.temporal_patch_size}, should be 2 for Qwen")


def _check_image_param(parameter_name: str, value: Union[float, int, List]) -> list[float]:
    """
    Validate image_mean or image_std parameters.

    Rules:
      - Accepts a single number or list of length 3.
      - image_mean: must be within [lower, upper] (inclusive/exclusive based on flag).
      - image_std: must be > 0.

    Args:
        parameter_name (str): "image_mean" or "image_std".
        value: a number or sequence of 3 numbers.

    Returns:
        list[float]: validated values.

    Raises:
        TypeError, ValueError
    """
    if isinstance(value, (int, float)):
        value = [value] * MEAN_STD_LENGTH
    if not isinstance(value, list) or len(value) != MEAN_STD_LENGTH:
        raise TypeError(f"{parameter_name} must be a number or list of length 3")
    if not all(isinstance(v, (int, float)) for v in value):
        raise ValueError(f"All elements in {parameter_name} must be numbers")

    if parameter_name == "image_mean":
        lower, upper = DEFAULT_LOWER_BOUNDARY_MEAN, DEFAULT_UPPER_BOUNDARY_MEAN
        if not all(lower <= v <= upper for v in value):
            raise ValueError(f"{parameter_name} values must be within {DEFAULT_LOWER_BOUNDARY_MEAN} and {DEFAULT_UPPER_BOUNDARY_MEAN}")
    elif parameter_name == "image_std":
        if not all(v > 0 for v in value):
            raise ValueError("All image_std values must be > 0")

    return [float(v) for v in value]



def _resolve_params(defaults: dict, overrides: dict, *args: str) -> dict:
    """
    Resolve parameters by applying defaults and overrides.

    Workflow:
      1. Copy defaults
      2. Apply overrides if provided
      3. Handle special case for "size" dictionary
      4. Validate image_mean and image_std values

    Args:
        defaults: dict of default values
        overrides: dict of values to override defaults
        *args: optional keys for the "size" dictionary (default: ("shortest_edge", "longest_edge"))

    Returns:
        dict: resolved parameters with validation applied

    Raises:
        ValueError: if size dict is incomplete, or mean/std out of range
        TypeError: if mean/std type is invalid
    """
    size_keys = args if args else ("shortest_edge", "longest_edge")
    result = defaults.copy()

    # Step 1 & 2: Apply overrides
    for key, value in overrides.items():
        if value is not None:
            result[key] = value

    # Step 3: Handle "size" dictionary
    if "size" in overrides and overrides["size"] is not None:
        size = overrides["size"]
        if not all(k in size for k in size_keys):
            raise ValueError(f"size must contain {size_keys} keys.")
        result["min_pixels"] = size[size_keys[0]]
        result["max_pixels"] = size[size_keys[1]]

    # Step 4: Validate normalization parameters
    result["image_mean"] = _check_image_param(
        "image_mean", result.get("image_mean")
    )
    result["image_std"] = _check_image_param(
        "image_std", result.get("image_std")
    )
    return result


def _validate_basic_param(cons: ImageConstraints):
    """
    Validate the basic preprocessing parameters for image size constraints.

    Checks that:
    - min_pixels is not below the default lower boundary
    - max_pixels is not above the default upper boundary
    - min_pixels is smaller than max_pixels

    Args:
        cons: An object with `min_pixels` and `max_pixels` attributes.

    Raises:
        ValueError: If any of the validation conditions are not met.
    """
    min_pixels = cons.min_pixels
    max_pixels = cons.max_pixels
    if min_pixels < DEFAULT_LOWER_BOUNDARY_MIN_PIXELS:
        raise ValueError(
            f"min_pixels ({min_pixels}) cannot be less than the lower boundary ({DEFAULT_LOWER_BOUNDARY_MIN_PIXELS})")
    if max_pixels > DEFAULT_UPPER_BOUNDARY_MAX_PIXELS:
        raise ValueError(
            f"max_pixels ({max_pixels}) cannot exceed the upper boundary ({DEFAULT_UPPER_BOUNDARY_MAX_PIXELS})")
    if min_pixels >= max_pixels:
        raise ValueError(f"min_pixels ({min_pixels}) must be smaller than max_pixels ({max_pixels})")


def _convert_to_acc_type(images: Optional[List[Image]], videos: Optional[List[List[Image]]]) -> Tuple[
    Optional[List[_acc.Image]], Optional[List[List[_acc.Image]]]]:
    """
       Convert user-provided Image or list-of-Image objects into internal _acc.Image type.

       This function:
       - Converts each Image in `images` to _acc.Image using `get_inner()`.
       - Converts each frame in each video in `videos` to _acc.Image.
       - Returns None for images or videos if the corresponding list is empty.

       Args:
           images: Optional list of Image objects to convert.
           videos: Optional list of lists of Image objects (videos) to convert.

       Returns:
           A tuple of (converted_images, converted_videos), where each element is either
           a list of _acc.Image objects or None if no images/videos were provided.
    """
    acc_imgs = []
    acc_videos = []
    if images is not None:
        for img in images:
            acc_imgs.append(img.get_inner())
    if videos is not None:
        for video in videos:
            acc_video = []
            for frame in video:
                acc_video.append(frame.get_inner())
            acc_videos.append(acc_video)
    if not acc_imgs:
        acc_imgs = None
    if not acc_videos:
        acc_videos = None
    return acc_imgs, acc_videos


class MultimodalQwen2VLImageProcessor(Qwen2VLImageProcessor):
    """Qwen2VL Preprocessor for images and videos."""

    def __init__(
            self,
            do_resize: bool = True,
            size: Dict[str, int] = None,
            resample: Resampling = Resampling.BICUBIC,
            do_rescale: bool = True,
            rescale_factor: Union[int, float] = RESCALE_FACTOR,
            do_normalize: bool = True,
            image_mean: Optional[Union[float, List[float]]] = None,
            image_std: Optional[Union[float, List[float]]] = None,
            do_convert_rgb: bool = True,
            min_pixels: Optional[int] = MIN_PIXELS,
            max_pixels: Optional[int] = MAX_PIXELS,
            patch_size: int = PATCH_SIZE,
            temporal_patch_size: int = TEMPORAL_PATCH_SIZE,
            merge_size: int = MERGE_SIZE,
            **kwargs,
    ) -> None:
        """Initialize the image processor with preprocessing parameters.

        This constructor sets up all preprocessing flags and parameters,
        logs warnings for unused or unsupported parameters, and enforces
        mandatory flags for resizing, rescaling, and normalization.

        Args:
            do_resize (bool): Whether resizing is applied. Always enabled internally.
            size (Dict[str, int], optional): Target size for resize (ignored).
            resample (Resampling): Interpolation method (only BICUBIC used).
            do_rescale (bool): Whether rescaling is applied. Always enabled internally.
            rescale_factor (Union[int, float]): Factor for rescaling (only 1/255 used).
            do_normalize (bool): Whether normalization is applied. Always enabled internally.
            image_mean (float or List[float], optional): Mean for normalization.
            image_std (float or List[float], optional): Std for normalization.
            do_convert_rgb (bool): Whether to convert images to RGB (ignored, always RGB).
            min_pixels (int, optional): Minimum allowed pixels for images.
            max_pixels (int, optional): Maximum allowed pixels for images.
            patch_size (int): Patch size for preprocessing.
            temporal_patch_size (int): Temporal patch size for video frames.
            merge_size (int): Merge size for patch merging.
            **kwargs: Additional unused parameters.

        Notes:
            - Parameters marked as ignored or unsupported are logged.
            - do_resize, do_rescale, and do_normalize flags are internally enforced
              and cannot be disabled by the user.
            """
        _log_param_warnings(GLOBAL_PARAM_WARNING)
        _check_do_flags(
            do_resize=do_resize,
            do_rescale=do_rescale,
            do_normalize=do_normalize
        )

        _warn_unused_params(
            size=size,
            do_convert_rgb=do_convert_rgb,
            resample=resample,
            rescale_factor=rescale_factor,
        )
        if image_mean is None:
            log.warn("Parameter 'image_mean' not provided, using default")
            image_mean = MEAN
        if image_std is None:
            log.warn("Parameter 'image_std' not provided, using default")
            image_std = STD
        super().__init__(do_resize=do_resize, resample=resample, do_rescale=do_rescale,
                         rescale_factor=rescale_factor,
                         do_normalize=do_normalize, image_mean=image_mean, image_std=image_std,
                         do_convert_rgb=do_convert_rgb,
                         min_pixels=min_pixels, max_pixels=max_pixels, patch_size=patch_size,
                         temporal_patch_size=temporal_patch_size,
                         merge_size=merge_size, **kwargs)

    def _preprocess_with_accelerate(self, images: ImageInput, image_mean: List[float], image_std: List[float],
                                    cons: ImageConstraints) -> [np.ndarray, tuple[int, int, int]]:
        """Internal preprocessing for images or frames.

        Args:
            images (list[Image]): List of input images.
            image_mean (list[float]): Mean for normalization.
            image_std (list[float]): Std for normalization.
            cons (ImageConstraints): Constraint object.

        Returns:
            tuple[np.ndarray, tuple[int,int,int]]:
                - Flattened patches.
                - Grid dimensions (T,H,W).
        """
        h, w = _smart_resize(images[0].height, images[0].width,
                             cons.patch_size * cons.merge_size, cons.min_pixels, cons.max_pixels)
        tensor_acc_list = _acc.Qwen2VLProcessor.Preprocess(images, image_mean, image_std, w, h)
        patches = np.stack([
            np.asarray(ObjectWrapper(t.numpy()))[0]
            for t in tensor_acc_list
        ])
        # NOTE: The following patch partitioning and reshaping strictly follows
        # the Qwen2-VL paper design. Do not modify!
        patches = patches.transpose(0, 3, 1, 2)
        if patches.shape[0] % cons.temporal_patch_size != 0:
            repeats = np.repeat(patches[-1][np.newaxis],
                                cons.temporal_patch_size - (patches.shape[0] % cons.temporal_patch_size), axis=0)
            patches = np.concatenate([patches, repeats], axis=0)

        channel = patches.shape[1]
        grid_t = patches.shape[0] // cons.temporal_patch_size
        grid_h, grid_w = h // cons.patch_size, w // cons.patch_size
        patches = patches.reshape(
            grid_t,
            cons.temporal_patch_size,
            channel,
            grid_h // cons.merge_size,
            cons.merge_size,
            cons.patch_size,
            grid_w // cons.merge_size,
            cons.merge_size,
            cons.patch_size,
        )
        patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        flatten_patches = patches.reshape(
            grid_t * grid_h * grid_w,
            channel * cons.temporal_patch_size * cons.patch_size * cons.patch_size,
        )
        return flatten_patches, (grid_t, grid_h, grid_w)

    def _prepare_and_validate_images_and_videos(
            self,
            images: ImageInput,
            videos: VideoInput,
            cons: ImageConstraints
    ) -> Tuple[Optional[List[Image]], Optional[List[List[Image]]]]:
        """Prepare and validate images and videos.

        This function performs the following steps:
            1. Flattens nested image lists into a single list for uniform processing.
            2. Converts videos into batched format suitable for frame-wise validation.
            3. Validates each image against the provided constraints.
            4. Validates each video frame against the provided constraints.

        Args:
            images (ImageInput): images to process.
            videos (VideoInput): Videos to process.
            cons (ImageConstraints): Constraints to validate images and video frames.

        Returns:
            Tuple:
                images (Optional[List[_acc.Image]]): Flattened and validated list of images,
                    or None if no images were provided.
                videos (Optional[List[List[_acc.Image]]]): Batched and validated list of videos
                    (each video is a list of frames), or None if no videos were provided.
            """
        _validate_basic_param(cons)
        if images is not None:
            images = _make_flat_list_of_images(images)
            with ThreadPoolExecutor() as executor:
                list(executor.map(lambda img: _check_image(img, cons), images))
        if videos is not None:
            videos = _make_batched_videos(videos)
            for video in videos:
                with ThreadPoolExecutor() as executor:
                    list(executor.map(lambda img: _check_image(img, cons), video))
        return images, videos

    def _preprocess_images_and_videos(
            self,
            images: Optional[List["_acc.Image"]],
            videos: Optional[List[List["_acc.Image"]]],
            cons: ImageConstraints,
            image_mean: Union[float, List[float]],
            image_std: Union[float, List[float]]
    ) -> Dict[str, np.ndarray]:
        """Preprocess images and videos into patch tensors.

        This function performs accelerated preprocessing for both images and videos:
            - Images are preprocessed individually and converted into patch tensors.
            - Videos are preprocessed frame-by-frame, then aggregated into patch tensors.

        Args:
            images (Optional[List[_acc.Image]]): List of images to preprocess (maybe None).
            videos (Optional[List[List[_acc.Image]]]): List of videos, where each video is a list of frames (maybe None).
            cons (ImageConstraints): Constraints applied during preprocessing.
            image_mean: Mean value(s) used for normalization.
            image_std: Standard deviation value(s) used for normalization.

        Returns:
            Dict[str, np.ndarray]:
                - If images are provided:
                    {
                        "pixel_values": np.ndarray of preprocessed image patches,
                        "image_grid_thw": np.ndarray of corresponding grid shapes
                    }
                - If videos are provided:
                    {
                        "pixel_values_videos": np.ndarray of preprocessed video patches,
                        "video_grid_thw": np.ndarray of corresponding video grid shapes
                    }
        """
        data = {}

        def process_inputs(inputs, is_video=False):
            pixel_values, vision_grid_thws = [], []
            for inp in inputs:
                patches, grid_thw = self._preprocess_with_accelerate(inp, image_mean, image_std, cons)
                pixel_values.extend(patches)
                vision_grid_thws.append(grid_thw)

            if is_video:
                return {
                    "pixel_values_videos": np.array(pixel_values),
                    "video_grid_thw": np.array(vision_grid_thws)
                }
            else:
                return {
                    "pixel_values": np.array(pixel_values),
                    "image_grid_thw": np.array(vision_grid_thws)
                }

        if images is not None and videos is None:
            data = process_inputs([[img] for img in images], is_video=False)
        elif videos is not None:
            data = process_inputs(videos, is_video=True)
        return data

    def preprocess(self,
                   images: ImageInput,
                   videos: VideoInput = None,
                   do_resize: Optional[bool] = None,
                   size: Dict[str, int] = None,
                   min_pixels: Optional[int] = None,
                   max_pixels: Optional[int] = None,
                   resample: Resampling = None,
                   do_rescale: Optional[bool] = None,
                   rescale_factor: Optional[float] = None,
                   do_normalize: Optional[bool] = None,
                   image_mean: Optional[Union[float, List[float]]] = None,
                   image_std: Optional[Union[float, List[float]]] = None,
                   patch_size: Optional[int] = None,
                   temporal_patch_size: Optional[int] = None,
                   merge_size: Optional[int] = None,
                   do_convert_rgb: Optional[bool] = None,
                   return_tensors: Optional[str] = None,
                   data_format: Optional[str] = None,
                   input_data_format: Optional[str] = None) -> BatchFeature:
        """Public preprocessing API.

        Args:
            images (ImageInput): Image input (single or list).
            videos (VideoInput, optional): Video input.
            do_resize (bool, optional): Resize flag (always enabled).
            size (dict, optional): Unused, ignored.
            min_pixels (int, optional): Minimum allowed pixels.
            max_pixels (int, optional): Maximum allowed pixels.
            resample (Interpolation, optional): Unused.
            do_rescale (bool, optional): Rescale flag (always enabled).
            rescale_factor (float, optional): Unused.
            do_normalize (bool, optional): Normalize flag (always enabled).
            image_mean (list[float], optional): Mean for normalization.
            image_std (list[float], optional): Std for normalization.
            patch_size (int, optional): Patch size.
            temporal_patch_size (int, optional): Temporal patch size.
            merge_size (int, optional): Merge size.
            do_convert_rgb (bool, optional): Unused. Only support RGB
            return_tensors (str, optional): Return tensor type.
            data_format (str, optional): Unused. Only suport CHW
            input_data_format (str, optional): Unused. Only support HWC(RGB Image)

        Returns:
            BatchFeature: Preprocessed feature batch.
        """
        _check_do_flags(do_resize=do_resize, do_rescale=do_rescale, do_normalize=do_normalize)
        _warn_unused_params(
            do_convert_rgb=do_convert_rgb,
            data_format=data_format,
            resample=resample,
            rescale_factor=rescale_factor,
            input_data_format=input_data_format
        )
        defaults = {
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "patch_size": self.patch_size,
            "temporal_patch_size": self.temporal_patch_size,
            "merge_size": self.merge_size,
        }
        overrides = {
            "min_pixels": min_pixels,
            "max_pixels": max_pixels,
            "image_mean": image_mean,
            "image_std": image_std,
            "patch_size": patch_size,
            "temporal_patch_size": temporal_patch_size,
            "merge_size": merge_size,
            "size": size,
        }
        params = _resolve_params(defaults, overrides)
        min_pixels = params["min_pixels"]
        max_pixels = params["max_pixels"]
        image_mean = params["image_mean"]
        image_std = params["image_std"]
        patch_size = params["patch_size"]
        temporal_patch_size = params["temporal_patch_size"]
        merge_size = params["merge_size"]
        cons = ImageConstraints(patch_size, merge_size, min_pixels, max_pixels, temporal_patch_size)
        images, videos = self._prepare_and_validate_images_and_videos(images, videos, cons)
        acc_imgs, acc_videos = _convert_to_acc_type(images, videos)
        data = self._preprocess_images_and_videos(acc_imgs, acc_videos, cons, image_mean, image_std)
        return BatchFeature(data=data, tensor_type=return_tensors)
