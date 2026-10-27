# Adapter

## `MultimodalQwen2VLImageProcessor`

Officially supported. This class inherits from `Qwen2VLImageProcessor` in the `transformers` library. It uses the in-house acceleration capabilities of Multimodal SDK to accelerate image and video preprocessing for the Qwen2VL model, and returns a `BatchFeature` type consistent with that of `transformers`.

**Basic Usage**

For the preprocessing of image and video data, Multimodal SDK currently supports only the processing capabilities provided by `transformers` 4.51.3.

### `__init__`

**Description**

Class initializer.

**Function Prototype**

```python
def __init__(
    self,
    do_resize: bool = True,
    size: Dict[str, int] = None,
    resample: Resampling = Resampling.BICUBIC,
    do_rescale: bool = True,
    rescale_factor: Union[int, float] = 1 / 255,
    do_normalize: bool = True,
    image_mean: Optional[Union[float, List[float]]] = None,
    image_std: Optional[Union[float, List[float]]] = None,
    do_convert_rgb: bool = True,
    min_pixels: Optional[int] = 56 * 56,
    max_pixels: Optional[int] = 28 * 28 * 1280,
    patch_size: int = 14,
    temporal_patch_size: int = 2,
    merge_size: int = 2,
    **kwargs,
) -> None:
```

**Parameters**

|Parameter|Type|Description|Optional/Required|Whether Configurable in Current Version|
|--|--|--|--|--|
|do_resize|bool|Whether to resize the input image. The default is `True`.|Optional|✗|
|size|Dict[str, int]|Minimum and maximum sizes of the input image. The format must be `{"shortest_edge": int, "longest_edge": int}`. If you set this parameter, it overrides `min_pixels` and `max_pixels`. The default is `None`.|Optional|✓|
|resample|Resampling|Interpolation method for image resizing. The default is `Resampling.BICUBIC`.|Optional|✗|
|do_rescale|bool|Whether to perform rescale, that is, scale pixel values to a specific range. The default is `True`.|Optional|✗|
|rescale_factor|Union[int, float]|Pixel scaling factor. The default is `1 / 255`.|Optional|✗|
|do_normalize|bool|Whether to normalize the image. The default is `True`.|Optional|✗|
|image_mean|Optional[Union[float, List[float]]]|Mean for image normalization. If this parameter is `None`, the default value is used. The default value is [0.48145466, 0.4578275, 0.40821073].|Optional|✓|
|image_std|Optional[Union[float, List[float]]]|Standard deviation for image normalization. If this parameter is `None`, the default value is used. The default value is [0.26862954, 0.26130258, 0.27577711].|Optional|✓|
|do_convert_rgb|bool|Whether to convert the input image to RGB. The default is `True`.|Optional|✗|
|min_pixels|Optional[int]|Minimum number of pixels in the input image. The default is `56 * 56`. If you set the `size` parameter, this value is overridden.|Optional|✓|
|max_pixels|Optional[int]|Maximum number of pixels in the input image. Exceeding this value triggers downsampling. The default is `28 * 28 * 1280`. If you set the `size` parameter, this value is overridden.|Optional|✓|
|patch_size|int|Size of the patch for slicing along the spatial dimension, in pixels. The default is 14. A patch is the smallest unit used by the vision encoder to process an image.|Optional|✓|
|temporal_patch_size|int|Size of the patch for slicing along the temporal dimension, in frames. The Qwen2-VL model requires this value to be 2, and passing any other value causes an error. The default is 2.|Optional|✗|
|merge_size|int|Size used when merging patches. The default is 2.|Optional|✓|
|kwargs|dict|Other extended parameters.|Optional|✗|

>[!NOTE]
>When you initialize this preprocessing module, you can pass values for the following parameters, but this version applies the following restrictions:
>
>- **Always-on parameters** (not configurable):
>   - `do_resize`: This parameter is always enabled and cannot be disabled.
>   - `do_rescale`: This parameter is always enabled and cannot be disabled.
>   - `do_normalize`: This parameter is always enabled and cannot be disabled.
>- **Unsupported parameters**:
>   - `do_convert_rgb`: This parameter is not supported. Only RGB images are allowed as input.
>   - `data_format`: This parameter is not supported. The output data format is fixed to channel-first (C, H, W).
>   - `resample`: This parameter is not supported. The interpolation method is fixed to bicubic interpolation (`BICUBIC`).
>   - `rescale_factor`: This parameter is not supported. The scaling factor is fixed to `1 / 255`.
>   - `input_data_format`: This parameter is not supported. Only the RGB format is supported, and the channel order is HWC.
>   - `kwargs`: Other extended parameters.

**Example**

```python
from mm import MultimodalQwen2VLImageProcessor
processor = MultimodalQwen2VLImageProcessor(
    min_pixels=3136,
    max_pixels=518400,
    patch_size=14,
    temporal_patch_size=2,
    merge_size=2,
    image_mean=[0.48145466, 0.4578275, 0.40821073],
    image_std=[0.26862954, 0.26130258, 0.27577711]
)
```

### `preprocess`

**Description**

This function preprocesses the input images and videos according to specific hyperparameters and returns the results in the generic `BatchFeature` format.

**Function Prototype**

```python
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
               input_data_format: Optional[str] = None) -> BatchFeature
```

**Parameters**

|Parameter|Type|Description|Optional/Required|Whether Configurable in Current Version|
|--|--|--|--|--|
|images|ImageInput|Input image (single or multiple). For the types, see [ImageInput description](#imageinput-and-videoinput-type-description). You can pass an empty list to process videos only.|Required|✓|
|videos|Optional[VideoInput]|Input video. The default is `None`. For the types, see [VideoInput description](#imageinput-and-videoinput-type-description).|Optional|✓|
|do_resize|Optional[bool]|Whether to resize the image. This parameter is not configurable.|Optional|✗|
|size|Dict[str, int]|Minimum and maximum sizes of the input image. The format must be `{"shortest_edge": int, "longest_edge": int}`. If you set this parameter, it overrides `min_pixels` and `max_pixels`. The default is `None`.|Optional|✓|
|min_pixels|Optional[int]|Minimum number of pixels in the input image. If the number of pixels is below this value, an error may be raised or the input may be rejected. The `min_pixels` range is [10 × 10, `max_pixels`). If you set the `size` parameter, this value is overridden.|Optional|✓|
|max_pixels|Optional[int]|Maximum number of pixels in the input image. Exceeding this value triggers downsampling. The `max_pixels` range is (`min_pixels`, 4096 × 4096]. If you set the `size` parameter, this value is overridden.|Optional|✓|
|resample|Optional[Resampling]|The interpolation method is fixed to `BICUBIC`.|Optional|✗|
|do_rescale|Optional[bool]|Whether to rescale pixel values.|Optional|✗|
|rescale_factor|Optional[float]|The scaling factor is fixed to `1 / 255`.|Optional|✗|
|do_normalize|Optional[bool]|Whether to normalize the image.|Optional|✗|
|image_mean|Optional[Union[float, List[float]]]|Normalization mean. If this parameter is `None`, the initialization value is used. The value range is [0, 1].|Optional|✓|
|image_std|Optional[Union[float, List[float]]]|Normalization standard deviation. If this parameter is `None`, the initialization value is used. The value range is (0, `FLOAT32_MAX`).|Optional|✓|
|patch_size|Optional[int]|Size of the patch for slicing along the spatial dimension, in pixels. If this parameter is `None`, the initialization value is used.|Optional|✓|
|temporal_patch_size|Optional[int]|Size of the patch for slicing along the temporal dimension, in frames. The Qwen2-VL model requires this value to be 2, and passing any other value causes an error. The default is 2.|Optional|✗|
|merge_size|Optional[int]|Size used when merging patches. If this parameter is `None`, the initialization value is used.|Optional|✓|
|do_convert_rgb|Optional[bool]|The input must be an RGB image.|Optional|✗|
|return_tensors|Optional[str]|Format of the returned tensor. It supports `"np"`, `"pt"`, and similar values.|Optional|✓|
|data_format|Optional[str]|The output data format is fixed to channel-first (C, H, W).|Optional|✗|
|input_data_format|Optional[str]|The input data format is fixed to RGB with the HWC layout.|Optional|✗|

#### `ImageInput` and `VideoInput` Type Description

`ImageInput` supports the following forms:

- **Single image** (any one of the following):
  - `Image`: An image object provided by Multimodal SDK.
  - `PIL.Image.Image`: A PIL image object.
  - `np.ndarray`: A NumPy array with the HWC layout (height × width × channels).
- **Multiple images**: A `List[...]` composed of any of the preceding types.

`VideoInput` supports the following forms:

- **Single-frame video**: One image is considered a 1-frame video, and the types are the same as those for a single image.
- **Single video** `SingleVideoInput`:
  - `np.ndarray`: A 4-dimensional tensor with the shape [T, H, W, C], representing a T-frame video.
  - `List[Union[np.ndarray, PIL.Image.Image, Image]]`: A list composed of frames, where each frame supports `np.ndarray`, PIL, or a Multimodal `Image`.
- **Multiple videos**: `List[SingleVideoInput]`, where each element is a video.

**General Constraints**: The size of an image or video frame is limited to `10×10` to `4096×4096`, and both the width and height must be greater than `patch_size * merge_size`. For the same video, the width, height, and data layout format of each frame must be consistent.

>[!NOTE]
>When you use this function, the following constraints apply regardless of whether you initialized the following parameters:
>
>- **Always-on parameters** (not configurable):
>   - `do_resize`: This parameter is always enabled and cannot be disabled.
>   - `do_rescale`: This parameter is always enabled and cannot be disabled.
>   - `do_normalize`: This parameter is always enabled and cannot be disabled.
>- **Unsupported parameters**:
>   - `do_convert_rgb`: This parameter is not supported. Only RGB images are allowed as input.
>   - `data_format`: This parameter is not supported. The output data format is fixed to channel-first (C, H, W).
>   - `resample`: This parameter is not supported. The interpolation method is fixed to bicubic interpolation (`BICUBIC`).
>   - `rescale_factor`: This parameter is not supported. The scaling factor is fixed to `1 / 255`.
>   - `input_data_format`: This parameter is not supported. Only the RGB format is supported, and the channel order is HWC.
>- **The following additional constraints must also be satisfied at runtime**:
>   - The input image or video frame must be an RGB image of the U8 type. The data layout is limited to HWC, and the size is limited to 10 × 10 to 4096 × 4096.
>   - The input `min_pixels` range is [10 × 10, `max_pixels`).
>   - The input `max_pixels` range is (`min_pixels`, 4096 × 4096].
>   - Both the width and height of the input `Image` are greater than `patch_size * merge_size`.
>   - For each input image or each video frame, if its width and height are w and h, respectively, the following must be satisfied: `min_pixels < max_pixels`, `max_pixels >= (patch_size * merge_size)^2 * w / h`, and `patch_size * merge_size <= h, w`.
>   - For an input video, the width, height, and data layout format of each video frame must be consistent.

**Example**

```python
from mm import MultimodalQwen2VLImageProcessor
import numpy as np
processor = MultimodalQwen2VLImageProcessor(
    min_pixels=3136,
    max_pixels=518400,
    patch_size=14,
    temporal_patch_size=2,
    merge_size=2,
    image_mean=[0.48145466, 0.4578275, 0.40821073],
    image_std=[0.26862954, 0.26130258, 0.27577711]
)

def random_image_array():
    h = 1024
    w = 1024
    arr = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    return arr

def random_video_list(num_frames=None):
    num_frames = num_frames or np.random.randint(2, 5)
    return [random_image_array() for _ in range(num_frames)]
video = [random_video_list()]
result = processor.preprocess(images=[], videos=video)
```

## `InternVL2PreProcessor`

This class has no custom `__init__` parameters. `preprocess_image` is a static method that you can call directly through the class or an instance.

### `preprocess_image`

**Description**

This function processes the input image using the InternVL preprocessing process.

**Function Prototype**

```python
def preprocess_image(
    image: Union[PIL.Image.Image, Image],
    input_size: int,
    min_num: int,
    max_num: int,
    use_thumbnail: bool,
) -> torch.Tensor:
```

**Parameters**

|Parameter|Type|Optional/Required|Description|
|--|--|--|--|
|image|[PIL.Image.Image, Image]|Required|`Image` supports a single image: a Multimodal `Image` object or a `PIL.Image.Image` object.|
|input_size|int|Required|The size to which each cropped image is resized in the InternVL process (limited to 10 to 8192).|
|min_num|int|Required|The minimum number used to calculate the target scaling ratio. The value range is limited to [1, 4], and the value must be smaller than `max_num`.|
|max_num|int|Required|The maximum number used to calculate the target scaling ratio. The value range is limited to (`min_num`, 32].|
|use_thumbnail|bool|Required|Whether to add a thumbnail of the original image.|

>[!NOTE]
>When you use this function, note that the scaling ratio limit for `input_size` is 8192. However, due to how the scaling ratio is calculated, the size to which the original image needs to be scaled may exceed 8192. In that case, the underlying interface blocks the operation and returns an error.

**Example**

```python
from mm import Image, InternVL2PreProcessor

image = Image.open("/home/test.jpeg", "cpu")

internvl2_preprocessor = InternVL2PreProcessor()
result = internvl2_preprocessor.preprocess_image(image, 448, 1, 12, True)
```
