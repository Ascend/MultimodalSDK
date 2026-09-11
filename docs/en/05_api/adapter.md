# Adapter

## `MultimodalQwen2VLImageProcessor`

This class is officially supported. It inherits from `Qwen2VLImageProcessor` in the `transformers` library and uses the Multimodal SDK's internal acceleration capabilities to accelerate image and video preprocessing for the `Qwen2VL` model. It returns a `BatchFeature` type that is compatible with `transformers`.

**Basic Usage**

For image and video preprocessing, the Multimodal SDK currently supports only the processing capabilities of `transformers` 4.51.3.

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

| Parameter | Type | Description | Required (Yes/No) | Configurable in Current Version |
| -- | -- | -- | -- | -- |
| `do_resize` | `bool` | Resizes the input image. The default is `True`. | No | ✗ |
| `size` | `Dict[str, int]` | Specifies the minimum and maximum sizes of the input image. The format must be `{"shortest_edge": int, "longest_edge": int}`. If this parameter is set, it overrides `min_pixels` and `max_pixels`. The default is `None`. | No | ✓ |
| `resample` | `Resampling` | Specifies the interpolation method for image resizing. The default is `Resampling.BICUBIC`. | No | ✗ |
| `do_rescale` | `bool` | Rescales pixel values to a specific range. The default is `True`. | No | ✗ |
| `rescale_factor` | `Union[int, float]` | Specifies the pixel rescaling factor. The default is `1 / 255`. | No | ✗ |
| `do_normalize` | `bool` | Normalizes the image. The default is `True`. | No | ✗ |
| `image_mean` | `Optional[Union[float, List[float]]]` | Specifies the mean value for image normalization. If `None`, the default value is used. Default value: `[0.48145466, 0.4578275, 0.40821073]`. | No | ✓ |
| `image_std` | `Optional[Union[float, List[float]]]` | Specifies the standard deviation for image normalization. If `None`, the default value is used. Default value: `[0.26862954, 0.26130258, 0.27577711]`. | No | ✓ |
| `do_convert_rgb` | `bool` | Converts the input image to RGB. The default is `True`. | No | ✗ |
| `min_pixels` | `Optional[int]` | Specifies the minimum number of pixels in the input image. The default is `56 * 56`. If `size` is set, this value is overridden. | No | ✓ |
| `max_pixels` | `Optional[int]` | Specifies the maximum number of pixels in the input image. Downsampling is triggered when this value is exceeded. The default is `28 * 28 * 1280`. If `size` is set, this value is overridden. | No | ✓ |
| `patch_size` | `int` | Specifies the patch size for slicing along the spatial dimensions, in pixels. The default is `14`. A patch is the smallest unit processed by the vision encoder when processing an image. | No | ✓ |
| `temporal_patch_size` | `int` | Specifies the patch size for slicing along the temporal dimension, in frames. The `Qwen2-VL` model requires this value to be `2`; passing any other value results in an error. The default is `2`. | No | ✗ |
| `merge_size` | `int` | Specifies the size used when merging patches. The default is `2`. | No | ✓ |
| `kwargs` | `dict` | Specifies other extended parameters. | No | ✗ |

> [!NOTE]
> When you initialize this preprocessing module, you can pass values for the following parameters, but this version applies the following restrictions:
>
> - **Always-on parameters** (not configurable):
>   - `do_resize`: This parameter is always enabled and cannot be disabled.
>   - `do_rescale`: This parameter is always enabled and cannot be disabled.
>   - `do_normalize`: This parameter is always enabled and cannot be disabled.
> - **Unsupported parameters**:
>   - `do_convert_rgb`: This parameter is not supported. Only RGB images are allowed.
>   - `data_format`: This parameter is not supported. The output data format is fixed to channel-first `(C, H, W)`.
>   - `resample`: This parameter is not supported. The interpolation method is fixed to bicubic interpolation `(BICUBIC)`.
>   - `rescale_factor`: This parameter is not supported. The scaling factor is fixed to `1 / 255`.
>   - `input_data_format`: This parameter is not supported. Only RGB format is supported, with the channel order fixed to HWC.
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

Preprocesses the input images and videos according to specific hyperparameters and returns them in the generic `BatchFeature` format.

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

| Parameter | Type | Description | Required (Yes/No) | Configurable in Current Version |
| -- | -- | -- | -- | -- |
| `images` | `ImageInput` | Specifies the input images, either a single image or multiple images. See [ImageInput and VideoInput Type Descriptions](#imageinput-and-videoinput-type-descriptions) for supported types. An empty list can be passed to process videos only. | Yes | ✓ |
| `videos` | `Optional[VideoInput]` | Specifies the input videos. The default is `None`. See [ImageInput and VideoInput Type Descriptions](#imageinput-and-videoinput-type-descriptions) for supported types. | No | ✓ |
| `do_resize` | `Optional[bool]` | Resizes the images. This parameter cannot be configured. | No | ✗ |
| `size` | `Dict[str, int]` | Specifies the minimum and maximum sizes of the input image. The format must be `{"shortest_edge": int, "longest_edge": int}`. If this parameter is set, it overrides `min_pixels` and `max_pixels`. The default is `None`. | No | ✓ |
| `min_pixels` | `Optional[int]` | Specifies the minimum number of pixels in the input image. An error may occur or the request may be rejected if the value is below this threshold. The value range is `[10 * 10, max_pixels)`. If `size` is set, this value is overridden. | No | ✓ |
| `max_pixels` | `Optional[int]` | Specifies the maximum number of pixels in the input image. Downsampling is triggered when this value is exceeded. The value range is `(min_pixels, 4096 * 4096]`. If `size` is set, this value is overridden. | No | ✓ |
| `resample` | `Optional[Resampling]` | Specifies the interpolation method. Bicubic interpolation `(BICUBIC)` is used. | No | ✗ |
| `do_rescale` | `Optional[bool]` | Rescales pixel values. | No | ✗ |
| `rescale_factor` | `Optional[float]` | Specifies the scaling factor. The factor is fixed at `1 / 255`. | No | ✗ |
| `do_normalize` | `Optional[bool]` | Normalizes the image. | No | ✗ |
| `image_mean` | `Optional[Union[float, List[float]]]` | Specifies the normalization mean. If `None`, the initialization value is used. The value range is `[0, 1]`. | No | ✓ |
| `image_std` | `Optional[Union[float, List[float]]]` | Specifies the normalization standard deviation. If `None`, the initialization value is used. The value range is `(0, FLOAT32_MAX)`. | No | ✓ |
| `patch_size` | `Optional[int]` | Specifies the patch size for slicing along the spatial dimensions, in pixels. If `None`, the initialization value is used. | No | ✓ |
| `temporal_patch_size` | `Optional[int]` | Specifies the patch size for slicing along the temporal dimension, in frames. The `Qwen2-VL` model requires this value to be `2`; passing any other value results in an error. The default is `2`. | No | ✗ |
| `merge_size` | `Optional[int]` | Specifies the size used when merging patches. If `None`, the initialization value is used. | No | ✓ |
| `do_convert_rgb` | `Optional[bool]` | The input must be an RGB image. | No | ✗ |
| `return_tensors` | `Optional[str]` | Specifies the returned tensor format. Supports `"np"`, `"pt"`, and similar values. | No | ✓ |
| `data_format` | `Optional[str]` | Uses a fixed output data format of channel-first `(C, H, W)`. | No | ✗ |
| `input_data_format` | `Optional[str]` | Uses a fixed input data format of RGB with HWC layout. | No | ✗ |

#### ImageInput and VideoInput Type Descriptions

`ImageInput` supports the following forms:

- **Single image** (any one of the following):
  - `Image`: Multimodal SDK image object.
  - `PIL.Image.Image`: PIL image object.
  - `np.ndarray`: NumPy array in HWC layout (height × width × channels).
- **Multiple images**: A `List[...]` containing any of the preceding types.

`VideoInput` supports the following forms:

- **Single-frame video**: A single image is treated as a one-frame video, using any of the image types described above.
- **Single video** (`SingleVideoInput`):
  - `np.ndarray`: A 4D tensor with the shape `[T, H, W, C]`, representing a `T`-frame video.
  - `List[Union[np.ndarray, PIL.Image.Image, Image]]`: A list of frames. Each frame can be an `np.ndarray`, `PIL.Image.Image`, or Multimodal SDK `Image` object.
- **Multiple videos**: A `List[SingleVideoInput]`, where each element is a video.

**General constraints**: The size of each image or video frame must be from `10 × 10` to `4096 × 4096`, and both the width and height must be greater than `patch_size * merge_size`. The width, height, and data layout of all frames in the same video must be consistent.

> [!NOTE]
> When you use this function, the following constraints apply regardless of whether these parameters were initialized:
>
> - **Always-on parameters** (not configurable):
>   - `do_resize`: This parameter is always enabled and cannot be disabled.
>   - `do_rescale`: This parameter is always enabled and cannot be disabled.
>   - `do_normalize`: This parameter is always enabled and cannot be disabled.
> - **Unsupported parameters**:
>   - `do_convert_rgb`: This parameter is not supported. Only RGB images are allowed.
>   - `data_format`: This parameter is not supported. The output data format is fixed to channel-first `(C, H, W)`.
>   - `resample`: This parameter is not supported. The interpolation method is fixed to bicubic interpolation `(BICUBIC)`.
>   - `rescale_factor`: This parameter is not supported. The scaling factor is fixed to `1 / 255`.
>   - `input_data_format`: This parameter is not supported. Only RGB format is supported, with the channel order fixed to HWC.
> - **Additional constraints that must also be met at runtime are as follows:**
>   - The input image or video frame must be an RGB image of U8 type, with HWC layout and a size from `10 * 10` to `4096 * 4096`.
>   - The input `min_pixels` range is `[10 * 10, max_pixels)`.
>   - The input `max_pixels` range is `(min_pixels, 4096 * 4096]`.
>   - Both the width and height of the input image must be greater than `patch_size * merge_size`.
>   - For each input image or video frame, if its width and height are `w` and `h`, respectively, `min_pixels < max_pixels`, `max_pixels >= (patch_size * merge_size)^2 * w / h`, and `patch_size * merge_size <= h, w` must be satisfied.
>   - For an input video, the width, height, and data layout of each video frame must be consistent.

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

This class has no custom `__init__` parameters. `preprocess_image` is a static method and can be called directly through either the class or an instance.

### `preprocess_image`

**Description**

Preprocesses the input image using the InternVL preprocessing pipeline.

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

| Parameter | Type | Required (Yes/No) | Description |
| -- | -- | -- | -- |
| `image` | `Union[PIL.Image.Image, Image]` | Yes | Accepts a single image. Supported types are the Multimodal SDK `Image` object and `PIL.Image.Image` object. |
| `input_size` | `int` | Yes | Specifies the size to which each cropped image is resized in the InternVL preprocessing pipeline. The value ranges from 10 to 8192. |
| `min_num` | `int` | Yes | Specifies the minimum value used to calculate the target scaling ratio. The value range is `[1, 4]`, and the value must be smaller than `max_num`. |
| `max_num` | `int` | Yes | Specifies the maximum value used to calculate the target scaling ratio. The value range is `(min_num, 32]`. |
| `use_thumbnail` | `bool` | Yes | Adds a thumbnail of the original image. |

> [!NOTE]
> When you use this function, note that the `input_size` scaling limit is 8192. However, because of how the scaling ratio is calculated, the final size to which the original image needs to be resized may exceed 8192. In this case, the underlying interface blocks the request and returns an error.

**Example**

```python
from mm import Image, InternVL2PreProcessor

image = Image.open("/home/test.jpeg", "cpu")

internvl2_preprocessor = InternVL2PreProcessor()
result = internvl2_preprocessor.preprocess_image(image, 448, 1, 12, True)
```
