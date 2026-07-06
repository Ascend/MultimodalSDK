# Adapter

## `MultimodalQwen2VLImageProcessor`

This class inherits from `Qwen2VLImageProcessor` in the `transformers` library and uses the in-house acceleration capabilities of the Multimodal SDK to speed up image and video preprocessing for the `Qwen2VL` model. It returns a `BatchFeature` type that is compatible with `transformers`.

**Basic Usage**

For image and video preprocessing, Multimodal SDK currently supports only the processing capabilities of `transformers` 4.51.3.

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
        rescale_factor: Union[int, float] = RESCALE_FACTOR,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = True,
        min_pixels: Optional[int] = 56 * 56,
        max_pixels: Optional[int] = 28 *28 *1280,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        merge_size: int = 2,
        **kwargs,
) -> None:
```

**Parameters**

|Parameter|Type|Description|Optional/Required|Whether Configurable in Current Version|
|--|--|--|--|--|
|do_resize|bool|Whether to resize the input image. The default is `True`.|Optional|×|
|size|Dict[str, int]|Minimum and maximum sizes of the input image. The format must be `{"shortest_edge": min_pixels, "longest_edge": max_pixels}`. The default is `None`.|Optional|√|
|resample|Resampling|Interpolation method for image resizing. The default is `Resampling.BICUBIC`.|Optional|×|
|do_rescale|bool|Whether to rescale pixel values to a specific range. The default is `True`.|Optional|×|
|rescale_factor|Union[int, float]|Pixel rescaling factor, usually `1/255`. The default is `RESCALE_FACTOR`.|Optional|×|
|do_normalize|bool|Whether to normalize the image. The default is `True`.|Optional|×|
|image_mean|Optional[Union[float, List[float]]]|Mean value for image normalization. If `None`, the default value is used. The default value is `[0.48145466, 0.4578275, 0.40821073]`.|Optional|√|
|image_std|Optional[Union[float, List[float]]]|Standard deviation for image normalization. If `None`, the default value is used. The default value is `[0.26862954, 0.26130258, 0.27577711]`.|Optional|√|
|do_convert_rgb|bool|Whether to convert the input image to RGB. The default is `True`.|Optional|×|
|min_pixels|Optional[int]|Minimum number of pixels in the input image. The default is 56 × 56.|Optional|√|
|max_pixels|Optional[int]|Maximum number of pixels in the input image. If the value is exceeded, downsampling is triggered. The default is 28 × 28 × 1280.|Optional|√|
|patch_size|int|Patch size for slicing along the spatial dimensions, in pixels. The default is 14.|Optional|√|
|temporal_patch_size|int|Patch size for slicing along the temporal dimension, in frames. The default is 2, and configuration is not supported.|Optional|×|
|merge_size|int|Size used when merging patches. The default is 2.|Optional|√|
|kwargs|dict|Other extended parameters.|Optional|×|

>[!CAUTION] Note
>When you initialize this preprocessing module, you can pass values for the following parameters, but this version applies the following restrictions:
>
>- **Always-on parameters** (not configurable):
>    - `do_resize`: This parameter is always enabled and cannot be disabled.
>    - `do_rescale`: This parameter is always enabled and cannot be disabled.
>    - `do_normalize`: This parameter is always enabled and cannot be disabled.
>- **Unsupported parameters**:
>    - `do_convert_rgb`: This parameter is not supported. Only RGB images are allowed.
>    - `data_format`: This parameter is not supported. The output data format is fixed to channel-first `(C, H, W)`.
>    - `resample`: This parameter is not supported. The interpolation method is fixed to bicubic interpolation `(BICUBIC)`.
>    - `rescale_factor`: This parameter is not supported. The scaling factor is fixed to `1/255`.
>    - `input_data_format`: This parameter is not supported. Only RGB format is supported, and the channel order is HWC.
>    - `kwargs`: Other extended parameters.

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

This function preprocesses the input images and videos according to specific hyperparameters and returns them in the generic `BatchFeature` format.

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
|images|ImageInput|`ImageInput` supports the following inputs:<p>**Single image**:</p><ul><li>`Image`: Multimodal image object.</li><li>`PIL.Image.Image`: PIL image object.</li><li>`np.ndarray`: NumPy-format image that supports HWC (height × width × channels).</li></ul><p>**Multiple images**:</p>`List[...]`: List of any of the preceding types, representing multiple images.<p>All images must be sized from 10 × 10 to 4096 × 4096, and both width and height must be greater than `patch_size * merge_size`.</p>|Required|√|
|videos|Optional[VideoInput]|`VideoInput` defaults to `None` and supports the following inputs:<ul><li>**Single-frame video**: Single image counts as a one-frame video.</li><li>**Single video sequence**: `SingleVideoInput`.</li><li>**Multiple videos**: `List[SingleVideoInput]`, where each element is a video.</li></ul>For a single-frame image, `NumPy`, `PIL`, or a Multimodal `Image` object is supported.<p>The following forms are supported for a single video input:</p><ul><li>`np.ndarray`: 4D tensor in the shape `[T, H, W, C]`, representing a `T`-frame video.</li><li>`List[FrameInput]`: List of frames.</li></ul>The size of video frames must be from 10 × 10 to 4096 × 4096, and both width and height must be greater than `patch_size * merge_size`.|Optional|√|
|do_resize|Optional[bool]|Whether to resize the image. This parameter cannot be configured.|Optional|×|
|size|Dict[str, int]|Minimum and maximum sizes of the input image. The format must be `{"shortest_edge": min_pixels, "longest_edge": max_pixels}`. The default is `None`. The input `min_pixels` and `max_pixels` must meet the following constraints.|Optional|√|
|min_pixels|Optional[int]|Minimum number of pixels in the input image. If the value is below this threshold, an error may occur or the request may be rejected. The `min_pixels` range is [10 × 10, `max_pixels`).|Optional|√|
|max_pixels|Optional[int]|Maximum number of pixels in the input image. If the value is exceeded, downsampling is triggered. The `max_pixels` range is (`min_pixels`, 4096 × 4096].|Optional|√|
|resample|Optional[Resampling]|The interpolation method is fixed to `BICUBIC`.|Optional|×|
|do_rescale|Optional[bool]|Whether to rescale pixel values.|Optional|×|
|rescale_factor|Optional[float]|The scaling factor is fixed to `1/255`.|Optional|×|
|do_normalize|Optional[bool]|Whether to normalize the image.|Optional|×|
|image_mean|Optional[Union[float, List[float]]]|Normalization mean. If `None`, the initialization value is used. The valid range is [0, 1].|Optional|√|
|image_std|Optional[Union[float, List[float]]]|Normalization standard deviation. If `None`, the initialization value is used. The valid range is (0, `FLOAT32_MAX`).|Optional|√|
|patch_size|Optional[int]|Patch size for slicing along the spatial dimensions, in pixels. If `None`, the initialization value is used.|Optional|√|
|temporal_patch_size|Optional[int]|Patch size for slicing along the temporal dimension, in frames. The default is 2.|Optional|×|
|merge_size|Optional[int]|Size used when merging patches. If `None`, the initialization value is used.|Optional|√|
|do_convert_rgb|Optional[bool]|The input must be an RGB image.|Optional|×|
|return_tensors|Optional[str]|Returned tensor format. It supports `"np"`, `"pt"`, and similar values.|Optional|√|
|data_format|Optional[str]|The output data format is fixed to channel-first `(C, H, W)`.|Optional|×|
|input_data_format|Optional[str]|The input data format is fixed to RGB with HWC layout.|Optional|×|

>[!CAUTION] Note
>When you use this function, the following constraints apply regardless of whether these parameters were initialized:
>
>- **Always-on parameters** (not configurable):
>    - `do_resize`: This parameter is always enabled and cannot be disabled.
>    - `do_rescale`: This parameter is always enabled and cannot be disabled.
>    - `do_normalize`: This parameter is always enabled and cannot be disabled.
>- **Unsupported parameters**:
>    - `do_convert_rgb`: This parameter is not supported. Only RGB images are allowed.
>    - `data_format`: This parameter is not supported. The output data format is fixed to channel-first `(C, H, W)`.
>    - `resample`: This parameter is not supported. The interpolation method is fixed to bicubic interpolation `(BICUBIC)`.
>    - `rescale_factor`: This parameter is not supported. The scaling factor is fixed to `1/255`.
>    - `input_data_format`: This parameter is not supported. Only RGB format is supported, and the channel order is HWC.
>- **Additional constraints that must also be met at runtime are as follows:**
>    - The input image or video frame must be an RGB image of the U8 type. The data layout is limited to HWC, and the size must be from 10 × 10 to 4096 × 4096.
>    - The input `min_pixels` range is [10 × 10, `max_pixels`).
>    - The input `max_pixels` range is (`min_pixels`, 4096 × 4096].
>    - Both the width and height of the input image must be greater than `patch_size * merge_size`.
>    - For each input image or video frame, if the width and height are `w` and `h`, respectively, then `min_pixels < max_pixels`, `max_pixels >= w/h*(patch_size * merge_size)^2`, and `patch_size * merge_size <= h, w` must be satisfied.
>    - For an input video, the width, height, and data layout format of each video frame must be consistent.

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

### `preprocess_image`

**Description**

This function uses the InternVL preprocessing pipeline to process the input image. It is a member function of `InternVL2PreProcessor`. Therefore, you must initialize an `InternVL2PreProcessor` object before use.

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
|image|Union[PIL.Image.Image, Image]|Required|`Image` supports the following inputs:<p>**Single image**:</p><ul><li>`Image`: Multimodal image object.</li><li>`PIL.Image.Image`: PIL image object.</li></ul>|
|input_size|int|Required|Resize size for each cropped image in the InternVL process. The value is limited to `10-8192`.|
|min_num|int|Required|Minimum value used to calculate the target scaling ratio. The value range is [1, 4], and the value must be smaller than `max_num`.|
|max_num|int|Required|Maximum value used to calculate the target scaling ratio. The value range is (`min_num`, 32].|
|use_thumbnail|bool|Required|Whether to add a thumbnail of the original image.|

>[!CAUTION] Note
>When you use this function, note that the limit on the `input_size` scaling ratio is 8192. However, because of how the scaling ratio is calculated, the final resized image may exceed 8192. In that case, the underlying interface blocks the request and returns an error.

**Example**

```python
from mm import Image, InternVL2PreProcessor

image = Image.open("/home/test.jpeg", "cpu")

internVL2PreProcessor = InternVL2PreProcessor()
result = internVL2PreProcessor.preprocess_image(image, 448, 1, 12, True)
```
