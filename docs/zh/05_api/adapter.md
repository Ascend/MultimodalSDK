# Adapter

## MultimodalQwen2VLImageProcessor

正式支持。该类继承 transformers 库中的 Qwen2VLImageProcessor，将使用多模态内部的加速能力对 Qwen2VL 模型的图像/视频预处理环节进行加速，返回与 transformers 一致的 BatchFeature 类型。

**使用基本说明**

对于图像/视频数据的预处理，多模态当前仅支持对接 transformers 4.51.3 版本的处理能力。

### `__init__`

**功能描述**

类初始化函数。

**函数原型**

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

**参数说明**

|参数名|类型|说明|可选/必选|当前版本是否支持设置|
|--|--|--|--|--|
|do_resize|bool|是否对输入图像进行缩放，默认为 True。|可选|✗|
|size|Dict[str, int]|输入图像的最小和最大尺寸，格式必须为{"shortest_edge": int, "longest_edge": int}。若设置此参数，将覆盖 min_pixels 和 max_pixels。默认为 None。|可选|✓|
|resample|Resampling|图像缩放插值方式，默认为 Resampling.BICUBIC。|可选|✗|
|do_rescale|bool|是否执行 rescale（缩放像素值到特定范围），默认为 True。|可选|✗|
|rescale_factor|Union[int, float]|像素缩放因子，默认为 1 / 255。|可选|✗|
|do_normalize|bool|是否对图像执行归一化，默认为 True。|可选|✗|
|image_mean|Optional[Union[float, List[float]]]|图像归一化均值，若为 None，则使用默认值。默认值为[0.48145466, 0.4578275, 0.40821073]。|可选|✓|
|image_std|Optional[Union[float, List[float]]]|图像归一化标准差，若为 None，则使用默认值。默认值为[0.26862954, 0.26130258, 0.27577711]。|可选|✓|
|do_convert_rgb|bool|是否将输入图像转换为 RGB，默认为 True。|可选|✗|
|min_pixels|Optional[int]|输入图像的最小像素数。默认为 56 \* 56。若设置 size 参数，此值将被覆盖。|可选|✓|
|max_pixels|Optional[int]|输入图像的最大像素数。高于该值时会触发降采样。默认为 28 \* 28 \* 1280。若设置 size 参数，此值将被覆盖。|可选|✓|
|patch_size|int|空间维度切分 patch 的大小（像素），默认为 14。patch 为视觉编码器处理图像时的最小单元。|可选|✓|
|temporal_patch_size|int|时间维度切分 patch 的大小（帧）。Qwen2-VL 模型要求该值必须为 2，传入其他值会报错。默认为 2。|可选|✗|
|merge_size|int|patch 合并时的大小，默认为 2。|可选|✓|
|kwargs|dict|其他扩展参数。|可选|✗|

>[!CAUTION] 注意
>初始化本预处理模块时，以下参数均可以传入值，但在此版本中具有以下限制：
>
>- **固定启用参数**（不可配置）：
>   - do_resize：该参数始终启用，不支持关闭。
>   - do_rescale：该参数始终启用，不支持关闭。
>   - do_normalize：该参数始终启用，不支持关闭。
>- **不支持的参数**：
>   - do_convert_rgb：不支持该参数，仅允许输入 RGB 图像。
>   - data_format：不支持该参数，输出数据格式固定为通道优先 (C, H, W)。
>   - resample：不支持该参数，插值方式固定为双三次插值 (BICUBIC)。
>   - rescale_factor：不支持该参数，缩放因子固定为 1 / 255。
>   - input_data_format：不支持该参数，仅支持 RGB 格式，通道顺序为 HWC。
>   - kwargs: 其他扩展参数。

**示例**

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

### preprocess

**功能描述**

预处理函数，对输入的图像/视频，根据特定的超参数进行预处理，以 BatchFeature 的通用形式返回。

**函数原型**

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

**参数说明**

|参数名|类型|说明|可选/必选|当前版本是否支持设置|
|--|--|--|--|--|
|images|ImageInput|输入图像（单张或多张），类型见下方 [ImageInput 说明](#imageinput-与-videoinput-类型说明)。可传空列表仅处理视频。|必选|✓|
|videos|Optional[VideoInput]|输入视频，默认为 `None`；类型见下方 [VideoInput 说明](#imageinput-与-videoinput-类型说明)。|可选|✓|
|do_resize|Optional[bool]|是否对图像进行缩放。该参数不可配置。|可选|✗|
|size|Dict[str, int]|输入图像的最小和最大尺寸，格式必须为{"shortest_edge": int, "longest_edge": int}。若设置此参数，将覆盖 min_pixels 和 max_pixels。默认为 None。|可选|✓|
|min_pixels|Optional[int]|输入图像的最小像素数，低于该值时可能报错或拒绝。min_pixels 取值范围为[10 * 10, max_pixels)。若设置 size 参数，此值将被覆盖。|可选|✓|
|max_pixels|Optional[int]|输入图像的最大像素数，高于该值时会触发降采样。max_pixels 取值范围为(min_pixels, 4096 * 4096]。若设置 size 参数，此值将被覆盖。|可选|✓|
|resample|Optional[Resampling]|插值方式固定为 BICUBIC。|可选|✗|
|do_rescale|Optional[bool]|是否执行像素值缩放。|可选|✗|
|rescale_factor|Optional[float]|缩放因子固定为 1 / 255。|可选|✗|
|do_normalize|Optional[bool]|是否执行归一化。|可选|✗|
|image_mean|Optional[Union[float, List[float]]]|归一化均值，若为 None，则使用初始化值，取值范围为[0, 1]。|可选|✓|
|image_std|Optional[Union[float, List[float]]]|归一化标准差，若为 None，则使用初始化值，取值范围在(0, FLOAT32_MAX)。|可选|✓|
|patch_size|Optional[int]|空间维度切分 patch 的大小（像素），若为 None，则使用初始化值。|可选|✓|
|temporal_patch_size|Optional[int]|时间维度切分 patch 的大小（帧）。Qwen2-VL 模型要求该值必须为 2，传入其他值会报错。默认为 2。|可选|✗|
|merge_size|Optional[int]|patch 合并时的大小，若为 None，则使用初始化值。|可选|✓|
|do_convert_rgb|Optional[bool]|输入必须为 RGB 图像。|可选|✗|
|return_tensors|Optional[str]|返回张量格式，支持"np"、"pt" 等。|可选|✓|
|data_format|Optional[str]|输出数据格式固定为通道优先 (C, H, W)。|可选|✗|
|input_data_format|Optional[str]|输入数据格式固定为 RGB 且为 HWC 排布。|可选|✗|

#### ImageInput 与 VideoInput 类型说明

`ImageInput` 支持以下形式：

- **单张图像**（任选其一）：
  - `Image`：多模态 SDK 的图像对象
  - `PIL.Image.Image`：PIL 图像对象
  - `np.ndarray`：HWC 排布（高度 × 宽度 × 通道）的 Numpy 数组
- **多张图像**：上述任意类型组成的 `List[...]`

`VideoInput` 支持以下形式：

- **单帧视频**：一张图像即视为 1 帧视频，类型同上
- **单个视频** `SingleVideoInput`：
  - `np.ndarray`：形状 `[T, H, W, C]` 的 4 维张量，表示 T 帧视频
  - `List[Union[np.ndarray, PIL.Image.Image, Image]]`：逐帧组成的列表，每帧支持 `np.ndarray` / PIL / 多模态 `Image`
- **多个视频**：`List[SingleVideoInput]`，每个元素是一个视频

**通用约束**：图像或视频帧大小限制为 `10×10` 至 `4096×4096`，且宽和高均需大于 `patch_size * merge_size`；同一视频中各帧的宽、高及数据排布格式需一致。

>[!CAUTION] 注意
>使用本函数时，无论是否在初始化中初始化过以下参数，其均具有以下约束：
>
>- **固定启用参数**（不可配置）：
>   - do_resize：该参数始终启用，不支持关闭。
>   - do_rescale：该参数始终启用，不支持关闭。
>   - do_normalize：该参数始终启用，不支持关闭。
>- **不支持的参数**：
>   - do_convert_rgb：不支持该参数，仅允许输入 RGB 图像。
>   - data_format：不支持该参数，输出数据格式固定为通道优先 (C, H, W)。
>   - resample：不支持该参数，插值方式固定为双三次插值 (BICUBIC)。
>   - rescale_factor：不支持该参数，缩放因子固定为 1 / 255。
>   - input_data_format：不支持该参数，仅支持 RGB 格式，通道顺序为 HWC。
>- **使用时还需满足的额外约束如下**：
>   - 输入图像或视频帧为 U8 类型 RGB 图像，数据排布限制为 HWC，大小限制为 10 \* 10 ~ 4096 \* 4096。
>   - 输入 min_pixels 范围为 [10 \* 10, max_pixels)。
>   - 输入 max_pixels 范围为 (min_pixels, 4096 \* 4096]。
>   - 输入 Image 的宽和高均大于 patch_size * merge_size。
>   - 对于传入的每一张图像，或者每一个视频帧，若其宽高分别为 w, h，则需满足 `min_pixels < max_pixels`，`max_pixels >= w/h * (patch_size * merge_size)^2`，`patch_size * merge_size <= h, w`。
>   - 对于传入的视频，每一个视频帧的宽、高及数据排布格式需一致。

**示例**

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

## InternVL2PreProcessor

该类无自定义 `__init__` 参数，`preprocess_image` 为静态方法，可直接通过类或实例调用。

### preprocess_image

**功能描述**

预处理函数，使用 InternVL 预处理流程处理输入的图像。

**函数原型**

```python
def preprocess_image(
    image: Union[PIL.Image.Image, Image],
    input_size: int,
    min_num: int,
    max_num: int,
    use_thumbnail: bool,
) -> torch.Tensor:
```

**参数说明**

|参数名|类型|可选/必选|说明|
|--|--|--|--|
|image|[PIL.Image.Image, Image]|必选|Image 支持单张图像：多模态 Image 对象；PIL.Image.Image 对象。|
|input_size|int|必选|InternVL 处理流程中每个被 crop 的图像缩放大小（限制为 10 ~ 8192）。|
|min_num|int|必选|用于计算目标缩放比例的最小数量。限制取值范围为[1, 4]且小于 max_num。|
|max_num|int|必选|用于计算目标缩放比例的最大数量。限制取值范围为(min_num, 32]。|
|use_thumbnail|bool|必选|是否加入原图的缩略图。|

>[!CAUTION] 注意
>使用本函数时请注意对于 input_size 缩放比的限制为 8192，但由于缩放比例计算的问题，最终计算出需要将原图缩放的大小可能会超过 8192，此时会被底层接口拦截报错。

**示例**

```python
from mm import Image, InternVL2PreProcessor

image = Image.open("/home/test.jpeg", "cpu")

internvl2_preprocessor = InternVL2PreProcessor()
result = internvl2_preprocessor.preprocess_image(image, 448, 1, 12, True)
```
