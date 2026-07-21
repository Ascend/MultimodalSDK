# 功能函数参考

## mm.Tensor

本文档适用于 Multimodal SDK 最新发布版本。接口异常通常以 `ValueError`、`TypeError`、`RuntimeError` 或 `ImportError` 抛出，错误码处理建议请参见[附录 - 错误码](../06_references/appendix.md#错误码)。

Tensor 类将被用于承载任意模态的通用数据，实现通用数据的创建、管理以及数据复制等操作。

### Tensor 属性列表

|属性名|类型|说明|备注|
|--|--|--|--|
|device|str|Tensor 所在设备。|默认为"cpu"。|
|dtype|DataType|Tensor 数据类型。|默认为 DataType.FLOAT32。|
|shape|list|Tensor 的维度信息。|默认为空 list。|
|format|TensorFormat|Tensor 数据排布类型。|默认为 TensorFormat.ND。|
|nbytes|int|Tensor 数据占用字节数。|默认为 0。|

### Tensor.set_format

**功能描述**

设置数据排布格式。

**函数原型**

```python
set_format(tensor_format: TensorFormat)
```

**输入参数说明**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|tensor_format|TensorFormat|必选|输入需要设置的排布格式。支持 ND、NHWC 和 NCHW。若设置为 NHWC 或者 NCHW，需要确保 tensor 的 shape 有四维。|

**示例**

```python
from mm import Tensor, TensorFormat
tensor = Tensor()
tensor.set_format(TensorFormat.ND)
```

### Tensor.clone

**功能描述**

深拷贝 Tensor 实例对象为新的 Tensor 实例。

**函数原型**

```python
clone()-> Tensor
```

**返回值说明**

|数据类型|说明|
|--|--|
|Tensor|新的 Tensor 实例。|

**示例**

```python
from mm import Tensor
tensor = Tensor()
tensor_new = tensor.clone()
```

### Tensor.from_numpy

**功能描述**

将 Numpy 数组转化为 Tensor 对象。

**函数原型**

```python
def from_numpy(nd_array: numpy.ndarray) -> Tensor:
```

**输入参数说明**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|nd_array|numpy.ndarray|必选|输入的 dtype 支持 int8、uint8 和 float32 数据类型；输入 numpy.ndarray 需为行主序，内存必须连续。|

**返回值说明**

|数据类型|说明|
|--|--|
|Tensor|通过 numpy.ndarray 创建的 Tensor 实例。|

>[!NOTE] 说明
>构造后的 Tensor 对象与 numpy.ndarray 共享数据，数据的生命周期由 numpy.ndarray 对象维护。

**示例**

```python
from mm import Tensor
import numpy as np

arr = np.zeros((1024, 768, 3), dtype=np.uint8)
tensor = Tensor.from_numpy(arr)
```

### Tensor.numpy

**功能描述**

将 Tensor 对象转化为 Numpy 数组。

**函数原型**

```python
numpy()-> np.ndarray
```

**返回值说明**

|数据类型|说明|
|--|--|
|numpy.ndarray|转换后的 Numpy 数组。|

>[!NOTE] 说明
>
>- Tensor 对象转换为 numpy.ndarray 对象，转换后的 numpy.ndarray 与 Tensor 共享数据，数据的生命周期由 Tensor 对象维护。
>- Tensor 所处的 device 必须为 CPU。

**示例**

```python
from mm import Tensor
import numpy as np

arr = np.zeros((1024, 768, 3), dtype=np.uint8)
tensor = Tensor.from_numpy(arr)
arr_new = tensor.numpy()
```

### Tensor.from_torch

**功能描述**

将 torch.Tensor 张量转化为 Tensor 对象。

**函数原型**

```python
def from_torch(torch_tensor: torch.Tensor) -> Tensor:
```

**输入参数说明**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|torch_tensor|torch.Tensor|必选|输入的 dtype 支持 int8、uint8 和 float32 数据类型；输入 torch.Tensor 需为行主序，内存必须连续；输入 torch.Tensor 的 Device 必须在 CPU 上。|

**返回值说明**

|数据类型|说明|
|--|--|
|Tensor|通过 torch.Tensor 创建的 Tensor 对象。|

>[!NOTE] 说明
>构造后的 Tensor 对象与 torch.Tensor 共享数据，数据的生命周期由 torch.Tensor 对象维护。

**示例**

```python
from mm import Tensor
import torch

tensor = torch.zeros((1024, 768, 3), dtype=torch.uint8)
mm_tensor = Tensor.from_torch(tensor)
```

### Tensor.torch

**功能描述**

将 Tensor 对象转化为 torch.Tensor 张量。

**函数原型**

```python
torch()-> torch.Tensor
```

**返回值说明**

|数据类型|说明|
|--|--|
|torch.Tensor|转换后的 torch.Tensor 张量。|

>[!NOTE] 说明
>
>- Tensor 对象转换为 torch.Tensor 对象，转换后的 torch.Tensor 与 Tensor 共享数据，数据的生命周期由 Tensor 对象维护。
>- Tensor 所处的 device 必须为 CPU。

**示例**

```python
from mm import Tensor
import torch

tensor = torch.zeros((1024, 768, 3), dtype=torch.uint8)
mm_tensor = Tensor.from_torch(tensor)
torch_tensor = mm_tensor.torch()
```

### Tensor.normalize

**功能描述**

Tensor 类成员函数，使用均值和标准差对当前对象进行归一化。给定 n 个通道的均值：\(mean\[1\],...,mean\[n\]\)和标准差：\(std\[1\],...,std\[n\]\)，此变换将对当前对象的每个通道进行归一化，即 output\[channel\] = \(src\[channel\] - mean\[channel\]\) / std\[channel\]，其中 src 为当前 Tensor 对象。

**函数原型**

```python
def normalize(mean: list[float], std: list[float], device_mode: DeviceMode = DeviceMode.CPU)-> Tensor
```

**输入参数说明**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|mean|list[float]|必选|均值数组，长度必须为 3，参数范围为[0, 1]。|
|std|list[float]|必选|标准差数组，长度必须为 3，参数范围为(0, 3.4028235e38]。|
|device_mode|DeviceMode|可选|运行的模式，当前仅支持 CPU。|

**返回值说明**

|数据类型|说明|
|--|--|
|Tensor|转换后的 Tensor 对象。|

>[!NOTE] 说明
>
>- Tensor 对象的 format 仅支持 NCHW 或 NHWC，N 仅支持 1，C 仅支持 3。
>- Tensor 对象的数据类型仅支持 Float32。
>- Tensor 对象所处的 device 必须为 CPU。

**示例**

```python
from mm import Tensor, TensorFormat
import torch

tensor = torch.randn(1, 3, 224, 224, dtype=torch.float32)
mm_tensor = Tensor.from_torch(tensor)
mm_tensor.set_format(TensorFormat.NCHW)
mean = [0.1, 0.1, 0.1]
std = [0.1, 0.1, 0.1]
dst_mm_tensor = mm_tensor.normalize(mean, std)
```

## mm.Image

Image 类将被用于承载图像数据，实现通用图像的创建、管理以及数据复制等操作。

### Image 属性列表

|属性名|类型|说明|
|--|--|--|
|device|str|Image 所在设备。仅支持 cpu。|
|dtype|DataType|Image 数据类型。仅支持 DataType.UINT8。|
|size|list|Image 的大小。|
|format|ImageFormat|Image 数据图像格式。|
|nbytes|int|Image 数据占用字节数。|
|height|int|Image 的高度。|
|width|int|Image 的宽度。|

### Image.open

**功能描述**

通过指定路径创建 Image。

**函数原型**

```python
open(path:str | bytes, device:str | bytes = b'cpu')
```

**输入参数说明**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|path|str \| bytes|必选|输入的路径必须是有效的，且长度不超过 4096，路径中不能包含软链接，大小不能超过 1GB，且文件权限不得超过 640，User/Group/Others 的权限分别不得超过 6、4、0。且文件后缀应为 jpg 或 jpeg；输入的图像必须是 jpg 和 jpeg 中的一种且宽和高均应在[10,8192]区间内；目前通过 Image.open 构建的仅为 RGB。|
|device|str \| bytes|可选|设备类型，目前只支持 cpu 且默认为 cpu。|

**返回值说明**

|数据类型|说明|
|--|--|
|Image|通过路径创建的新 Image 实例。|

**示例**

```python
from mm import Image
img = Image.open("/home/test.jpg", "cpu")
```

### Image.from_numpy

**功能描述**

将 Numpy 数组转化为 Image 实例。

**函数原型**

```python
from_numpy(
    nd_array: numpy.ndarray,
    image_format: ImageFormat,
    device: str | bytes = b"cpu"
) -> Image:
```

**输入参数说明**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|nd_array|numpy.ndarray|必选|输入的 Numpy 数组。需满足以下条件：输入的 dtype 仅支持 uint8 数据类型，输入 Numpy 数组维度必须为 3，不能为空；输入三维数组时 RGB 和 BGR 图像格式对应数组形状必须为[H, W, 3]；BGR_PLANAR 和 RGB_PLANAR 图像格式对应数组形状必须为[3, H, W]；输入 Numpy 数组各元素数值范围为[0, 255]，且为行主序，内存必须连续；宽和高均应在[10,8192]区间内。 |
|image_format|ImageFormat|必选|图像格式，支持 RGB、BGR、BGR_PLANAR 和 RGB_PLANAR，输入的类型需与 numpy 的数据维度对应。|
|device|str \| bytes|可选|设备类型，目前只支持 cpu 且默认为 cpu。|

**返回值说明**

|数据类型|说明|
|--|--|
|Image|通过 numpy.ndarray 创建的 Image 实例。|

>[!NOTE] 说明
>构造后的 Image 对象与 numpy.ndarray 共享数据，数据的生命周期由 numpy.ndarray 对象维护。

**示例**

```python
from mm import Image, ImageFormat
import numpy as np

arr = np.zeros((1024, 768, 3), dtype=np.uint8)
img = Image.from_numpy(arr, ImageFormat.RGB, "cpu")
```

### Image.numpy

**功能描述**

将 Image 实例对象转换为 Numpy 数组。

**函数原型**

```python
numpy()-> numpy.ndarray
```

**返回值说明**

|数据类型|说明|
|--|--|
|numpy.ndarray|转换后的 Numpy 数组。|

>[!NOTE] 说明
>
>- 输出 ndarray 形状根据图像格式决定：
> 当 Image 实例对象格式为 RGB 和 BGR 时为\[H, W, 3\]；当 format 为 RGB_PLANAR 和 BGR_PLANAR 时为\[3, H, W\]。
>- Image 所处的 device 必须为 CPU。

**示例**

```python
from mm import Image
import numpy as np

img = Image.open("/home/test.jpg", "cpu")
arr = img.numpy()
```

### Image.from_torch

**功能描述**

将 torch.Tensor 转化为 Image 实例。

**函数原型**

```python
from_torch(
    torch_tensor: torch.Tensor,
    image_format: ImageFormat,
    device: str | bytes = b"cpu"
) -> Image:
```

**参数说明**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|torch_tensor|torch.Tensor|必选|输入的 pytorch 张量。需满足：dtype 仅支持 uint8；维度必须为 3；RGB/BGR 形状为 [H, W, 3]，PLANAR 格式为 [3, H, W]；数值范围 [0, 255]，行主序且内存连续，设备在 cpu 上；宽高均在 [10, 8192] 区间内。|
|image_format|ImageFormat|必选|支持 RGB、BGR、BGR_PLANAR 和 RGB_PLANAR，需与 torch.Tensor 的数据维度对应。|
|device|str \| bytes|可选|设备类型，目前只支持 cpu 且默认为 cpu。|

>[!NOTE] 说明
>构造后的 Image 对象与 torch.Tensor 共享数据，数据的生命周期由 torch.Tensor 对象维护。

**返回值说明**

|数据类型|说明|
|--|--|
|Image|通过 torch.Tensor 创建的 Image 实例。|

**示例**

```python
from mm import Image, ImageFormat
import torch

tensor = torch.zeros((1024, 768, 3), dtype=torch.uint8)
img = Image.from_torch(tensor, ImageFormat.RGB, "cpu")
```

### Image.torch

**功能描述**

将 Image 实例对象转换为 torch Tensor 张量。

**函数原型**

```python
torch()-> torch.Tensor
```

**返回值说明**

|数据类型|说明|
|--|--|
|torch.Tensor|转换后的 torch 张量。|

>[!NOTE] 说明
>
>- 输出 Tensor 形状根据图像格式决定：
> 当 Image 实例对象 format 为 RGB 和 BGR 时为\[H, W, 3\]，当 format 为 BGR_PLANAR 和 RGB_PLANAR 时为\[3, H, W\]。
>- Image 所处的 device 必须为 CPU。

**示例**

```python
from mm import Image
import torch

img = Image.open("/home/test.jpg", "cpu")
tensor = img.torch()
```

### Image.from_pillow

**功能描述**

将 PIL 图像对象转换为 Image 对象。

**函数原型**

```python
from_pillow(pillow_image: PIL.Image.Image) -> Image
```

**输入参数说明**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|pillow_image|PIL.Image.Image|必选|输入 Pillow Image 对象，且其 mode 属性仅支持"L"（灰度图）、"RGB"（RGB 图像）和"RGBA"（包含透明度的 RGB 图像）。|

**返回值说明**

|数据类型|说明|
|--|--|
|Image|构建的 Image 对象。|

**示例**

```python
from mm import Image
from PIL import Image as PImage
import numpy as np

img_np = np.random.randint(0, 255, size=(1, 400, 400, 3), dtype=np.uint8)
img_pil = PImage.fromarray(img_np[0])
img_mm = Image.from_pillow(img_pil)
```

### Image.pillow

**功能描述**

将 Image 对象转换为 PIL 图像对象。

**函数原型**

```python
pillow()-> PIL.Image.Image
```

**返回值说明**

|数据类型|说明|
|--|--|
|PIL.Image.Image|转换后的 Pillow 图像实例。|

>[!NOTE] 说明
>
>- 输出 PIL.Image.Image dtype 需与 Image 实例对象保持一致，当前仅支持 uint8。
>- 输出 PIL 的 Image 实例对象 mode 为"RGB"。
>- Image 所处的 device 必须为 CPU。

**示例**

```python
from mm import Image
from PIL import Image as PImage

pillow_image = PImage.open("/home/test.jpg")
img = Image.from_pillow(pillow_image)
pillow_image_new = img.pillow()
```

### Image.clone

**功能描述**

深拷贝 Image 实例对象为新的 Image 实例。

**函数原型**

```python
clone()-> Image
```

**返回值说明**

|数据类型|说明|
|--|--|
|Image|通过深拷贝创建的新 Image 实例。|

**示例**

```python
from mm import Image
img = Image.open("/home/test.jpg", "cpu")
img_copy = img.clone()
```

### Image.resize

**功能描述**

对 Image 实例对象进行缩放操作。

**函数原型**

```python
resize(size: Tuple[int, int], interpolation: Interpolation, device_mode: DeviceMode = DeviceMode.CPU) -> "Image"
```

**参数说明**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|size|Tuple[int, int]|必选|resize 之后的图像宽高。size 需为两维，第 1 维是 width，第 2 维是 height，且宽和高均应在[10, 8192]区间内。|
|interpolation|Interpolation|必选|resize 插值算法，当前仅支持 BICUBIC。|
|device_mode|DeviceMode|可选|resize 运行的模式，当前仅支持 CPU。|

**返回值说明**

|数据类型|说明|
|--|--|
|Image|resize 操作后获得的新 Image 实例。|

>[!NOTE] 说明
>当前仅支持图像格式为 RGB 或 BGR，数据格式为 UINT8，各元素值范围在\[0,255\]。

**示例**

```python
from mm import Image, DeviceMode, Interpolation
img = Image.open("/home/test.jpg", "cpu")
img_resize = img.resize((10, 10), Interpolation.BICUBIC, DeviceMode.CPU)
```

### Image.crop

**功能描述**

对 Image 实例对象进行裁剪操作。

**函数原型**

```python
crop(top: int, left: int, height: int, width: int, device_mode: DeviceMode = DeviceMode.CPU) -> "Image":
```

**参数说明**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|top|int|必选|裁剪的上起始坐标。从 0 取值，(top+height)不能超过原图高度。|
|left|int|必选|裁剪的左起始坐标。从 0 取值，(left+width)不能超过原图宽度。|
|height|int|必选|裁剪区域高度，范围为[10, 原图高度-top]。|
|width|int|必选|裁剪区域宽度，范围为[10, 原图宽度-left]。|
|device_mode|DeviceMode|可选|crop 运行的模式，当前仅支持 CPU。|

**返回值说明**

|数据类型|说明|
|--|--|
|Image|crop 操作后获得的新 Image 实例。|

>[!NOTE] 说明
>当前仅支持图像格式为 RGB 或 BGR，数据格式为 UINT8，各元素值范围在\[0,255\]。

**示例**

```python
from mm import Image, DeviceMode
img = Image.open("/home/test.jpg", "cpu")
img_crop = img.crop(10, 10, 10, 10, DeviceMode.CPU)
```

### Image.to_tensor

**功能描述**

对 Image 实例对象进行\[0,255\]至\[0.0, 1.0\]缩放，以及格式转换能力（HWC-\>CHW）。

>[!NOTE] 说明
>当前仅支持图像格式为 RGB 或 BGR，数据格式为 UINT8，各元素值范围在\[0,255\]。
>输出的 Tensor 实例数据类型默认为 DataType.FLOAT32。

**函数原型**

```python
def to_tensor(target_format: TensorFormat = TensorFormat.NCHW, device_mode: DeviceMode = DeviceMode.CPU) -> "Tensor":
```

**参数说明**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|target_format|TensorFormat|可选|转换后 Tensor 实例的格式，支持 NHWC 和 NCHW，默认值为 NCHW。|
|device_mode|DeviceMode|可选|运行的模式，当前仅支持 CPU。|

**返回值说明**

|数据类型|说明|
|--|--|
|Tensor|操作后获得的 Tensor 实例。|

**示例**

```python
from mm import Image, TensorFormat, DeviceMode
img = Image.open("/home/test.jpg", "cpu")
dst_tensor = img.to_tensor(TensorFormat.NCHW, DeviceMode.CPU)
```

## 日志注册

用户注册日志级别与日志回调函数。

### register_log_conf

日志注册函数。

**函数原型**

```python
register_log_conf(min_level: LogLevel, callback: Callable[[LogLevel, str, str, int, str], None])
```

**输入参数说明**

|参数名|类型|可选/必选|说明|
|--|--|--|--|
|min_level|LogLevel|必选|最小日志级别，只有大于等于该级别的日志才会输出。不允许传入 None。|
|callback|Callable[[LogLevel, str, str, int, str], None]|必选|日志回调函数，传入 None 时使用内部默认的日志输出函数。|

>[!CAUTION] 注意
>在日志回调函数中抛异常会触发 C++侧抛出异常，引起程序 coredump，建议在回调中捕获异常并处理。

**示例**

```python
from mm import register_log_conf, LogLevel
def custom_log_handler(level: LogLevel, file: str, func: str, line: int, msg: str) -> None:
    print(f"[custom][{level}] {file}:{line} ({func}) - {msg}")
register_log_conf(LogLevel.ERROR, custom_log_handler)
```

## mm.video_decode

**功能描述**

将传入的视频文件解码，并返回 Image 对象列表。

**函数原型**

```python
def video_decode(video_path: str | bytes, device: str | bytes, frame_indices: set = None, sample_num: int = -1) -> list:
```

**输入参数说明**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|video_path|str \| bytes|必选|解码视频路径，当前仅支持 mp4 格式文件，分辨率支持 480P-4K。|
|device|str \| bytes|必选|解码设备，当前仅支持 cpu。|
|frame_indices|set|可选|期望解码的视频帧 ID。|
|sample_num|int|可选|期望解码后获取的总帧数。|

**返回值说明**

|数据类型|说明|
|--|--|
|list[Image]|解码后的 Image 对象列表。格式为 RGB，数据类型为 uint8。|

>[!NOTE] 说明
>
>- 期望解码的视频帧 ID 取值范围为\[0, 视频总帧数-1\)，默认为空集合，参数优先级高于期望解码后获取的总帧数。该参数意为目标解码的帧 ID 集合。
>- 期望解码后获取的总帧数取值范围为\(0, 视频总帧数\]，默认值为 -1，最终解码 ID 集合为通过视频帧总数计算等间隔抽取。
>- 若 frame_indices 和 sample_num 均未指定，会返回失败。
>- 分辨率支持说明：目前支持的视频帧宽高为\[480, 480\] - \[4096, 4096\]。
>- 输入的路径必须是有效的，且长度不超过 4096，路径中不能包含软链接，且文件权限不得超过 640，User/Group/Others 的权限分别不得超过 6、4、0。

**示例**

```python
from mm import video_decode

file_path = "test.mp4"
mm_images = video_decode(file_path, "cpu", set(), 32)
```

<a id="mmnormalize"></a>

## mm.normalize

**功能描述**

使用均值和标准差对 Tensor 对象进行归一化。给定 n 个通道的均值：\(mean\[1\],...,mean\[n\]\)和标准差：\(std\[1\],...,std\[n\]\)，此变换将对输入 Tensor 对象的每个通道进行归一化，即 output\[channel\] = \(src\[channel\] - mean\[channel\]\) / std\[channel\]。

**函数原型**

```python
def normalize(src: Tensor, mean: list[float], std: list[float], device_mode: DeviceMode = DeviceMode.CPU)-> Tensor
```

**输入参数说明**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|src|Tensor|必选|输入 Tensor 实例。|
|mean|list[float]|必选|均值数组，长度必须为 3，参数范围为[0, 1]。|
|std|list[float]|必选|标准差数组，长度必须为 3，参数范围为(0, 3.4028235e38]。|
|device_mode|DeviceMode|可选|运行的模式，当前仅支持 CPU。|

**返回值说明**

|数据类型|说明|
|--|--|
|Tensor|转换后的 Tensor 对象。|

>[!NOTE] 说明
>
>- 输入 Tensor 对象的 format 仅支持 NCHW 或 NHWC，N 仅支持 1，C 仅支持 3。
>- 输入 Tensor 对象的数据类型仅支持 Float32。
>- Tensor 对象所处的 device 必须为 CPU。

**示例**

```python
from mm import Tensor, TensorFormat, normalize, DeviceMode
import torch

tensor = torch.randn(1, 3, 224, 224, dtype=torch.float32)
mm_tensor = Tensor.from_torch(tensor)
mm_tensor.set_format(TensorFormat.NCHW)
mean = [0.1, 0.1, 0.1]
std = [0.1, 0.1, 0.1]
dst_mm_tensor = normalize(mm_tensor, mean, std, DeviceMode.CPU)
```

<a id="mmload_audio"></a>

## mm.load_audio

**功能描述**

使用音频加载接口，对给定的音频实现加载。支持对单音频文件加载以及对批量音频的并行加载，支持开关重采样。

**函数原型**

```python
def load_audio(audio_inputs: Union[str, List[str]], sr: Optional[int] = None)
-> Union[Tuple[Tensor, int], List[Tuple[Tensor, int]]]
```

**输入参数说明**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|audio_inputs|Union[str, List[str]]|必选|输入单音频路径 / 多音频所在目录 / 音频列表。|
|sr|Optional[int]|可选|若用户指定重采样率，则按该采样率进行重采样，否则不进行重采样。|

**返回值说明**

|数据类型|说明|
|--|--|
|Union[Tuple[Tensor, int], List[Tuple[Tensor, int]]]|单音频返回(音频 tensor 数据, 采样率)，多音频返回(音频 tensor 数据，采样率)列表。|

>[!NOTE] 说明
>
>- 输入音频仅支持 wav 文件。
>- 可加载音频文件的数量范围为[1,128]。
>- 用户输入的采样率必须为[1,64000]范围内的正整数。
>- 多通道音频会自动转换为单通道音频。

**示例**

```python
from mm import load_audio
single_audio_path = "/path/to/speech.wav"
audio_file_paths = ["/path/to/audio1.wav", "/path/to/audio2.wav"]
audio_directory = "/path/to/audio_dir"

waveform, sr = load_audio(single_audio_path)
batch_from_filelist = load_audio(audio_file_paths)
batch_from_directory = load_audio(audio_directory)
```

<a id="mmbaseframeselector"></a>

## mm.BaseFrameSelector

基于文本-图像匹配的关键帧选择器抽象基类，封装模型初始化、特征提取、边界定位、输入校验和相似度计算等公共能力。该类不可直接实例化，需通过子类 KRangFrameSelector 或 KFrameSelector 使用。

### BaseFrameSelector 属性列表

|属性名|类型|说明|
|--|--|--|
|model_path|str|CLIP 模型权重路径。|
|device_id|int|NPU 设备索引。|
|model_type|str|模型类型，取值为"clip"（英文）或"cn_clip"（中文）。|
|batch_size|int|图像特征提取批次大小，默认为 64。|
|similar_threshold|float|文本-图像相似度衰减阈值，相似度低于（最大相似度 - 该值）的帧将被过滤。默认为 0.025。|
|image_similar_threshold|float|图像相似度梯度阈值，用于场景边界检测，相邻帧间余弦相似度梯度超过该值时判定为场景切换。默认为 0.015。|
|image_size|tuple|输入图像缩放尺寸，需与模型训练尺寸一致。默认为(672, 672)。|

### BaseFrameSelector.__init__

**功能描述**

初始化关键帧选择器，加载指定模型并完成参数校验。

**函数原型**

```python
__init__(model_path: str, device_id: int, model_type: str = "clip", similar_threshold: float = 0.025, image_similar_threshold: float = 0.015, image_size: tuple = (672, 672))
```

**输入参数说明**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|model_path|str|必选|CLIP 模型权重路径，必须为非空字符串，且指向有效的目录。|
|device_id|int|必选|NPU 设备索引，必须为整数。|
|model_type|str|可选|模型类型，取值为"clip"（英文）或"cn_clip"（中文），默认为"clip"。|
|similar_threshold|float|可选|文本-图像相似度衰减阈值，取值范围为[0, 1]，默认为 0.025。|
|image_similar_threshold|float|可选|图像相似度梯度阈值，用于场景边界检测，取值范围为[0, 1]，默认为 0.015。|
|image_size|tuple|可选|输入图像缩放尺寸，形如(width, height)，宽高取值范围为[10, 8192]，默认为(672, 672)。|

>[!NOTE] 说明
>
>- BaseFrameSelector 为抽象类，不可直接实例化，需通过子类 KRangFrameSelector 或 KFrameSelector 创建实例。
>- model_path 所指向的目录必须存在，且目录权限不能高于 750，目录属主必须与当前用户一致。

### BaseFrameSelector.select_keyframes

**功能描述**

从视频帧序列中选择与查询文本相关的关键帧，为抽象方法，由子类实现具体选择策略。

**函数原型**

```python
select_keyframes(query: str, frames: List[np.ndarray], sample_num: int, do_resample: bool) -> Tuple[List[int], List[np.ndarray]]
```

**输入参数说明**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|query|str|必选|描述目标视觉内容的查询文本，必须为非空字符串。|
|frames|List[np.ndarray]|必选|视频帧列表。|
|sample_num|int|必选|最大关键帧数量，必须为正整数。|
|do_resample|bool|必选|是否在区间内进行自适应重采样。|

**返回值说明**

|数据类型|说明|
|--|--|
|Tuple[List[int], List[np.ndarray]]|元组，包含关键帧索引列表和关键帧图像列表。|

## mm.KRangFrameSelector

基于区间合并的关键帧选择器，继承自 BaseFrameSelector。定位与查询文本相关的连续场景区间，并在区间内自适应采样关键帧，适用于需要时序上下文的任务。

### KRangFrameSelector.select_keyframes

**功能描述**

区间关键帧选择主流程：提取特征并计算相似度后，贪心选择候选帧并扩展为场景区间，合并相邻语义相似区间，最后在区间内自适应重采样。

**函数原型**

```python
select_keyframes(query: str, frames: List[np.ndarray], sample_num: int, do_resample: bool) -> Tuple[List[int], List[np.ndarray]]
```

**输入参数说明**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|query|str|必选|描述目标视觉内容的查询文本，必须为非空字符串。|
|frames|List[np.ndarray]|必选|视频帧列表。|
|sample_num|int|必选|最大关键帧数量，必须为正整数。|
|do_resample|bool|必选|是否在合并区间内进行自适应重采样。为 True 时，在区间内按相似度 top-k 与均匀填充策略重采样；为 False 时，仅返回合并后的区间端点。|

**返回值说明**

|数据类型|说明|
|--|--|
|Tuple[List[int], List[np.ndarray]]|元组，包含关键帧索引列表（已去重排序）和关键帧图像列表。|

**示例**

```python
from mm import KRangFrameSelector
import numpy as np

selector = KRangFrameSelector(model_path="/path/to/clip_model", device_id=0, model_type="clip")
frames = [np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8) for _ in range(100)]
indices, key_frames = selector.select_keyframes(query="a cat sitting on a sofa", frames=frames, sample_num=8, do_resample=True)
```

## mm.KFrameSelector

离散关键帧选择器，继承自 BaseFrameSelector。选择与查询文本相关且视觉多样的离散关键帧，适用于需要视觉多样性的任务。

### KFrameSelector.select_keyframes

**功能描述**

离散关键帧选择主流程：提取特征并计算相似度后，贪心选择候选帧，并通过特征距离去重保证视觉多样性。

**函数原型**

```python
select_keyframes(query: str, frames: List[np.ndarray], sample_num: int, do_resample: bool = False) -> Tuple[List[int], List[np.ndarray]]
```

**输入参数说明**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|query|str|必选|描述目标视觉内容的查询文本，必须为非空字符串。|
|frames|List[np.ndarray]|必选|视频帧列表。|
|sample_num|int|必选|最大关键帧数量，必须为正整数。|
|do_resample|bool|可选|该参数在 KFrameSelector 中未使用，默认为 False。|

**返回值说明**

|数据类型|说明|
|--|--|
|Tuple[List[int], List[np.ndarray]]|元组，包含关键帧索引列表（已去重排序）和关键帧图像列表。|

**示例**

```python
from mm import KFrameSelector
import numpy as np

selector = KFrameSelector(model_path="/path/to/clip_model", device_id=0, model_type="clip")
frames = [np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8) for _ in range(100)]
indices, key_frames = selector.select_keyframes(query="a dog running in the park", frames=frames, sample_num=8)
```
