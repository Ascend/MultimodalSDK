# 功能函数参考

## mm.Tensor

Tensor类将被用于承载任意模态的通用数据，实现通用数据的创建、管理以及数据复制等操作。

### Tensor属性列表

|属性名|类型|说明|备注|
|--|--|--|--|
|device|str|Tensor所在设备。|默认为"cpu"。|
|dtype|DataType|Tensor数据类型。|默认为DataType.FLOAT32。|
|shape|list|Tensor的维度信息。|默认为空list。|
|format|TensorFormat|Tensor数据排布类型。|默认为TensorFormat.ND。|
|nbytes|int|Tensor数据占用字节数。|默认为0。|

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
|tensor_format|TensorFormat|必选|输入需要设置的排布格式。支持ND、NHWC和NCHW。若设置为NHWC或者NCHW，需要确保tensor的shape有四维。|

**示例**

```python
from mm import Tensor, TensorFormat
tensor = Tensor()
tensor.set_format(TensorFormat.ND)
```

### Tensor.clone

**功能描述**

深拷贝Tensor实例对象为新的Tensor实例。

**函数原型**

```python
clone()-> Tensor
```

**返回值说明**

|数据类型|说明|
|--|--|
|Tensor|新的Tensor实例。|

**示例**

```python
from mm import Tensor
tensor = Tensor()
tensor_new = tensor.clone()
```

### Tensor.from_numpy

**功能描述**

将Numpy数组转化为Tensor对象。

**函数原型**

```python
from_numpy(nd_array: numpy.ndarray) -> Tensor:
```

**输入参数说明**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|nd_array|numpy.ndarray|必选|输入的dtype支持int8、uint8和float32数据类型; 输入numpy.ndarray需为行主序，内存必须连续。|

**返回值说明**

|数据类型|说明|
|--|--|
|Tensor|通过numpy.ndarray创建的Tensor实例。|

>[!NOTE] 说明
>构造后的Tensor对象与numpy.ndarray共享数据，数据的生命周期由numpy.ndarray对象维护。

**示例**

```python
from mm import Tensor
import numpy as np

arr = np.zeros((1024, 768, 3), dtype=np.uint8)
tensor = Tensor.from_numpy(arr)
```

### Tensor.numpy

**功能描述**

将Tensor对象转化为Numpy数组。

**函数原型**

```python
numpy()-> np.ndarray
```

**返回值说明**

|数据类型|说明|
|--|--|
|numpy.ndarray|转换后的Numpy数组。|

>[!NOTE] 说明
>
>- Tensor对象转换为numpy.ndarray对象，转换后的numpy.ndarray与Tensor共享数据，数据的生命周期由Tensor对象维护。
>- Tensor所处的device必须为CPU。

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

将torch.Tensor张量转化为Tensor对象。

**函数原型**

```python
from_torch(torch_tensor: torch.Tensor) -> Tensor:
```

**输入参数说明**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|torch_tensor|torch.Tensor|必选|输入的dtype支持int8、uint8和float32数据类型；输入torch.Tensor需为行主序，内存必须连续；输入torch.Tensor的Device必须在CPU上。|

**返回值说明**

|数据类型|说明|
|--|--|
|Tensor|通过torch.Tensor创建的Tensor对象。|

>[!NOTE] 说明
>构造后的Tensor对象与torch.Tensor共享数据，数据的生命周期由torch.Tensor对象维护。

**示例**

```python
from mm import Tensor
import torch

tensor = torch.zeros((1024, 768, 3), dtype=torch.uint8)
mm_tensor = Tensor.from_torch(tensor)
```

### Tensor.torch

**功能描述**

将Tensor对象转化为torch.Tensor张量。

**函数原型**

```python
torch()-> torch.Tensor
```

**返回值说明**

|数据类型|说明|
|--|--|
|torch.Tensor|转换后的torch.Tensor张量。|

>[!NOTE] 说明
>
>- Tensor对象转换为torch.Tensor对象，转换后的torch.Tensor与Tensor共享数据，数据的生命周期由Tensor对象维护。
>- Tensor所处的device必须为CPU。

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

Tensor类成员函数，使用均值和标准差对当前对象进行归一化。给定n个通道的均值：\(mean\[1\],...,mean\[n\]\)和标准差：\(std\[1\],..,std\[n\]\)，此变换将对当前对象的每个通道进行归一化，即output\[channel\] = \(src\[channel\] - mean\[channel\]\) / std\[channel\]，其中src为当前Tensor对象。

**函数原型**

```python
def normalize(mean: list[float], std: list[float], device_mode: DeviceMode = DeviceMode.CPU)-> Tensor
```

**输入参数说明**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|mean|list[float]|必选|均值数组，长度必须为3，参数范围为[0, 1]。|
|std|list[float]|必选|标准差数组，长度必须为3，参数范围大于0，小于3.4028235e38。|
|device_mode|DeviceMode|可选|运行的模式，当前仅支持CPU。|

**返回值说明**

|数据类型|说明|
|--|--|
|Tensor|转换后的Tensor对象。|

>[!NOTE] 说明
>
>- Tensor对象的format仅支持NCHW或NHWC，N仅支持1，C仅支持3。
>- Tensor对象的数据类型仅支持Float32。
>- Tensor对象所处的device必须为CPU。

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

Image类将被用于承载图像数据，实现通用图像的创建、管理以及数据复制等操作。

### Image属性列表

|属性名|类型|说明|
|--|--|--|
|device|str|Image所在设备。仅支持cpu。|
|dtype|DataType|Image数据类型。仅支持DataType.UINT8。|
|size|list|Image的大小。|
|format|ImageFormat|Image数据图像格式。|
|nbytes|int|Image数据占用字节数。|
|height|int|Image的高度。|
|width|int|Image的宽度。|

### Image.open

**功能描述**

通过指定路径创建Image。

**函数原型**

```python
open(path:str | bytes, device:str | bytes = b'cpu')
```

**输入参数说明**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|path|str \| bytes|必选|输入的路径必须是有效的，且长度不超过4096，路径中不能包含软链接，大小不能超过1GB，且文件权限不得超过640，User/Group/Others的权限分别不得超过6、4、0。且文件后缀应为jpg或jpeg；输入的图像必须是jpg和jpeg中的一种且宽和高均应在[10,8192]区间内；目前通过Image.open构建的仅为RGB。|
|device|str \| bytes|可选|设备类型，目前只支持cpu且默认为cpu。|

**返回值说明**

|数据类型|说明|
|--|--|
|Image|通过路径创建的新Image实例。|

**示例**

```python
from mm import Image
img= Image.open("/home/test.jpg", "cpu")
```

### Image.from_numpy

**功能描述**

从Numpy数组转化为Image实例。

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
|nd_array|numpy.ndarray|必选|输入的Numpy数组。需满足以下条件：输入的dtype仅支持uint8数据类型，输入Numpy数组维度必须为3，不能为空；输入三维数组时RGB和BGR图像格式对应数组形状必须为[H, W, 3]；BGR_PLANAR和RGB_PLANAR图像格式对应数组形状必须为[3, H, W]；输入Numpy数组各元素数值范围为[0, 255]，且为行主序，内存必须连续；宽和高均应在[10,8192]区间内。 |
|image_format|ImageFormat|必选|图像格式，支持RGB、BGR、BGR_PLANAR和RGB_PLANAR，输入的类型需与numpy的数据维度对应。|
|device|str \| bytes|可选|设备类型，目前只支持cpu且默认为cpu。|

**返回值说明**

|数据类型|说明|
|--|--|
|Image|通过numpy.ndarray创建的Image实例。|

>[!NOTE] 说明
>构造后的Image对象与numpy.ndarray共享数据，数据的生命周期由numpy.ndarray对象维护。

**示例**

```python
from mm import Image, ImageFormat
import numpy as np

arr = np.zeros((1024, 768, 3), dtype=np.uint8)
img = Image.from_numpy(arr, ImageFormat.RGB, "cpu")
```

### Image.numpy

**功能描述**

将Image实例对象转换为Numpy数组。

**函数原型**

```python
numpy()-> numpy.ndarray
```

**返回值说明**

|数据类型|说明|
|--|--|
|numpy.ndarray|转换后的Numpy数组。|

>[!NOTE] 说明
>
>- 输出ndarray形状根据图像格式决定：
> 当Image实例对象格式为RGB和BGR时为\[H, W, 3\]；当format为RGB_PLANAR和BGR_PLANAR时为\[3, H, W\]。
>- Image所处的device必须为CPU。

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
>构造后的Image对象与torch.Tensor共享数据，数据的生命周期由torch.Tensor对象维护。

**返回值说明**

|数据类型|说明|
|--|--|
|Image|通过torch.Tensor创建的Image实例。|

**示例**

```python
from mm import Image, ImageFormat
import torch

tensor = torch.zeros((1024, 768, 3), dtype=torch.uint8)
img = Image.from_torch(tensor, ImageFormat.RGB, "cpu")
```

### Image.torch

**功能描述**

将Image实例对象转换为torch Tensor张量。

**函数原型**

```python
torch()-> torch.Tensor
```

**返回值说明**

|数据类型|说明|
|--|--|
|torch.Tensor|转换后的torch张量。|

>[!NOTE] 说明
>
>- 输出Tensor形状根据图像格式决定：
> 当Image实例对象format为RGB和BGR时为\[H, W, 3\]，当format为BGR_PLANAR和RGB_PLANAR时为\[3, H, W\]。
>- Image所处的device必须为CPU。

**示例**

```python
from mm import Image
import torch

img = Image.open("/home/test.jpg", "cpu")
tensor = img.torch()
```

### Image.from_pillow

**功能描述**

将PIL图像对象转换为Image对象。

**函数原型**

```python
from_pillow(pillow_image: PIL.Image.Image) -> Image
```

**输入参数说明**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|pillow_image|PIL.Image.Image|必选|输入Pillow Image对象，且其mode属性仅支持"L"（灰度图）、"RGB"（RGB图像）和"RGBA"（包含透明度的RGB图像）。|

**返回值说明**

|数据类型|说明|
|--|--|
|Image|构建的Image对象。|

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

将Image对象转换为PIL图像对象。

**函数原型**

```python
pillow()-> PIL.Image.Image
```

**返回值说明**

|数据类型|说明|
|--|--|
|PIL.Image.Image|转换后的Pillow图像实例。|

>[!NOTE] 说明
>
>- 输出PIL.Image.Image dtype需与Image实例对象保持一致，当前仅支持uint8。
>- 输出PIL的Image实例对象mode为“RGB”。
>- Image所处的device必须为CPU。

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

深拷贝Image实例对象为新的Image实例。

**函数原型**

```python
clone()-> Image
```

**返回值说明**

|数据类型|说明|
|--|--|
|Image|通过深拷贝创建的新Image实例。|

**示例**

```python
from mm import Image
img = Image.open("/home/test.jpg", "cpu")
img_copy = img.clone()
```

### Image.resize

**功能描述**

对Image实例对象进行缩放操作。

**函数原型**

```python
resize(size: Tuple[int, int], interpolation: Interpolation, device_mode: DeviceMode) -> "Image"
```

**参数说明**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|size|Tuple[int, int]|必选|resize之后的图像宽高。size需为两维，第1维是width，第2维是height，且宽和高均应在[10, 8192]区间内。|
|interpolation|Interpolation|必选|resize插值算法，当前仅支持BICUBIC。|
|device_mode|DeviceMode|可选|resize运行的模式，当前仅支持CPU。|

**返回值说明**

|数据类型|说明|
|--|--|
|Image|resize操作后获得的新Image实例。|

>[!NOTE] 说明
>当前仅支持图像格式为RGB或BGR，数据格式为UINT8，各元素值范围在\[0,255\]。

**示例**

```python
from mm import Image, DeviceMode, Interpolation
img = Image.open("/home/test.jpg", "cpu")
img_resize = img.resize((10,10), Interpolation.BICUBIC, DeviceMode.CPU)
```

### Image.crop

**功能描述**

对Image实例对象进行裁剪操作。

**函数原型**

```python
crop(top: int, left: int, height: int, width: int, device_mode: DeviceMode) -> "Image":
```

**参数说明**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|top|int|必选|裁剪的上起始坐标。从0取值，(top+height)不能超过原图高度。|
|left|int|必选|裁剪的左起始坐标。从0取值，(left+width)不能超过原图宽度。|
|height|int|必选|裁剪区域高度，范围为[10, 原图高度-top]。|
|width|int|必选|裁剪区域宽度，范围为[10, 原图宽度-left]。|
|device_mode|DeviceMode|可选|crop运行的模式，当前仅支持CPU。|

**返回值说明**

|数据类型|说明|
|--|--|
|Image|crop操作后获得的新Image实例。|

>[!NOTE] 说明
>当前仅支持图像格式为RGB或BGR，数据格式为UINT8，各元素值范围在\[0,255\]。

**示例**

```python
from mm import Image, DeviceMode
img = Image.open("/home/test.jpg", "cpu")
img_crop = img.crop(10, 10, 10, 10, DeviceMode.CPU)
```

### Image.to_tensor

**功能描述**

对Image实例对象进行\[0,255\]至\[0.0, 1.0\]缩放，以及格式转换能力（HWC-\>CHW）。

>[!NOTE] 说明
>当前仅支持图像格式为RGB或BGR，数据格式为UINT8，各元素值范围在\[0,255\]。
>输出的Tensor实例数据类型默认为DataType.FLOAT32。

**函数原型**

```python
def to_tensor(target_format: TensorFormat = TensorFormat.NCHW, device_mode: DeviceMode = DeviceMode.CPU) -> "Tensor":
```

**参数说明**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|target_format|TensorFormat|可选|转换后Tensor实例的格式，支持NHWC和NCHW，默认值为NCHW。|
|device_mode|DeviceMode|可选|运行的模式，当前仅支持CPU。|

**返回值说明**

|数据类型|说明|
|--|--|
|Tensor|操作后获得的Tensor实例。|

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
|min_level|LogLevel|必选|最小日志级别，只有大于等于该级别的日志才会输出。不允许传入None。|
|callback|Callable[[LogLevel, str, str, int, str], None]|必选|日志回调函数，传入None时使用内部默认的日志输出函数。|

>[!CAUTION] 注意
>在日志回调函数中抛异常会触发C++侧抛出异常，引起程序coredump，建议在回调中捕获异常并处理。

**示例**

```python
from mm import register_log_conf, LogLevel
def custom_log_handler(level: LogLevel, file: str, func: str, line: int, msg: str) -> None:
    print(f"[custom][{level}] {file}:{line} ({func}) - {msg}")
register_log_conf(LogLevel.ERROR, custom_log_handler)
```

## mm.video_decode

**功能描述**

将传入的视频文件解码，并返回Image对象列表。

**函数原型**

```python
def video_decode(video_path: str | bytes, device: str | bytes, frame_indices: set = None, sample_num: int = -1) -> list:
```

**输入参数说明**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|video_path|str \| bytes|必选|解码视频路径，当前仅支持mp4格式文件，分辨率支持480P-4K。|
|device|str \| bytes|必选|解码设备，当前仅支持cpu。|
|frame_indices|set|可选|期望解码的视频帧ID。|
|sample_num|int|可选|期望解码后获取的总帧数。|

**返回值说明**

|数据类型|说明|
|--|--|
|list[Image]|解码后的Image对象列表。格式为RGB，数据类型为uint8。|

>[!NOTE] 说明
>
>- 期望解码的视频帧ID取值范围为\[0, 视频总帧数-1），默认为空集合，参数优先级高于期望解码后获取的总帧数。该参数意为目标解码的帧ID集合。
>- 期望解码后获取的总帧数取值范围为\(0, 视频总帧数\]，默认值为-1，最终解码ID集合为通过视频帧总数计算等间隔抽取。
>- 若frame_indices和sample_num均未指定，会返回失败。
>- 分辨率支持说明：目前支持的视频帧宽高为\[480, 480\] - \[4096, 4096\]。
>- 输入的路径必须是有效的，且长度不超过4096，路径中不能包含软链接，且文件权限不得超过640，User/Group/Others的权限分别不得超过6、4、0。

**示例**

```python
from mm import video_decode

file_path = "test.mp4"
mm_images = video_decode(file_path, "cpu", [], 32)
```

<a id="mmnormalize"></a>

## mm.normalize

**功能描述**

使用均值和标准差对Tensor对象进行归一化。给定n个通道的均值：\(mean\[1\],...,mean\[n\]\)和标准差：\(std\[1\],..,std\[n\]\)，此变换将对输入Tensor对象的每个通道进行归一化，即output\[channel\] = \(src\[channel\] - mean\[channel\]\) / std\[channel\]。

**函数原型**

```python
def normalize(src: Tensor, mean: list[float], std: list[float], device_mode: DeviceMode = DeviceMode.CPU)-> Tensor
```

**输入参数说明**

|参数名|数据类型|可选/必选|说明|
|--|--|--|--|
|src|Tensor|必选|输入Tensor实例。|
|mean|list[float]|必选|均值数组，长度必须为3，参数范围为[0, 1]。|
|std|list[float]|必选|方差数组，长度必须为3，参数范围为[0，3.4028235e38]。|
|device_mode|DeviceMode|可选|运行的模式，当前仅支持CPU。|

**返回值说明**

|数据类型|说明|
|--|--|
|Tensor|转换后的Tensor对象。|

>[!NOTE] 说明
>
>- 输入Tensor对象的format仅支持NCHW或NHWC，N仅支持1，C仅支持3。
>- 输入Tensor对象的数据类型仅支持Float32。
>- Tensor对象所处的device必须为CPU。

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
|Union[Tuple[Tensor, int], List[Tuple[Tensor, int]]]|单音频返回(音频tensor数据, 采样率)，多音频返回(音频tensor数据，采样率)列表。|

>[!NOTE] 说明
>
>- 输入音频仅支持wav文件。
>- 可加载音频文件的范围为[1,128]。
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
