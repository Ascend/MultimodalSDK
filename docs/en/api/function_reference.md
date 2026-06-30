# Function Reference

## `mm.Tensor`

The Tensor class serves as a container for general-purpose data of any modality. It supports creating, managing, and copying that data.

### Tensor Properties

|Property|Type|Description|Remarks|
|--|--|--|--|
|device|str|Device where the Tensor resides.|Default: `"cpu"`.|
|dtype|DataType|Tensor data type.|Default: `DataType.FLOAT32`.|
|shape|list|Tensor dimension information.|Default: an empty list.|
|format|TensorFormat|Tensor data layout type.|Default: `TensorFormat.ND`.|
|nbytes|int|Number of bytes occupied by the Tensor data.|Default: 0.|

### `Tensor.set_format`

**Description**

Set the data layout format.

**Function Prototype**

```typescript
set_format(tensor_format: TensorFormat)
```

**Parameters**

|Parameter|Data Type|Optional/Required|Description|
|--|--|--|--|
|tensor_format|TensorFormat|Required|Layout format to set. It supports ND, NHWC, and NCHW. If you set NHWC or NCHW, ensure that the Tensor shape has four dimensions.|

**Example**

```python
from mm import Tensor, TensorFormat
tensor = Tensor()
tensor.set_format(TensorFormat.ND)
```

### `Tensor.clone`

**Description**

Deep copy the Tensor instance into a new Tensor instance.

**Function Prototype**

```typescript
clone()-> Tensor
```

**Returns**

|Data Type|Description|
|--|--|
|Tensor|New Tensor instance.|

**Example**

```python
from mm import Tensor
tensor = Tensor()
tensor_new = tensor.clone()
```

### `Tensor.from_numpy`

**Description**

Convert a NumPy array into a Tensor object.

**Function Prototype**

```typescript
from_numpy(nd_array: numpy.ndarray) -> Tensor:
```

**Parameters**

|Parameter|Data Type|Optional/Required|Description|
|--|--|--|--|
|nd_array|numpy.ndarray|Required|<ul><li>The input dtype supports only int8, uint8, and float32.</li><li>The input `numpy.ndarray` must be row-major and contiguous in memory.</li></ul>|

**Returns**

|Data Type|Description|
|--|--|
|Tensor|Tensor instance created from `numpy.ndarray`.|

>[!NOTE] Note
>The constructed Tensor object shares data with the `numpy.ndarray`, and the `numpy.ndarray` object manages the lifetime of the data.

**Example**

```python
from mm import Tensor
import numpy as np

arr = np.zeros((1024, 768, 3), dtype=np.uint8)
tensor = Tensor.from_numpy(arr)
```

### `Tensor.numpy`

**Description**

Convert a Tensor object into a NumPy array.

**Function Prototype**

```typescript
numpy()-> np.ndarray
```

**Returns**

|Data Type|Description|
|--|--|
|numpy.ndarray|Converted NumPy array.|

>[!NOTE] Note
>
>- The Tensor object is converted into a `numpy.ndarray` object. The converted `numpy.ndarray` shares data with the Tensor, and the Tensor object manages the lifetime of the data.
>- The Tensor must reside on the CPU.

**Example**

```python
from mm import Tensor
import numpy as np

arr = np.zeros((1024, 768, 3), dtype=np.uint8)
tensor = Tensor.from_numpy(arr)
arr_new = tensor.numpy()
```

### `Tensor.from_torch`

**Description**

Convert a `torch.Tensor` into a Tensor object.

**Function Prototype**

```typescript
from_torch(torch_tensor: torch.Tensor) -> Tensor:
```

**Parameters**

|Parameter|Data Type|Optional/Required|Description|
|--|--|--|--|
|torch_tensor|torch.Tensor|Required|<ul><li>The input dtype supports only int8, uint8, and float32.</li><li>The input `torch.Tensor` must be row-major and contiguous in memory.</li><li>The input `torch.Tensor` device must be on the CPU.</li></ul>|

**Returns**

|Data Type|Description|
|--|--|
|Tensor|Tensor object created from `torch.Tensor`.|

>[!NOTE] Note
>The constructed Tensor object shares data with the `torch.Tensor`, and the `torch.Tensor` object manages the lifetime of the data.

**Example**

```python
from mm import Tensor
import torch

tensor = torch.zeros((1024, 768, 3), dtype=torch.uint8)
mm_tensor = Tensor.from_torch(tensor)
```

### `Tensor.torch`

**Description**

Convert a Tensor object into a `torch.Tensor`.

**Function Prototype**

```typescript
torch()-> torch.Tensor
```

**Returns**

|Data Type|Description|
|--|--|
|torch.Tensor|Converted `torch.Tensor` tensor.|

>[!NOTE] Note
>
>- The Tensor object is converted into a `torch.Tensor` object. The converted `torch.Tensor` shares data with the Tensor, and the Tensor object manages the lifetime of the data.
>- The Tensor must reside on the CPU.

**Example**

```python
from mm import Tensor
import torch

tensor = torch.zeros((1024, 768, 3), dtype=torch.uint8)
mm_tensor = Tensor.from_torch(tensor)
torch_tensor = mm_tensor.torch()
```

### `Tensor.normalize`

**Description**

Normalize the current Tensor object using the mean and standard deviation. Given the mean values of `n` channels, `mean[1], ..., mean[n]`, and the standard deviation values, `std[1], ..., std[n]`, this operation normalizes each channel of the current Tensor object. That is, `output[channel] = (src[channel] - mean[channel]) / std[channel]`, where `src` is the current Tensor object.

**Function Prototype**

```typescript
def normalize(mean: list[float], std: list[float], device_mode: DeviceMode = DeviceMode.CPU)-> Tensor
```

**Parameters**

|Parameter|Data Type|Optional/Required|Description|
|--|--|--|--|
|mean|list[float]|Required|Mean array. Its length must be 3, and the valid range is [0, 1].|
|std|list[float]|Required|Standard deviation array. Its length must be 3, and the valid range is greater than 0 and less than 3.4028235e38.|
|device_mode|DeviceMode|Optional|Runtime mode. Currently only CPU is supported.|

**Returns**

|Data Type|Description|
|--|--|
|Tensor|Converted Tensor object.|

>[!NOTE] Note
>
>- The Tensor format supports only NCHW or NHWC. N supports only 1, and C supports only 3.
>- The Tensor data type supports only Float32.
>- The Tensor must reside on the CPU.

**Example**

```python
from mm import Tensor
import torch

tensor = torch.zeros((1024, 768, 3), dtype=torch.uint8)
mm_tensor = Tensor.from_torch(tensor)
mean = [0.1, 0.1, 0.1]
std = [0.1, 0.1, 0.1]
dst_mm_tensor = mm_tensor.normalize(mean, std)
```

## `mm.Image`

The Image class serves as a container for image data. It supports creating, managing, and copying image data.

### Image Property List

|Property|Type|Description|
|--|--|--|
|device|str|Device where the Image resides. Only `cpu` is supported.|
|dtype|DataType|Image data type. Only `DataType.UINT8` is supported.|
|size|list|Image size.|
|format|ImageFormat|Image format of the Image data.|
|nbytes|int|Number of bytes occupied by the Image data.|
|height|int|Image height.|
|width|int|Image width.|

### `Image.open`

**Description**

Create an Image from a specified path.

**Function Prototype**

```typescript
open(path:str | bytes, device:str | bytes = b'cpu')
```

**Parameters**

|Parameter|Data Type|Optional/Required|Description|
|--|--|--|--|
|path|str \| bytes|Required|<ul><li>The input path must be valid, no longer than 4096 characters, contain no symbolic links, be no larger than 1 GB, and the file permissions must not exceed 640. The User, Group, and Others permissions must not exceed 6, 4, and 0, respectively. The file suffix must be jpg or jpeg.</li><li>The input image must be either jpg or jpeg, and both the width and height must be within [10, 8192].</li><li>Images created by `Image.open` are currently RGB only.</li></ul>|
|device|str \| bytes|Optional|Device type. Currently only `cpu` is supported, and the default is `cpu`.|

**Returns**

|Data Type|Description|
|--|--|
|Image|New Image instance created from the path.|

**Example**

```python
from mm import Image
img= Image.open("/home/test.jpg", "cpu")
```

### `Image.from_numpy`

**Description**

Convert a NumPy array into an Image instance.

**Function Prototype**

```typescript
from_numpy(
        nd_array: numpy.ndarray,
        image_format: ImageFormat,
        device: str | bytes = b"cpu"
) -> Image:
```

**Parameters**

|Parameter|Data Type|Optional/Required|Description|
|--|--|--|--|
|nd_array|numpy.ndarray|Required|Input NumPy array. It must meet the following conditions: <ul><li>The input dtype supports only uint8. The input NumPy array must have 3 dimensions and cannot be empty.</li><li>For a 3D array, the array shape for RGB and BGR image formats must be `[H, W, 3]`. The array shape for BGR_PLANAR and RGB_PLANAR image formats must be `[3, H, W]`.</li><li>The values of each element in the input NumPy array must be within [0, 255]. The array must be row-major and contiguous in memory.</li><li>Both the width and height must be within [10, 8192].</li></ul>|
|image_format|ImageFormat|Required|Image format. It supports RGB, BGR, BGR_PLANAR, and RGB_PLANAR. The input type must match the NumPy data dimensions.|
|device|str \| bytes|Optional|Device type. Currently only `cpu` is supported, and the default is `cpu`.|

**Returns**

|Data Type|Description|
|--|--|
|Image|Image instance created from `numpy.ndarray`.|

>[!NOTE] Note
>The constructed Image object shares data with the `numpy.ndarray`, and the `numpy.ndarray` object manages the lifetime of the data.

**Example**

```python
from mm import Image, ImageFormat
import numpy as np

arr = np.zeros((1024, 768, 3), dtype=np.uint8)
img = Image.from_numpy(arr, ImageFormat.RGB, "cpu")
```

### `Image.numpy`

**Description**

Convert an Image instance into a NumPy array.

**Function Prototype**

```typescript
numpy()-> numpy.ndarray
```

**Returns**

|Data Type|Description|
|--|--|
|numpy.ndarray|Converted NumPy array.|

>[!NOTE] Note
>
>- The output `ndarray` shape depends on the image format. When the Image instance format is RGB or BGR, the shape is `[H, W, 3]`. When `format` is RGB_PLANAR or BGR_PLANAR, the shape is `[3, H, W]`.
>- The Image must reside on the CPU.

**Example**

```python
from mm import Image
import numpy as np

img = Image.open("/home/test.jpg", "cpu")
arr = img.numpy()
```

### `Image.from_torch`

**Description**

Convert a `torch.Tensor` tensor into an Image instance.

**Function Prototype**

```typescript
from_torch(
        torch_tensor: torch.Tensor,
        image_format: ImageFormat,
        device: str | bytes = b"cpu"
) -> Image:
```

**Parameters**

|Parameter|Data Type|Optional/Required|Description|
|--|--|--|--|
|torch_tensor|torch.Tensor|Required|Input PyTorch tensor. It must meet the following conditions: <ul><li>The input tensor dtype supports only uint8. The input `torch.Tensor` must have 3 dimensions and cannot be empty.</li><li>For a 3D array, the array shape for RGB and BGR image formats must be `[H, W, 3]`. The array shape for BGR_PLANAR and RGB_PLANAR image formats must be `[3, H, W]`.</li><li>The values of each element in the input `torch.Tensor` array must be within [0, 255]. The array must be row-major and contiguous in memory, and the device must be on the CPU.</li><li>The tensor width and height must both be within [10, 8192].</li></ul>|
|image_format|ImageFormat|Required|Image format. It supports RGB, BGR, BGR_PLANAR, and RGB_PLANAR. The input type must match the `torch.Tensor` data dimensions.|
|device|str \| bytes|Optional|Device type. Currently only `cpu` is supported, and the default is `cpu`.|

>[!NOTE] Note
>The constructed Image object shares data with the `torch.Tensor`, and the `torch.Tensor` object manages the lifetime of the data.

**Returns**

|Data Type|Description|
|--|--|
|Image|Image instance created from `torch.Tensor`.|

**Example**

```python
from mm import Image, ImageFormat
import torch

tensor = torch.zeros((1024, 768, 3), dtype=torch.uint8)
img = Image.from_torch(tensor, ImageFormat.RGB, "cpu")
```

### `Image.torch`

**Description**

Convert an Image instance into a `torch.Tensor` tensor.

**Function Prototype**

```typescript
torch()-> torch.Tensor
```

**Returns**

|Data Type|Description|
|--|--|
|torch.Tensor|Converted `torch.Tensor` tensor.|

>[!NOTE] Note
>
>- The output Tensor shape depends on the image format. When the Image instance `format` is RGB or BGR, the shape is `[H, W, 3]`. When `format` is BGR_PLANAR or RGB_PLANAR, the shape is `[3, H, W]`.
>- The Image must reside on the CPU.

**Example**

```python
from mm import Image
import torch

img = Image.open("/home/test.jpg", "cpu")
tensor = img.torch()
```

### `Image.from_pillow`

**Description**

Convert a PIL image object into an Image object.

**Function Prototype**

```typescript
from_pillow(pillow_image: PIL.Image.Image) -> Image
```

**Parameters**

|Parameter|Data Type|Optional/Required|Description|
|--|--|--|--|
|pillow_image|PIL.Image.Image|Required|Input Pillow Image object. Its `mode` attribute supports only `"L"` for grayscale images, `"RGB"` for RGB images, and `"RGBA"` for RGB images with an alpha channel.|

**Returns**

|Data Type|Description|
|--|--|
|Image|Constructed Image object.|

**Example**

```python
from mm import Image
from PIL import Image as PImage
import numpy as np

img_np = np.random.randint(0, 255, size=(1, 400, 400, 3), dtype=np.uint8)
img_pil = PImage.fromarray(img_np[0])
img_mm = Image.from_pillow(img_pil)
```

### `Image.pillow`

**Description**

Convert an Image object into a PIL image object.

**Function Prototype**

```typescript
pillow()-> PIL.Image.Image
```

**Returns**

|Data Type|Description|
|--|--|
|PIL.Image.Image|Converted Pillow image instance.|

>[!NOTE] Note
>
>- The output `PIL.Image.Image` dtype must match the Image instance. Currently, only uint8 is supported.
>- The output PIL Image instance `mode` is `"RGB"`.
>- The Image must reside on the CPU.

**Example**

```python
from mm import Image
from PIL import Image as PImage

pillow_image = PImage.open("/home/test.jpg")
img = Image.from_pillow(pillow_image)
pillow_image_new = img.pillow()
```

### `Image.clone`

**Description**

Deep copy the Image instance into a new Image instance.

**Function Prototype**

```typescript
clone()-> Image
```

**Returns**

|Data Type|Description|
|--|--|
|Image|New Image instance created by a deep copy.|

**Example**

```python
from mm import Image
img = Image.open("/home/test.jpg", "cpu")
img_copy = img.clone()
```

### `Image.resize`

**Description**

Resize an Image instance.

**Function Prototype**

```typescript
resize(size: Tuple[int, int], interpolation: Interpolation, device_mode: DeviceMode) -> "Image"
```

**Parameters**

|Parameter|Data Type|Optional/Required|Description|
|--|--|--|--|
|size|Tuple[int, int]|Required|Image width and height after resize. `size` must have two dimensions. The first dimension is `width`, and the second dimension is `height`. Both width and height must be within [10, 8192].|
|interpolation|Interpolation|Required|Resize interpolation algorithm. Currently only BICUBIC is supported.|
|device_mode|DeviceMode|Optional|Runtime mode for resize. Currently only CPU is supported.|

**Returns**

|Data Type|Description|
|--|--|
|Image|New Image instance obtained after resize.|

>[!NOTE] Note
>Currently, only the RGB or BGR image formats are supported, the data type is UINT8, and each element value is within [0, 255].

**Example**

```python
from mm import Image, DeviceMode, Interpolation
img = Image.open("/home/test.jpg", "cpu")
img_resize = img.resize((10,10), Interpolation.BICUBIC, DeviceMode.CPU)
```

### `Image.crop`

**Description**

Crop an Image instance.

**Function Prototype**

```typescript
crop(top: int, left: int, height: int, width: int, device_mode: DeviceMode) -> "Image":
```

**Parameters**

|Parameter|Data Type|Optional/Required|Description|
|--|--|--|--|
|top|int|Required|Top starting coordinate of the crop. It starts from 0. `(top + height)` must not exceed the original image height.|
|left|int|Required|Left starting coordinate of the crop. It starts from 0. `(left + width)` must not exceed the original image width.|
|height|int|Required|Crop height. The range is [10, original image height - top].|
|width|int|Required|Crop width. The range is [10, original image width - left].|
|device_mode|DeviceMode|Optional|Runtime mode for crop. Currently only CPU is supported.|

**Returns**

|Data Type|Description|
|--|--|
|Image|New Image instance obtained after crop.|

>[!NOTE] Note
>Currently, only the RGB or BGR image formats are supported, the data type is UINT8, and each element value is within [0, 255].

**Example**

```python
from mm import Image, DeviceMode
img = Image.open("/home/test.jpg", "cpu")
img_crop = img.crop(10, 10, 10, 10, DeviceMode.CPU)
```

### `Image.to_tensor`

**Description**

Scale an Image instance from [0, 255] to `[0.0, 1.0]`, and convert the format from HWC to CHW.

>[!NOTE] Note
>Currently, only the RGB or BGR image formats are supported, the data type is UINT8, and each element value is within [0, 255].
>The output Tensor instance data type defaults to `DataType.FLOAT32`.

**Function Prototype**

```typescript
def to_tensor(target_format: TensorFormat = TensorFormat.NCHW, device_mode: DeviceMode = DeviceMode.CPU) -> "Tensor":
```

**Parameters**

|Parameter|Data Type|Optional/Required|Description|
|--|--|--|--|
|target_format|TensorFormat|Optional|Format of the converted Tensor instance. It supports NHWC and NCHW. The default value is NCHW.|
|device_mode|DeviceMode|Optional|Runtime mode. Currently only CPU is supported.|

**Returns**

|Data Type|Description|
|--|--|
|Tensor|Tensor instance obtained after the operation.|

**Example**

```python
from mm import Image, TensorFormat, DeviceMode
img = Image.open("/home/test.jpg", "cpu")
dst_tensor = img.to_tensor(TensorFormat.NCHW, DeviceMode.CPU)
```

## Log Registration

Register the log level and log callback function.

### `register_log_conf`

Log registration function.

**Function Prototype**

```typescript
register_log_conf(min_level: LogLevel, callback: Callable[[LogLevel, str, str, int, str], None])
```

**Parameters**

|Parameter|Type|Optional/Required|Description|
|--|--|--|--|
|min_level|LogLevel|Required|Minimum log level. Only logs at or above this level are output. `None` is not allowed.|
|callback|Callable[[LogLevel, str, str, int, str], None]|Required|Log callback function. When `None` is passed, the built-in default log output function is used.|

>[!CAUTION] Caution
>If the log callback function throws an exception, a C++ exception is triggered and the program core dumps. You are advised to catch and handle exceptions in the callback.

**Example**

```python
from mm import register_log_conf, LogLevel
def custom_log_handler(level: LogLevel, file: str, func: str, line: int, msg: str) -> None:
    print(f"[custom][{level}] {file}:{line} ({func}) - {msg}")
register_log_conf(LogLevel.ERROR, custom_log_handler)
```

## `mm.video_decode`

**Description**

Decode the input video file and return a list of Image objects.

**Function Prototype**

```typescript
def video_decode(video_path: str | bytes, device: str | bytes, frame_indices: set = None, sample_num: int = -1) -> list:
```

**Parameters**

|Parameter|Data Type|Optional/Required|Description|
|--|--|--|--|
|video_path|str \| bytes|Required|Path of the video file to decode. Currently only MP4 files are supported, and the resolution supports 480P to 4K.|
|device|str \| bytes|Required|Decode device. Currently only `cpu` is supported.|
|frame_indices|set|Optional|Video frame IDs to decode.|
|sample_num|int|Optional|Total number of frames expected after decoding.|

**Returns**

|Data Type|Description|
|--|--|
|list[Image]|List of decoded Image objects. The format is RGB, and the data type is uint8.|

>[!NOTE] Note
>
>- The expected frame ID range is `[0, total number of video frames - 1)`. The default is an empty set. This parameter has higher priority than the expected total number of frames after decoding. This parameter indicates the set of target frame IDs to decode.
>- The expected total number of frames after decoding is in the range `(0, total number of video frames]`. The default value is -1. The final decoded ID set is sampled at equal intervals based on the total number of video frames.
>- If neither `frame_indices` nor `sample_num` is specified, the call fails.
>- Resolution support: currently supported video frame sizes are [480, 480] to [4096, 4096].
>- The input path must be valid, no longer than 4096 characters, contain no symbolic links, and the file permissions must not exceed 640. The User, Group, and Others permissions must not exceed 6, 4, and 0, respectively.

**Example**

```python
from mm import video_decode

file_path = "test.mp4"
mm_images = video_decode(file_path, "cpu", [], 32)
```

## `mm.normalize`

**Description**

Normalize a Tensor object using the mean and standard deviation. Given the mean values of `n` channels, `mean[1], ..., mean[n]`, and the standard deviation values, `std[1], ..., std[n]`, this transform normalizes each channel of the input Tensor object. That is, `output[channel] = (src[channel] - mean[channel]) / std[channel]`.

**Function Prototype**

```typescript
def normalize(src: Tensor, mean: list[float], std: list[float], device_mode: DeviceMode = DeviceMode.CPU)-> Tensor
```

**Parameters**

|Parameter|Data Type|Optional/Required|Description|
|--|--|--|--|
|src|Tensor|Required|Input Tensor instance.|
|mean|list[float]|Required|Mean array. Its length must be 3, and the valid range is [0, 1].|
|std|list[float]|Required|Standard deviation array. Its length must be 3, and the valid range is [0, 3.4028235e38].|
|device_mode|DeviceMode|Optional|Runtime mode. Currently only CPU is supported.|

**Returns**

|Data Type|Description|
|--|--|
|Tensor|Converted Tensor object.|

>[!NOTE] Note
>
>- The input Tensor format supports only NCHW or NHWC. N supports only 1, and C supports only 3.
>- The input Tensor data type supports only Float32.
>- The Tensor must reside on the CPU.

**Example**

```python
from mm import Tensor
import torch

tensor = torch.zeros((1024, 768, 3), dtype=torch.uint8)
mm_tensor = Tensor.from_torch(tensor)
mean = [0.1, 0.1, 0.1]
std = [0.1, 0.1, 0.1]
dst_mm_tensor = normalize(mm_tensor, mean, std, DeviceMode.CPU)
```

## `mm.load_audio`

**Description**

Load the given audio through the audio loading API. It supports loading a single audio file, parallel loading of a batch of audio files, and toggling resampling.

**Function Prototype**

```typescript
def load_audio(audio_inputs: Union[str, List[str]], sr: Optional[int] = None)
-> Union[Tuple[Tensor, int], List[Tuple[Tensor, int]]]
```

**Parameters**

|Parameter|Data Type|Optional/Required|Description|
|--|--|--|--|
|audio_inputs|Union[str, List[str]]|Required|Input single audio path, the directory that contains multiple audio files, or an audio file list.|
|sr|Optional[int]|Optional|If you specify a resampling rate, the audio is resampled at that rate. Otherwise, resampling is not performed.|

**Returns**

|Data Type|Description|
|--|--|
|Union[Tuple[Tensor, int], List[Tuple[Tensor, int]]]|A single audio file returns `(audio tensor data, sample rate)`. Multiple audio files return a list of `(audio tensor data, sample rate)`.|

>[!NOTE] Note
>
>- Input audio supports only wav files.
>- The number of audio files that can be loaded ranges from 1 to 128.
>- The sample rate that you specify must be a positive integer in the range [1, 64000].
>- Multichannel audio is automatically converted to mono audio.

**Example**

```python
from mm import load_audio
single_audio_path = "/path/to/speech.wav"
audio_file_paths = ["/path/to/audio1.wav", "/path/to/audio2.wav"]
audio_directory = "/path/to/audio_dir"

waveform, sr = load_audio(single_audio_path)
batch_from_filelist = load_audio(audio_file_paths)
batch_from_directory = load_audio(audio_directory)
```
