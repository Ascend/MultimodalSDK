# Function Reference

## `mm.Tensor`

This document applies to the latest release of Multimodal SDK. Interface exceptions are usually thrown as `ValueError`, `TypeError`, `RuntimeError`, or `ImportError`. For guidance on handling error codes, see [Appendix > Error Codes](../06_references/appendix.md#error-codes).

The Tensor class serves as a container for general-purpose data of any modality. It supports data creation, management, and copy operations.

### Tensor Properties

|Property|Type|Description|Remarks|
|--|--|--|--|
|device|str|Device where the Tensor resides.|The default is `"cpu"`.|
|dtype|DataType|Tensor data type.|The default is `DataType.FLOAT32`.|
|shape|list|Tensor dimension information.|The default is an empty list.|
|format|TensorFormat|Tensor data layout type.|The default is `TensorFormat.ND`.|
|nbytes|int|Number of bytes occupied by the Tensor data.|The default is 0.|

### `Tensor.set_format`

**Description**

Set the data layout format.

**Function Prototype**

```python
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

```python
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

```python
def from_numpy(nd_array: numpy.ndarray) -> Tensor:
```

**Parameters**

|Parameter|Data Type|Optional/Required|Description|
|--|--|--|--|
|nd_array|numpy.ndarray|Required|The input dtype supports int8, uint8, and float32 data types. The input `numpy.ndarray` must be row-major, and its memory must be contiguous.|

**Returns**

|Data Type|Description|
|--|--|
|Tensor|Tensor instance created from `numpy.ndarray`.|

>[!NOTE]
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

```python
numpy()-> np.ndarray
```

**Returns**

|Data Type|Description|
|--|--|
|numpy.ndarray|Converted NumPy array.|

>[!NOTE]
>
>- The Tensor object is converted into a `numpy.ndarray` object. The converted `numpy.ndarray` shares data with the Tensor, and the Tensor object manages the lifetime of the data.
>- The device on which the Tensor resides must be the CPU.

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

Convert a `torch.Tensor` tensor into a Tensor object.

**Function Prototype**

```python
def from_torch(torch_tensor: torch.Tensor) -> Tensor:
```

**Parameters**

|Parameter|Data Type|Optional/Required|Description|
|--|--|--|--|
|torch_tensor|torch.Tensor|Required|The input dtype supports int8, uint8, and float32 data types. The input `torch.Tensor` must be row-major, and its memory must be contiguous. The input `torch.Tensor` must be on the CPU.|

**Returns**

|Data Type|Description|
|--|--|
|Tensor|Tensor object created from `torch.Tensor`.|

>[!NOTE]
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

Convert a Tensor object into a `torch.Tensor` tensor.

**Function Prototype**

```python
torch()-> torch.Tensor
```

**Returns**

|Data Type|Description|
|--|--|
|torch.Tensor|Converted `torch.Tensor` tensor.|

>[!NOTE]
>
>- The Tensor object is converted into a `torch.Tensor` object. The converted `torch.Tensor` shares data with the Tensor, and the Tensor object manages the lifetime of the data.
>- The device on which the Tensor resides must be the CPU.

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

A member function of the Tensor class that uses the mean and standard deviation to normalize the current object. Given the mean values of `n` channels, `mean[1], ..., mean[n]`, and the standard deviation values, `std[1], ..., std[n]`, this transform normalizes each channel of the current object. That is, `output[channel] = (src[channel] - mean[channel]) / std[channel]`, where `src` is the current Tensor object.

**Function Prototype**

```python
def normalize(mean: list[float], std: list[float], device_mode: DeviceMode = DeviceMode.CPU)-> Tensor
```

**Parameters**

|Parameter|Data Type|Optional/Required|Description|
|--|--|--|--|
|mean|list[float]|Required|Mean array. Its length must be 3, and the valid range is [0, 1].|
|std|list[float]|Required|Standard deviation array. Its length must be 3, and the valid range is (0, 3.4028235e38].|
|device_mode|DeviceMode|Optional|Runtime mode. Currently only CPU is supported.|

**Returns**

|Data Type|Description|
|--|--|
|Tensor|Converted Tensor object.|

>[!NOTE]
>
>- The format of the Tensor object supports only NCHW or NHWC. N supports only 1, and C supports only 3.
>- The data type of the Tensor object supports only Float32.
>- The device on which the Tensor object resides must be the CPU.

**Example**

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

## `mm.Image`

The Image class serves as a container for general-purpose image data. It supports image creation, management, and copy operations.

### Image Properties

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

```python
open(path:str | bytes, device:str | bytes = b'cpu')
```

**Parameters**

|Parameter|Data Type|Optional/Required|Description|
|--|--|--|--|
|path|str \| bytes|Required|The input path must be valid, no longer than 4096 characters, contain no symbolic links, and be no larger than 1 GB. The file permissions must not exceed 640, and the User, Group, and Others permissions must not exceed 6, 4, and 0, respectively. The file suffix must be jpg or jpeg. The input image must be jpg or jpeg, and both the width and height must be within [10, 8192]. Currently, only RGB images can be created by `Image.open`.|
|device|str \| bytes|Optional|Device type. Currently only `cpu` is supported, and the default is `cpu`.|

**Returns**

|Data Type|Description|
|--|--|
|Image|New Image instance created from the path.|

**Example**

```python
from mm import Image
img = Image.open("/home/test.jpg", "cpu")
```

### `Image.from_numpy`

**Description**

Convert a NumPy array into an Image instance.

**Function Prototype**

```python
from_numpy(
    nd_array: numpy.ndarray,
    image_format: ImageFormat,
    device: str | bytes = b"cpu"
) -> Image:
```

**Parameters**

|Parameter|Data Type|Optional/Required|Description|
|--|--|--|--|
|nd_array|numpy.ndarray|Required|Input NumPy array. It must meet the following conditions: the input dtype supports only the uint8 data type, and the input NumPy array must have 3 dimensions and cannot be empty. For a 3D array, the shape for the RGB and BGR image formats must be [H, W, 3], and the shape for the BGR_PLANAR and RGB_PLANAR image formats must be [3, H, W]. Each element of the input NumPy array must be in the range [0, 255], the array must be row-major with contiguous memory, and both the width and height must be within [10, 8192].|
|image_format|ImageFormat|Required|Image format. It supports RGB, BGR, BGR_PLANAR, and RGB_PLANAR. The input type must correspond to the NumPy data dimensions.|
|device|str \| bytes|Optional|Device type. Currently only `cpu` is supported, and the default is `cpu`.|

**Returns**

|Data Type|Description|
|--|--|
|Image|Image instance created from `numpy.ndarray`.|

>[!NOTE]
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

```python
numpy()-> numpy.ndarray
```

**Returns**

|Data Type|Description|
|--|--|
|numpy.ndarray|Converted NumPy array.|

>[!NOTE]
>
>- The output ndarray shape depends on the image format:
> When the Image instance format is RGB or BGR, the shape is [H, W, 3]. When `format` is RGB_PLANAR or BGR_PLANAR, the shape is [3, H, W].
>- The device on which the Image resides must be the CPU.

**Example**

```python
from mm import Image
import numpy as np

img = Image.open("/home/test.jpg", "cpu")
arr = img.numpy()
```

### `Image.from_torch`

**Description**

Convert a `torch.Tensor` into an Image instance.

**Function Prototype**

```python
from_torch(
    torch_tensor: torch.Tensor,
    image_format: ImageFormat,
    device: str | bytes = b"cpu"
) -> Image:
```

**Parameters**

|Parameter|Data Type|Optional/Required|Description|
|--|--|--|--|
|torch_tensor|torch.Tensor|Required|Input PyTorch tensor. It must meet the following conditions: the dtype supports only uint8, and the number of dimensions must be 3. The shape is [H, W, 3] for the RGB and BGR formats and [3, H, W] for the PLANAR formats. The value range is [0, 255], the tensor must be row-major with contiguous memory and reside on the CPU, and both the width and height must be within [10, 8192].|
|image_format|ImageFormat|Required|It supports RGB, BGR, BGR_PLANAR, and RGB_PLANAR, and must correspond to the data dimensions of the `torch.Tensor`.|
|device|str \| bytes|Optional|Device type. Currently only `cpu` is supported, and the default is `cpu`.|

>[!NOTE]
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

```python
torch()-> torch.Tensor
```

**Returns**

|Data Type|Description|
|--|--|
|torch.Tensor|Converted torch tensor.|

>[!NOTE]
>
>- The output Tensor shape depends on the image format:
> When the Image instance `format` is RGB or BGR, the shape is [H, W, 3]. When `format` is BGR_PLANAR or RGB_PLANAR, the shape is [3, H, W].
>- The device on which the Image resides must be the CPU.

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

```python
from_pillow(pillow_image: PIL.Image.Image) -> Image
```

**Parameters**

|Parameter|Data Type|Optional/Required|Description|
|--|--|--|--|
|pillow_image|PIL.Image.Image|Required|Input Pillow Image object. Its `mode` attribute supports only `"L"` (grayscale image), `"RGB"` (RGB image), and `"RGBA"` (RGB image with an alpha channel).|

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

```python
pillow()-> PIL.Image.Image
```

**Returns**

|Data Type|Description|
|--|--|
|PIL.Image.Image|Converted Pillow image instance.|

>[!NOTE]
>
>- The dtype of the output `PIL.Image.Image` must be the same as that of the Image instance. Currently, only uint8 is supported.
>- The `mode` of the output PIL Image instance is `"RGB"`.
>- The device on which the Image resides must be the CPU.

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

```python
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

```python
resize(size: Tuple[int, int], interpolation: Interpolation, device_mode: DeviceMode = DeviceMode.CPU) -> "Image"
```

**Parameters**

|Parameter|Data Type|Optional/Required|Description|
|--|--|--|--|
|size|Tuple[int, int]|Required|Image width and height after resize. `size` must have two dimensions. The first dimension is `width`, and the second dimension is `height`. Both the width and height must be within [10, 8192].|
|interpolation|Interpolation|Required|Interpolation algorithm for resize. Currently only BICUBIC is supported.|
|device_mode|DeviceMode|Optional|Runtime mode for resize. Currently only CPU is supported.|

**Returns**

|Data Type|Description|
|--|--|
|Image|New Image instance obtained after the resize operation.|

>[!NOTE]
>Currently, only the RGB or BGR image formats are supported, the data type is UINT8, and each element value is within [0, 255].

**Example**

```python
from mm import Image, DeviceMode, Interpolation
img = Image.open("/home/test.jpg", "cpu")
img_resize = img.resize((10, 10), Interpolation.BICUBIC, DeviceMode.CPU)
```

### `Image.crop`

**Description**

Crop an Image instance.

**Function Prototype**

```python
crop(top: int, left: int, height: int, width: int, device_mode: DeviceMode = DeviceMode.CPU) -> "Image"
```

**Parameters**

|Parameter|Data Type|Optional/Required|Description|
|--|--|--|--|
|top|int|Required|Top starting coordinate of the crop. It starts from 0, and `(top+height)` must not exceed the original image height.|
|left|int|Required|Left starting coordinate of the crop. It starts from 0, and `(left+width)` must not exceed the original image width.|
|height|int|Required|Height of the crop area. The range is [10, original image height - top].|
|width|int|Required|Width of the crop area. The range is [10, original image width - left].|
|device_mode|DeviceMode|Optional|Runtime mode for crop. Currently only CPU is supported.|

**Returns**

|Data Type|Description|
|--|--|
|Image|New Image instance obtained after the crop operation.|

>[!NOTE]
>Currently, only the RGB or BGR image formats are supported, the data type is UINT8, and each element value is within [0, 255].

**Example**

```python
from mm import Image, DeviceMode
img = Image.open("/home/test.jpg", "cpu")
img_crop = img.crop(10, 10, 10, 10, DeviceMode.CPU)
```

### `Image.to_tensor`

**Description**

Scale an Image instance from [0, 255] to [0.0, 1.0] and convert its format from HWC to CHW.

>[!NOTE]
>Currently, only the RGB or BGR image formats are supported, the data type is UINT8, and each element value is within [0, 255].
>The data type of the output Tensor instance defaults to `DataType.FLOAT32`.

**Function Prototype**

```python
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

You can register the log level and the log callback function.

### `register_log_conf`

Log registration function.

**Function Prototype**

```python
register_log_conf(min_level: LogLevel, callback: Callable[[LogLevel, str, str, int, str], None])
```

**Parameters**

|Parameter|Type|Optional/Required|Description|
|--|--|--|--|
|min_level|LogLevel|Required|Minimum log level. Only logs at this level or higher are output. `None` is not allowed.|
|callback|Callable[[LogLevel, str, str, int, str], None]|Required|Log callback function. When `None` is passed, the built-in default log output function is used.|

>[!NOTE]
>Throwing an exception in the log callback function triggers an exception on the C++ side and causes the program to core dump. You are advised to catch and handle exceptions in the callback.

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

```python
def video_decode(video_path: str | bytes, device: str | bytes, frame_indices: set = None, sample_num: int = -1) -> list:
```

**Parameters**

|Parameter|Data Type|Optional/Required|Description|
|--|--|--|--|
|video_path|str \| bytes|Required|Path of the video to decode. Currently only mp4 files are supported, and the resolution supports 480P to 4K.|
|device|str \| bytes|Required|Decoding device. Currently only `cpu` is supported.|
|frame_indices|set|Optional|Video frame IDs expected to be decoded.|
|sample_num|int|Optional|Total number of frames expected after decoding.|

**Returns**

|Data Type|Description|
|--|--|
|list[Image]|List of decoded Image objects. The format is RGB, and the data type is uint8.|

>[!NOTE]
>
>- The value range of the video frame IDs expected to be decoded is [0, total number of video frames - 1). The default is an empty set. This parameter has a higher priority than the total number of frames expected after decoding. It indicates the set of frame IDs to decode.
>- The total number of frames expected after decoding is in the range (0, total number of video frames]. The default value is -1. The final decoded ID set is obtained by sampling at equal intervals based on the total number of video frames.
>- If neither `frame_indices` nor `sample_num` is specified, the call fails.
>- Resolution: the currently supported video frame width and height are [480, 480] to [4096, 4096].
>- The input path must be valid, no longer than 4096 characters, contain no symbolic links, and have file permissions no greater than 640. The User, Group, and Others permissions must not exceed 6, 4, and 0, respectively.

**Example**

```python
from mm import video_decode

file_path = "test.mp4"
mm_images = video_decode(file_path, "cpu", set(), 32)
```

<a id="mmnormalize"></a>

## `mm.normalize`

**Description**

Normalize a Tensor object using the mean and standard deviation. Given the mean values of `n` channels, `mean[1], ..., mean[n]`, and the standard deviation values, `std[1], ..., std[n]`, this transform normalizes each channel of the input Tensor object. That is, `output[channel] = (src[channel] - mean[channel]) / std[channel]`.

**Function Prototype**

```python
def normalize(src: Tensor, mean: list[float], std: list[float], device_mode: DeviceMode = DeviceMode.CPU)-> Tensor
```

**Parameters**

|Parameter|Data Type|Optional/Required|Description|
|--|--|--|--|
|src|Tensor|Required|Input Tensor instance.|
|mean|list[float]|Required|Mean array. Its length must be 3, and the valid range is [0, 1].|
|std|list[float]|Required|Standard deviation array. Its length must be 3, and the valid range is (0, 3.4028235e38].|
|device_mode|DeviceMode|Optional|Runtime mode. Currently only CPU is supported.|

**Returns**

|Data Type|Description|
|--|--|
|Tensor|Converted Tensor object.|

>[!NOTE]
>
>- The format of the input Tensor object supports only NCHW or NHWC. N supports only 1, and C supports only 3.
>- The data type of the input Tensor object supports only Float32.
>- The device on which the Tensor object resides must be the CPU.

**Example**

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

## `mm.load_audio`

**Description**

Use the audio loading API to load the given audio. It supports loading a single audio file and parallel loading of a batch of audio files, and it supports toggling resampling.

**Function Prototype**

```python
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
|Union[Tuple[Tensor, int], List[Tuple[Tensor, int]]]|A single audio file returns (audio tensor data, sample rate). Multiple audio files return a list of (audio tensor data, sample rate).|

>[!NOTE]
>
>- Input audio supports only wav files.
>- The number of audio files that can be loaded ranges from [1, 128].
>- The sample rate that you input must be a positive integer in the range [1, 64000].
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

<a id="mmbaseframeselector"></a>

## `mm.BaseFrameSelector`

An abstract base class for keyframe selectors based on text-image matching. It encapsulates common capabilities such as model initialization, feature extraction, boundary location, input validation, and similarity calculation. This class cannot be instantiated directly. Use it through the `KRangFrameSelector` or `KFrameSelector` subclass.

### BaseFrameSelector Properties

|Property|Type|Description|
|--|--|--|
|model_path|str|Path of the CLIP model weights.|
|device_id|int|NPU device index.|
|model_type|str|Model type. The value is `"clip"` for English or `"cn_clip"` for Chinese.|
|batch_size|int|Batch size for image feature extraction. The default is 64.|
|similar_threshold|float|Text-image similarity decay threshold. Frames whose similarity is lower than (maximum similarity - this value) are filtered out. The default is 0.025.|
|image_similar_threshold|float|Image similarity gradient threshold, used for scene boundary detection. When the cosine similarity gradient between adjacent frames exceeds this value, a scene switch is determined. The default is 0.015.|
|image_size|tuple|Resize size of the input image. It must be the same as the model training size. The default is (672, 672).|

### `BaseFrameSelector.__init__`

**Description**

Initialize the keyframe selector, load the specified model, and complete parameter validation.

**Function Prototype**

```python
__init__(model_path: str, device_id: int, model_type: str = "clip", similar_threshold: float = 0.025, image_similar_threshold: float = 0.015, image_size: tuple = (672, 672))
```

**Parameters**

|Parameter|Data Type|Optional/Required|Description|
|--|--|--|--|
|model_path|str|Required|Path of the CLIP model weights. It must be a non-empty string and point to a valid directory.|
|device_id|int|Required|NPU device index. It must be an integer.|
|model_type|str|Optional|Model type. The value is `"clip"` for English or `"cn_clip"` for Chinese. The default is `"clip"`.|
|similar_threshold|float|Optional|Text-image similarity decay threshold. The valid range is [0, 1]. The default is 0.025.|
|image_similar_threshold|float|Optional|Image similarity gradient threshold, used for scene boundary detection. The valid range is [0, 1]. The default is 0.015.|
|image_size|tuple|Optional|Resize size of the input image, in the form of (width, height). The width and height must be within [10, 8192]. The default is (672, 672).|

>[!NOTE]
>
>- `BaseFrameSelector` is an abstract class and cannot be instantiated directly. Create instances through the `KRangFrameSelector` or `KFrameSelector` subclass.
>- The directory pointed to by `model_path` must exist, its permissions must not be higher than 750, and its owner must be the current user.

### `BaseFrameSelector.select_keyframes`

**Description**

Select keyframes related to the query text from a video frame sequence. This is an abstract method, and the subclass implements the specific selection strategy.

**Function Prototype**

```python
select_keyframes(query: str, frames: List[np.ndarray], sample_num: int, do_resample: bool) -> Tuple[List[int], List[np.ndarray]]
```

**Parameters**

|Parameter|Data Type|Optional/Required|Description|
|--|--|--|--|
|query|str|Required|Query text that describes the target visual content. It must be a non-empty string.|
|frames|List[np.ndarray]|Required|List of video frames.|
|sample_num|int|Required|Maximum number of keyframes. It must be a positive integer.|
|do_resample|bool|Required|Whether to perform adaptive resampling within the interval.|

**Returns**

|Data Type|Description|
|--|--|
|Tuple[List[int], List[np.ndarray]]|A tuple that contains the list of keyframe indices and the list of keyframe images.|

## `mm.KRangFrameSelector`

A keyframe selector based on interval merging that inherits from `BaseFrameSelector`. It locates contiguous scene intervals related to the query text and adaptively samples keyframes within those intervals. It applies to tasks that require temporal context.

### `KRangFrameSelector.select_keyframes`

**Description**

Main process for interval keyframe selection: after feature extraction and similarity calculation, greedily select candidate frames and expand them into scene intervals, merge adjacent semantically similar intervals, and finally perform adaptive resampling within the intervals.

**Function Prototype**

```python
select_keyframes(query: str, frames: List[np.ndarray], sample_num: int, do_resample: bool) -> Tuple[List[int], List[np.ndarray]]
```

**Parameters**

|Parameter|Data Type|Optional/Required|Description|
|--|--|--|--|
|query|str|Required|Query text that describes the target visual content. It must be a non-empty string.|
|frames|List[np.ndarray]|Required|List of video frames.|
|sample_num|int|Required|Maximum number of keyframes. It must be a positive integer.|
|do_resample|bool|Required|Whether to perform adaptive resampling within the merged intervals. When it is True, resampling is performed within the intervals by using the top-k similarity and uniform filling strategy. When it is False, only the endpoints of the merged intervals are returned.|

**Returns**

|Data Type|Description|
|--|--|
|Tuple[List[int], List[np.ndarray]]|A tuple that contains the list of keyframe indices (deduplicated and sorted) and the list of keyframe images.|

**Example**

```python
from mm import KRangFrameSelector
import numpy as np

selector = KRangFrameSelector(model_path="/path/to/clip_model", device_id=0, model_type="clip")
frames = [np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8) for _ in range(100)]
indices, key_frames = selector.select_keyframes(query="a cat sitting on a sofa", frames=frames, sample_num=8, do_resample=True)
```

## `mm.KFrameSelector`

A discrete keyframe selector that inherits from `BaseFrameSelector`. It selects discrete keyframes that are related to the query text and visually diverse. It applies to tasks that require visual diversity.

### `KFrameSelector.select_keyframes`

**Description**

Main process for discrete keyframe selection: after feature extraction and similarity calculation, greedily select candidate frames and deduplicate them by feature distance to ensure visual diversity.

**Function Prototype**

```python
select_keyframes(query: str, frames: List[np.ndarray], sample_num: int, do_resample: bool = False) -> Tuple[List[int], List[np.ndarray]]
```

**Parameters**

|Parameter|Data Type|Optional/Required|Description|
|--|--|--|--|
|query|str|Required|Query text that describes the target visual content. It must be a non-empty string.|
|frames|List[np.ndarray]|Required|List of video frames.|
|sample_num|int|Required|Maximum number of keyframes. It must be a positive integer.|
|do_resample|bool|Optional|This parameter is not used in KFrameSelector. The default is False.|

**Returns**

|Data Type|Description|
|--|--|
|Tuple[List[int], List[np.ndarray]]|A tuple that contains the list of keyframe indices (deduplicated and sorted) and the list of keyframe images.|

**Example**

```python
from mm import KFrameSelector
import numpy as np

selector = KFrameSelector(model_path="/path/to/clip_model", device_id=0, model_type="clip")
frames = [np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8) for _ in range(100)]
indices, key_frames = selector.select_keyframes(query="a dog running in the park", frames=frames, sample_num=8)
```

## `mm.core.processor.resize_and_normalize`

**Description**

Resize a batch of images or video frames and normalize their pixel values on the NPU, and return a `torch.Tensor`. Internally, it calls `acc.Qwen2VLProcessor.PreprocessTensor` and stays consistent with the preprocessing path in the Qwen2-VL and Qwen3-VL processors.

**Function Prototype**

```python
from typing import List
import torch

def resize_and_normalize(
    frames: torch.Tensor,
    height: int,
    width: int,
    image_mean: List[float],
    image_std: List[float],
) -> torch.Tensor
```

**Parameters**

|Parameter|Data Type|Optional/Required|Description|
|--|--|--|--|
|frames|torch.Tensor|Required|Input 4D tensor. Its `shape` must be (N, 3, H, W) or (N, H, W, 3) (NCHW or NHWC), and its `dtype` must be `torch.uint8`.|
|height|int|Required|Target resize height in pixels. It must be > 0.|
|width|int|Required|Target resize width in pixels. It must be > 0.|
|image_mean|List[float]|Required|Per-channel normalization mean. Its length must be 3, each element must be in the range [0.0, 1.0], and a `ValueError` is thrown when an element is out of range.|
|image_std|List[float]|Required|Per-channel normalization standard deviation. Its length must be 3, and all elements must be > 0.|

**Returns**

|Data Type|Description|
|--|--|
|torch.Tensor|Tensor after resize and normalize. The shape is (N, 3, height, width).|

**Exceptions**

- `ValueError`: `frames` is not a 4D tensor or is not of the `uint8` type, height or width is not positive, the length of `image_mean` or `image_std` is not 3, an element of `image_mean` is outside [0.0, 1.0], an element of `image_std` is not positive, or the shape is neither NCHW nor NHWC.

**Example**

```python
import torch
from mm.core.processor import resize_and_normalize

frames = torch.zeros((4, 3, 480, 640), dtype=torch.uint8)  # 4 frames in the NHWC layout are also supported
out = resize_and_normalize(
    frames,
    height=224,
    width=224,
    image_mean=[0.5, 0.5, 0.5],
    image_std=[0.5, 0.5, 0.5],
)
assert tuple(out.shape) == (4, 3, 224, 224)
```

## `mm.core.scc.scc_should_run`

**Description**

Determine whether SCC compression is worth applying to the current sample. The per-item computation cost of SCC grows with the number of input tokens. Beyond a certain scale, the compression time instead exceeds the savings on the LLM side. When the number of tokens exceeds the heuristic threshold (8192 tokens by default), compression is skipped.

**Function Prototype**

```python
def scc_should_run(n: int, max_tokens_per_item: int = 8192) -> bool
```

**Parameters**

|Parameter|Data Type|Optional/Required|Description|
|--|--|--|--|
|n|int|Required|Number of input tokens of the current sample (a single image or a single video). It must be > 0.|
|max_tokens_per_item|int|Optional|Heuristic threshold. When the number of tokens is less than or equal to this value, True is returned. A value of `<= 0` disables the check, and compression always runs. The default is 8192.|

**Returns**

|Data Type|Description|
|--|--|
|bool|True is returned when `n <= max_tokens_per_item` (or the threshold is disabled). Otherwise, False is returned.|

**Exceptions**

- `ValueError`: `n <= 0`.

**Example**

```python
from mm.core.scc import scc_should_run

scc_should_run(1024)              # True  -- default threshold 8192
scc_should_run(20000)             # False -- exceeds the default threshold
scc_should_run(20000, max_tokens_per_item=0)  # True -- threshold disabled
```

## `mm.core.scc.scc_shrink`

**Description**

Calculate the target number of tokens after SCC compression (that is, the placeholder or feature length) based on the given compression ratio.

**Function Prototype**

```python
def scc_shrink(tokens_per_frame: int, ratio: float) -> int
```

**Parameters**

|Parameter|Data Type|Optional/Required|Description|
|--|--|--|--|
|tokens_per_frame|int|Required|Number of tokens of each frame before compression.|
|ratio|float|Required|Compression ratio. It must be in (0, 1]. A value of 1.0 indicates no compression.|

**Returns**

|Data Type|Description|
|--|--|
|int|`max(1, ceil(tokens_per_frame * ratio))`, which ensures that at least 1 is returned.|

**Exceptions**

- `ValueError`: `ratio` is not in (0, 1].

**Example**

```python
from mm.core.scc import scc_shrink

scc_shrink(1024, 0.5)  # 512
scc_shrink(7,    0.1)  # 1
scc_shrink(1024, 1.0)  # 1024
```

## `mm.core.scc.scc_compress_to_target`

**Description**

Compress a set of visual tokens to exactly `k_target` tokens. In the video scenario, SCC runs independently on each frame and then concatenates the results, without feature fusion across frames. Then the number is adjusted to `k_target` (the centers with the lowest connectivity are removed, or the mean is used for padding).

**Function Prototype**

```python
def scc_compress_to_target(
    features: torch.Tensor,
    k_target: int,
    num_frames: int = 1,
    tokens_per_frame: int = 0,
    tau: float = 0.98,
    epsilon: float = 0.05,
) -> torch.Tensor
```

**Parameters**

|Parameter|Data Type|Optional/Required|Description|
|--|--|--|--|
|features|torch.Tensor|Required|Input token features. `shape=(N, D)`, and `dtype` must be one of `bfloat16`, `float16`, or `float32`.|
|k_target|int|Required|Target number of tokens after compression. It must satisfy `1 <= k_target <= N`, and when `num_frames > 1`, `k_target >= num_frames`.|
|num_frames|int|Optional|Number of video frames. `1` indicates an image, and `>1` indicates a video that is compressed independently by frame. The default is 1.|
|tokens_per_frame|int|Optional|Number of original tokens of each frame. It takes effect only when `num_frames > 1` and `tokens_per_frame > 0`. In this case, an error is thrown if `num_frames * tokens_per_frame != N`. The default is 0.|
|tau|float|Optional|Cosine similarity threshold. Tokens with `cos > tau` are considered the same component. The value must be in (0, 1]. The default is 0.98.|
|epsilon|float|Optional|Error tolerance of the approximate Union-Find sampling. It is used only in the CPU fallback path, and its value is in (0, 1). The default is 0.05.|

**Returns**

|Data Type|Description|
|--|--|
|torch.Tensor|Compressed features with the shape (k_target, D). The `dtype` is the same as that of the input.|

**Exceptions**

- `ValueError`: The input `dtype` is not supported, `num_frames < 1`, `k_target` is out of range, or in video mode `N != num_frames * tokens_per_frame`.

**Example**

```python
import torch
from mm.core.scc import scc_compress_to_target

# Single-frame image: compress 1024 tokens to 256
features = torch.randn(1024, 128, dtype=torch.float32)
compressed = scc_compress_to_target(features, k_target=256)
assert compressed.shape == (256, 128)

# Video: compress 8 frames × 256 tokens per frame to 128 tokens
video_features = torch.randn(8 * 256, 128, dtype=torch.float32)
compressed = scc_compress_to_target(
    video_features,
    k_target=128,
    num_frames=8,
    tokens_per_frame=256,
)
assert compressed.shape == (128, 128)
```

## `mm.core.scc.set_uniform_true`

**Description**

Generate a bool tensor of length `n` that contains exactly `k` evenly distributed `True` positions. Use it to manually construct a sparse sampling mask when you do not call `scc_compress_to_target`.

**Function Prototype**

```python
def set_uniform_true(n: int, k: int) -> torch.Tensor
```

**Parameters**

|Parameter|Data Type|Optional/Required|Description|
|--|--|--|--|
|n|int|Required|Length of the output tensor. It must be > 0.|
|k|int|Required|Number of `True` values. It must satisfy `0 < k <= n`.|

**Returns**

|Data Type|Description|
|--|--|
|torch.Tensor|Tensor with the shape (n,) and `dtype=bool` that contains `k` `True` values. When multiple sampling points are rounded to the same index, that position is recorded only once.|

**Exceptions**

- `ValueError`: `n <= 0`, or `k` is not in (0, `n`].

**Example**

```python
import torch
from mm.core.scc import set_uniform_true

mask = set_uniform_true(10, 3)
# tensor([ True, False, False, False,  True, False, False, False, False,  True])
assert mask.sum().item() == 3
```
