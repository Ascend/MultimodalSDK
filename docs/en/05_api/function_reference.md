# Function Reference

## `mm.Tensor`

This document applies to the latest released version of the Multimodal SDK. Interface exceptions are typically thrown as `ValueError`, `TypeError`, `RuntimeError`, or `ImportError`. For recommended error code handling, see [Appendix - Error Codes](../06_references/appendix.md#error-codes).

The `Tensor` class is used to hold general-purpose data of any modality and provides operations for creating, managing, and copying general-purpose data.

### Tensor Attributes

| Attribute | Type         | Description                                  | Remarks                            |
| --------- | ------------ | -------------------------------------------- | ---------------------------------- |
| device    | str          | Device where the Tensor is located.          | Default value: `"cpu"`.            |
| dtype     | DataType     | Data type of the Tensor.                     | Default value: `DataType.FLOAT32`. |
| shape     | list         | Shape of the Tensor.                         | Default value: an empty list.      |
| format    | TensorFormat | Data layout of the Tensor.                   | Default value: `TensorFormat.ND`.  |
| nbytes    | int          | Number of bytes occupied by the Tensor data. | Default value: `0`.                |

### `Tensor.set_format`

**Description**

Sets the data layout.

**Prototype**

```python
set_format(tensor_format: TensorFormat)
```

**Parameters**

| Parameter | Type | Required (Yes/No) | Description |
| -- | -- | -- | -- |
| tensor_format | TensorFormat | Yes | Layout to set. `ND`, `NHWC`, and `NCHW` are supported. If `NHWC` or `NCHW` is specified, the Tensor must have a four-dimensional shape. |

**Example**

```python
from mm import Tensor, TensorFormat
tensor = Tensor()
tensor.set_format(TensorFormat.ND)
```

### `Tensor.clone`

**Description**

Creates a deep copy of the `Tensor` instance.

**Prototype**

```python
clone() -> Tensor
```

**Return Values**

| Type   | Description                |
| ------ | -------------------------- |
| Tensor | The new `Tensor` instance. |

**Example**

```python
from mm import Tensor
tensor = Tensor()
tensor_new = tensor.clone()
```

### `Tensor.from_numpy`

**Description**

Creates a `Tensor` object from a NumPy array.

**Prototype**

```python
def from_numpy(nd_array: numpy.ndarray) -> Tensor:
```

**Parameters**

| Parameter | Type | Required (Yes/No) | Description |
|--|--|--|--|
| nd_array | numpy.ndarray | Yes | The input dtype must be `int8`, `uint8`, or `float32`. The input `numpy.ndarray` must be in row-major order and have contiguous memory. |

**Return Values**

| Type   | Description                                             |
| ------ | ------------------------------------------------------- |
| Tensor | The `Tensor` instance created from the `numpy.ndarray`. |

> [!NOTE]
> The created `Tensor` object shares data with the `numpy.ndarray`. The data lifetime is managed by the `numpy.ndarray` object.

**Example**

```python
from mm import Tensor
import numpy as np

arr = np.zeros((1024, 768, 3), dtype=np.uint8)
tensor = Tensor.from_numpy(arr)
```

### `Tensor.numpy`

**Description**

Converts the `Tensor` object to a NumPy array.

**Prototype**

```python
numpy() -> np.ndarray
```

**Return Values**

| Type          | Description                |
| ------------- | -------------------------- |
| numpy.ndarray | The resulting NumPy array. |

> [!NOTE]
>
>- The converted `numpy.ndarray` shares data with the `Tensor` object. The data lifetime is managed by the `Tensor` object.
>- The `Tensor` must be located on the CPU.

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

Creates a `Tensor` object from a `torch.Tensor`.

**Prototype**

```python
def from_torch(torch_tensor: torch.Tensor) -> Tensor:
```

**Parameters**

| Parameter | Type | Required (Yes/No) | Description |
| -- | -- | -- | -- |
| torch_tensor | torch.Tensor | Yes | The input dtype must be `int8`, `uint8`, or `float32`. The input `torch.Tensor` must be in row-major order and have contiguous memory. The input `torch.Tensor` must be located on the CPU. |

**Return Values**

| Type   | Description                                          |
| ------ | ---------------------------------------------------- |
| Tensor | The `Tensor` object created from the `torch.Tensor`. |

> [!NOTE]
> The created `Tensor` object shares data with the `torch.Tensor`. The data lifetime is managed by the `torch.Tensor` object.

**Example**

```python
from mm import Tensor
import torch

tensor = torch.zeros((1024, 768, 3), dtype=torch.uint8)
mm_tensor = Tensor.from_torch(tensor)
```

### `Tensor.torch`

**Description**

Converts the `Tensor` object to a `torch.Tensor`.

**Prototype**

```python
torch() -> torch.Tensor
```

**Return Values**

| Type         | Description                   |
| ------------ | ----------------------------- |
| torch.Tensor | The resulting `torch.Tensor`. |

> [!NOTE]
>
>- The converted `torch.Tensor` shares data with the `Tensor` object. The data lifetime is managed by the `Tensor` object.
>- The `Tensor` must be located on the CPU.

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

Normalizes the current `Tensor` object using the mean and standard deviation. Given the mean values `mean[1], ..., mean[n]` and standard deviation values `std[1], ..., std[n]` for `n` channels, this operation normalizes each channel of the current tensor object as follows: `output[channel] = (src[channel] - mean[channel]) / std[channel]`, where `src` is the current `Tensor` object.

**Prototype**

```python
def normalize(mean: list[float], std: list[float], device_mode: DeviceMode = DeviceMode.CPU) -> Tensor
```

**Parameters**

| Parameter | Type | Required (Yes/No) | Description |
| -- | -- | -- | -- |
| mean | list[float] | Yes | Array of mean values. The array must contain three values, each in the range [0, 1]. |
| std | list[float] | Yes | Array of standard deviation values. The array must contain three values, each in the range (0, 3.4028235e38]. |
| device_mode | DeviceMode | No | Execution mode. Currently, only `CPU` is supported. |

**Return Values**

| Type   | Description                    |
| ------ | ------------------------------ |
| Tensor | The resulting `Tensor` object. |

> [!NOTE]
>
>- The `format` of the `Tensor` object must be `NCHW` or `NHWC`. `N` must be 1 and `C` must be 3.
>- The data type of the `Tensor` object must be `Float32`.
>- The `Tensor` must be located on the CPU.

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

The `Image` class is used to hold image data and provides operations for creating, managing, and copying image data.

### Image Attributes

| Attribute | Type        | Description                                                 |
| --------- | ----------- | ----------------------------------------------------------- |
| device    | str         | Device where the Image is located. Only `cpu` is supported. |
| dtype     | DataType    | Data type of the Image. Only `DataType.UINT8` is supported. |
| size      | list        | Size of the Image.                                          |
| format    | ImageFormat | Image format.                                               |
| nbytes    | int         | Number of bytes occupied by the Image data.                 |
| height    | int         | Height of the Image.                                        |
| width     | int         | Width of the Image.                                         |

### `Image.open`

**Description**

Creates an `Image` from the specified path.

**Prototype**

```python
open(path: str | bytes, device: str | bytes = b'cpu')
```

**Parameters**

| Parameter | Type | Required (Yes/No) | Description |
| --------- | ---- | ----------------- | ----------- |
| path | str \| bytes | Yes | The input path must be valid and no longer than 4096 characters. The path must not contain symbolic links. The file size must not exceed 1 GB, and the file permissions must not be more permissive than `640`, with the permissions for User/Group/Others not exceeding 6/4/0, respectively. The file extension must be `jpg` or `jpeg`. The input image must be in JPG or JPEG format, and both its width and height must be in the range [10, 8192]. Currently, `Image.open` creates only RGB images. |
| device | str \| bytes | No | Device type. Currently, only `cpu` is supported. Default value: `cpu`. |

**Return Values**

| Type  | Description                                               |
| ----- | --------------------------------------------------------- |
| Image | The new `Image` instance created from the specified path. |

**Example**

```python
from mm import Image
img = Image.open("/home/test.jpg", "cpu")
```

### `Image.from_numpy`

**Description**

Creates an `Image` instance from a NumPy array.

**Prototype**

```python
from_numpy(
    nd_array: numpy.ndarray,
    image_format: ImageFormat,
    device: str | bytes = b"cpu"
) -> Image:
```

**Parameters**

| Parameter | Type | Required (Yes/No) | Description |
| --------- | ---- | ----------------- | ----------- |
| nd_array | numpy.ndarray | Yes | The input NumPy array must have a dtype of `uint8`, be three-dimensional, and not be empty. For RGB and BGR image formats, the shape of the three-dimensional array must be [H, W, 3]. For BGR_PLANAR and RGB_PLANAR image formats, the shape must be [3, H, W]. The values of all elements in the NumPy array must be in the range [0, 255]. The array must be in row-major order and have contiguous memory. Both the width and height must be in the range [10, 8192]. |
| image_format | ImageFormat | Yes | Image format. `RGB`, `BGR`, `BGR_PLANAR`, and `RGB_PLANAR` are supported. The specified format must match the dimensions of the NumPy array. |
| device | str \| bytes | No | Device type. Currently, only `cpu` is supported. Default value: `cpu`. |

**Return Values**

| Type  | Description                                        |
| ----- | -------------------------------------------------- |
| Image | The `Image` instance created from the NumPy array. |

> [!NOTE]
> The created `Image` object shares data with the NumPy array. The data lifetime is managed by the NumPy array.

**Example**

```python
from mm import Image, ImageFormat
import numpy as np

arr = np.zeros((1024, 768, 3), dtype=np.uint8)
img = Image.from_numpy(arr, ImageFormat.RGB, "cpu")
```

### `Image.numpy`

**Description**

Converts the `Image` instance to a NumPy array.

**Prototype**

```python
numpy() -> numpy.ndarray
```

**Return Values**

| Type          | Description                |
| ------------- | -------------------------- |
| numpy.ndarray | The resulting NumPy array. |

> [!NOTE]
>
>- The shape of the output `ndarray` depends on the image format.
>- For an `Image` with the `RGB` or `BGR` format, the shape is [H, W, 3]. For an `Image` with the `RGB_PLANAR` or `BGR_PLANAR` format, the shape is [3, H, W].
>- The `Image` must be located on the CPU.

**Example**

```python
from mm import Image
import numpy as np

img = Image.open("/home/test.jpg", "cpu")
arr = img.numpy()
```

### `Image.from_torch`

**Description**

Creates an `Image` instance from a `torch.Tensor`.

**Prototype**

```python
from_torch(
    torch_tensor: torch.Tensor,
    image_format: ImageFormat,
    device: str | bytes = b"cpu"
) -> Image:
```

**Parameters**

| Parameter | Type | Required (Yes/No) | Description |
| --------- | ---- | ----------------- | ----------- |
| torch_tensor | torch.Tensor | Yes | The input PyTorch tensor must have a dtype of `uint8`, be three-dimensional, and have a shape of [H, W, 3] for the RGB/BGR formats or [3, H, W] for the PLANAR formats. Its values must be in the range [0, 255]. The tensor must be in row-major order, have contiguous memory, and be located on the CPU. Both the width and height must be in the range [10, 8192]. |
| image_format | ImageFormat | Yes | Image format. `RGB`, `BGR`, `BGR_PLANAR`, and `RGB_PLANAR` are supported. The specified format must match the dimensions of the `torch.Tensor`. |
| device | str \| bytes | No | Device type. Currently, only `cpu` is supported. Default value: `cpu`. |

> [!NOTE]
> The created `Image` object shares data with the `torch.Tensor`. The data lifetime is managed by the `torch.Tensor` object.

**Return Values**

| Type  | Description                                           |
| ----- | ----------------------------------------------------- |
| Image | The `Image` instance created from the `torch.Tensor`. |

**Example**

```python
from mm import Image, ImageFormat
import torch

tensor = torch.zeros((1024, 768, 3), dtype=torch.uint8)
img = Image.from_torch(tensor, ImageFormat.RGB, "cpu")
```

### `Image.torch`

**Description**

Converts the `Image` instance to a `torch.Tensor`.

**Prototype**

```python
torch() -> torch.Tensor
```

**Return Values**

| Type         | Description                   |
| ------------ | ----------------------------- |
| torch.Tensor | The resulting `torch.Tensor`. |

> [!NOTE]
>
>- The shape of the output `Tensor` depends on the image format.
>- For an `Image` with the `RGB` or `BGR` format, the shape is [H, W, 3]. For an `Image` with the `BGR_PLANAR` or `RGB_PLANAR` format, the shape is [3, H, W].
>- The `Image` must be located on the CPU.

**Example**

```python
from mm import Image
import torch

img = Image.open("/home/test.jpg", "cpu")
tensor = img.torch()
```

### `Image.from_pillow`

**Description**

Creates an `Image` object from a PIL image object.

**Prototype**

```python
from_pillow(pillow_image: PIL.Image.Image) -> Image
```

**Parameters**

| Parameter | Type | Required (Yes/No) | Description |
| --------- | ---- | ----------------- | ----------- |
| pillow_image | PIL.Image.Image | Yes | The input Pillow Image object must have a `mode` of `"L"` (grayscale), `"RGB"` (RGB image), or `"RGBA"` (RGB image with transparency). |

**Return Values**

| Type  | Description                 |
| ----- | --------------------------- |
| Image | The created `Image` object. |

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

Converts the `Image` object to a PIL image object.

**Prototype**

```python
pillow() -> PIL.Image.Image
```

**Return Values**

| Type            | Description                          |
| --------------- | ------------------------------------ |
| PIL.Image.Image | The resulting Pillow image instance. |

> [!NOTE]
>
>- The dtype of the output `PIL.Image.Image` must be consistent with that of the `Image` instance. Currently, only `uint8` is supported.
>- The mode of the output PIL Image instance is `"RGB"`.
>- The `Image` must be located on the CPU.

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

Creates a deep copy of the `Image` instance.

**Prototype**

```python
clone() -> Image
```

**Return Values**

| Type  | Description                                                             |
| ----- | ----------------------------------------------------------------------- |
| Image | The new `Image` instance created by deep copying the original instance. |

**Example**

```python
from mm import Image
img = Image.open("/home/test.jpg", "cpu")
img_copy = img.clone()
```

### `Image.resize`

**Description**

Resizes the `Image` instance.

**Prototype**

```python
resize(size: Tuple[int, int], interpolation: Interpolation, device_mode: DeviceMode = DeviceMode.CPU) -> "Image"
```

**Parameters**

| Parameter | Type | Required (Yes/No) | Description |
| --------- | ---- | ----------------- | ----------- |
| size | Tuple[int, int] | Yes | Width and height of the resized image. `size` must have two dimensions. The first dimension is `width`, and the second dimension is `height`. Both the width and height must be in the range [10, 8192]. |
| interpolation | Interpolation | Yes | Interpolation algorithm used for resizing. Currently, only `BICUBIC` is supported. |
| device_mode | DeviceMode | No | Execution mode for resizing. Currently, only `CPU` is supported. |

**Return Values**

| Type  | Description                                       |
| ----- | ------------------------------------------------- |
| Image | The new `Image` instance obtained after resizing. |

> [!NOTE]
> Currently, only images with the `RGB` or `BGR` format, a data type of `UINT8`, and element values in the range [0, 255] are supported.

**Example**

```python
from mm import Image, DeviceMode, Interpolation
img = Image.open("/home/test.jpg", "cpu")
img_resize = img.resize((10, 10), Interpolation.BICUBIC, DeviceMode.CPU)
```

### `Image.crop`

**Description**

Crops the `Image` instance.

**Prototype**

```python
crop(top: int, left: int, height: int, width: int, device_mode: DeviceMode = DeviceMode.CPU) -> "Image"
```

**Parameters**

| Parameter | Type | Required (Yes/No) | Description |
| --------- | ---- | ----------------- | ----------- |
| top | int | Yes | Top starting coordinate of the crop region. The value must be greater than or equal to 0. `top + height` must not exceed the height of the original image. |
| left | int | Yes | Left starting coordinate of the crop region. The value must be greater than or equal to 0. `left + width` must not exceed the width of the original image. |
| height | int | Yes | Height of the crop region. The value must be in the range [10, original image height - `top`]. |
| width | int | Yes | Width of the crop region. The value must be in the range [10, original image width - `left`]. |
| device_mode | DeviceMode | No | Execution mode for cropping. Currently, only `CPU` is supported. |

**Return Values**

| Type  | Description                                       |
| ----- | ------------------------------------------------- |
| Image | The new `Image` instance obtained after cropping. |

> [!NOTE]
> Currently, only images with the `RGB` or `BGR` format, a data type of `UINT8`, and element values in the range [0, 255] are supported.

**Example**

```python
from mm import Image, DeviceMode
img = Image.open("/home/test.jpg", "cpu")
img_crop = img.crop(10, 10, 10, 10, DeviceMode.CPU)
```

### `Image.to_tensor`

**Description**

Converts the `Image` instance from the [0, 255] range to the [0.0, 1.0] range and converts the data layout from HWC to CHW.

> [!NOTE]
> Currently, only images with the `RGB` or `BGR` format, a data type of `UINT8`, and element values in the range [0, 255] are supported. The data type of the output `Tensor` instance is `DataType.FLOAT32` by default.

**Prototype**

```python
def to_tensor(target_format: TensorFormat = TensorFormat.NCHW, device_mode: DeviceMode = DeviceMode.CPU) -> "Tensor":
```

**Parameters**

| Parameter | Type | Required (Yes/No) | Description |
| --------- | ---- | ----------------- | ----------- |
| target_format | TensorFormat | No | Layout of the resulting `Tensor` instance. `NHWC` and `NCHW` are supported. Default value: `NCHW`. |
| device_mode | DeviceMode | No | Execution mode. Currently, only `CPU` is supported. |

**Return Values**

| Type   | Description                      |
| ------ | -------------------------------- |
| Tensor | The resulting `Tensor` instance. |

**Example**

```python
from mm import Image, TensorFormat, DeviceMode

img = Image.open("/home/test.jpg", "cpu")
dst_tensor = img.to_tensor(TensorFormat.NCHW, DeviceMode.CPU)
```

## Log Registration

Registers the log level and log callback function.

### `register_log_conf`

Registers logging configuration.

**Prototype**

```python
register_log_conf(min_level: LogLevel, callback: Callable[[LogLevel, str, str, int, str], None])
```

**Parameters**

| Parameter | Type | Required (Yes/No) | Description |
| --------- | ---- | ----------------- | ----------- |
| min_level | LogLevel | Yes | Minimum log level. Only logs at or above this level are output. `None` is not allowed. |
| callback | Callable[[LogLevel, str, str, int, str], None] | Yes | Log callback function. If `None` is passed, the internal default log output function is used. |

> [!NOTE]
> Raising an exception in the log callback function causes an exception to be thrown on the C++ side, which may cause the program to coredump. It is recommended that you catch and handle exceptions in the callback.

**Example**

```python
from mm import register_log_conf, LogLevel
def custom_log_handler(level: LogLevel, file: str, func: str, line: int, msg: str) -> None:
    print(f"[custom][{level}] {file}:{line} ({func}) - {msg}")
register_log_conf(LogLevel.ERROR, custom_log_handler)
```

## `mm.video_decode`

**Description**

Decodes the specified video file and returns a list of `Image` objects.

**Prototype**

```python
def video_decode(video_path: str | bytes, device: str | bytes, frame_indices: set = None, sample_num: int = -1) -> list:
```

**Parameters**

| Parameter | Type | Required (Yes/No) | Description |
| --------- | ---- | ----------------- | ----------- |
| video_path | str \| bytes | Yes | Path of the video to decode. Currently, only MP4 files are supported. Resolutions from 480P to 4K are supported. |
| device | str \| bytes | Yes | Device used for decoding. Currently, only `cpu` is supported. |
| frame_indices | set | No | IDs of the video frames to decode. |
| sample_num | int | No | Total number of frames to obtain after decoding. |

**Return Values**

| Type        | Description                                                                                      |
| ----------- | ------------------------------------------------------------------------------------------------ |
| list[Image] | A list of decoded `Image` objects. The images are in RGB format and have a data type of `uint8`. |

> [!NOTE]
>
>- The IDs of the video frames to decode must be in the range [0, `total number of video frames - 1`). Default value: an empty set. This parameter takes precedence over `sample_num`. The specified frame IDs are the target frame IDs to decode.
>- `sample_num` must be in the range (0, total number of video frames]. Default value: `-1`. The frame IDs are generated by sampling frames at equal intervals based on the total number of video frames.
>- Decoding fails if neither `frame_indices` nor `sample_num` is specified.
>- Supported video frame dimensions range from [480, 480] to [4096, 4096].
>- The input path must be valid and no longer than 4096 characters. The path must not contain symbolic links, and the file permissions must not be more permissive than `640`, with the permissions for User/Group/Others not exceeding `6`/`4`/`0`, respectively.

**Example**

```python
from mm import video_decode

file_path = "test.mp4"
mm_images = video_decode(file_path, "cpu", set(), 32)
```

<a id="mmnormalize"></a>

## `mm.normalize`

**Description**

Normalizes a `Tensor` object using the mean and standard deviation. Given the mean values `mean[1], ..., mean[n]` and standard deviation values `std[1], ..., std[n]` for `n` channels, this operation normalizes each channel of the input `Tensor` object as follows: `output[channel] = (src[channel] - mean[channel]) / std[channel]`.

**Prototype**

```python
def normalize(src: Tensor, mean: list[float], std: list[float], device_mode: DeviceMode = DeviceMode.CPU) -> Tensor
```

**Parameters**

| Parameter | Type | Required (Yes/No) | Description |
| --------- | ---- | ----------------- | ----------- |
| src | Tensor | Yes | Input `Tensor` instance. |
| mean | list[float] | Yes | Array of mean values. The array must contain three values, each in the range [0, 1]. |
| std | list[float] | Yes | Array of standard deviation values. The array must contain three values, each in the range (0, 3.4028235e38]. |
| device_mode | DeviceMode | No | Execution mode. Currently, only `CPU` is supported. |

**Return Values**

| Type   | Description                    |
| ------ | ------------------------------ |
| Tensor | The resulting `Tensor` object. |

> [!NOTE]
>
>- The `format` of the input `Tensor` object must be `NCHW` or `NHWC`. `N` must be 1 and `C` must be 3.
>- The data type of the input `Tensor` object must be `Float32`.
>- The `Tensor` must be located on the CPU.

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

Loads audio using the audio loading interface. Both individual audio files and batches of audio files can be loaded in parallel, and resampling can be enabled or disabled.

**Prototype**

```python
def load_audio(audio_inputs: Union[str, List[str]], sr: Optional[int] = None)
-> Union[Tuple[Tensor, int], List[Tuple[Tensor, int]]]
```

**Parameters**

| Parameter | Type | Required (Yes/No) | Description |
| --------- | ---- | ----------------- | ----------- |
| audio_inputs | Union[str, List[str]] | Yes | Path to a single audio file, a directory containing multiple audio files, or a list of audio file paths. |
| sr | Optional[int] | No | If a resampling rate is specified, the audio is resampled to that rate. Otherwise, no resampling is performed. |

**Return Values**

| Type | Description |
| ---- | ----------- |
| Union[Tuple[Tensor, int], List[Tuple[Tensor, int]]] | For a single audio file, returns the audio Tensor data and sampling rate. For multiple audio files, returns a list of audio Tensor data and sampling rate pairs. |

> [!NOTE]
>
>- Only WAV audio files are supported.
>- The number of audio files that can be loaded must be in the range [1, 128].
>- The specified sampling rate must be a positive integer in the range [1, 64000].
>- Multi-channel audio is automatically converted to single-channel (mono).

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

An abstract base class for selecting keyframes based on text-image matching. It encapsulates common capabilities such as model initialization, feature extraction, boundary localization, input validation, and similarity calculation. This class cannot be instantiated directly. Use the `KRangFrameSelector` or `KFrameSelector` subclass instead.

### `BaseFrameSelector` Attributes

| Attribute | Type | Description |
| --------- | ---- | ----------- |
| model_path | str | Path to the CLIP model weights. |
| device_id | int | NPU device index. |
| model_type | str | Model type. The value can be `"clip"` (English) or `"cn_clip"` (Chinese). |
| batch_size | int | Batch size for image feature extraction. Default value: 64. |
| similar_threshold | float | Text-image similarity decay threshold. Frames with similarity lower than the maximum similarity minus this value are filtered out. Default value: 0.025. |
| image_similar_threshold | float | Image similarity gradient threshold used for scene boundary detection. A scene switch is identified when the cosine similarity gradient between adjacent frames exceeds this value. Default value: 0.015. |
| image_size | tuple | Image size for input resizing. The size must match the model training size. Default value: `(672, 672)`. |

### `BaseFrameSelector.__init__`

**Description**

Initializes the keyframe selector, loads the specified model, and validates the parameters.

**Prototype**

```python
__init__(model_path: str, device_id: int, model_type: str = "clip", similar_threshold: float = 0.025, image_similar_threshold: float = 0.015, image_size: tuple = (672, 672))
```

**Parameters**

| Parameter | Type | Required (Yes/No) | Description |
| --------- | ---- | ----------------- | ----------- |
| model_path | str | Yes | Path to the CLIP model weights. The value must be a non-empty string and must point to a valid directory. |
| device_id | int | Yes | NPU device index. The value must be an integer. |
| model_type | str | No | Model type. The value can be `"clip"` (English) or `"cn_clip"` (Chinese). Default value: `"clip"`. |
| similar_threshold | float | No | Text-image similarity decay threshold. The value must be in the range [0, 1]. Default value: 0.025. |
| image_similar_threshold | float | No | Image similarity gradient threshold used for scene boundary detection. The value must be in the range [0, 1]. Default value: 0.015. |
| image_size | tuple | No | Input image size in the form (width, height). Both the width and height must be in the range [10, 8192]. Default value: (672, 672). |

> [!NOTE]
>
>- `BaseFrameSelector` is an abstract class and cannot be instantiated directly. Create an instance through the `KRangFrameSelector` or `KFrameSelector` subclass.
>- The directory specified by `model_path` must exist. Its permissions must not be more permissive than `750`, and its owner must be the current user.

### `BaseFrameSelector.select_keyframes`

**Description**

Selects keyframes related to the query text from a sequence of video frames. This is an abstract method, and the specific selection strategy is implemented by subclasses.

**Prototype**

```python
select_keyframes(query: str, frames: List[np.ndarray], sample_num: int, do_resample: bool) -> Tuple[List[int], List[np.ndarray]]
```

**Parameters**

| Parameter | Type | Required (Yes/No) | Description |
| --------- | ---- | ----------------- | ----------- |
| query | str | Yes | Query text describing the target visual content. The value must be a non-empty string. |
| frames | List[np.ndarray] | Yes | List of video frames. |
| sample_num | int | Yes | Maximum number of keyframes. The value must be a positive integer. |
| do_resample | bool | Yes | Whether to perform adaptive resampling within the interval. |

**Return Values**

| Type | Description |
| ---- | ----------- |
| Tuple[List[int], List[np.ndarray]] | A tuple containing a list of keyframe indices and a list of keyframe images. |

## `mm.KRangFrameSelector`

A keyframe selector based on interval merging that inherits from `BaseFrameSelector`. It identifies continuous scene intervals related to the query text and adaptively samples keyframes within the intervals. It is suitable for tasks that require temporal context.

### `KRangFrameSelector.select_keyframes`

**Description**

Runs the interval-based keyframe selection workflow. After extracting features and calculating similarities, it greedily selects candidate frames and expands them into scene intervals, merges adjacent semantically similar intervals, and finally performs adaptive resampling within the intervals.

**Prototype**

```python
select_keyframes(query: str, frames: List[np.ndarray], sample_num: int, do_resample: bool) -> Tuple[List[int], List[np.ndarray]]
```

**Parameters**

| Parameter | Type | Required (Yes/No) | Description |
| --------- | ---- | ----------------- | ----------- |
| query | str | Yes | Query text describing the target visual content. The value must be a non-empty string. |
| frames | List[np.ndarray] | Yes | List of video frames. |
| sample_num | int | Yes | Maximum number of keyframes. The value must be a positive integer. |
| do_resample | bool | Yes | Whether to perform adaptive resampling within the merged intervals. When set to `True`, keyframes are resampled within the intervals using a combination of top-k selection based on similarity and uniform filling. When set to `False`, only the endpoints of the merged intervals are returned. |

**Return Values**

| Type | Description |
| ---- | ----------- |
| Tuple[List[int], List[np.ndarray]] | A tuple containing a list of keyframe indices, which are deduplicated and sorted, and a list of keyframe images. |

**Example**

```python
from mm import KRangFrameSelector
import numpy as np

selector = KRangFrameSelector(model_path="/path/to/clip_model", device_id=0, model_type="clip")
frames = [np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8) for _ in range(100)]
indices, key_frames = selector.select_keyframes(query="a cat sitting on a sofa", frames=frames, sample_num=8, do_resample=True)
```

## `mm.KFrameSelector`

A discrete keyframe selector that inherits from `BaseFrameSelector`. It selects discrete keyframes that are relevant to the query text and visually diverse. It is suitable for tasks that require visual diversity.

### `KFrameSelector.select_keyframes`

**Description**

Runs the discrete keyframe selection workflow. After extracting features and calculating similarities, it greedily selects candidate frames and removes duplicates based on feature distance to ensure visual diversity.

**Prototype**

```python
select_keyframes(query: str, frames: List[np.ndarray], sample_num: int, do_resample: bool = False) -> Tuple[List[int], List[np.ndarray]]
```

**Parameters**

| Parameter | Type | Required (Yes/No) | Description |
| --------- | ---- | ----------------- | ----------- |
| query | str | Yes | Query text describing the target visual content. The value must be a non-empty string. |
| frames | List[np.ndarray] | Yes | List of video frames. |
| sample_num | int | Yes | Maximum number of keyframes. The value must be a positive integer. |
| do_resample | bool | No | This parameter is not used by `KFrameSelector`. Default value: `False`. |

**Return Values**

| Type | Description |
| ---- | ----------- |
| Tuple[List[int], List[np.ndarray]] | A tuple containing a list of keyframe indices, which are deduplicated and sorted, and a list of keyframe images. |

**Example**

```python
from mm import KFrameSelector
import numpy as np

selector = KFrameSelector(model_path="/path/to/clip_model", device_id=0, model_type="clip")
frames = [np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8) for _ in range(100)]
indices, key_frames = selector.select_keyframes(query="a dog running in the park", frames=frames, sample_num=8)
```
