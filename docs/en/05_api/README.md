# Python API Reference

> [!NOTE]
>
> - The classes and APIs identified in the documentation are public and available for users to call. APIs of other classes are for internal use only and are not recommended for direct use. If necessary, you can inspect the source code.
> - When you import Multimodal SDK, it explicitly sets the `HF_DATASETS_OFFLINE` and `HF_HUB_OFFLINE` environment variables to `1`, enabling Hugging Face offline mode and preventing data from being retrieved over the network.
> - This document applies to the latest release of Multimodal SDK. Python 3.10 or 3.11 and Ubuntu 22.04 or openEuler 24.03 are recommended.

**Source Code Lookup**

You can use the following sample code to locate the installation directory and then access the source files.

```python
import mm
print(mm.__file__)
```

The printed output is the path to the source file.

## Contents

- [Data Types](#data-types)
  - [DataType](#datatype)
  - [TensorFormat](#tensorformat)
  - [ImageFormat](#imageformat)
  - [LogLevel](#loglevel)
  - [DeviceMode](#devicemode)
  - [Interpolation](#interpolation)
- [Function Reference](./function_reference.md)
  - [mm.Tensor](./function_reference.md#mmtensor)
  - [mm.Image](./function_reference.md#mmimage)
  - [Log Registration](./function_reference.md#log-registration)
  - [mm.video_decode](./function_reference.md#mmvideo_decode)
  - [mm.normalize](./function_reference.md#mmnormalize)
  - [mm.load_audio](./function_reference.md#mmload_audio)
  - [Key Frame Extraction](./function_reference.md#mmbaseframeselector)
- [Adapter](./adapter.md)
  - [MultimodalQwen2VLImageProcessor](./adapter.md#multimodalqwen2vlimageprocessor)
  - [InternVL2PreProcessor](./adapter.md#internvl2preprocessor)
- [patcher](./patcher.md)
  - [video_patcher](./patcher.md#video_patcher)
  - [qwen2_vl_image_processor_patcher](./patcher.md#qwen2_vl_image_processor_patcher)
  - [image_patcher](./patcher.md#image_patcher)
  - [internvl2_image_processor_patcher](./patcher.md#internvl2_image_processor_patcher)

## Data Types

### `DataType`

Enumeration of data types.

| Property           | Description  |
| ------------------ | ------------ |
| `DataType.INT8`    | int8 type    |
| `DataType.UINT8`   | uint8 type   |
| `DataType.FLOAT32` | float32 type |

### `TensorFormat`

Enumeration of tensor data layout formats.

| Property | Description |
| -------- | ----------- |
| `TensorFormat.ND` | ND layout, indicating a general N-dimensional array. |
| `TensorFormat.NHWC` | NHWC layout, indicating that the tensor layout is batch, height, width, channel. |
| `TensorFormat.NCHW` | NCHW layout, indicating that the tensor layout is batch, channel, height, width. |

### `ImageFormat`

Enumeration of image formats.

| Property | Description |
| -------- | ----------- |
| `ImageFormat.RGB` | RGB format, with an array channel order of `[H, W, 3]` and channels representing R/G/B. |
| `ImageFormat.BGR` | BGR format, with an array channel order of `[H, W, 3]` and channels representing B/G/R. |
| `ImageFormat.RGB_PLANAR` | RGB_PLANAR format, with an array channel order of `[3, H, W]` and channels representing R/G/B. |
| `ImageFormat.BGR_PLANAR` | BGR_PLANAR format, with an array channel order of `[3, H, W]` and channels representing B/G/R. |

### `LogLevel`

Enumeration of log levels.

| Property         | Description       |
| ---------------- | ----------------- |
| `LogLevel.DEBUG` | Debug level       |
| `LogLevel.INFO`  | Info level        |
| `LogLevel.WARN`  | Warning level     |
| `LogLevel.ERROR` | Error level       |
| `LogLevel.FATAL` | Fatal error level |

### `DeviceMode`

Operation mode.

| Property         | Description                             |
| ---------------- | --------------------------------------- |
| `DeviceMode.CPU` | The current operation runs in CPU mode. |

### `Interpolation`

Interpolation algorithm used in `resize` operations.

| Property                | Description                     |
| ----------------------- | ------------------------------- |
| `Interpolation.BICUBIC` | Bicubic interpolation algorithm |
