# Python API Reference

> [!NOTE]
>
> - The classes and APIs marked as public in the documentation are available for you to call. The APIs of other classes are for internal use only, and you are advised not to call them directly. If you have special requirements, you can inspect the source code.
> - This document applies to the latest release of Multimodal SDK. You are advised to use Python 3.12 with Ubuntu 22.04 or openEuler 24.03.

**How to View the Source Code**

You can use the following sample code to view the installation path of the source code, and then open that path to view the source files.

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
  - [Keyframe Extraction](./function_reference.md#mmbaseframeselector)
  - [mm.core.processor.resize_and_normalize](./function_reference.md#mmcoreprocessorresize_and_normalize)
  - [mm.core.scc.scc_should_run](./function_reference.md#mmcoresccscc_should_run)
  - [mm.core.scc.scc_shrink](./function_reference.md#mmcoresccscc_shrink)
  - [mm.core.scc.scc_compress_to_target](./function_reference.md#mmcoresccscc_compress_to_target)
  - [mm.core.scc.set_uniform_true](./function_reference.md#mmcoresccset_uniform_true)
- [Adapter](./adapter.md)
  - [MultimodalQwen2VLImageProcessor](./adapter.md#multimodalqwen2vlimageprocessor)
  - [InternVL2PreProcessor](./adapter.md#internvl2preprocessor)
- [patcher](./patcher.md)
  - [Environment Variables](./patcher.md#environment-variables)
  - [Applicable Models](./patcher.md#applicable-models)
  - [Version and Branch Mapping](./patcher.md#version-and-branch-mapping)
  - [Starting vLLM](./patcher.md#starting-vllm)

## Data Types

### DataType

Enumeration of data types.

|Property|Description|
| -- | -- |
|DataType.INT8|The int8 type.|
|DataType.UINT8|The uint8 type.|
|DataType.FLOAT32|The float32 type.|

### TensorFormat

Enumeration of tensor data layout formats.

|Property|Description|
| -- | -- |
|TensorFormat.ND|The ND format, indicating a general N-dimensional array.|
|TensorFormat.NHWC|The NHWC format, indicating that the tensor layout is Batch, Height, Width, Channel.|
|TensorFormat.NCHW|The NCHW format, indicating that the tensor layout is Batch, Channel, Height, Width.|

### ImageFormat

Enumeration of image formats.

|Property|Description|
| -- | -- |
|ImageFormat.RGB|The RGB type. The array channel order is [H, W, 3], and the channels are R, G, and B.|
|ImageFormat.BGR|The BGR type. The array channel order is [H, W, 3], and the channels are B, G, and R.|
|ImageFormat.RGB_PLANAR|The RGB_PLANAR type. The array channel order is [3, H, W], and the channels are R, G, and B.|
|ImageFormat.BGR_PLANAR|The BGR_PLANAR type. The array channel order is [3, H, W], and the channels are B, G, and R.|

### LogLevel

Enumeration of log levels.

|Property|Description|
| -- | -- |
|LogLevel.DEBUG|Debug level.|
|LogLevel.INFO|Info level.|
|LogLevel.WARN|Warning level.|
|LogLevel.ERROR|Error level.|
|LogLevel.FATAL|Fatal error level.|

### DeviceMode

Mode in which an operation runs.

|Property|Description|
| -- | -- |
|DeviceMode.CPU|The current operation runs in CPU mode.|

### Interpolation

Interpolation algorithm used in resize operations.

|Property|Description|
| -- | -- |
|Interpolation.BICUBIC|Bicubic interpolation algorithm.|
