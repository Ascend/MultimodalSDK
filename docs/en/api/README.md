# Python API Reference<a name="ZH-CN_TOPIC_0000002456910817"></a>

>[!NOTE] Note
>
>- The classes and APIs identified in the documentation are public and available for users to call. The APIs of other classes are for internal use only and are not recommended for direct use. If you have special requirements, you can inspect the source code.
>- When you import Multimodal SDK, it explicitly sets the `HF_DATASETS_OFFLINE` and `HF_HUB_OFFLINE` environment variables to 1. Therefore, it enables Hugging Face offline mode and does not connect to the network to retrieve data.

**Source Code Lookup<a name="section12411139493"></a>**

You can use the following sample code to locate the installation directory of the source code and then open that directory to view the source files.

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
- [Patcher](./patcher.md)
    - [video_patcher](./patcher.md#video_patcher)
    - [qwen2_vl_image_processor_patcher](./patcher.md#qwen2_vl_image_processor_patcher)
    - [image_patcher](./patcher.md#image_patcher)
    - [internvl2_image_processor_patcher](./patcher.md#internvl2_image_processor_patcher)

## Data Types<a name="ZH-CN_TOPIC_0000002423192124"></a>

### DataType<a name="ZH-CN_TOPIC_0000002423192160"></a>

Enumeration of data types.

|Property|Description|
|--|--|
|DataType.INT8|int8 type.|
|DataType.UINT8|uint8 type.|
|DataType.FLOAT32|float32 type.|

### TensorFormat<a name="ZH-CN_TOPIC_0000002423352008"></a>

Enumeration of tensor data layout formats.

|Property|Description|
|--|--|
|TensorFormat.ND|ND format, indicating a general N-dimensional array.|
|TensorFormat.NHWC|NHWC format, indicating that the tensor layout is batch, height, width, channel.|
|TensorFormat.NCHW|NCHW format, indicating that the tensor layout is batch, channel, height, width.|

### ImageFormat<a name="ZH-CN_TOPIC_0000002456790953"></a>

Enumeration of image format values.

|Property|Description|
|--|--|
|ImageFormat.RGB|RGB type.|
|ImageFormat.BGR|BGR type.|
|ImageFormat.RGB_PLANAR|RGB_PLANAR type.|
|ImageFormat.BGR_PLANAR|BGR_PLANAR type.|

### LogLevel<a name="ZH-CN_TOPIC_0000002423192156"></a>

Enumeration of log levels.

|Property|Description|
|--|--|
|LogLevel.DEBUG|Debug level.|
|LogLevel.INFO|Info level.|
|LogLevel.WARN|Warning level.|
|LogLevel.ERROR|Error level.|
|LogLevel.FATAL|Fatal error level.|

### DeviceMode<a name="ZH-CN_TOPIC_0000002456910801"></a>

Operation mode.

|Property|Description|
|--|--|
|DeviceMode.CPU|The current operation runs in CPU mode.|

### Interpolation<a name="ZH-CN_TOPIC_0000002423192132"></a>

Interpolation algorithm used in resize operations.

|Property|Description|
|--|--|
|Interpolation.BICUBIC|Bicubic interpolation algorithm.|
