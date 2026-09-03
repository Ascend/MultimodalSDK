# Python 接口说明

> [!NOTE]
>
> - 对于资料中注明的类、接口为公开的，可供用户调用。对于其他类，接口均为内部使用，不建议直接调用。如有特殊需要可以通过源码查看。
> - 本文档适用于 Multimodal SDK 最新发布版本，建议 Python 版本为 3.12，操作系统为 Ubuntu 22.04 / openEuler 24.03。

**源码查看方式**

可以通过以下示例代码查看源码所在安装地址，并进入安装地址查看源码文件。

```python
import mm
print(mm.__file__)
```

打印的内容即为源码文件所在的地址。

## 目录

- [数据类型](#数据类型)
  - [DataType](#datatype)
  - [TensorFormat](#tensorformat)
  - [ImageFormat](#imageformat)
  - [LogLevel](#loglevel)
  - [DeviceMode](#devicemode)
  - [Interpolation](#interpolation)
- [功能函数参考](./function_reference.md)
  - [mm.Tensor](./function_reference.md#mmtensor)
  - [mm.Image](./function_reference.md#mmimage)
  - [日志注册](./function_reference.md#日志注册)
  - [mm.video_decode](./function_reference.md#mmvideo_decode)
  - [mm.normalize](./function_reference.md#mmnormalize)
  - [mm.load_audio](./function_reference.md#mmload_audio)
  - [关键帧提取](./function_reference.md#mmbaseframeselector)
  - [mm.core.processor.resize_and_normalize](./function_reference.md#mmcoreprocessorresize_and_normalize)
  - [mm.core.scc.scc_should_run](./function_reference.md#mmcoresccscc_should_run)
  - [mm.core.scc.scc_shrink](./function_reference.md#mmcoresccscc_shrink)
  - [mm.core.scc.scc_compress_to_target](./function_reference.md#mmcoresccscc_compress_to_target)
  - [mm.core.scc.set_uniform_true](./function_reference.md#mmcoresccset_uniform_true)
- [Adapter](./adapter.md)
  - [MultimodalQwen2VLImageProcessor](./adapter.md#multimodalqwen2vlimageprocessor)
  - [InternVL2PreProcessor](./adapter.md#internvl2preprocessor)
- [patcher](./patcher.md)
  - [环境变量](./patcher.md#环境变量)
  - [适用模型](./patcher.md#适用模型)
  - [版本与分支对应](./patcher.md#版本与分支对应)
  - [启动 vLLM](./patcher.md#启动-vllm)

## 数据类型

### DataType

数据类型枚举类

| 属性名 | 说明 |
| -- | -- |
| DataType.INT8 | int8 类型。 |
| DataType.UINT8 | uint8 类型。 |
| DataType.FLOAT32 | float32 类型。 |

### TensorFormat

Tensor 数据排布格式枚举类

| 属性名 | 说明 |
| -- | -- |
| TensorFormat.ND | ND 格式，表示通用 N 维数组。 |
| TensorFormat.NHWC | NHWC 格式，表示张量排布为 Batch, Height, Width, Channel。 |
| TensorFormat.NCHW | NCHW 格式，表示张量排布为 Batch, Channel, Height, Width。 |

### ImageFormat

ImageFormat 枚举类

| 属性名 | 说明 |
| -- | -- |
| ImageFormat.RGB | RGB 类型，数组通道顺序为 `[H, W, 3]`，通道含义为 R/G/B。 |
| ImageFormat.BGR | BGR 类型，数组通道顺序为 `[H, W, 3]`，通道含义为 B/G/R。 |
| ImageFormat.RGB_PLANAR | RGB_PLANAR 类型，数组通道顺序为 `[3, H, W]`，通道含义为 R/G/B。 |
| ImageFormat.BGR_PLANAR | BGR_PLANAR 类型，数组通道顺序为 `[3, H, W]`，通道含义为 B/G/R。 |

### LogLevel

日志级别枚举类

| 属性名 | 说明 |
| -- | -- |
| LogLevel.DEBUG | 调试级别。 |
| LogLevel.INFO | 提示级别。 |
| LogLevel.WARN | 警告级别。 |
| LogLevel.ERROR | 错误级别。 |
| LogLevel.FATAL | 致命错误级别。 |

### DeviceMode

操作运行的模式

| 属性名 | 说明 |
| -- | -- |
| DeviceMode.CPU | 当前操作运行在 CPU 模式下。 |

### Interpolation

resize 操作中使用的插值算法

| 属性名 | 说明 |
| -- | -- |
| Interpolation.BICUBIC | 双立方插值算法。 |
