# 前置知识

本文档介绍使用 Multimodal SDK 所需的背景知识、专业名词解释及必备基础。

## 专业名词说明

### SDK 与框架

| 名词 | 说明 |
|------|------|
| Multimodal SDK | 多模态软件开发工具包，通过提供一系列高性能的昇腾设备亲和性接口，加速大模型推理预处理流程 |
| CANN | Compute Architecture for Neural Networks，昇腾神经网络计算架构，是华为昇腾处理器的 AI 计算引擎 |
| vLLM | 大模型推理框架，Multimodal SDK 提供 vLLM 预处理插件以加速推理 |
| Qwen2VL | 通义千问视觉语言模型，Multimodal SDK 提供其图像/视频预处理加速能力 |
| InternVL2 | 视觉语言模型，Multimodal SDK 提供其图像/视频预处理加速能力 |
| CLIP | Contrastive Language-Image Pre-Training，对比语言-图像预训练模型，用于关键帧选择中的文本-图像匹配 |

### 硬件与设备

| 名词 | 说明 |
|------|------|
| NPU | Neural Processing Unit，神经网络处理单元，昇腾处理器的核心计算单元 |
| Atlas 800I A2 | 华为 Atlas 推理服务器型号，Multimodal SDK 支持的硬件平台 |
| davinci | 昇腾 NPU 设备在 Linux 系统中的设备文件名，位于 `/dev/davinci*` |

### 数据类型与格式

| 名词 | 说明 |
|------|------|
| Tensor | 张量，多维数组的数据结构，用于承载任意模态的通用数据 |
| DataType | 数据类型枚举，支持 INT8、UINT8、FLOAT32 |
| TensorFormat | Tensor 数据排布格式，支持 ND（通用 N 维数组）、NHWC（Batch-Height-Width-Channel）、NCHW（Batch-Channel-Height-Width） |
| ImageFormat | 图像格式枚举，支持 RGB、BGR、RGB_PLANAR、BGR_PLANAR |
| DeviceMode | 设备运行模式，当前仅支持 CPU 模式 |
| Interpolation | 插值算法枚举，resize 操作中使用，当前仅支持 BICUBIC（双立方插值） |

### 图像与视频处理

| 名词 | 说明 |
|------|------|
| resize | 图像缩放操作，将图像调整为指定尺寸 |
| crop | 图像裁剪操作，从图像中截取指定区域 |
| decode | 解码操作，将压缩格式（如 jpg、mp4）转换为原始数据 |
| 关键帧 | 视频中具有代表性的帧，用于视频内容理解和分析 |
| 帧 ID | 视频帧的索引编号，从 0 开始 |
| 采样率 | 音频每秒采样的点数，单位为 Hz |

### 依赖库

| 名词 | 说明 |
|------|------|
| FFmpeg | 开源音视频处理库，Multimodal SDK 使用其进行视频解码 |
| libjpeg-turbo | JPEG 图像处理加速库，Multimodal SDK 使用其进行图像解码 |
| Pillow (PIL) | Python 图像处理库，Multimodal SDK 支持与 PIL Image 相互转换 |
| PyTorch | 深度学习框架，Multimodal SDK 支持与 torch.Tensor 相互转换 |
| NumPy | Python 科学计算库，Multimodal SDK 支持与 numpy.ndarray 相互转换 |
| transformers | Hugging Face 的预训练模型库，Multimodal SDK 的依赖之一 |

## 必备知识

### 基本概念理解

#### 多模态数据处理

Multimodal SDK 主要处理以下模态数据：

- **图像**：支持 jpg/jpeg 格式，宽高范围 [10, 8192]
- **视频**：支持 mp4 格式，分辨率范围 480P-4K
- **音频**：支持 wav 格式，采样率范围 [1, 64000]

#### 数据流向

SDK 的典型数据流向：

1. **输入**：文件路径 → 解码 → 原始数据
2. **处理**：resize、crop、normalize 等预处理操作
3. **输出**：Tensor 对象，可转换为 NumPy 数组或 PyTorch 张量
