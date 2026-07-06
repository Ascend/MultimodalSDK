# Prerequisites

This document introduces the background knowledge, terminology explanations, and essential foundations required for using Multimodal SDK.

## Terminology

### SDK and Framework

| Term | Description |
|------|------|
| Multimodal SDK | Multimodal Software Development Kit, accelerates LLM inference preprocessing by providing a set of high-performance interfaces optimized for Ascend devices |
| CANN | Compute Architecture for Neural Networks, the AI computing engine for Huawei Ascend processors |
| vLLM | Large language model inference framework, Multimodal SDK provides vLLM preprocessing plugins for acceleration |
| Qwen2VL | Tongyi Qianwen Vision-Language Model, Multimodal SDK provides image/video preprocessing acceleration for it |
| InternVL2 | Vision-Language Model, Multimodal SDK provides image/video preprocessing acceleration for it |
| CLIP | Contrastive Language-Image Pre-Training, used for text-image matching in key frame selection |

### Hardware and Devices

| Term | Description |
|------|------|
| NPU | Neural Processing Unit, the core computing unit of Ascend processors |
| Atlas 800I A2 | Huawei Atlas inference server model, a hardware platform supported by Multimodal SDK |
| Ascend 310/910 | Ascend chip series models |
| davinci | Device file name for Ascend NPU in Linux systems, located at `/dev/davinci*` |

### Data Types and Formats

| Term | Description |
|------|------|
| Tensor | Multi-dimensional array data structure, used to carry universal data of any modality |
| DataType | Data type enumeration, supports INT8, UINT8, FLOAT32 |
| TensorFormat | Tensor data layout format, supports ND (general N-dimensional array), NHWC (Batch-Height-Width-Channel), NCHW (Batch-Channel-Height-Width) |
| ImageFormat | Image format enumeration, supports RGB, BGR, RGB_PLANAR, BGR_PLANAR |
| DeviceMode | Device operation mode, currently only supports CPU mode |
| Interpolation | Interpolation algorithm enumeration, used in resize operations, currently only supports BICUBIC (bicubic interpolation) |

### Image and Video Processing

| Term | Description |
|------|------|
| resize | Image scaling operation, adjusts image to specified dimensions |
| crop | Image cropping operation, extracts a specified region from an image |
| decode | Decoding operation, converts compressed formats (such as jpg, mp4) to raw data |
| Key frame | Representative frames in a video, used for video content understanding and analysis |
| Frame ID | Index number of a video frame, starting from 0 |
| Sample rate | Number of audio samples per second, measured in Hz |

### Dependencies

| Term | Description |
|------|------|
| FFmpeg | Open-source audio/video processing library, Multimodal SDK uses it for video decoding |
| libjpeg-turbo | JPEG image processing acceleration library, Multimodal SDK uses it for image decoding |
| Pillow (PIL) | Python image processing library, Multimodal SDK supports mutual conversion with PIL Image |
| PyTorch | Deep learning framework, Multimodal SDK supports mutual conversion with torch.Tensor |
| NumPy | Python scientific computing library, Multimodal SDK supports mutual conversion with numpy.ndarray |
| transformers | Hugging Face's pre-trained model library, one of Multimodal SDK's dependencies |

## Essential Knowledge

### Basic Concept Understanding

#### Multimodal Data Processing

Multimodal SDK primarily processes the following modality data:

- **Images**: Supports jpg/jpeg format, width and height range [10, 8192]
- **Videos**: Supports mp4 format, resolution range 480P-4K
- **Audio**: Supports wav format, sample rate range [1, 64000]

#### Data Flow

Typical data flow of the SDK:

1. **Input**: File path → Decode → Raw data
2. **Processing**: Preprocessing operations such as resize, crop, normalize
3. **Output**: Tensor object, convertible to NumPy array or PyTorch tensor
