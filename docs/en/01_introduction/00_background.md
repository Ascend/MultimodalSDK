# Background

This document introduces the background knowledge, terminology, and essential concepts required to use the Multimodal SDK.

## Terminology

### SDK and Framework

| Term | Description |
|------|-------------|
| Multimodal SDK | Multimodal Software Development Kit that accelerates preprocessing for foundation model inference by providing a set of high-performance APIs optimized for Ascend devices. |
| CANN | Compute Architecture for Neural Networks, Huawei Ascend's AI computing engine. |
| vLLM | Large language model inference framework. The Multimodal SDK provides vLLM preprocessing plugins to accelerate inference. |
| Qwen2VL | Qwen vision-language model. The Multimodal SDK provides image/video preprocessing acceleration for this model. |
| InternVL2 | Vision-language model. The Multimodal SDK provides image/video preprocessing acceleration for this model. |
| CLIP | Contrastive Language-Image Pre-Training, used for text-image matching in keyframe selection. |

### Hardware and Devices

| Term | Description |
|------|-------------|
| NPU | Neural Processing Unit, the core computing unit of Ascend processors |
| Atlas 800I A2 | Huawei Atlas inference server model, a hardware platform supported by the Multimodal SDK |
| davinci | Device node name for Ascend NPUs in Linux systems, located at `/dev/davinci*` |

### Data Types and Formats

| Term | Description |
|------|-------------|
| Tensor | Multi-dimensional array data structure used to store data of different modalities. |
| DataType | Data type enumeration that supports INT8, UINT8, and FLOAT32. |
| TensorFormat | Tensor data layout format that supports ND (general N-dimensional array), NHWC (Batch-Height-Width-Channel), and NCHW (Batch-Channel-Height-Width). |
| ImageFormat | Image format enumeration that supports RGB, BGR, RGB_PLANAR, and BGR_PLANAR. |
| DeviceMode | Device operation mode. Currently, only CPU mode is supported. |
| Interpolation | Interpolation algorithm enumeration used for resize operations. Currently, only BICUBIC (bicubic interpolation) is supported. |

### Image, Video, and Audio Processing

| Term | Description |
|------|-------------|
| resize | Image scaling operation that adjusts an image to the specified dimensions |
| crop | Image cropping operation that extracts a specified region from an image |
| decode | Decoding operation that converts compressed formats (such as JPG and MP4) to raw data |
| Keyframe | Representative frame in a video used for video content understanding and analysis |
| Frame ID | Index number of a video frame, starting from 0 |
| Sample rate | Number of audio samples per second, measured in Hz |

### Dependencies

| Term | Description |
|------|-------------|
| FFmpeg | Open-source audio/video processing library used by the Multimodal SDK for video decoding |
| libjpeg-turbo | JPEG image processing acceleration library used by the Multimodal SDK for image decoding |
| Pillow (PIL) | Python image processing library. The Multimodal SDK supports conversion to and from PIL Image |
| PyTorch | Deep learning framework. The Multimodal SDK supports conversion to and from `torch.Tensor` |
| NumPy | Python scientific computing library. The Multimodal SDK supports conversion to and from `numpy.ndarray` |
| transformers | Hugging Face's pretrained model library |

## Essential Knowledge

### Basic Concepts

#### Multimodal Data Processing

The Multimodal SDK primarily processes the following types of data:

- **Images**: JPG/JPEG format, with a width and height range of [10, 8192]
- **Videos**: MP4 format, with a resolution range of [480, 4096]
- **Audio**: WAV format, with a sample rate range of [1, 64000] Hz

#### Data Flow

The typical data flow of the SDK is as follows:

1. **Input**: File path → Decode → Raw data
2. **Processing**: Preprocessing operations such as `resize`, `crop`, and `normalize`
3. **Output**: Tensor object, which can be converted to a NumPy array or PyTorch tensor
