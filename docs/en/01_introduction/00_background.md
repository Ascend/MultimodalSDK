# Prerequisites

This document describes the background knowledge, terminology, and essential basics required to use Multimodal SDK.

## Terminology

### SDK and Frameworks

|Term|Description|
|------|------|
| Multimodal SDK | A multimodal software development kit that accelerates the LLM inference preprocessing process by providing a series of high-performance Ascend device affinity interfaces. |
| CANN | Compute Architecture for Neural Networks. The Ascend neural network computing architecture is the AI computing engine of Huawei Ascend processors. |
| vLLM | An LLM inference framework. Multimodal SDK provides a vLLM preprocessing plugin to accelerate inference. |
| Qwen2VL | The Tongyi Qianwen visual language model. Multimodal SDK provides image and video preprocessing acceleration for it. |
| InternVL2 | A visual language model. Multimodal SDK provides image and video preprocessing acceleration for it. |
| CLIP | Contrastive Language-Image Pre-Training, a contrastive language-image pre-training model used for text-image matching in key frame selection. |

### Hardware and Devices

|Term|Description|
|------|------|
| NPU | Neural Processing Unit. The neural network processing unit is the core computing unit of Ascend processors. |
| Atlas 800I A2 | A Huawei Atlas inference server model and a hardware platform that Multimodal SDK supports. |
| davinci | The device file name of an Ascend NPU device in the Linux system, located at `/dev/davinci*`. |

### Data Types and Formats

|Term|Description|
|------|------|
| Tensor | A tensor, which is a multidimensional array data structure that carries generic data of any modality. |
| DataType | The data type enumeration. INT8, UINT8, and FLOAT32 are supported. |
| TensorFormat | The Tensor data layout format. Generic N-dimensional array (ND), Batch-Height-Width-Channel (NHWC), and Batch-Channel-Height-Width (NCHW) are supported. |
| ImageFormat | The image format enumeration. RGB, BGR, RGB_PLANAR, and BGR_PLANAR are supported. |
| DeviceMode | The device operation mode. Currently, only CPU mode is supported. |
| Interpolation | The interpolation algorithm enumeration used in resize operations. Currently, only BICUBIC (bicubic interpolation) is supported. |

### Image, Audio, and Video Processing

|Term|Description|
|------|------|
| resize | An image scaling operation that resizes an image to a specified size. |
| crop | An image cropping operation that cuts a specified region from an image. |
| decode | A decoding operation that converts a compressed format (such as jpg or mp4) into raw data. |
| Key frame | A representative frame in a video, used for video content understanding and analysis. |
| Frame ID | The index number of a video frame, starting from 0. |
| Sampling rate | The number of audio samples per second, in Hz. |

### Dependency Libraries

|Term|Description|
|------|------|
| FFmpeg | An open-source audio and video processing library that Multimodal SDK uses for video decoding. |
| libjpeg-turbo | A JPEG image processing acceleration library that Multimodal SDK uses for image decoding. |
| Pillow (PIL) | A Python image processing library. Multimodal SDK supports conversion to and from PIL Image. |
| PyTorch | A deep learning framework. Multimodal SDK supports conversion to and from `torch.Tensor`. |
| NumPy | A Python scientific computing library. Multimodal SDK supports conversion to and from `numpy.ndarray`. |
| transformers | The Hugging Face pre-trained model library. |

## Required Knowledge

### Understanding Basic Concepts

#### Multimodal Data Processing

Multimodal SDK mainly processes the following data modalities:

- **Image**: Supports the jpg/jpeg format, with width and height in the range [10, 8192].
- **Video**: Supports the mp4 format, with resolution in the range [480, 4096].
- **Audio**: Supports the wav format, with a sampling rate in the range [1, 64000].

#### Data Flow

The typical data flow of the SDK is as follows:

1. **Input**: File path → Decode → Raw data
2. **Processing**: Preprocessing operations such as resize, crop, and normalize
3. **Output**: A Tensor object, which can be converted into a NumPy array or a PyTorch tensor
