# Introduction

The multimodal LLM inference process needs to handle large amounts of complex data. Multimodal SDK accelerates the LLM inference preprocessing process by providing a series of high-performance Ascend device affinity interfaces.

- It includes common preprocessing operations such as image and video loading and decoding, resize, and crop.
- It supports conversion between multiple open-source data structures and acceleration library data structures, which makes it easy to use and quick to port.

## Software Architecture

<img src="../figures/mmsdk_arch.svg" alt="Multimodal SDK software architecture diagram" width="1200"/>

**Modules in the Architecture Diagram**

| Module | Description |
| -- | -- |
| vLLM framework preprocessing plugin | Provides acceleration when you use vLLM for LLM inference. Qwen2VL: Provides accelerated image and video preprocessing when you use the Qwen2VL model, and greatly shortens the preprocessing latency compared with transformers. InternVL2: Provides accelerated image and video preprocessing when you use the InternVL2 model. |
| Acceleration library | Provides a series of high-performance image and tensor processing interfaces. |

## Supported Hardware and OSs

> **Querying the Device Product Model**
>
> In the Linux system, you can query the device product model in the following ways:
>
> ```bash
> # Use the dmidecode command:
> dmidecode -s system-product-name
>
> # Read the sysfs file:
> cat /sys/class/dmi/id/product_name
> ```
>
> Both methods can retrieve the product model information of the device. Choose one as needed.

| Product Series | Product Model | OS Version |
| -- | -- | -- |
| Atlas A2 inference products | Atlas 800I A2 | Ubuntu 22.04/openEuler 24.03 |
