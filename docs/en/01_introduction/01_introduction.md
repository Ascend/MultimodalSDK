# Introduction

Multimodal foundation model inference workflows involve processing large volumes of complex data. The Multimodal SDK accelerates foundation model inference preprocessing by providing a set of high-performance APIs optimized for Ascend devices.

- It includes common preprocessing operations such as image and video loading and decoding, resizing, and cropping.
- It supports conversion between various open-source data structures and accelerator library data structures, facilitating rapid adoption and porting.

## Software Architecture

<img src="../figures/mmsdk_arch.svg" alt="Multimodal SDK software architecture diagram" width="1200"/>

**Modules in the Architecture**

| Module | Description |
| -- | -- |
| vLLM framework preprocessing plugin | Provides acceleration when you use vLLM for foundation model inference. For Qwen2VL, it provides accelerated image and video preprocessing when you use the Qwen2VL model, significantly reducing preprocessing latency compared with Transformers. For InternVL2, it provides accelerated image and video preprocessing when you use the InternVL2 model. |
| Acceleration library | Provides a set of high-performance image and tensor processing APIs. |

## Supported Hardware and Operating Systems

> **Querying the Device Product Model**
>
> In Linux, you can use either of the following methods to query the device product model:
>
> ```bash
> # Use the `dmidecode` command:
> dmidecode -s system-product-name
>
> # Read the sysfs file:
> cat /sys/class/dmi/id/product_name
> ```
>
> Both methods return the device product model. Use either method as needed.

| Product Series | Product Model | Operating System Version |
| -- | -- | -- |
| Atlas A2 inference products | Atlas 800I A2 inference server | Ubuntu 22.04 / openEuler 24.03 |
