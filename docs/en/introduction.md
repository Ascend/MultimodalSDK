# Introduction

Multimodal inference workflows need to process large volumes of complex data. Multimodal SDK accelerates LLM preprocessing by providing a set of high-performance interfaces optimized for Ascend devices.

- It includes common preprocessing operations such as image and video loading and decoding, resizing, and cropping.
- It supports conversion between multiple open-source data structures and accelerator library data structures, which makes it easy to use and quick to port.

**User Guide**

If you are using this software for the first time, start with the examples in the quick start document, and ensure that you prepare the required environment and software packages as described there.

If you are already familiar with the process, you can go directly to the API section to find the required functions and interfaces and accelerate your data processing.

# Software Architecture

The software architecture of Multimodal SDK is shown in [Figure 1](#fig92951326193419).

**Figure 1** software architecture of Multimodal SDK<a name="fig92951326193419"></a>
![](figures/Multimodal-SDK-software-architecture.png "Multimodal SDK software architecture")

**Table 1** modules in the architecture

|Module|Description|
|--|--|
|vLLM framework preprocessing plugin|Provides acceleration when you use vLLM for LLM inference. <ul><li>Qwen2-VL: Provides accelerated image and video preprocessing when you use the Qwen2-VL model. Compared with preprocessing in Transformers, latency is greatly reduced.</li><li>InternVL2: Provides accelerated image and video preprocessing when you use the InternVL2 model.</li></ul>|
|Acceleration library|Provides a set of high-performance image and tensor processing interfaces.|

# Supported Hardware and OSs

|Product Series|Product Model|OS Version|
|--|--|--|
|Atlas A2 inference products|Atlas 800I A2 inference server|Ubuntu 22.04|
