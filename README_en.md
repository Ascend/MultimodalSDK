<h1 align="center">Multimodal SDK</h1>

<div align="center">

[![Ascend](https://img.shields.io/badge/Community-MindSDK-blue.svg)](https://www.hiascend.com/eng/developer/software/mindsdk)
[![License](https://badgen.net/badge/License/MulanPSL-2.0/blue)](./LICENSE.md)
[![Zread](https://img.shields.io/badge/Zread-Ask_AI-_.svg?style=flat&color=0052D9&labelColor=000000&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTQuOTYxNTYgMS42MDAxSDIuMjQxNTZDMS44ODgxIDEuNjAwMSAxLjYwMTU2IDEuODg2NjQgMS42MDE1NiAyLjI0MDFWNC45NjAxQzEuNjAxNTYgNS4zMTM1NiAxLjg4ODEgNS42MDAxIDIuMjQxNTYgNS42MDAxSDQuOTYxNTZDNS4zMTUwMiA1LjYwMDEgNS42MDE1NiA1LjMxMzU2IDUuNjAxNTYgNC45NjAxVjIuMjQwMUM1LjYwMTU2IDEuODg2NjQgNS4zMTUwMiAxLjYwMDEgNC45NjE1NiAxLjYwMDFaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00Ljk2MTU2IDEwLjM5OTlIMi4yNDE1NkMxLjg4ODEgMTAuMzk5OSAxLjYwMTU2IDEwLjY4NjQgMS42MDE1NiAxMS4wMzk5VjEzLjc1OTlDMS42MDE1NiAxNC4xMTM0IDEuODg4MSAxNC4zOTk5IDIuMjQxNTYgMTQuMzk5OUg0Ljk2MTU2QzUuMzE1MDIgMTQuMzk5OSA1LjYwMTU2IDE0LjExMzQgNS42MDE1NiAxMy43NTk5VjExLjAzOTlDNS42MDE1NiAxMC42ODY0IDUuMzE1MDIgMTAuMzk5OSA0Ljk2MTU2IDEwLjM5OTlaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik0xMy43NTg0IDEuNjAwMUgxMS4wMzg0QzEwLjY4NSAxLjYwMDEgMTAuMzk4NCAxLjg4NjY0IDEwLjM5ODQgMi4yNDAxVjQuOTYwMUMxMC4zOTg0IDUuMzEzNTYgMTAuNjg1IDUuNjAwMSAxMS4wMzg0IDUuNjAwMUgxMy43NTg0QzE0LjExMTkgNS42MDAxIDE0LjM5ODQgNS4zMTM1NiAxNC4zOTg0IDQuOTYwMVYyLjI0MDFDMTQuMzk4NCAxLjg4NjY0IDE0LjExMTkgMS42MDAxIDEzLjc1ODQgMS42MDAxWiIgZmlsbD0iI2ZmZiIvPgo8cGF0aCBkPSJNNCAxMkwxMiA0TDQgMTJaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00IDEyTDEyIDQiIHN0cm9rZT0iI2ZmZiIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgo8L3N2Zz4K&logoColor=ffffff)](https://zread.ai/Ascend/MultimodalSDK)
[![DeepWiki](https://img.shields.io/badge/DeepWiki-Ask_AI-_.svg?style=flat&color=0052D9&labelColor=000000&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACwAAAAyCAYAAAAnWDnqAAAAAXNSR0IArs4c6QAAA05JREFUaEPtmUtyEzEQhtWTQyQLHNak2AB7ZnyXZMEjXMGeK/AIi+QuHrMnbChYY7MIh8g01fJoopFb0uhhEqqcbWTp06/uv1saEDv4O3n3dV60RfP947Mm9/SQc0ICFQgzfc4CYZoTPAswgSJCCUJUnAAoRHOAUOcATwbmVLWdGoH//PB8mnKqScAhsD0kYP3j/Yt5LPQe2KvcXmGvRHcDnpxfL2zOYJ1mFwrryWTz0advv1Ut4CJgf5uhDuDj5eUcAUoahrdY/56ebRWeraTjMt/00Sh3UDtjgHtQNHwcRGOC98BJEAEymycmYcWwOprTgcB6VZ5JK5TAJ+fXGLBm3FDAmn6oPPjR4rKCAoJCal2eAiQp2x0vxTPB3ALO2CRkwmDy5WohzBDwSEFKRwPbknEggCPB/imwrycgxX2NzoMCHhPkDwqYMr9tRcP5qNrMZHkVnOjRMWwLCcr8ohBVb1OMjxLwGCvjTikrsBOiA6fNyCrm8V1rP93iVPpwaE+gO0SsWmPiXB+jikdf6SizrT5qKasx5j8ABbHpFTx+vFXp9EnYQmLx02h1QTTrl6eDqxLnGjporxl3NL3agEvXdT0WmEost648sQOYAeJS9Q7bfUVoMGnjo4AZdUMQku50McDcMWcBPvr0SzbTAFDfvJqwLzgxwATnCgnp4wDl6Aa+Ax283gghmj+vj7feE2KBBRMW3FzOpLOADl0Isb5587h/U4gGvkt5v60Z1VLG8BhYjbzRwyQZemwAd6cCR5/XFWLYZRIMpX39AR0tjaGGiGzLVyhse5C9RKC6ai42ppWPKiBagOvaYk8lO7DajerabOZP46Lby5wKjw1HCRx7p9sVMOWGzb/vA1hwiWc6jm3MvQDTogQkiqIhJV0nBQBTU+3okKCFDy9WwferkHjtxib7t3xIUQtHxnIwtx4mpg26/HfwVNVDb4oI9RHmx5WGelRVlrtiw43zboCLaxv46AZeB3IlTkwouebTr1y2NjSpHz68WNFjHvupy3q8TFn3Hos2IAk4Ju5dCo8B3wP7VPr/FGaKiG+T+v+TQqIrOqMTL1VdWV1DdmcbO8KXBz6esmYWYKPwDL5b5FA1a0hwapHiom0r/cKaoqr+27/XcrS5UwSMbQAAAABJRU5ErkJggg==)](https://deepwiki.com/Ascend/MultimodalSDK)

</div>

## ✨ Latest Updates

<span style="font-size:14px;">

🔹 **[2026.09.30]**: 🚀 [Multimodal SDK 26.2.0 Release is now available](https://atomgit.com/Ascend/MultimodalSDK/releases/v26.2.0).<br>
🔹 **[2026.07.31]**: 🚀 [Multimodal SDK 26.1.0 Release is now available](https://atomgit.com/Ascend/MultimodalSDK/releases/v26.1.0).<br>
🔹 **[2026.04.25]**: 🚀 [Multimodal SDK 26.0.0 Release is now available](https://atomgit.com/Ascend/MultimodalSDK/releases/v26.0.0).<br>
🔹 **[2025.12.30]**: 🚀 Multimodal SDK is released as open source.<br>

</span>

## ℹ️ Introduction

Multimodal SDK is a multimodal LLM inference preprocessing acceleration suite optimized for Ascend devices. It provides commonly used interfaces for image, video, and audio processing, covering preprocessing processes such as image and video loading and decoding, resizing, cropping, and audio loading. It also supports conversion between multiple open-source data structures and accelerator library data structures, helping you quickly onboard and migrate multimodal inference applications.

<img src="./docs/en/figures/mmsdk_arch.svg" width="1200"/>

## ⚙️ Features

| Feature | Description | Interface |
| --- | --- | --- |
| [Image Processing](./docs/en/04_user_guide/user_guide.md#image-processing) | Supports image reading, resizing, and cropping, as well as NumPy data conversion for image preprocessing scenarios in multimodal models. | [Link](./docs/en/05_api/function_reference.md) |
| [Video Processing](./docs/en/04_user_guide/user_guide.md#video-processing) | Supports decoding by specified frame ID or uniform sampling by target frame count, and returns a list of `Image` objects that can be further processed. | [Link](./docs/en/05_api/function_reference.md) |
| [Audio Processing](./docs/en/04_user_guide/user_guide.md#audio-processing) | Supports audio loading from a single file, a file list, or a directory, and allows you to specify the resampling rate. | [Link](./docs/en/05_api/function_reference.md) |
| [Inference Framework Patches](./docs/en/05_api/patcher.md) | Provides vLLM framework patches to integrate acceleration capabilities into existing inference processes. | [Link](./docs/en/05_api/patcher.md) |

## 🚀 Quick Start

Multimodal SDK provides a Docker-based quick start process that helps you start the environment, load variables, and verify your first Python example. For details, see the [Quick Start](./docs/en/02_quickstart/quickstart.md).

If you need to modify the source code and rebuild it, see the [Contribution Guide](./CONTRIBUTING.md).

## 📦 Installation Guide

Multimodal SDK supports offline installation, image installation, and source code installation. For details, see the [Installation Guide](./docs/en/03_installation_guide/installation_guide.md).

## 📘 User Guide

Multimodal SDK provides examples and operation guidance for the basic preprocessing interfaces in typical scenarios: images, videos, and audio. For details, see [Examples and Guidance](./docs/en/04_user_guide/user_guide.md).

For API descriptions, see the following documents.

| Document | Description |
| --- | --- |
| [Function Reference](./docs/en/05_api/function_reference.md) | Describes interfaces such as `Tensor`, `Image`, `video_decode`, and `load_audio`. |
| [Patcher](./docs/en/05_api/patcher.md) | Describes the vLLM framework preprocessing acceleration patches. |
| [Python API Reference](./docs/en/05_api/README.md) | Data type enumerations and the entry point to the API documentation. |

## 🗺️ Roadmap

[Roadmap (2026Q3)](https://atomgit.com/Ascend/MultimodalSDK/issues/40)

## 🔀 Version Maintenance Strategy

| Version | Maintenance Strategy | Current Status | Release Date | Next Status | EOL Date |
| --- | --- | --- | --- | --- | --- |
| master | Long-term support | Development | Under active development and not yet released | Continuous development | - |
| v26.2.0 | Regular branch | Maintenance | 2026-09-30 | Expected to enter unsupported status on March 31, 2027. | 2027-03-31 |
| v26.1.0 | Regular branch | Maintenance | 2026-07-31 | Expected to enter unsupported status on January 31, 2027. | 2027-01-31 |
| v26.0.0 | Regular branch | Maintenance | 2026-04-25 | Expected to enter unsupported status on October 25, 2026. | 2026-10-25 |

## 🛠️ Contribution Guide

You are welcome to contribute to the project. For the contribution process and guidelines, see the [Contribution Guide](./CONTRIBUTING.md).

## ⚖️ Related Notices

🔹 [Release Notes](./docs/en/release_notes_mm.md)<br>
🔹 [Security Hardening](./docs/en/06_references/security_hardening.md)<br>
🔹 [License Statement](./LICENSE.md)<br>
🔹 [Documentation License Statement](./docs/LICENSE)<br>
🔹 [Disclaimer](./docs/en/01_introduction/02_disclaimer.md)<br>
🔹 [Third-Party Open-Source Software Notice](./Third_Party_Open_Source_Software_Notice)

## 🤝 Suggestions and Communication

You are welcome to raise questions and exchange ideas in the following ways.

| Resource | Description |
| :-- | :-- |
| [FAQ](./docs/en/06_references/faq.md) | Answers to frequently asked questions and usage inquiries. |
| [Create an Issue](https://atomgit.com/Ascend/MultimodalSDK/issues/create/choose) | Submit bugs, feature requests, or suggestions. |
| [Community Tasks](https://atomgit.com/Ascend/MultimodalSDK/issues) | View and claim community tasks. |
| [Meeting Calendar](https://meeting.ascend.osinfra.cn/?sig=sig-MindSeriesSDK) | Schedule of regular community meetings and activities. |
