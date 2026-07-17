# Multimodal SDK

- [Latest Updates](#latest-updates)
- [Introduction](#introduction)
- [Directory Structure](#directory-structure)
- [Version Description](#version-description)
- [Environment Deployment](#environment-deployment)
- [Build Process](#build-process)
- [Quick Start](#quick-start)
- [Features](#features)
- [API Reference](#api-reference)
- [FAQ](#faq)
- [Security Statement](#security-statement)
- [Disclaimer](#disclaimer)
- [License](#license)
- [Suggestions and Communication](#suggestions-and-communication)

# Latest Updates

- [Dec. 30, 2025]: 🚀 Multimodal SDK is released as open source.

# Introduction

    Multimodal SDK is an LLM inference preprocessing acceleration toolkit optimized for Ascend devices. It provides high-performance interfaces for processing large volumes of multimodal data.
    - It includes common preprocessing operations such as image and video loading and decoding, resizing, and cropping.
    - It supports conversion between multiple open-source data structures and accelerator library data structures, which makes it easy to use and port.
<div align="center">

For details, see [Introduction](docs/en/introduction.md).

[![Zread](https://img.shields.io/badge/Zread-Ask_AI-_.svg?style=flat&color=0052D9&labelColor=000000&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTQuOTYxNTYgMS42MDAxSDIuMjQxNTZDMS44ODgxIDEuNjAwMSAxLjYwMTU2IDEuODg2NjQgMS42MDE1NiAyLjI0MDFWNC45NjAxQzEuNjAxNTYgNS4zMTM1NiAxLjg4ODEgNS42MDAxIDIuMjQxNTYgNS42MDAxSDQuOTYxNTZDNS4zMTUwMiA1LjYwMDEgNS42MDE1NiA1LjMxMzU2IDUuNjAxNTYgNC45NjAxVjIuMjQwMUM1LjYwMTU2IDEuODg2NjQgNS4zMTUwMiAxLjYwMDEgNC45NjE1NiAxLjYwMDFaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00Ljk2MTU2IDEwLjM5OTlIMi4yNDE1NkMxLjg4ODEgMTAuMzk5OSAxLjYwMTU2IDEwLjY4NjQgMS42MDE1NiAxMS4wMzk5VjEzLjc1OTlDMS42MDE1NiAxNC4xMTM0IDEuODg4MSAxNC4zOTk5IDIuMjQxNTYgMTQuMzk5OUg0Ljk2MTU2QzUuMzE1MDIgMTQuMzk5OSA1LjYwMTU2IDE0LjExMzQgNS42MDE1NiAxMy43NTk5VjExLjAzOTlDNS42MDE1NiAxMC42ODY0IDUuMzE1MDIgMTAuMzk5OSA0Ljk2MTU2IDEwLjM5OTlaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik0xMy43NTg0IDEuNjAwMUgxMS4wMzg0QzEwLjY4NSAxLjYwMDEgMTAuMzk4NCAxLjg4NjY0IDEwLjM5ODQgMi4yNDAxVjQuOTYwMUMxMC4zOTg0IDUuMzEzNTYgMTAuNjg1IDUuNjAwMSAxMS4wMzg0IDUuNjAwMUgxMy43NTg0QzE0LjExMTkgNS42MDAxIDE0LjM5ODQgNS4zMTM1NiAxNC4zOTg0IDQuOTYwMVYyLjI0MDFDMTQuMzk4NCAxLjg4NjY0IDE0LjExMTkgMS42MDAxIDEzLjc1ODQgMS42MDAxWiIgZmlsbD0iI2ZmZiIvPgo8cGF0aCBkPSJNNCAxMkwxMiA0TDQgMTJaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00IDEyTDEyIDQiIHN0cm9rZT0iI2ZmZiIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgo8L3N2Zz4K&logoColor=ffffff)](https://zread.ai/Ascend/MultimodalSDK)&nbsp;&nbsp;&nbsp;&nbsp;
[![DeepWiki](https://img.shields.io/badge/DeepWiki-Ask_AI-_.svg?style=flat&color=0052D9&labelColor=000000&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACwAAAAyCAYAAAAnWDnqAAAAAXNSR0IArs4c6QAAA05JREFUaEPtmUtyEzEQhtWTQyQLHNak2AB7ZnyXZMEjXMGeK/AIi+QuHrMnbChYY7MIh8g01fJoopFb0uhhEqqcbWTp06/uv1saEDv4O3n3dV60RfP947Mm9/SQc0ICFQgzfc4CYZoTPAswgSJCCUJUnAAoRHOAUOcATwbmVLWdGoH//PB8mnKqScAhsD0kYP3j/Yt5LPQe2KvcXmGvRHcDnpxfL2zOYJ1mFwrryWTz0advv1Ut4CJgf5uhDuDj5eUcAUoahrdY/56ebRWeraTjMt/00Sh3UDtjgHtQNHwcRGOC98BJEAEymycmYcWwOprTgcB6VZ5JK5TAJ+fXGLBm3FDAmn6oPPjR4rKCAoJCal2eAiQp2x0vxTPB3ALO2CRkwmDy5WohzBDwSEFKRwPbknEggCPB/imwrycgxX2NzoMCHhPkDwqYMr9tRcP5qNrMZHkVnOjRMWwLCcr8ohBVb1OMjxLwGCvjTikrsBOiA6fNyCrm8V1rP93iVPpwaE+gO0SsWmPiXB+jikdf6SizrT5qKasx5j8ABbHpFTx+vFXp9EnYQmLx02h1QTTrl6eDqxLnGjporxl3NL3agEvXdT0WmEost648sQOYAeJS9Q7bfUVoMGnjo4AZdUMQku50McDcMWcBPvr0SzbTAFDfvJqwLzgxwATnCgnp4wDl6Aa+Ax283gghmj+vj7feE2KBBRMW3FzOpLOADl0Isb5587h/U4gGvkt5v60Z1VLG8BhYjbzRwyQZemwAd6cCR5/XFWLYZRIMpX39AR0tjaGGiGzLVyhse5C9RKC6ai42ppWPKiBagOvaYk8lO7DajerabOZP46Lby5wKjw1HCRx7p9sVMOWGzb/vA1hwiWc6jm3MvQDTogQkiqIhJV0nBQBTU+3okKCFDy9WwferkHjtxib7t3xIUQtHxnIwtx4mpg26/HfwVNVDb4oI9RHmx5WGelRVlrtiw43zboCLaxv46AZeB3IlTkwouebTr1y2NjSpHz68WNFjHvupy3q8TFn3Hos2IAk4Ju5dCo8B3wP7VPr/FGaKiG+T+v+TQqIrOqMTL1VdWV1DdmcbO8KXBz6esmYWYKPwDL5b5FA1a0hwapHiom0r/cKaoqr+27/XcrS5UwSMbQAAAABJRU5ErkJggg==)](https://deepwiki.com/Ascend/MultimodalSDK)

</div>

# Directory Structure

```text
├── build_script
│   └── build.sh
├── script
│   ├── help.info
│   ├── install.sh
│   ├── set_env.sh
│   └── uninstall.sh
├── setup.py
├── source
│   └── mm
│       ├── acc
│       │   ├── _impl
│       │   │   └── __init__.py
│       │   ├── __init__.py
│       │   └── wrapper
│       │       ├── data_type.py
│       │       ├── image_wrapper.py
│       │       ├── __init__.py
│       │       ├── tensor_wrapper.py
│       │       ├── util.py
│       │       └── video_wrapper.py
│       ├── adapter
│       │   ├── __init__.py
│       │   ├── internvl2_preprocessor.py
│       │   └── qwen2_vl_preprocessor.py
│       ├── comm
│       │   ├── __init__.py
│       │   └── log.py
│       ├── core
│       │   └── __init__.py
│       ├── __init__.py
│       └── patcher
│           ├── __init__.py
│           └── vllm
│               ├── image_patcher.py
│               ├── __init__.py
│               ├── internvl2_image_processor_patcher.py
│               ├── qwen2_vl_image_processor_patcher.py
│               └── video_patcher.py
└── test
    ├── assets
    │   ├── dog_1920_1080.jpg
    │   ├── dog_1920_1080.png
    │   └── test_aac.mp4
    ├── test_demo.py
    ├── test_image.py
    ├── test_internvl2_preprocessor.py
    ├── test_log.py
    ├── test_qwen2_vl_preprocessor.py
    ├── test_tensor.py
    └── test_video.py

```

# Version Description

The release notes for Multimodal SDK include software version compatibility and the feature changes in each version. See the following table.

| Product | Version |
| :--- | :--- |
| Ascend HDK | 26.0.RC1 |
| CANN | 9.0.0 |

For details, see [Release Notes](docs/en/release_notes.md).

# Environment Deployment

This section describes how to install Multimodal SDK. For details, see [Installation Guide](docs/en/installation_guide.md).

# Build Process

This section uses the CANN 8.3.RC2 compatible release as an example to describe how to build Multimodal SDK from source. The NPU driver, firmware, and CANN package can be downloaded from the Ascend Community website.

1. Download the build dependencies.

   **Note**: Building requires Python 3.11. Ensure that this version is installed in the environment.

   ```bash
   # Download the source code
   git clone https://gitcode.com/Ascend/MultimodalSDK.git
   # Enter the project root directory
   cd MultimodalSDK

   # Download the makeself dependency into the project root directory. The script automatically applies patches and builds it. The project uses a customized version of makeself for packaging, so you need to download makeself v2.5 and the corresponding patch
   git clone -b v2.5.0.x https://gitcode.com/cann-src-third-party/makeself.git makeself_patch
   git clone -b release-2.5.0 https://gitcode.com/gh_mirrors/ma/makeself.git
   ```

   The project requires several open-source components. Download the following source code:

   ```bash
   # AccSDK dependency
   cd MultimodalSDK/AccSDK
   wget https://mindcluster.obs.cn-north-4.myhuaweicloud.com/opensource.tar.gz
   # Acc_data dependency
   cd MultimodalSDK/AccSDK/acc_data/3rdparty/pybind
   git clone -b v2.13.6 https://gitcode.com/GitHub_Trending/py/pybind11.git
   cd MultimodalSDK/AccSDK/acc_data/3rdparty/gtest
   git clone -b release-1.11.0 https://gitcode.com/GitHub_Trending/go/googletest.git
   ```

2. Run the build.

   Run the following commands to build the project:

    ```bash
    source /path/to/Ascend/ascend-toolkit/set_env.sh
    bash MultimodalSDK/build_script/build_merge.sh
    ```

3. The generated run package is located in `MultimodalSDK/output` as `Ascend-mindxsdk-multimodal_${SDK_VERSION}_linux-aarch64.run`.

4. Run the test cases.

   First install `lcov` 2.0 to collect test coverage and generate a visual report.

   ```bash
   apt update
   apt install -y libcapture-tiny-perl libdatetime-perl libtimedate-perl
   wget https://github.com/linux-test-project/lcov/releases/download/v2.0/lcov-2.0.tar.gz
   tar -xzf lcov-2.0.tar.gz && cd lcov-2.0
   make install
   ```

   Then run the following command to run the test cases.

   ```bash
   bash MultimodalSDK/build_script/build_merge.sh test
   ```

# Quick Start

Multimodal SDK provides a set of high-performance, easy-to-use interfaces. For details, see [Quick Start](docs/en/quickstart.md) and [Examples and Guidance](docs/en/user_guide.md).

# Features

Multimodal SDK provides high-performance interfaces optimized for Ascend devices for LLM inference preprocessing.
It includes common preprocessing operations such as image and video loading and decoding, resizing, and cropping.
It supports conversion between multiple open-source data structures and accelerator library data structures for ease of use and portability.

# API Reference

For the API reference, see the following documents.

[Python API Reference](docs/en/api/README.md)

[Adapter](docs/en/api/adapter.md)

[Patcher](docs/en/api/patcher.md)

[Function Reference](docs/en/api/function_reference.md)

# FAQ

## Issue Symptom

Even after `lzma` is installed, calling `torchvision` still reports that the `lzma` module is missing.

## Solution

Install the `lzma` module.

```shell
pip install backports.lzma
```

Go to the Python library directory. The following example uses Python 3.11.4.

```shell
cd /xx/xx/python-3.11.4/lib/python3.11
```

Modify `lzma.py` as follows.

```shell
from _lzma import *
from _lzma import _encode_filter_properties, _decode_filter_properties
```

Replace it with the following content.

```shell
from backports.lzma import *
from backports.lzma import _encode_filter_properties, _decode_filter_properties
```

# Security Statement

- When you use APIs to read a file, ensure that the file owner is you and that the permissions are no greater than 640 to avoid privilege escalation and other security issues. Software or programs downloaded from external sources may be risky. You are responsible for ensuring their security.
- Communication matrix: Multimodal SDK does not actively open or depend on any ports. Therefore, it does not involve a communication matrix.
- Public network addresses: The URLs in the Multimodal SDK installation package are removed after installation and are not accessed. Therefore, they do not pose a risk.

For details, see [Security Statement](docs/en/security_hardening.md) and [Appendix](docs/en/appendix.md).

# Disclaimer

- This repository contains multiple development branches. These branches may include unfinished, experimental, or untested features. Before official release, these branches should not be used in any production environment or in projects that depend on critical business systems. Ensure that you use our official release version to guarantee code stability and security.
  The project and its contributors are not responsible for any issues, losses, or data corruption caused by using development branches.
- For the official release, see <https://gitcode.com/ascend/MultimodalSDK/releases>.

# License

Multimodal SDK is licensed under Apache 2.0. The corresponding license text is available in [LICENSE](LICENSE.md).

The documents in the `docs` directory of Multimodal SDK are licensed under CC-BY 4.0. For details, see [LICENSE](./docs/LICENSE).

# Suggestions and Communication

Everyone is welcome to contribute to the community. If you have any questions or suggestions, submit an [issue](https://gitcode.com/Ascend/MultimodalSDK/issues), and we will reply as soon as possible. Thank you for your support.
