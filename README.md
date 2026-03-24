# MultimodalSDK
-   [最新消息](#最新消息)
-   [简介](#简介)
-   [目录结构](#目录结构)
-   [版本说明](#版本说明)
-   [环境部署](#环境部署)
-   [编译流程](#编译流程)
-   [快速入门](#快速入门)
-   [功能介绍&特性介绍](#功能介绍&特性介绍)
-   [API参考](#API参考)
-   [FAQ](#FAQ)
-   [安全声明](#安全声明)
-   [免责声明](#免责声明)
-   [License](#License)
-   [建议与交流](#建议与交流)

# 最新消息
- [2025.12.30]: 🚀 Multimodal SDK 开源发布

# 简介

    多模态大模型推理流程中需要处理大量复杂的数据。Multimodal SDK通过提供一系列高性能的昇腾设备亲和性接口，加速大模型推理预处理流程。
    - 包括图像视频加载和解码，resize、crop等预处理常用操作。
    - 支持多种开源数据结构与加速库数据结构的相互转换，方便快速使用和移植。
<div align="center">

更多详情请查看[简介](docs/zh/introduction.md)

[![Zread](https://img.shields.io/badge/Zread-Ask_AI-_.svg?style=flat&color=0052D9&labelColor=000000&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTQuOTYxNTYgMS42MDAxSDIuMjQxNTZDMS44ODgxIDEuNjAwMSAxLjYwMTU2IDEuODg2NjQgMS42MDE1NiAyLjI0MDFWNC45NjAxQzEuNjAxNTYgNS4zMTM1NiAxLjg4ODEgNS42MDAxIDIuMjQxNTYgNS42MDAxSDQuOTYxNTZDNS4zMTUwMiA1LjYwMDEgNS42MDE1NiA1LjMxMzU2IDUuNjAxNTYgNC45NjAxVjIuMjQwMUM1LjYwMTU2IDEuODg2NjQgNS4zMTUwMiAxLjYwMDEgNC45NjE1NiAxLjYwMDFaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00Ljk2MTU2IDEwLjM5OTlIMi4yNDE1NkMxLjg4ODEgMTAuMzk5OSAxLjYwMTU2IDEwLjY4NjQgMS42MDE1NiAxMS4wMzk5VjEzLjc1OTlDMS42MDE1NiAxNC4xMTM0IDEuODg4MSAxNC4zOTk5IDIuMjQxNTYgMTQuMzk5OUg0Ljk2MTU2QzUuMzE1MDIgMTQuMzk5OSA1LjYwMTU2IDE0LjExMzQgNS42MDE1NiAxMy43NTk5VjExLjAzOTlDNS42MDE1NiAxMC42ODY0IDUuMzE1MDIgMTAuMzk5OSA0Ljk2MTU2IDEwLjM5OTlaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik0xMy43NTg0IDEuNjAwMUgxMS4wMzg0QzEwLjY4NSAxLjYwMDEgMTAuMzk4NCAxLjg4NjY0IDEwLjM5ODQgMi4yNDAxVjQuOTYwMUMxMC4zOTg0IDUuMzEzNTYgMTAuNjg1IDUuNjAwMSAxMS4wMzg0IDUuNjAwMUgxMy43NTg0QzE0LjExMTkgNS42MDAxIDE0LjM5ODQgNS4zMTM1NiAxNC4zOTg0IDQuOTYwMVYyLjI0MDFDMTQuMzk4NCAxLjg4NjY0IDE0LjExMTkgMS42MDAxIDEzLjc1ODQgMS42MDAxWiIgZmlsbD0iI2ZmZiIvPgo8cGF0aCBkPSJNNCAxMkwxMiA0TDQgMTJaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00IDEyTDEyIDQiIHN0cm9rZT0iI2ZmZiIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgo8L3N2Zz4K&logoColor=ffffff)](https://zread.ai/Ascend/MultimodalSDK)&nbsp;&nbsp;&nbsp;&nbsp;
[![DeepWiki](https://img.shields.io/badge/DeepWiki-Ask_AI-_.svg?style=flat&color=0052D9&labelColor=000000&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACwAAAAyCAYAAAAnWDnqAAAAAXNSR0IArs4c6QAAA05JREFUaEPtmUtyEzEQhtWTQyQLHNak2AB7ZnyXZMEjXMGeK/AIi+QuHrMnbChYY7MIh8g01fJoopFb0uhhEqqcbWTp06/uv1saEDv4O3n3dV60RfP947Mm9/SQc0ICFQgzfc4CYZoTPAswgSJCCUJUnAAoRHOAUOcATwbmVLWdGoH//PB8mnKqScAhsD0kYP3j/Yt5LPQe2KvcXmGvRHcDnpxfL2zOYJ1mFwrryWTz0advv1Ut4CJgf5uhDuDj5eUcAUoahrdY/56ebRWeraTjMt/00Sh3UDtjgHtQNHwcRGOC98BJEAEymycmYcWwOprTgcB6VZ5JK5TAJ+fXGLBm3FDAmn6oPPjR4rKCAoJCal2eAiQp2x0vxTPB3ALO2CRkwmDy5WohzBDwSEFKRwPbknEggCPB/imwrycgxX2NzoMCHhPkDwqYMr9tRcP5qNrMZHkVnOjRMWwLCcr8ohBVb1OMjxLwGCvjTikrsBOiA6fNyCrm8V1rP93iVPpwaE+gO0SsWmPiXB+jikdf6SizrT5qKasx5j8ABbHpFTx+vFXp9EnYQmLx02h1QTTrl6eDqxLnGjporxl3NL3agEvXdT0WmEost648sQOYAeJS9Q7bfUVoMGnjo4AZdUMQku50McDcMWcBPvr0SzbTAFDfvJqwLzgxwATnCgnp4wDl6Aa+Ax283gghmj+vj7feE2KBBRMW3FzOpLOADl0Isb5587h/U4gGvkt5v60Z1VLG8BhYjbzRwyQZemwAd6cCR5/XFWLYZRIMpX39AR0tjaGGiGzLVyhse5C9RKC6ai42ppWPKiBagOvaYk8lO7DajerabOZP46Lby5wKjw1HCRx7p9sVMOWGzb/vA1hwiWc6jm3MvQDTogQkiqIhJV0nBQBTU+3okKCFDy9WwferkHjtxib7t3xIUQtHxnIwtx4mpg26/HfwVNVDb4oI9RHmx5WGelRVlrtiw43zboCLaxv46AZeB3IlTkwouebTr1y2NjSpHz68WNFjHvupy3q8TFn3Hos2IAk4Ju5dCo8B3wP7VPr/FGaKiG+T+v+TQqIrOqMTL1VdWV1DdmcbO8KXBz6esmYWYKPwDL5b5FA1a0hwapHiom0r/cKaoqr+27/XcrS5UwSMbQAAAABJRU5ErkJggg==)](https://deepwiki.com/Ascend/MultimodalSDK)

</div>

# 目录结构

``` 
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

# 版本说明
Multimodal的版本说明包含Multimodal的软件版本配套关系和软件包下载以及每个版本的特性变更说明，参考下表：

| 产品名称 | 版本 |
| :--- | :--- |
| Ascend HDK | 25.5.0 |
| CANN | 8.5.0 |

更多详情请查看[版本说明](docs/zh/release_notes.md)。

# 环境部署

介绍Multimodal的安装方式。更多详情请查看[安装指南](docs/zh/installation_guide.md)。

# 编译流程
本节以CANN 8.5.0相关配套为例，介绍如何通过源码编译生成 Multimodal SDK，其中NPU驱动、固件和CANN软件包可以通过昇腾社区下载。

1. 编译依赖下载

   ```bash
   # 下载源码
   git clone https://gitcode.com/Ascend/MultimodalSDK.git
   # 进入项目根目录
   cd MultimodalSDK

   # 将makeself依赖下载在项目根目录，脚本会自动进行patch/编译，项目使用定制版的makeself进行打包，需要下载makeselfv2.5和对应的patch
   git clone -b v2.5.0.x https://gitcode.com/cann-src-third-party/makeself.git makeself_patch
   git clone -b release-2.5.0 https://gitcode.com/gh_mirrors/ma/makeself.git
   ```

   项目需要一些开源组件，需要下载以下源码：
   ```bash
   # AccSDK依赖
   cd MultimodalSDK/AccSDK
   wget https://mindcluster.obs.cn-north-4.myhuaweicloud.com/opensource.tar.gz
   # acc_data依赖
   cd MultimodalSDK/AccSDK/acc_data/3rdparty/pybind
   git clone -b v2.13.6 https://gitcode.com/GitHub_Trending/py/pybind11.git
   cd MultimodalSDK/AccSDK/acc_data/3rdparty/gtest
   git clone -b release-1.11.0 https://gitcode.com/GitHub_Trending/go/googletest.git
   ```

2. 执行编译

   执行以下命令编译：
    ```bash
    source /path/to/Ascend/ascend-toolkit/set_env.sh
    bash MultimodalSDK/build_script/build_merge.sh
	```

3. 生成的 run 包在 ```MultimodalSDK/output``` 下：```Ascend-mindxsdk-multimodal_${SDK_VERSION}_linux-aarch64.run```

4. 执行测试用例

   首先安装lcov2.0用于统计测试覆盖率和生成可视化报告：
   ```bash
   apt update
   apt install -y libcapture-tiny-perl libdatetime-perl libtimedate-perl
   wget https://github.com/linux-test-project/lcov/releases/download/v2.0/lcov-2.0.tar.gz
   tar -xzf lcov-2.0.tar.gz && cd lcov-2.0
   make install
   ```

   然后执行以下命令运行测试用例：
   ```bash
   bash MultimodalSDK/build_script/build_merge.sh test
   ```

# 快速入门
Multimodal SDK提供了一系列CPU高性能和易用性的接口，更多详情请查看[快速入门](docs/zh/quick_start.md)与[样例与指导](docs/zh/user_guide.md)。

# 功能介绍&特性介绍
多模态大模型推理流程中需要处理大量复杂的数据。Multimodal SDK通过提供一系列高性能的昇腾设备亲和性接口，加速大模型推理预处理流程。
包括图像视频加载和解码，resize、crop等预处理常用操作。
支持多种开源数据结构与加速库数据结构的相互转换，方便快速使用和移植。

# API参考

API参考详见：

[Python接口说明](docs/zh/api/README.md)

[Adapter](docs/zh/api/adapter.md)

[Patcher](docs/zh/api/patcher.md)

[功能函数](docs/zh/api/function_reference.md)

# FAQ

### 问题现象
即使已安装lzma模块，调用torchvision时，依然提示缺少lzma模块
### 解决方案
安装lzma模块
```shell
pip install backports.lzma
```
进入python的库目录，以使用的python3.11.4为例
```shell
cd /xx/xx/python-3.11.4/lib/python3.11
```
修改lzma.py，将下面的内容
```shell
from _lzma import *
from _lzma import _encode_filter_properties, _decode_filter_properties
```
修改为
```shell
from backports.lzma import *
from backports.lzma import _encode_filter_properties, _decode_filter_properties
```

# 安全声明

- 使用API读取文件时，用户需要保证该文件的owner必须为自己，且权限不高于640，避免发生提权等安全问题。 外部下载的软件代码或程序可能存在风险，功能的安全性需由用户保证。
- 通信矩阵：目前Multimodal SDK开发套件包不会主动打开或者依赖任意端口，因此不涉及通信矩阵。
- 公网地址：Multimodal SDK的安装包中的网址安装结束后会被清除，并不会访问，不会造成风险.

更多详情请查看[安全声明](docs/zh/security_hardening.md)与[附录](docs/zh/appendix.md)。

# 免责声明

- 本仓库代码中包含多个开发分支，这些分支可能包含未完成、实验性或未测试的功能。在正式发布前，这些分支不应被应用于任何生产环境或者依赖关键业务的项目中。请务必使用我们的正式发行版本，以确保代码的稳定性和安全性。
  使用开发分支所导致的任何问题、损失或数据损坏，本项目及其贡献者概不负责。
- 正式版本请参考release版本 <https://gitcode.com/ascend/MultimodalSDK/releases>。

# License

MultimodalSDK以Apache 2.0许可证许可，对应许可证文本可查阅[LICENSE](LICENSE.md)。

MultimodalSDK docs目录下的文档适用CC-BY 4.0许可证，具体请参见[LICENSE](./docs/LICENSE)文件。

# 贡献声明

1. 提交错误报告：如果您在MultimodalSDK中发现了一个不存在安全问题的漏洞，请在MultimodalSDK仓库中的Issues中搜索，以防该漏洞被重复提交，如果找不到漏洞可以创建一个新的Issues。如果发现了一个安全问题请不要将其公开，请参阅安全问题处理方式。提交错误报告时应该包含完整信息。
2. 安全问题处理：本项目中对安全问题处理的形式，请通过邮箱通知项目核心人员确认编辑。
3. 解决现有问题：通过查看仓库的Issues列表可以发现需要处理的问题信息, 可以尝试解决其中的某个问题。
4. 如何提出新功能：请使用Issues的Feature标签进行标记，我们会定期处理和确认开发。
5. 开始贡献：
   - Fork本项目的仓库。
   - Clone到本地。
   - 创建开发分支。
   - 本地自测，提交前请通过所有的单元测试，包括为您要解决的问题新增的单元测试。
   - 提交代码。
   - 新建Pull Request。
   - 代码检视，您需要根据评审意见修改代码，并重新提交更新。此流程可能涉及多轮迭代。
   - 当您的PR获得足够数量的检视者批准后，Committer会进行最终审核。
   - 审核和测试通过后，CI会将您的PR合并入到项目的主干分支。

# 建议与交流

欢迎大家为社区做贡献。如果有任何疑问或建议，请提交[issue](https://gitcode.com/Ascend/MultimodalSDK/issues)，我们会尽快回复。感谢您的支持。