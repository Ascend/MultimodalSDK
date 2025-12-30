# MultiModal
-   [简介](#简介)
-   [目录结构](#目录结构)
-   [版本说明](#版本说明)
-   [环境部署](#环境部署)
-   [快速入门](#快速入门)
-   [功能介绍&特性介绍](#功能介绍&特性介绍)
-   [API参考](#API参考)
-   [FAQ](#FAQ)
-   [安全声明](#安全声明)
-   [免责声明](#免责声明)
-   [License](#License)
-   [建议与交流](#建议与交流)

# 简介

    多模态大模型推理流程中需要处理大量复杂的数据。Multimodal SDK通过提供一系列高性能的昇腾设备亲和性接口，加速大模型推理预处理流程。
    - 包括图像视频加载和解码，resize、crop等预处理常用操作。
    - 支持多种开源数据结构与加速库数据结构的相互转换，方便快速使用和移植。


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

# 环境部署

介绍Multimodal的安装方式。

## 安装依赖

### 安装Ubuntu系统依赖
| 依赖名称 | 版本建议 | 获取建议 |
| :--- | :--- | :--- |
| Python | 3.9及以上 | 建议通过获取源码包编译安装。 |
| CMake | 3.14及以上 | 建议通过包管理模块安装。 |
| Make | 4.1及以上 | 建议通过包管理模块安装。 |
| GCC | 9.4及以上 | 建议通过包管理模块安装。 |

### 安装NPU驱动固件和CANN

安装前，请参考[CANN 软件安装指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1/softwareinst/instg/instg_0000.html?Mode=PmIns&InstallType=local&OS=Ubuntu&Software=cannToolKit)安装CANN开发套件包、昇腾NPU驱动和昇腾NPU固件。
CANN软件提供进程级环境变量设置脚本，供用户在进程中引用，以自动完成环境变量设置。用户进程结束后自动失效。可在程序启动的Shell脚本中使用如下命令设置CANN的相关环境变量，也可通过命令行执行如下命令（以root用户默认安装路径“/usr/local/Ascend”为例）：

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### 安装Python软件包依赖

```shell
pip install transformer==4.51.3
pip install pillow==11.2.1
pip install numpy==1.26.4
pip install torch==2.5.1
```

## 安装Multimodal SDK

Multimodal SDK安装包[获取链接](https://gitcode.com/Ascend/MultimodalSDK/releases/7.2.RC1)

安装Multimodal SDK过程如下：
1. 以软件包的安装用户登录安装环境。
2. 将Multimodal SDK软件包上传到安装环境的任意路径下并进入软件包所在路径。
3. 增加对软件包的可执行权限。
    ```shell
    chmod u+x Ascend-mindxsdk-multimodal_{version}_linux-{arch}.run
    ```
4. 执行如下命令，校验软件包的一致性和完整性。
    ```shell
    ./Ascend-mindxsdk-multimodal_{version}_linux-{arch}.run --check
    ```
    如果系统没有shasum或者sha256sum工具则会校验失败，此时需要自行安装shasum或者sha256sum工具。 
    若显示如下信息，说明软件包已通过校验。
    ```shell
    Verifying archive integrity...  100%   SHA256 checksums are OK. All good.    
    ```
5. 创建Multimodal SDK软件包的安装路径。不建议在“/tmp”路径下安装。

    若用户未指定安装路径，软件会默认安装到Multimodal SDK软件包所在的路径。
    若用户想指定安装路径，需要先创建安装路径。以安装路径“/home/work/Mind_SDK”为例：
    ```shell
    mkdir -p /home/work/Mind_SDK
    ```
6. 进入Multimodal SDK软件包所在路径，参考以下命令安装Multimodal SDK。

    - 若用户指定了安装路径，将安装在指定的路径下。以安装路径“/home/work/Mind_SDK”为例：
    ```shell
    ./Ascend-mindxsdk-multimodal_{version}_linux-{arch}.run --install --install-path=/home/work/Mind_SDK
    ```
   - 若用户未指定安装路径，将安装在当前路径下。
    ```shell
    ./Ascend-mindxsdk-multimodal_{version}_linux-{arch}.run --install
    ```
7. 安装完成后，若显示如下信息，表示软件安装成功。
    ```shell
    Successfully installed Multimodal SDK
    ```
8. 进入Multimodal SDK安装路径下的“script”目录，执行以下命令使Multimodal SDK的环境变量生效。
    ```shell
    source set_env.sh
    ```
--install安装命令可选参数表

| 输入参数 | 含义                                                                                                                                                                                                                                                  |
| --- |-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--help` \| `-h` | 查询帮助信息。                                                                                                                                                                                                                                             |
| `--info` | 查询包构建信息。                                                                                                                                                                                                                                            |
| `--list` | 查询文件列表。                                                                                                                                                                                                                                             |
| `--check` | 查询包完整性。                                                                                                                                                                                                                                             |
| `--quiet` \| `-q` | 启用静默模式。<br>需要和 `--install` 或 `--upgrade` 参数配合使用。                                                                                                                                                                |
| `--nox11` | 不生成一个xterm终端                                                                                                                                                                                                                                        |
| `--noexec` | 不运行嵌入的脚本                                                                                                                                                                                                                                            |
| `--extract=<path>` | 直接提取到目标目录（绝对路径或相对路径）<br>通常与 `--noexec` 选项一起使用，仅用于提取文件而不运行它们                                                                                                                                                                                         |
| `--tar arg1 [arg2 ...]` | 通过 tar 命令访问归档文件的内容                                                                                                                                                                                                                                  |
| `--install` | Multimodal SDK软件包安装操作命令。<br>• 当前路径和安装路径不能存在非法字符，仅支持大小写字母、数字、`-`、`_`、`.`、`/`特殊字符。<br>• 安装路径下不能存在名为multimodal的文件或文件夹。<br>• 若存在名为multimodal的软链接，则会被覆盖。                                                                                                 |
| `--install-path=<path>` | （可选）自定义软件包安装根目录。如未设置，默认为当前命令执行所在目录。<br>• 建议用户使用绝对路径安装Multimodal SDK，指定安装路径时请避免使用相对路径。<br>• 需要和 `--install` 或 `--upgrade` 参数配合使用。<br>• 与 `--upgrade` 参数配合使用时，`--install-path` 代表旧软件包的安装目录，并在该目录下执行升级。<br>• 传入的路径参数不能存在非法字符，仅支持大小写字母、数字、`-`、`_`/特殊字符。 |
| `--upgrade` | Multimodal SDK软件包升级操作命令。升级需要确保已经安装过目录完整的Multimodal SDK。                                                                                                                                                                                             |
| `--version` | 查询软件包Multimodal SDK版本。                                                                                                                                                                                                                              |

# 快速入门

1. 使用高性能接口

    Multimodal SDK提供了一系列CPU和NPU高性能接口，用户可以根据这些接口自行选用集成到自己的业务流程中，下面提供几个高性能接口的示例作为参考。 
    高性能图像解码接口：
    ```shell
    from mm import Image
    img= Image.open("/home/test.jpg", "cpu")  # 样例代码的图片需要开发者自行替换
    ```
    高性能图像数据处理resize接口：
    ```shell
    from mm import Image, DeviceMode, Interpolation
    img = Image.open("/home/test.jpg", "cpu") # 样例代码的图片需要开发者自行替换
    img_resize = img.resize((500,500), Interpolation.BICUBIC, DeviceMode.CPU)
    ```
    高性能图像数据处理crop接口：
    ```shell
    from mm import Image, DeviceMode
    img = Image.open("/home/test.jpg", "cpu") # 样例代码的图片需要开发者自行替换
    img_crop = img.crop(10, 10, 10, 10, DeviceMode.CPU)
    ```
2. 使用开源推理框架对接接口

    基于Multimodal SDK提供的高性能接口，Multimodal SDK也提供了对接开源推理框架的vllm的适配方案，基于vllm以及vllm-ascend的patch机制
    - 使用补丁时，目前仅支持从vllm-ascend社区获取0.8.5.rc1镜像版
    - 镜像的安装方式请参见[vllm-ascend](https://docs.vllm.ai/projects/ascend/en/v0.8.5rc1/installation.html)，安装时请选择Using docker（从容器中安装）。
    - 在镜像中使用Multimodal SDK能力时，请首先执行以下命令：
    ```shell
    export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH
    ```
    您需要在vllm包的utils.py文件中添加如下内容，该文件路径在镜像中的位置为“/vllm-workspace/vllm/vllm/multimodal/utils.py”。
    ```shell
    import mm.patcher.vllm.video_patcher
    ```
    添加完成后，当使用vllm服务并传入视频文件数据时，若可以看到如下提示信息，则表示使用成功。
    ```shell
    load_file: Multimodal SDK Video Patcher Enabled!
    ```

# 功能介绍&特性介绍
多模态大模型推理流程中需要处理大量复杂的数据。Multimodal SDK通过提供一系列高性能的昇腾设备亲和性接口，加速大模型推理预处理流程。
包括图像视频加载和解码，resize、crop等预处理常用操作。
支持多种开源数据结构与加速库数据结构的相互转换，方便快速使用和移植。

# API参考

## mm.Tensor
Tensor类将被用于承载任意模态的通用数据，实现通用数据的创建、管理以及数据复制等操作。
## mm.Image
Image类将被用于承载图像数据，实现通用图像的创建、管理以及数据复制等操作。
## mm.video_decode
将传入的视频文件解码，并返回Image对象列表。
## mm.normalize
使用均值和标准差对Tensor对象进行归一化。
## mm.register_log_conf
日志注册函数。
## Adapter
使用多模态内部的加速能力对Qwen2VL模型的图像/视频预处理环节进行加速，继承transformers库中的前处理函数，当前仅支持对接transformers4.51.3版本的处理能力
## patcher
该补丁为vllm图像、视频解码提供加速能力，也为部分模型的图像/视频预处理提供加速能力

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


# 免责声明

- 本仓库代码中包含多个开发分支，这些分支可能包含未完成、实验性或未测试的功能。在正式发布前，这些分支不应被应用于任何生产环境或者依赖关键业务的项目中。请务必使用我们的正式发行版本，以确保代码的稳定性和安全性。
  使用开发分支所导致的任何问题、损失或数据损坏，本项目及其贡献者概不负责。
- 正式版本请参考release版本 <https://gitcode.com/ascend/MultimodalSDK/releases>

# License

MultimodalSDK以Apache 2.0许可证许可，对应许可证文本可查阅[LICENSE](LICENSE.md)。

# 贡献声明

1. 提交错误报告：如果您在MultimodalSDK中发现了一个不存在安全问题的漏洞，请在MultimodalSDK仓库中的Issues中搜索，以防该漏洞被重复提交，如果找不到漏洞可以创建一个新的Issues。如果发现了一个安全问题请不要将其公开，请参阅安全问题处理方式。提交错误报告时应该包含完整信息。
2. 安全问题处理：本项目中对安全问题处理的形式，请通过邮箱通知项目核心人员确认编辑。
3. 解决现有问题：通过查看仓库的Issues列表可以发现需要处理的问题信息, 可以尝试解决其中的某个问题。
4. 如何提出新功能：请使用Issues的Feature标签进行标记，我们会定期处理和确认开发。
5. 开始贡献：
   - Fork本项目的仓库
   - Clone到本地
   - 创建开发分支
   - 本地自测，提交前请通过所有的单元测试，包括为您要解决的问题新增的单元测试。
   - 提交代码
   - 新建Pull Request
   - 代码检视，您需要根据评审意见修改代码，并重新提交更新。此流程可能涉及多轮迭代。
   - 当您的PR获得足够数量的检视者批准后，Committer会进行最终审核。
   - 审核和测试通过后，CI会将您的PR合并入到项目的主干分支。

# 建议与交流

欢迎大家为社区做贡献。如果有任何疑问或建议，请提交[issue](https://gitcode.com/Ascend/MultimodalSDK/issues)，我们会尽快回复。感谢您的支持。