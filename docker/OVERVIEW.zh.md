# MultimodalSDK

> [English](./OVERVIEW.md) | 中文

## 快速参考

- 从哪里获取帮助
  - [issue 反馈](https://gitcode.com/Ascend/MultimodalSDK/issues)
  - [MultimodalSDK 代码](https://gitcode.com/Ascend/MultimodalSDK)
  - [MultimodalSDK 文档入口](https://gitcode.com/Ascend/MultimodalSDK/blob/master/README.md)
  - [社区入口](https://www.hiascend.com/)

## MultimodalSDK简介

多模态大模型推理流程中需要处理大量复杂的数据。MultimodalSDK 通过提供一系列高性能的昇腾设备亲和性接口，加速大模型推理预处理流程。预处理接口当前在 CPU 上执行（`DeviceMode.CPU`），通常与 CANN/NPU 推理框架配合部署。

- 包括图像视频加载和解码，resize、crop等预处理常用操作。
- 支持多种开源数据结构与加速库数据结构的相互转换，方便快速使用和移植。

## 支持的 Tags 及 Dockerfile 链接

### Tag 规范

Tag 遵循以下格式：

```bash
<MultimodalSDK版本>-<芯片系列>-<操作系统>-<python版本>
```

| 字段         | 示例值                          | 说明             |
| ------------ | ------------------------------- | ---------------- |
| `MultimodalSDK版本`   | `26.1.0`              | MultimodalSDK 版本号      |
| `芯片系列`   | `910`          | 目标芯片系列 |
| `操作系统`   | `ubuntu22.04`、`openeuler24.03` | 基础操作系统     |
| `python版本` | `py3.11`    | Python 版本      |

### 支持的tags及Dockerfile

| Tag                                | Dockerfile                                                   |
| ---------------------------------- | ------------------------------------------------------------ |
| `26.1.0-910b-openeuler24.03-py3.11`   | [Dockerfile.910b.openEuler](./Dockerfile.910b.openEuler) |
| `26.1.0-910b-ubuntu22.04-py3.11`    | [Dockerfile.910b.ubuntu](./Dockerfile.910b.ubuntu)      |

---

## 快速开始

### 前置要求（可选）

#### 安装驱动

主机上必须安装与容器内 CANN 版本兼容的 NPU 驱动。请参阅 [CANN 兼容性矩阵](https://www.hiascend.com/document) 了解驱动与 CANN 版本的对应关系。

---

### 如何本地构建

```bash
docker build -t {your_repo}/multimodal:latest -f Dockerfile.<芯片系列>.<操作系统> .
```

### 运行 MultimodalSDK 容器

```bash
docker run \
    --name multimodal_container \
    --device /dev/davinci0 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -it ascend/multimodal:tag bash
```

## 进入容器

```bash
docker exec -it multimodal_container bash
```

### 如何二次开发

```bash
FROM swr.cn-south-1.myhuaweicloud.com/ascendhub/multimodalsdk:26.1.0-910b-ubuntu22.04-py3.11-aarch64

RUN apt update -y && \
    apt install gcc ...

...
```

---

## MultimodalSDK使用说明

MultimodalSDK 提供丰富的示例代码，帮助开发者快速上手。您可以通过以下链接获取最新的示例：

- [MultimodalSDK 示例代码](https://gitcode.com/Ascend/MultimodalSDK/blob/master/docs/zh/02_quickstart/quickstart.md)

## 支持的硬件

| 产品型号                       | 架构           |
| ------------------------------- | -------------- |
| Atlas 800I A2                   | ARM64 |

---

## 许可证

查看这些镜像中包含的 CANN 和 Mind 系列软件的[许可证信息](https://github.com/Ascend/cann-container-image/blob/main/LICENSE)。

与所有容器镜像一样，预装软件包（Python、系统库等）可能受其自身许可证约束。
