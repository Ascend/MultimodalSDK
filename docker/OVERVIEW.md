# MultimodalSDK

> English | [中文](./OVERVIEW.zh.md)

## Quick Reference

- Where to get help
  - [Issue Feedback](https://gitcode.com/Ascend/MultimodalSDK/issues)
  - [MultimodalSDK Code](https://gitcode.com/Ascend/MultimodalSDK)
  - [MultimodalSDK Documentation](https://gitcode.com/Ascend/MultimodalSDK/blob/master/README.md)
  - [Community](https://www.hiascend.com/)

## MultimodalSDK Overview

In the inference pipeline of multimodal large models, massive and complex data needs to be processed. The MultimodalSDK accelerates the preprocessing workflow of large model inference by providing a set of high-performance Ascend-affinity interfaces. Preprocessing APIs currently run on CPU (`DeviceMode.CPU`) and are typically deployed alongside CANN/NPU inference frameworks.

- It covers common preprocessing operations such as image and video loading and decoding, as well as resize, crop, and other typical processing steps.
- It supports mutual conversion between various open-source data structures and acceleration library data structures, enabling rapid application and easy migration.

## Supported Tags and Dockerfile Links

### Tag Naming Convention

Tags follow this pattern:

```bash
<multimodalsdk_version>-<vllm-ascend_version>-<torch-npu_version>-<chip_series>-<os>-<python_version>-aarch64
```

| Field            | Example Values                  | Description               |
| ---------------- | ------------------------------- | ------------------------- |
| `multimodalsdk_version` | `26.1.0`   | MultimodalSDK version              |
| `cann_version`   | `9.1.0`              | cann version      |
| `torch-npu_version`   | `2.6.0.post5`              | torch-npu version      |
| `chip_series`    | `910b`         | Target Atlas chip family |
| `os`             | `ubuntu22.04`, `openeuler24.03` | Base operating system     |
| `python_version` | `py3.12`    | Python version            |

### Tags and Dockerfile

| Tag                                | Dockerfile                                                   |
| ---------------------------------- | ------------------------------------------------------------ |
| `26.1.0-vllm_cann9.1.0-torch_npu2.6.0.post5-910b-openeuler24.03-py3.12-aarch64`   | [Dockerfile.910b.openEuler](./Dockerfile.910b.openEuler) |
| `26.1.0-vllm_cann9.1.0-torch_npu2.6.0.post5-910b-ubuntu22.04-py3.12-aarch64`    | [Dockerfile.910b.ubuntu](./Dockerfile.910b.ubuntu)      |

---

## Quick Start

### Prerequisites (optional)

#### Install Driver

An NPU driver compatible with the container's CANN version must be installed on the host. See the [CANN Compatibility Matrix](https://www.hiascend.com/document) for driver ↔ CANN version mapping.

---

### How to build

```bash
docker build -t {your_repo}/multimodal:latest -f Dockerfile.<chip_series>.<os> .
```

### Running MultimodalSDK Container

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

## Enter the Container

```bash
docker exec -it multimodal_container bash
```

## MultimodalSDK Usage

MultimodalSDK provides sample code to help developers get started quickly. You can access the examples through the following link:

- [MultimodalSDK Samples](https://gitcode.com/Ascend/MultimodalSDK/blob/master/docs/zh/02_quickstart/quickstart.md)

### Development

```bash
# Add required software by developer
FROM swr.cn-south-1.myhuaweicloud.com/ascendhub/multimodalsdk:26.1.0-vllm_cann9.1.0-torch_npu2.6.0.post5-910b-ubuntu22.04-py3.12-aarch64

RUN apt update -y && \
    ...

...
```

---

## Supported Hardware

| Product Examples                | Architecture   |
| ------------------------------- | -------------- |
| Atlas 800I A2                   | ARM64|

---

## License

View the [license information](https://github.com/Ascend/cann-container-image/blob/main/LICENSE) for CANN and MindSeries software included in these images.

As with all container images, the pre-installed packages (Python, system libraries, etc.) may be subject to their own licenses.
