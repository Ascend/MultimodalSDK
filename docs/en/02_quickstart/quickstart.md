# Quick Start

Multimodal SDK provides multimodal preprocessing acceleration capabilities, including image decoding, resize/crop, video frame decoding, and audio loading. This document helps you start the environment with Docker and get your first Python example running.

For native installation on the host, see [Installation Guide](../03_installation_guide/installation_guide.md).

## Prerequisites

Before you start, ensure that:

- **Hardware**: Atlas 800I A2. For details, see [Supported Hardware and OSs](../01_introduction/01_introduction.md#supported-hardware-and-oss).
- **Docker**: Docker is installed, and the current user can run containers.
- **Test image**: The image already provides `/data/test.jpg`. You do not need to mount an extra test image directory.

## Step 1: Pulling the Image

1. **Selecting a matching version**
   - Visit the Ascend community [image repository](https://www.hiascend.com/developer/ascendhub/detail/e0081aa3c4dd441dbd6a379bee8cc4c9).
   - Select the image version that matches the current hardware model (Atlas 800I A2).
   - Note that currently only the aarch64 CPU architecture is supported.

2. **Checking the environment in advance**
   - Run the following command to verify the NPU driver status:

   ```bash
   npu-smi info
   ```

   - Check that the driver version matches the CANN version in the image (see [Firmware and Drivers](https://www.hiascend.com/hardware/firmware-drivers/community)).

3. **Image pull example**

   The image tag format is `{version}-{cann}-{torch_npu}-910b-{os}-{python}-aarch64`. The following table describes each variable:

   | Variable        | Description            | Example Value                      |
   | --------------- | ---------------------- | ---------------------------------- |
   | `{version}`   | Multimodal SDK version | `26.2.0`                         |
   | `{cann}`      | CANN version           | `cann9.2.0`                      |
   | `{torch_npu}` | torch_npu version      | `torch_npu2.10.0.post4`          |
   | `{os}`        | Base operating system  | `ubuntu22.04`/`openeuler24.03` |
   | `{python}`    | Python version         | `py3.12`                         |

   ```bash
   TAG={version}-{cann}-{torch_npu}-910b-{os}-{python}-aarch64
   docker pull swr.cn-south-1.myhuaweicloud.com/ascendhub/multimodalsdk:${TAG}
   docker tag swr.cn-south-1.myhuaweicloud.com/ascendhub/multimodalsdk:${TAG} \
       multimodalsdk:${TAG}
   ```

   For example, for version 26.2.0, Ubuntu 22.04, and Python 3.12:

   ```bash
   docker pull swr.cn-south-1.myhuaweicloud.com/ascendhub/multimodalsdk:26.2.0-cann9.2.0-torch_npu2.10.0.post4-910b-ubuntu22.04-py3.12-aarch64
   docker tag swr.cn-south-1.myhuaweicloud.com/ascendhub/multimodalsdk:26.2.0-cann9.2.0-torch_npu2.10.0.post4-910b-ubuntu22.04-py3.12-aarch64 \
       multimodalsdk:26.2.0-cann9.2.0-torch_npu2.10.0.post4-910b-ubuntu22.04-py3.12-aarch64
   ```

## Step 2: Starting the Container

> [!NOTE]
>
> - Adjust the device number in `--device /dev/davinci0` according to the actual NPU number on the host (for example, `davinci1`).

Before starting the container, check whether a container with the same name already exists:

```bash
docker ps -a | grep multimodal_container
```

If `multimodal_container` is found, stop and remove it by running the following commands:

```bash
docker stop multimodal_container
docker rm multimodal_container
```

Run the following command to start the container, and check whether the container starts successfully:

```bash
docker run \
    --name multimodal_container \
    --device /dev/davinci0 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64 \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -itd multimodalsdk:26.2.0-cann9.2.0-torch_npu2.10.0.post4-910b-ubuntu22.04-py3.12-aarch64 bash

docker ps -a | grep multimodal_container
```

Enter the container. Perform all subsequent operations inside the container.

```bash
docker exec -it multimodal_container bash
```

## Step 3: Loading the Environment

> [!NOTE]
>
> The `MULTIMODAL_SDK_HOME` environment variable indicates the Multimodal SDK installation path. Its default value is `/usr/local/multimodal`.

```bash
export MULTIMODAL_SDK_HOME="/usr/local/multimodal"
source ${MULTIMODAL_SDK_HOME}/script/set_env.sh
```

## Step 4: Running the Verification Script

If the image does not have the built-in `/data/test.jpg`, download the test image into the container. The download command is as follows:

```bash
mkdir -p /data
wget --tries=3 --timeout=30 --waitretry=5 -O /data/test.jpg https://raw.atomgit.com/Ascend/MultimodalSDK/blobs/f1f648b7a8b8a67c7509b3425a89f743bbf59563/dog_1920_1080.jpg
```

> [!NOTE]
>
> The second parameter of `Image.open` is the decoding device string, and currently only `"cpu"` is supported. The run mode of operator interfaces such as `resize` uses the `DeviceMode.CPU` enumeration.

```bash
export TEST_IMAGE="/data/test.jpg"
chmod 640 "$TEST_IMAGE"
python3 - <<'EOF'
import os
from mm import Image, DeviceMode, Interpolation

test_image = os.environ["TEST_IMAGE"]
img = Image.open(test_image, "cpu")
img_resize = img.resize((500, 500), Interpolation.BICUBIC, DeviceMode.CPU)
print(f"resize output shape: {img_resize.numpy().shape}")
EOF
```

If the output is as follows, the verification is successful:

```text
resize output shape: (500, 500, 3)
```

## Step 5: Cleaning Up the Environment

After the verification is complete, exit the container. You are advised to clean up the container environment to release resources.

```bash
exit
docker stop multimodal_container
docker rm multimodal_container
```

## Step 6: Next Steps

| Goal | Document |
| -- | -- |
| Image resize/crop visualization example | [Examples and Guidance > Image Processing](../04_user_guide/user_guide.md#image-processing) |
| Video frame decoding | [Examples and Guidance > Video Processing](../04_user_guide/user_guide.md#video-processing) |
| Audio loading | [Examples and Guidance > Audio Processing](../04_user_guide/user_guide.md#audio-processing) |
| vLLM inference framework integration | [Patcher](../05_api/patcher.md) |
| Complete API reference | [Function Reference](../05_api/function_reference.md) |

# Common Issues Quick Reference

| Symptom | Solution |
| -- | -- |
| File permission error | Ensure that the permissions on the image file are no higher than 640: `chmod 640 "$TEST_IMAGE"`. |
| Test image not found in the container | Confirm that the image version you use has `/data/test.jpg` built in, and that `TEST_IMAGE` uses the container path `/data/test.jpg`. |
| The container cannot access the NPU | Check the NPU driver mounts and the `--device /dev/davinci*` device number. |
| Failure to import `mm` | Confirm that you have run `source ${MULTIMODAL_SDK_HOME}/script/set_env.sh`. |
| More issues | [FAQ](../06_references/faq.md), [Appendix](../06_references/appendix.md) |
