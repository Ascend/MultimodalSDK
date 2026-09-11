# Quick Start

Multimodal SDK provides multimodal preprocessing acceleration capabilities, including image decoding, resize/crop, video frame decoding, and audio loading. This document helps you start the environment with Docker and run your first Python example.

For native installation on the host machine, see [Installation and Deployment](../03_installation_guide/installation_guide.md).

## Prerequisites

Before you begin, ensure that:

- **Hardware**: Atlas 800I A2 inference server. See [Supported Hardware and Operating Systems](../01_introduction/01_introduction.md#supported-hardware-and-operating-systems).
- **Docker**: Docker is installed, and the current user can run containers.
- **Test image**: The container image already includes `/data/test.jpg`. Therefore, you do not need to mount an additional directory for test images.

## Step 1: Pulling the Image

1. **Selecting the matching version**
   - Access the Ascend community [image repository](https://www.hiascend.com/developer/ascendhub/detail/e0081aa3c4dd441dbd6a379bee8cc4c9).
   - Select the corresponding image version based on your hardware model (Atlas 800I A2 inference server).
   - Note that only the aarch64 CPU architecture is currently supported.

2. **Prechecking the environment**
   - Run the following command to verify the NPU driver status:

   ```bash
   npu-smi info
   ```

   - Check that the driver version matches the CANN version in the image (see [Firmware and Drivers](https://www.hiascend.com/hardware/firmware-drivers/community))

3. **Image pull example**

   The image tag format is `{version}-{cann}-{torch_npu}-910b-{os}-{python}-aarch64`. The variables are described as follows:

   | Variable | Description | Example Value |
   |------|------|--------|
   | `{version}` | Multimodal SDK version | `26.1.0` |
   | `{cann}` | CANN version | `9.1.0` |
   | `{torch_npu}` | torch_npu version | `2.6.0.rc1` |
   | `{os}` | Base operating system | `ubuntu22.04`/`openeuler24.03` |
   | `{python}` | Python version | `py3.12` |

   ```bash
   TAG={version}-{cann}-{torch_npu}-910b-{os}-{python}-aarch64
   docker pull swr.cn-south-1.myhuaweicloud.com/ascendhub/multimodalsdk:${TAG}
   docker tag swr.cn-south-1.myhuaweicloud.com/ascendhub/multimodalsdk:${TAG} \
       multimodalsdk:${TAG}
   ```

   The following example uses version 26.1.0, Ubuntu 22.04, and Python 3.12:

   ```bash
   docker pull swr.cn-south-1.myhuaweicloud.com/ascendhub/multimodalsdk:26.1.0-cann9.1.0-torch_npu2.6.0.post5-910b-ubuntu22.04-py3.12-aarch64
   docker tag swr.cn-south-1.myhuaweicloud.com/ascendhub/multimodalsdk:26.1.0-cann9.1.0-torch_npu2.6.0.post5-910b-ubuntu22.04-py3.12-aarch64 \
       multimodalsdk:26.1.0-cann9.1.0-torch_npu2.6.0.post5-910b-ubuntu22.04-py3.12-aarch64
   ```

## Step 2: Starting the Container

> [!NOTE]
>
> - Adjust the device number in `--device /dev/davinci0` according to the actual NPU number on the host machine (for example, `davinci1`).

First check whether a container with the same name already exists. If one exists, delete it:

```bash
docker stop multimodal_container
docker rm multimodal_container
```

Run the following commands to start the container and verify that the container starts successfully:

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
    -itd multimodalsdk:26.1.0-cann9.1.0-torch_npu2.6.0.post5-910b-ubuntu22.04-py3.12-aarch64 bash

docker ps -a | grep multimodal_container
```

Enter the container and perform all subsequent operations inside the container.

```bash
docker exec -it multimodal_container bash
```

## Step 3: Loading the Environment

> [!NOTE]
>
> The `MULTIMODAL_SDK_HOME` environment variable indicates the Multimodal SDK installation path, and its default value is `/usr/local/multimodal`.

```bash
export MULTIMODAL_SDK_HOME="/usr/local/multimodal"
source ${MULTIMODAL_SDK_HOME}/script/set_env.sh
```

## Step 4: Running the Verification Script

If `/data/test.jpg` is not preinstalled in the container image, run the following command to download a test image into the container:

```bash
mkdir -p /data
wget --tries=3 --timeout=30 --waitretry=5 -O /data/test.jpg https://raw.atomgit.com/Ascend/MultimodalSDK/blobs/f1f648b7a8b8a67c7509b3425a89f743bbf59563/dog_1920_1080.jpg
```

> [!NOTE]
>
> The second parameter of `Image.open` is the decoding device string, and currently only `"cpu"` is supported. Operator interfaces such as `resize` use the `DeviceMode.CPU` enum value as their running mode.

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

If the following output is displayed, the verification is successful:

```text
resize output shape: (500, 500, 3)
```

## Step 5: Cleaning Up the Environment

After you complete the verification, exit the container. You are advised to clean up the container environment to release resources.

```bash
exit
docker stop multimodal_container
docker rm multimodal_container
```

## Step 6: Next Steps

| Goal | Document |
| -- | -- |
| Image resize/crop visualization example | [Examples and Guide > Image Processing](../04_user_guide/user_guide.md#image-processing) |
| Video frame decoding | [Examples and Guide > Video Processing](../04_user_guide/user_guide.md#video-processing) |
| Audio loading | [Examples and Guide > Audio Processing](../04_user_guide/user_guide.md#audio-processing) |
| Integration with the vLLM inference framework | [patcher](../05_api/patcher.md) |
| Complete API reference | [Function Reference](../05_api/function_reference.md) |

# Common Issues Quick Reference

| Symptom | Handling |
| -- | -- |
| File permission error | Ensure that the permissions of the image do not exceed 640: `chmod 640 "$TEST_IMAGE"`. |
| Test image not found in the container | Confirm that the image version in use has `/data/test.jpg` built in and that `TEST_IMAGE` uses the container path `/data/test.jpg`. |
| The container cannot access the NPU | Check the NPU driver mounts and the device numbers in `--device /dev/davinci*`. |
| Failed to import `mm` | Confirm that you have run `source ${MULTIMODAL_SDK_HOME}/script/set_env.sh`. |
| More issues | [FAQ](../06_references/faq.md) and [Appendix](../06_references/appendix.md) |
