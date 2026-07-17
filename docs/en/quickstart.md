# Quick Start

Multimodal SDK provides multimodal preprocessing acceleration capabilities, including image decoding, resize/crop, video frame decoding, and audio loading. This document will help you start the environment through Docker and run your first Python example.

For native installation on the host, please refer to [Installation Guide](./installation_guide.md).

## Prerequisites

Before you begin, please confirm:

- **Hardware**: Atlas 800I A2 inference server, see [Introduction - Supported Hardware](./introduction.md#supported-hardware-and-oss)
- **Docker**: Docker is installed and the current user can run containers
- **Test Image**: Prepare a jpg/jpeg image with file permissions no higher than 640 (`chmod 640`)

## Step 1: Pull the Image

1. **Select Matching Version**
   - Visit the Ascend Community [Image Repository](https://www.hiascend.com/developer/ascendhub/detail/e0081aa3c4dd441dbd6a379bee8cc4c9)
   - Select the corresponding image version based on your current hardware model (e.g., Atlas 800I A2 inference server)
   - Note the distinction between CPU architecture (x86_64 / aarch64) and Ascend chip model (Ascend 310 / 910, etc.)

2. **Environment Pre-check**
   - Use the `npu-smi info` command to verify NPU driver status
   - Check driver version compatibility with the image CANN version (refer to [Firmware and Driver documentation](https://www.hiascend.com/hardware/firmware-drivers/community))

3. **Image Pull Example**

   The image Tag format is `{version}-{chip}-{os}-{python}-{arch}`, with the following variables:

   | Variable | Description | Example |
   |------|------|--------|
   | `{version}` | Multimodal SDK version | `26.0.0` |
   | `{chip}` | Ascend chip series | `910b` |
   | `{os}` | Base operating system | `ubuntu22.04` / `openeuler24.03` |
   | `{python}` | Python version | `py3.11` |
   | `{arch}` | CPU architecture | `aarch64` / `x86_64` |

   ```bash
   TAG={version}-{chip}-{os}-{python}-{arch}
   docker pull swr.cn-south-1.myhuaweicloud.com/ascendhub/multimodalsdk:${TAG}
   docker tag swr.cn-south-1.myhuaweicloud.com/ascendhub/multimodalsdk:${TAG} \
       multimodalsdk:${TAG}
   ```

   Using version 26.0.0, 910b chip, Ubuntu 22.04, Python 3.11, aarch64 architecture as an example:

   ```bash
   docker pull swr.cn-south-1.myhuaweicloud.com/ascendhub/multimodalsdk:26.0.0-910b-ubuntu22.04-py3.11-aarch64
   docker tag swr.cn-south-1.myhuaweicloud.com/ascendhub/multimodalsdk:26.0.0-910b-ubuntu22.04-py3.11-aarch64 \
       multimodalsdk:26.0.0-910b-ubuntu22.04-py3.11-aarch64
   ```

## Step 2: Start the Container

> [!NOTE]
>
> - The device number in `--device /dev/davinci0` needs to be adjusted according to the actual NPU number on the host (e.g., `davinci1`).
> - `-v /path/to/testdata:/data` mounts the host test image directory into the container for Step 4 to read.

Replace `/path/to/testdata` with the directory on the host that stores test images (must contain at least one jpg/jpeg file):

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
    -v /path/to/testdata:/data \
    -itd multimodalsdk:26.0.0-910b-ubuntu22.04-py3.11-aarch64 bash
```

## Step 3: Enter the Container and Load the Environment

```bash
docker exec -it multimodal_container bash
source ${MULTIMODAL_SDK_HOME}/script/set_env.sh
```

## Step 4: Run the Verification Script

Use an image in the mounted path inside the container (example: `/data/test.jpg`), then execute:

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

### Verification Successful

If the following output is displayed, the SDK is ready:

```text
resize output shape: (500, 500, 3)
```

## Next Steps

| Goal | Document |
| -- | -- |
| Image resize/crop visualization examples | [Examples and Guidance - Image Processing](./user_guide.md#image-processing) |
| Video frame decoding | [Examples and Guidance - Video Processing](./user_guide.md#video-processing) |
| Audio loading | [Examples and Guidance - Audio Processing](./user_guide.md#audio-processing) |
| Qwen2VL / InternVL2 preprocessing acceleration | [Adapter](./api/adapter.md) |
| vLLM inference framework integration | [patcher](./api/patcher.md) |
| Complete API reference | [Function Reference](./api/function_reference.md) |

## Quick Troubleshooting

| Symptom | Solution |
| -- | -- |
| File permission error | Ensure image permissions are no higher than 640: `chmod 640 "$TEST_IMAGE"` |
| Test image not found in container | Confirm Step 2 mounted the host directory, and `TEST_IMAGE` uses the container path (e.g., `/data/test.jpg`) |
| Container cannot access NPU | Check NPU driver mounting and `--device /dev/davinci*` device number |
| Failed to import `mm` | Confirm `source ${MULTIMODAL_SDK_HOME}/script/set_env.sh` was executed |
| More issues |[Appendix](./appendix.md) |
