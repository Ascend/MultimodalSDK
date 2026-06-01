# 快速入门

Multimodal SDK 提供多模态预处理加速能力，包括图像解码、resize/crop、视频帧解码与音频加载等。本文将帮助你在 **5 分钟内** 通过 Docker 启动环境并跑通第一个 Python 示例。

如需在宿主机原生安装，请参阅 [安装部署](./installation_guide.md)。

## 前置条件

开始之前，请确认：

- **硬件**：Atlas 800I A2 推理服务器（ARM64），详见 [简介 - 支持的硬件](./introduction.md#支持的硬件和操作系统)
- **驱动**：宿主机已安装与 CANN 9.0.0 兼容的 NPU 驱动，请参阅 [CANN 兼容性矩阵](https://www.hiascend.com/document)
- **Docker**：已安装 Docker，且当前用户可运行容器
- **测试图片**：准备一张 jpg/jpeg 图片，文件权限不高于 640（`chmod 640`）

## 步骤 1：拉取镜像

```bash
docker pull swr.cn-south-1.myhuaweicloud.com/ascendhub/multimodalsdk:26.0.0-910b-ubuntu22.04-py3.11-aarch64
docker tag swr.cn-south-1.myhuaweicloud.com/ascendhub/multimodalsdk:26.0.0-910b-ubuntu22.04-py3.11-aarch64 \
  multimodalsdk:26.0.0-910b-ubuntu22.04-py3.11-aarch64
```

## 步骤 2：启动容器

> [!NOTE] 说明
>
> - `--device /dev/davinci0` 中的设备编号需按宿主机实际 NPU 编号调整（如 `davinci1`）。
> - `-v /path/to/testdata:/data` 将宿主机测试图片目录挂载到容器内，便于步骤 4 读取。

将 `/path/to/testdata` 替换为宿主机上存放测试图片的目录（需包含至少一张 jpg/jpeg 文件）：

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

## 步骤 3：进入容器并加载环境

```bash
docker exec -it multimodal_container bash
source ${MULTIMODAL_SDK_HOME}/script/set_env.sh
```

## 步骤 4：运行验证脚本

使用容器内挂载路径下的图片（示例为 `/data/test.jpg`），然后执行：

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

### 验证成功

若输出以下结果，说明 SDK 已就绪：

```text
resize output shape: (500, 500, 3)
```

## 下一步

| 目标 | 文档 |
| -- | -- |
| 图像 resize/crop 可视化样例 | [样例和指导 - 图片处理](./user_guide.md#图片处理) |
| 视频帧解码 | [样例和指导 - 视频处理](./user_guide.md#视频处理) |
| 音频加载 | [样例和指导 - 音频处理](./user_guide.md#音频处理) |
| Qwen2VL / InternVL2 预处理加速 | [Adapter](./api/adapter.md) |
| vLLM 推理框架集成 | [patcher](./api/patcher.md) |
| API 完整参考 | [功能函数参考](./api/function_reference.md) |

## 常见问题速查

| 现象 | 处理方式 |
| -- | -- |
| 文件权限报错 | 确保图片权限不高于 640：`chmod 640 "$TEST_IMAGE"` |
| 容器内找不到测试图片 | 确认步骤 2 已挂载宿主机目录，且 `TEST_IMAGE` 使用容器内路径（如 `/data/test.jpg`） |
| 容器无法访问 NPU | 检查 NPU 驱动挂载与 `--device /dev/davinci*` 设备号 |
| 导入 `mm` 失败 | 确认已执行 `source ${MULTIMODAL_SDK_HOME}/script/set_env.sh` |
| 更多问题 | [FAQ](./faq.md)、[附录](./appendix.md) |
