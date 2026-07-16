# 快速入门

Multimodal SDK 提供多模态预处理加速能力，包括图像解码、resize/crop、视频帧解码与音频加载等。本文将帮助你通过 Docker 启动环境并跑通第一个 Python 示例。

如需在宿主机原生安装，请参阅 [安装部署](../03_installation_guide/installation_guide.md)。

## 前置条件

开始之前，请确认：

- **硬件**：Atlas 800I A2 推理服务器，详见 [简介 - 支持的硬件](../01_introduction/01_introduction.md#支持的硬件和操作系统)
- **Docker**：已安装 Docker，且当前用户可运行容器
- **测试图片**：镜像内已提供 `/data/test.jpg`，无需额外挂载测试图片目录。

## 步骤 1：拉取镜像

1. **选择匹配版本**
   - 访问昇腾社区[镜像仓](https://www.hiascend.com/developer/ascendhub/detail/e0081aa3c4dd441dbd6a379bee8cc4c9)
   - 根据当前硬件型号（Atlas 800I A2 推理服务器）选择对应的镜像版本
   - 注意当前仅支持 aarch64 CPU 架构

2. **环境预检查**
   - 使用命令验证 NPU 驱动状态

   ```bash
   npu-smi info
   ```

   - 检查驱动版本与镜像 CANN 版本匹配性（参考[《固件与驱动》文档](https://www.hiascend.com/hardware/firmware-drivers/community)）

3. **镜像拉取示例**

   镜像 Tag 格式为 `{version}-910b-{os}-{python}-aarch64`，各变量含义如下：

   | 变量 | 含义 | 示例值 |
   |------|------|--------|
   | `{version}` | Multimodal SDK 版本 | `26.0.0` |
   | `{os}` | 基础操作系统 | `ubuntu22.04` / `openeuler24.03` |
   | `{python}` | Python 版本 | `py3.11` |

   ```bash
   TAG={version}-910b-{os}-{python}-aarch64
   docker pull swr.cn-south-1.myhuaweicloud.com/ascendhub/multimodalsdk:${TAG}
   docker tag swr.cn-south-1.myhuaweicloud.com/ascendhub/multimodalsdk:${TAG} \
       multimodalsdk:${TAG}
   ```

   以 26.0.0 版本、Ubuntu 22.04、Python 3.11 为例：

   ```bash
   docker pull swr.cn-south-1.myhuaweicloud.com/ascendhub/multimodalsdk:26.0.0-910b-ubuntu22.04-py3.11-aarch64
   docker tag swr.cn-south-1.myhuaweicloud.com/ascendhub/multimodalsdk:26.0.0-910b-ubuntu22.04-py3.11-aarch64 \
       multimodalsdk:26.0.0-910b-ubuntu22.04-py3.11-aarch64
   ```

## 步骤 2：启动容器

> [!NOTE] 说明
>
> - `--device /dev/davinci0` 中的设备编号需按宿主机实际 NPU 编号调整（如 `davinci1`）。

先检查是否已经存在同名容器，若存在则先删除：

```bash
docker stop multimodal_container
docker rm multimodal_container
```

执行以下命令启动容器，并检查容器是否启动成功：

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
    -itd multimodalsdk:26.0.0-910b-ubuntu22.04-py3.11-aarch64 bash

docker ps -a | grep multimodal_container
```

进入容器，后续操作都在容器内进行。

```bash
docker exec -it multimodal_container bash
```

## 步骤 3：加载环境

> [!NOTE] 说明
>
> 使用 `MULTIMODAL_SDK_HOME` 环境变量表示 Multimodal SDK 安装路径，默认值为 `/usr/local/multimodal`。根据镜像版本，该路径可能不同。

```bash
export MULTIMODAL_SDK_HOME="/usr/local/multimodal"
source ${MULTIMODAL_SDK_HOME}/script/set_env.sh
```

## 步骤 4：运行验证脚本

如果镜像内未预置图片 `/data/test.jpg`，下载测试图片到容器内，下载命令如下：

```bash
mkdir /data
wget --tries=3 --timeout=30 --waitretry=5 -O /data/test.jpg https://raw.gitcode.com/Ascend/MultimodalSDK/blobs/f1f648b7a8b8a67c7509b3425a89f743bbf59563/dog_1920_1080.jpg
```

> [!NOTE] 说明
>
> `Image.open` 的第二个参数为解码设备字符串，当前仅支持 `"cpu"`；`resize` 等算子接口的运行模式使用 `DeviceMode.CPU` 枚举。

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

若输出以下结果，说明验证成功：

```text
resize output shape: (500, 500, 3)
```

## 步骤 5：清理环境

完成验证后，退出容器，建议清理容器环境，释放资源。

```bash
exit
docker stop multimodal_container
docker rm multimodal_container
```

## 步骤 6：下一步

| 目标 | 文档 |
| -- | -- |
| 图像 resize/crop 可视化样例 | [样例和指导 - 图片处理](../04_user_guide/user_guide.md#图片处理) |
| 视频帧解码 | [样例和指导 - 视频处理](../04_user_guide/user_guide.md#视频处理) |
| 音频加载 | [样例和指导 - 音频处理](../04_user_guide/user_guide.md#音频处理) |
| Qwen2VL / InternVL2 预处理加速 | [Adapter](../05_api/adapter.md) |
| vLLM 推理框架集成 | [patcher](../05_api/patcher.md) |
| API 完整参考 | [功能函数参考](../05_api/function_reference.md) |

# 常见问题速查

| 现象 | 处理方式 |
| -- | -- |
| 文件权限报错 | 确保图片权限不高于 640：`chmod 640 "$TEST_IMAGE"` |
| 容器内找不到测试图片 | 确认使用的镜像版本已内置 `/data/test.jpg`，且 `TEST_IMAGE` 使用容器内路径 `/data/test.jpg` |
| 容器无法访问 NPU | 检查 NPU 驱动挂载与 `--device /dev/davinci*` 设备号 |
| 导入 `mm` 失败 | 确认已执行 `source ${MULTIMODAL_SDK_HOME}/script/set_env.sh` |
| 更多问题 | [FAQ](../06_references/faq.md)、[附录](../06_references/appendix.md) |
