# 安装部署

Multimodal SDK 提供宿主机原生安装方式。完整安装（含 NPU 驱动、CANN 及 SDK）通常需要 **1–2 小时**，具体取决于网络与硬件环境。

## 适用读者

| 场景                  | 推荐路径                              |
| ------------------- | --------------------------------- |
| 首次体验 / 开发调试（约 5 分钟） | [快速入门 - Docker](./quickstart.md) |
| 生产环境宿主机原生部署         | 本文                                |
| 版本兼容性查询             | [版本配套说明](./release_notes.md)      |

## 版本配套

安装前请确认各组件版本相互配套：

| 产品名称           | 版本       |
| -------------- | -------- |
| Multimodal SDK | 26.0.0   |
| Ascend HDK     | 26.0.RC1 |
| CANN           | 9.0.0    |

更多兼容性说明请参阅 [版本配套说明](./release_notes.md)。

## 硬件与操作系统要求

| 项目   | 要求                                                                       |
| ---- | ------------------------------------------------------------------------ |
| 硬件   | Atlas 800I A2 推理服务器（ARM64），详见 [简介 - 支持的硬件](./introduction.md#支持的硬件和操作系统) |
| 操作系统 | Ubuntu 22.04（aarch64）                                                    |
| 安装用户 | 建议使用**同一普通用户**完成 CANN、NPU 驱动固件及 Multimodal SDK 的安装与运行                    |

## 安装前检查清单

开始安装前，请逐项确认：

- 硬件为 Atlas 800I A2，操作系统为 Ubuntu 22.04 aarch64
- 已安装 Ascend HDK **26.0.RC1** 及 CANN **9.0.0**（或兼容补丁版本）
- 已执行 CANN 环境变量脚本，`npu-smi info` 可正常回显
- Python ≥ 3.9（推荐 3.10 或 3.11）、GCC ≥ 9.4、CMake ≥ 3.14 已就绪
- 表 3 所列 Python 依赖已安装（使用 [patcher](./api/patcher.md) 镜像时可跳过，见下文决策说明）
- Multimodal SDK 安装包已下载并通过 `--check` 校验

---

## 1. 获取安装包

从 [Multimodal SDK Releases](https://gitcode.com/Ascend/MultimodalSDK/releases) 页面获取与当前版本配套的安装包，或使用以下命令下载：

```bash
wget https://gitcode.com/Ascend/MultimodalSDK/releases/download/v26.0.0/Ascend-mindxsdk-multimodal_26.0.0_linux-aarch64.run
```

后续步骤以该文件名为例，请根据实际版本替换：

```bash
export MMSDK_PACKAGE=Ascend-mindxsdk-multimodal_26.0.0_linux-aarch64.run
```

> [!NOTE] 说明
> 文件名格式为 `Ascend-mindxsdk-multimodal_{version}_linux-{arch}.run`，其中 `{version}` 为 SDK 版本号，`{arch}` 为 CPU 架构（如 `aarch64`）。

---

## 2. 安装依赖

### 2.1 安装 Ubuntu 系统依赖

Multimodal SDK 在 Ubuntu 环境中依赖的软件及版本要求见[表 1](#table-ubuntu-deps)。

<a id="table-ubuntu-deps"></a>

**表 1** Ubuntu 系统依赖名称对应版本

| 依赖名称   | 版本要求                      | 安装方式                                      |
| ------ | ------------------------- | ----------------------------------------- |
| Python | 最低 3.9；**推荐 3.10 或 3.11** | 包管理器安装（`python3`）；若系统自带版本过低，请从源码编译或安装更高版本 |
| CMake  | 3.14 及以上                  | 包管理器安装                                    |
| Make   | 4.1 及以上                   | 包管理器安装（随 `build-essential` 安装）            |
| GCC    | 9.4 及以上                   | 包管理器安装（随 `build-essential` 安装）            |

**安装命令**

以 Ubuntu 22.04 为例，执行以下命令安装 GCC、Make、CMake 及 Python：

```bash
sudo apt update
sudo apt install -y build-essential cmake python3 python3-pip python3-dev
```

**校验安装**

```bash
gcc --version
make --version
cmake --version
python3 --version
```

若分别返回如下信息，说明相应软件已安装（以下回显仅为示例，请以实际情况为准）：

```text
gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
GNU Make 4.1
cmake version 3.14.2
Python 3.10.12
```

### 2.2 安装 NPU 驱动固件和 CANN

**下载依赖软件包**

<a id="table-npu-cann-deps"></a>
**表 2** NPU/CANN 软件包清单

| 软件类型      | 软件包名称                                                         | 版本要求                    |
| --------- | ------------------------------------------------------------- | ----------------------- |
| 昇腾 NPU 驱动 | `Ascend-hdk-{npu_type}-npu-driver_{version}_linux-{arch}.run` | Ascend HDK **26.0.RC1** |
| 昇腾 NPU 固件 | `Ascend-hdk-{npu_type}-npu-firmware_{version}.run`            | Ascend HDK **26.0.RC1** |
| CANN 软件包  | `Ascend-cann-toolkit_{version}_linux-{arch}.run`              | CANN **9.0.0**          |

> [!NOTE] 说明
>
> - `{npu_type}` 表示芯片名称。
> - `{version}` 表示软件版本号。
> - `{arch}` 表示 CPU 架构。

请根据您使用的 CANN 版本类型，从对应下载入口获取[表 2](#table-npu-cann-deps)所列软件包：

- [昇腾商用版资源下载](https://www.hiascend.com/developer/download/commercial/result?module=cann)
- [昇腾社区版资源下载](https://www.hiascend.com/developer/download/community/result)

**安装步骤**

请根据您使用的 CANN 版本类型，参考对应官方文档完成安装：

| 版本类型 | 安装 NPU 驱动和固件 | 安装 CANN | 快速安装 FAQ |
| ---- | [安装 NPU 驱动和固件（商用版 9.0.0）](https://www.hiascend.com/document/detail/zh/canncommercial/900/softwareinst/instg/instg_0005.html) | [CANN 软件安装指南（商用版 9.0.0）](https://www.hiascend.com/document/detail/zh/canncommercial/900/softwareinst/instg/instg_0000.html) | [快速安装 FAQ（商用版 9.0.0）](https://www.hiascend.com/document/detail/zh/canncommercial/900/softwareinst/instg/instg_0050.html) |
| 社区版 | [安装 NPU 驱动和固件（社区版 9.0.0）](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/softwareinst/instg/instg_0005.html) | [CANN 软件安装指南（社区版 9.0.0）](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/softwareinst/instg/instg_0000.html) | [快速安装 FAQ（社区版 9.0.0）](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/softwareinst/instg/instg_0050.html) |

> [!NOTE] 说明
>
> - 安装 CANN 时，为确保 Multimodal SDK 正常使用，CANN 的相关依赖也需要一并安装。
> - CANN 版本兼容性请参阅 [CANN 兼容性矩阵](https://www.hiascend.com/document)。

**安装 CANN 后验证环境**

完成 NPU 驱动固件及 CANN 安装后，执行以下命令确认环境就绪：

```bash
# 加载 CANN 环境变量（路径以实际安装为准）
source /usr/local/Ascend/cann/set_env.sh

# 确认 NPU 驱动已加载
npu-smi info

# 确认 CANN 环境变量已生效
echo $ASCEND_HOME
```

若 `npu-smi info` 能正常回显芯片信息，且 `ASCEND_HOME` 非空，说明 NPU 与 CANN 环境已就绪。

### 2.3 安装 Python 软件包依赖

**是否需要安装表 3 依赖？**

```text
是否使用 vLLM + patcher？
├─ 是 → 使用 patcher 文档中的镜像，跳过表 3
└─ 否 → 必须安装表 3 中的全部依赖
```

<a id="table-python-deps"></a>

**表 3** Python 依赖名称对应版本

| 依赖名称         | 版本建议   | 说明                 |
| ------------ | ------ | ------------------ |
| transformers | 4.51.3 | 通过 pip 安装          |
| pillow       | 11.2.1 | 通过 pip 安装          |
| numpy        | 1.26.4 | 通过 pip 安装          |
| torch        | 2.5.1  | 见下方昇腾 aarch64 安装说明 |

> [!NOTE] 说明
> Multimodal SDK 安装包（`.run`）会自动安装 `mm` Python 包（whl），**不会**自动安装表 3 中的第三方依赖，请用户提前安装。

**安装命令**

```bash
pip3 install transformers==4.51.3 pillow==11.2.1 numpy==1.26.4
```

**torch 安装说明（aarch64 + 昇腾）**

在 ARM64 架构的昇腾环境中，`torch` 不能直接使用默认 pip 源安装，需按 CANN 配套要求安装 CPU 版 PyTorch 及对应插件。请参考以下文档，选择与 CANN 9.0.0 配套的 torch / torch_npu 版本：

- [Ascend PyTorch 版本配套表（GitHub）](https://github.com/Ascend/pytorch#version-branch-mapping)
- [CANN 商用版 9.0.0 - 安装 PyTorch 前必读](https://www.hiascend.com/document/detail/zh/canncommercial/900/envdeployment/instg/instg_0046.html)

以 Python 3.10 + torch 2.5.1 为例（具体 wheel 包名请以配套表为准）：

```bash
# 1. 安装 CPU 版 PyTorch（aarch64 wheel，版本需与配套表一致）
pip3 install torch==2.5.1

# 2. 安装与 CANN 9.0.0 配套的 torch_npu 插件
pip3 install torch-npu==2.5.1.post1
```

安装完成后可验证：

```bash
python3 -c "import torch; print('torch:', torch.__version__)"
```

**安全建议**

- 建议使用 `pip install package==version` 固定依赖版本，避免意外升级引入兼容性问题。
- 定期关注依赖库的安全公告，必要时执行 `pip audit`（需 pip ≥ 23.0）检查已知漏洞。

---

## 3. 安装 Multimodal SDK

**安装须知**

- 建议使用**同一普通用户**完成 CANN、NPU 驱动固件及 Multimodal SDK 的安装与运行。
- 软件包的安装、升级、卸载及版本查询相关的日志会保存至 `~/log/mindxsdk/deployment.log`；完整性校验、提取文件、tar 命令访问相关的日志会保存至 `~/log/makeself/makeself.log`。用户可查看相应文件，完成后续的日志跟踪及审计。

**安装准备**

确保安装环境中已执行 CANN 环境变量配置脚本：

```bash
source /usr/local/Ascend/cann/set_env.sh   # 默认路径，请按实际安装路径修改
```

**安装步骤**

1. 以软件包的安装用户登录安装环境。
2. 将 Multimodal SDK 软件包上传到安装环境的任意路径下并进入软件包所在路径。
3. 增加对软件包的可执行权限：

   ```bash
   chmod u+x ${MMSDK_PACKAGE}
   ```

4. 执行如下命令，校验软件包的一致性和完整性：

   ```bash
    ./${MMSDK_PACKAGE} --check
   ```

   若系统缺少 `sha256sum` 工具会导致校验失败，可执行以下命令安装：

   ```bash
   sudo apt install -y coreutils
   ```

   若显示 `Verifying archive integrity... OK`，说明软件包已通过校验。

5. 创建 Multimodal SDK 软件包的安装路径（可选）。

   > [!NOTE] 说明
   > 不建议在 `/tmp` 路径下安装：系统重启后 `/tmp` 内容可能被清除，且该目录权限与空间不稳定，不适合持久化部署。

   - 若用户未指定安装路径，软件会默认安装到 Multimodal SDK 软件包所在的路径。
   - 若用户想指定安装路径，需要先创建安装路径。以安装路径 `/home/work/Mind_SDK` 为例：

   ```bash
   mkdir -p /home/work/Mind_SDK
   ```

6. 进入 Multimodal SDK 软件包所在路径，参考以下命令安装 Multimodal SDK（安装路径约束请参见[附录 - 安装命令参数表](#table-install-params)）。

   **指定安装路径**（以 `/home/work/Mind_SDK` 为例）：

   ```bash
   ./${MMSDK_PACKAGE} --install --install-path=/home/work/Mind_SDK
   ```

      无人值守安装（自动确认交互提示，适用于脚本/CI 场景）：

   ```bash
   echo y | ./${MMSDK_PACKAGE} --install --install-path=/home/work/Mind_SDK
   ```

      **默认安装到当前路径**：

   ```bash
   ./${MMSDK_PACKAGE} --install
   ```

   无人值守安装：

   ```bash
   echo y | ./${MMSDK_PACKAGE} --install
   ```

   常用可选参数：`--quiet`（静默模式，需与 `--install` 配合）、`--version`（查询版本）。完整参数说明见[附录](#table-install-params)。

---

## 4. 配置环境变量

安装完成后，加载 Multimodal SDK 环境变量：

```bash
source ${MULTIMODAL_SDK_HOME}/script/set_env.sh
```

若 `${MULTIMODAL_SDK_HOME}` 未设置，请先进入安装目录下的 `script` 目录再执行：

```bash
# 示例：安装路径为 /home/work/Mind_SDK
source /home/work/Mind_SDK/multimodal/script/set_env.sh
```

若需每次登录自动生效，可将上述命令追加至 `~/.bashrc`：

```bash
echo 'source ${MULTIMODAL_SDK_HOME}/script/set_env.sh' >> ~/.bashrc
source ~/.bashrc
```

环境变量说明请参阅 [附录 - 环境变量说明](./appendix.md#环境变量说明)。

---

## 5. 安装验证

确认环境变量已加载后，执行以下验证：

```bash
echo "MULTIMODAL_SDK_HOME=${MULTIMODAL_SDK_HOME}"
python3 -c "import mm; print('mm import: OK')"
```

使用一张 jpg/jpeg 测试图片进行功能验证（将 `$TEST_IMAGE` 替换为实际路径）：

```bash
export TEST_IMAGE="/path/to/your/test.jpg"
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

**验证成功**

若输出以下结果，说明 SDK 已就绪：

```text
resize output shape: (500, 500, 3)
```

更多样例请参阅 [快速入门](./quickstart.md) 与 [样例和指导](./user_guide.md)。

---

## 6. 升级

升级前请确认当前已存在目录完整的 Multimodal SDK 安装，且新版本与 CANN / HDK 版本配套。

1. 下载新版本安装包并进入软件包所在目录。
2. 加载 CANN 及 Multimodal SDK 环境变量。
3. 执行升级命令（`--install-path` 需指向**旧版本的安装根目录**）：

   ```bash
   ./Ascend-mindxsdk-multimodal_{version}_linux-{arch}.run --upgrade --install-path=/home/work/Mind_SDK
   ```

   无人值守升级：

   ```bash
   echo y | ./Ascend-mindxsdk-multimodal_{version}_linux-{arch}.run --upgrade --install-path=/home/work/Mind_SDK
   ```

4. 重新加载环境变量并执行[安装验证](#5-安装验证)中的命令。

> [!NOTE] 说明
> 升级过程中安装程序会先卸载旧版本再安装新版本。若升级失败，请查看 `~/log/mindxsdk/deployment.log` 定位原因。

---

## 7. 卸载

Multimodal SDK 默认安装目录结构为 `{install-path}/multimodal/`，卸载脚本位于 `{install-path}/multimodal/script/uninstall.sh`。

**操作步骤**

1. 进入 Multimodal SDK 安装路径下的 `script` 目录，确认 `uninstall.sh` 是否有可执行权限：

   ```bash
   cd ${MULTIMODAL_SDK_HOME}/script
   ls -l uninstall.sh
   ```

      若脚本没有可执行权限：

   ```bash
   chmod u+x uninstall.sh
   ```

2. 执行卸载脚本。脚本会卸载已安装的 Python whl 包并删除安装目录：

   ```bash
   ./uninstall.sh
   ```

**卸载验证**

```bash
# 确认安装目录已删除
ls ${MULTIMODAL_SDK_HOME} 2>/dev/null && echo "目录仍存在" || echo "目录已删除"

# 确认 Python 包已卸载
pip3 show mm 2>/dev/null && echo "mm 仍存在" || echo "mm 已卸载"
```

> [!NOTE] 说明
> `uninstall.sh` 仅适用于正常安装途径且安装后未修改目录结构的情况。若安装异常，请手动删除安装目录下所有 `multimodal` 相关文件夹，并执行 `pip uninstall mm -y` 卸载 Python 包。

---

## 常见问题

| 现象                        | 处理方式                                                        |
| ------------------------- | ----------------------------------------------------------- |
| 导入 `mm` 失败                | 确认已执行 `source ${MULTIMODAL_SDK_HOME}/script/set_env.sh`     |
| `npu-smi info` 无输出        | 检查 NPU 驱动/固件是否安装成功，必要时重启后重试                                 |
| CANN 环境变量未生效              | 确认已 source CANN 的 `set_env.sh`，路径以实际安装为准                    |
| torch / transformers 版本冲突 | 对照[表 3](#table-python-deps)固定版本安装                           |
| 更多问题                      | [FAQ - 安装与环境](./faq.md#安装与环境)、[附录 - 错误码](./appendix.md#错误码) |

---

## 附录：安装命令参数表

<a id="table-install-params"></a>

**表 4** `--install` 安装命令可选参数表

| 输入参数 | 含义 |
| --- | --- |
| `--help`，`-h` | 查询帮助信息 |
| `--info` | 查询包构建信息 |
| `--list` | 查询文件列表 |
| `--check` | 查询包完整性 |
| `--quiet`，`-q` | 启用静默模式。需和 `--install` 或 `--upgrade` 配合使用 |
| `--noexec` | 不运行嵌入的脚本 |
| `--extract=` | 直接提取到目标目录（绝对路径或相对路径）。通常与 `--noexec` 一起使用，仅提取文件而不运行 |
| `--tar arg1 [arg2 ...]` | 通过 tar 命令访问归档文件的内容 |
| `--install` | Multimodal SDK 安装操作。约束：当前路径和安装路径不能存在非法字符，仅支持大小写字母、数字、`-_./`；安装路径下不能存在名为 `multimodal` 的文件或文件夹；若存在名为 `multimodal` 的软链接，则会被覆盖 |
| `--install-path=` | （可选）自定义安装根目录。未设置时默认为当前命令执行所在目录。建议使用绝对路径；需和 `--install` 或 `--upgrade` 配合使用；与 `--upgrade` 配合时代表旧软件包的安装目录 |
| `--upgrade` | Multimodal SDK 升级操作。需确保已存在目录完整的旧版本安装 |
| `--version` | 查询软件包版本 |

内部参数（请勿直接使用）

以下参数未展示在 `--help` 中：

- `--nox11`：废弃接口，无实际作用
- `--xwin`：使用 xwin 模式运行
- `--phase2`：要求执行第二步动作
