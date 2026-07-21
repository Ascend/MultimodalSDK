# 安装部署

## 安装说明

本文档适用于 Multimodal SDK 最新版本的安装，仅支持硬件为 Atlas 800I A2 推理服务器，操作系统为 Ubuntu 22.04 或 openEuler 24.03，建议预留不少于 16GB 可用磁盘空间和不少于 8GB 可用内存。Multimodal SDK 当前为正式支持版本，搭配 9.1.0 版本的 CANN 和相关配套。

Multimodal SDK 支持[离线安装](#离线安装)（`run` 包 / `Wheel` 包）、[源码安装](#源码安装)、[镜像安装](#镜像安装)三种方式。`run` 包：自解压安装脚本，含完整依赖；`Wheel` 包：Python 二进制分发包；镜像安装：基于容器化镜像部署。

若采用离线安装或源码安装，请首先[安装相关依赖](#安装依赖说明)，若采用镜像安装请跳过该步骤。

**注意事项**

如需安装 Multimodal SDK 软件包以外的第三方软件，请注意及时升级最新版本，关注并修补存在的漏洞。

## 安装依赖说明

### 安装 NPU 驱动固件和 CANN

请参考 [CANN 安装指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/softwareinst/instg/instg_0000.html)，使用 CANN（Compute Architecture for Neural Networks）9.1.0 及 HDK（Hardware Development Kit）26.1.0 完成 NPU（Neural Processing Unit）驱动固件和 CANN 的安装。

### 其他依赖

| 依赖名称       | 版本建议                     | 获取建议                                                                 |
| ------------ | ------------------------- | ---------------------------------------------------------------------- |
| CMake        | 3.14 及以上                  | 建议通过包管理器安装：<br>Ubuntu：`sudo apt-get install -y cmake`<br>openEuler：`sudo yum install -y cmake`<br>若版本不符合最低要求，可通过源码编译安装 |
| Make         | 4.1 及以上                   | 建议通过包管理器安装：<br>Ubuntu：`sudo apt-get install -y make`<br>openEuler：`sudo yum install -y make`<br>若版本不符合最低要求，可通过源码编译安装 |
| GCC          | 9.4 及以上                   | 建议通过包管理器安装：<br>Ubuntu：`sudo apt-get install -y build-essential`<br>openEuler：`sudo yum install -y gcc gcc-c++` |
| Python       | 最低 3.10；**推荐 3.11** | 建议通过包管理器安装：<br>Ubuntu：`sudo apt-get install -y python3 python3-pip python3-dev`<br>openEuler：`sudo yum install -y python3 python3-pip python3-devel`<br>若系统自带版本过低，可从源码编译或安装更高版本 |
| SWIG         | 4.3 及以上                  | 建议通过源码安装 |
| transformers | 4.51.3                    | 通过 pip 安装：<br>`pip3 install transformers==4.51.3`                       |
| einops       | 0.8.2                     | 通过 pip 安装：<br>`pip3 install einops==0.8.2`                            |
| pillow       | 11.2.1 及以上               | 通过 pip 安装：<br>`pip3 install pillow==11.2.1`                           |
| numpy        | 1.26.4                    | 通过 pip 安装：<br>`pip3 install numpy==1.26.4`                            |
| torch        | 2.5.1                     | 通过 pip 安装：<br>`pip3 install torch==2.5.1` |
| torch-npu    | 2.5.1.post1               | 通过 pip 安装：<br>`pip3 install torch-npu==2.5.1.post1`<br>需与 `vllm-ascend==v0.8.5rc1` 配套 |

## 安装方式

Multimodal SDK 提供三种安装方式：离线安装（`run` 包 / `Wheel` 包）、源码安装和镜像安装，可根据场景选择合适的方式。

### 离线安装

#### 方式一：`run` 包安装

**前提条件**

- 完成[安装相关依赖](#安装依赖说明)。

**安装须知**

- 建议使用**同一普通用户**完成 CANN、NPU 驱动固件及 Multimodal SDK 的安装与运行。
- 软件包的安装、升级、卸载及版本查询相关的日志会保存至 `~/log/mindxsdk/deployment.log`；完整性校验、提取文件、tar 命令访问相关的日志会保存至 `~/log/makeself/makeself.log`。用户可查看相应文件，完成后续的日志跟踪及审计。

**安装准备**

请从 [Multimodal SDK Releases](https://gitcode.com/Ascend/MultimodalSDK/releases) 下载 Multimodal SDK 软件包（Ascend-mindxsdk-multimodal_{version}_linux-{arch}.run）。其中 `{version}` 为 SDK 版本号（例如 `26.1.0`），`{arch}` 为 CPU 架构（`x86_64` 或 `aarch64`）。

以下命令统一使用 `${MMSDK_PACKAGE}` 表示已下载的 `.run` 包文件名。请按实际文件名设置，例如：

```bash
export MMSDK_PACKAGE=Ascend-mindxsdk-multimodal_26.1.0_linux-aarch64.run
```

确保安装环境中已执行 CANN 环境变量配置脚本：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh   # 默认路径，请根据实际安装路径修改
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
   sudo apt-get install -y coreutils
   ```

   若显示 `Verifying archive integrity... OK`，说明软件包已通过校验。

5. 创建 Multimodal SDK 软件包的安装路径（可选）。

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

   **默认安装到当前路径**：

   ```bash
   ./${MMSDK_PACKAGE} --install
   ```

   常用可选参数：`--quiet`（静默模式，需与 `--install` 配合）、`--version`（查询版本）。完整参数说明见[安装命令参数表](#table-install-params)。

<a id="安装验证"></a>

**安装验证**

`run` 完整安装完成后，加载 Multimodal SDK 环境变量：

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

环境变量说明请参阅 [附录 - 环境变量说明](../06_references/appendix.md#环境变量说明)。

执行以下验证：

```bash
# .run 安装时可查看安装路径；Wheel 安装可跳过此行
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

若输出以下结果，说明 SDK 已就绪：

```text
resize output shape: (500, 500, 3)
```

更多样例请参阅 [快速入门](../02_quickstart/quickstart.md) 与 [样例和指导](../04_user_guide/user_guide.md)。

**安装命令参数表**

<a id="table-install-params"></a>

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
| `--install` | Multimodal SDK 安装操作。约束：当前路径和安装路径不能存在非法字符，仅支持大小写字母、数字、`-`、`_`、`.`、`/`；安装路径下不能存在名为 `multimodal` 的文件或文件夹；若存在名为 `multimodal` 的软链接，则会被覆盖 |
| `--install-path=` | （可选）自定义安装根目录。未设置时默认为当前命令执行所在目录。建议使用绝对路径；需和 `--install` 或 `--upgrade` 配合使用；与 `--upgrade` 配合时代表旧软件包的安装目录 |
| `--upgrade` | Multimodal SDK 升级操作。需确保已存在目录完整的旧版本安装 |
| `--version` | 查询软件包版本 |

**内部参数**

> [!WARNING] 注意
> 以下参数供内部使用，请勿直接调用。

以下参数未展示在 `--help` 中：

- `--nox11`：废弃接口，无实际作用
- `--xwin`：使用 xwin 模式运行
- `--phase2`：要求执行第二步动作

#### 方式二：`Wheel` 包安装

**前提条件**

- 完成[安装相关依赖](#安装依赖说明)。

**安装须知**

若仅需在 Python 环境中使用 `mm` 包，可跳过 `run` 安装包，直接安装 `Wheel` 包。`Wheel` 内已捆绑 `libcore.so` 及 FFmpeg、libjpeg-turbo、soxr 等原生依赖，**无需**执行 `source ${MULTIMODAL_SDK_HOME}/script/set_env.sh` 或配置 `MULTIMODAL_SDK_HOME`。

> [!NOTE] 说明
>
> - `Wheel` 包**不包含** CANN（`libascendcl.so` 等）及 Python 第三方依赖，使用前须按 [安装依赖说明](#安装依赖说明) 完成安装。
> - 每次启动 Python 前，须加载 CANN 环境变量（路径以实际安装为准）：
>
>   ```bash
>   source /usr/local/Ascend/ascend-toolkit/set_env.sh
>   ```

**安装准备**

可通过以下任一方式获取与 SDK 版本配套的 `mm-*.whl`：

1. **从 Release 页面下载**：在 [Multimodal SDK Releases](https://gitcode.com/Ascend/MultimodalSDK/releases) 获取与 `run` 包同版本的 `Wheel` 文件（若已随发布提供）。
2. **从 `run` 包中提取**：

   ```bash
   ./${MMSDK_PACKAGE} --noexec --extract=/tmp/mmsdk_extract
   find /tmp/mmsdk_extract -name 'mm-*.whl'
   ```

3. **自行构建**：按 [CONTRIBUTING.md](../../../CONTRIBUTING.md) 完成源码构建后，在 `MultimodalSDK/dist/` 目录获取生成的 `mm-*.whl`。

**安装步骤**

1. 确认 CANN 环境变量已加载（见上方说明）。
2. 确认[安装依赖说明](#安装依赖说明)所列 Python 依赖已安装。
3. 安装 `Wheel` 包（将 `mm-1.0.0-py3-none-any.whl` 替换为实际文件名）：

   ```bash
   pip3 install /path/to/mm-1.0.0-py3-none-any.whl
   ```

   若环境中已存在旧版本，可强制重装：

   ```bash
   pip3 install --force-reinstall --no-deps /path/to/mm-1.0.0-py3-none-any.whl
   ```

   > [!NOTE] 说明
   > 建议使用 `--no-deps`，避免 pip 自动升级或降级成固定的依赖版本。

4. 参考[安装验证](#安装验证)的步骤进行验证，确认 SDK 已就绪（`Wheel` 包安装无需额外设置 `MULTIMODAL_SDK_HOME` 环境变量）。

### 源码安装

如需从源码构建 Multimodal SDK，请参考 [CONTRIBUTING.md](../../../CONTRIBUTING.md) 完成编译环境准备与源码构建。构建完成后：

- 若生成 `mm-*.whl`，可按 [离线安装：`Wheel` 包安装](#方式二wheel-包安装) 中的步骤完成安装。
- 若生成 `run` 安装包，可按 [离线安装：`run` 包安装](#方式一run-包安装) 中的步骤完成安装。

### 镜像安装

Multimodal SDK 支持容器化部署，可通过以下两种方式获取镜像：

**方式一：拉取官方镜像**

从昇腾社区镜像仓库直接拉取预构建镜像，详细请参阅 [Multimodal SDK 镜像仓库](https://www.hiascend.com/developer/ascendhub/detail/e0081aa3c4dd441dbd6a379bee8cc4c9)。

**方式二：本地构建镜像**

使用项目 `docker/` 目录提供的 Dockerfile 自行构建，详细构建说明请参阅 [docker/OVERVIEW.zh.md](../../../docker/OVERVIEW.zh.md)。

**启动容器**

获取镜像后，执行以下命令启动容器：

```bash
docker run -it \
    --name mmsdk_container \
    --device /dev/davinci0 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    ${镜像名}:${tag} bash
```

进入容器后可直接使用：

```bash
python3 -c "import mm; print('mm import: OK')"
```

> [!NOTE] 说明
>
> - 镜像内已预装 CANN、NPU 驱动及 Multimodal SDK，进入容器后无需额外配置环境变量。
> - 主机上需安装与容器内 CANN 版本兼容的 NPU 驱动（详见 [安装 NPU 驱动固件和 CANN](#安装-npu-驱动固件和-cann)）。

## 升级

### `run` 安装升级

升级前请确认当前已存在目录完整的 Multimodal SDK 安装，且新版本与 CANN / Ascend HDK 版本配套。

1. 下载新版本安装包并进入软件包所在目录。
2. 加载 CANN 及 Multimodal SDK 环境变量。
3. 执行升级命令（`--install-path` 需指向**旧版本的安装根目录**）：

   ```bash
   ./Ascend-mindxsdk-multimodal_{version}_linux-{arch}.run --upgrade --install-path=/home/work/Mind_SDK
   ```

4. 重新加载环境变量并执行[安装验证](#安装验证)中的命令。

> [!NOTE] 说明
> 升级过程中安装程序会先卸载旧版本再安装新版本。若升级失败，请查看 `~/log/mindxsdk/deployment.log` 定位原因。

### `Wheel` 包升级

1. 获取新版 `Wheel` 包（[Multimodal SDK Releases](https://gitcode.com/Ascend/MultimodalSDK/releases)）。
2. 加载 CANN 环境变量。
3. 执行升级命令（将 `mm-x.x.x-py3-none-any.whl` 替换为实际文件名）：

   ```bash
   pip3 install --upgrade /path/to/mm-x.x.x-py3-none-any.whl
   ```

   若需强制覆盖且不更新依赖：

   ```bash
   pip3 install --force-reinstall --no-deps /path/to/mm-x.x.x-py3-none-any.whl
   ```

4. 执行[安装验证](#安装验证)中的命令确认升级成功。

> [!NOTE] 说明
>
> - 建议使用 `--no-deps` 避免 pip 自动升级或降级已固定的依赖版本。
> - 若需回退，可使用 `pip3 install --force-reinstall /path/to/mm-旧版本.whl` 重新安装旧版本。

## 卸载

### `run` 安装卸载

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
> `uninstall.sh` 仅适用于正常安装途径且安装后未修改目录结构的情况。若安装异常，请手动删除安装目录下所有 `multimodal` 相关文件夹，并执行 `pip3 uninstall mm -y` 卸载 Python 包。

### `Wheel` 包安装卸载

```bash
pip3 uninstall -y mm
pip3 show mm 2>/dev/null && echo "mm 仍存在" || echo "mm 已卸载"
```

## 常见问题

| 现象                        | 处理方式                                                        |
| ------------------------- | ----------------------------------------------------------- |
| 导入 `mm` 失败（`run` 安装）      | 确认已执行 `source ${MULTIMODAL_SDK_HOME}/script/set_env.sh`     |
| 导入 `mm` 失败（`Wheel` 安装）       | 确认已 source CANN 环境变量；执行 `pip3 show mm` 确认 whl 已安装；必要时 `pip3 install --force-reinstall --no-deps mm-*.whl` 重装 |
| `libcore.so` / `libascendcl.so` 找不到 | `Wheel` 安装：确认使用配套版本 whl 并 source CANN；`.run` 安装：确认已 source `set_env.sh` |
| `npu-smi info` 无输出        | 检查 NPU 驱动/固件是否安装成功，必要时重启后重试                                 |
| CANN 环境变量未生效              | 确认已 source CANN 的 `set_env.sh`，路径以实际安装为准                    |
| torch / transformers 版本冲突 | 对照[表](#其他依赖)固定版本安装                           |
| 更多问题                      | [FAQ - 安装与环境](../06_references/faq.md#安装与环境) |
