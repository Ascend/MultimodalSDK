# 安装部署

## 获取安装包

请参考本章获取所需软件包和对应的数字签名文件。

**表 1** 软件包

| 组件名称 | 软件包名称 |
| -- | -- |
| Multimodal SDK | `Ascend-mindxsdk-multimodal_{version}_linux-{arch}.run` |

**软件数字签名验证**

为了防止软件包在传递过程中或存储期间被恶意篡改，下载软件包时请下载对应的数字签名文件用于完整性验证。

在软件包下载之后，请参考《OpenPGP签名验证指南》，对下载的软件包进行 PGP 数字签名校验。如果校验失败，请勿使用该软件包并联系华为技术支持工程师解决。

使用软件包安装/升级前，也需要按照上述过程，验证软件包的数字签名，确保软件包未被篡改。

运营商客户请访问：[https://support.huawei.com/carrier/digitalSignatureAction](https://support.huawei.com/carrier/digitalSignatureAction)

企业客户请访问：[https://support.huawei.com/enterprise/zh/tool/software-digital-signature-openpgp-validation-tool-TL1000000054](https://support.huawei.com/enterprise/zh/tool/software-digital-signature-openpgp-validation-tool-TL1000000054)

## 安装依赖

### 安装 Ubuntu 系统依赖

Ubuntu 系统环境中所需依赖名称、对应版本及获取建议请参见[表 2](#table-ubuntu-deps)。

<a id="table-ubuntu-deps"></a>

**表 2** Ubuntu 系统依赖名称对应版本

| 依赖名称 | 版本建议 | 获取建议 |
| -- | -- | -- |
| Python | 3.9 及以上 | 建议通过获取源码包编译安装。 |
| CMake | 3.14 及以上 | 建议通过包管理模块安装。 |
| Make | 4.1 及以上 | 建议通过包管理模块安装。 |
| GCC | 9.4 及以上 | 建议通过包管理模块安装。 |

参考如下命令，检查是否已安装 GCC、Make、CMake 等依赖软件。

```bash
gcc --version
make --version
cmake --version
python3 --version
```

若分别返回如下信息，说明相应软件已安装（以下回显仅为示例，请以实际情况为准）。

```text
gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
GNU Make 4.1
cmake version 3.14.2
Python 3.10.12
```

### 安装 NPU 驱动固件和 CANN

**下载依赖软件包**

<a id="table-npu-cann"></a>

**表 3** NPU/CANN 软件包清单

| 软件类型 | 软件包名称 | 获取方式 |
| -- | -- | -- |
| 昇腾 NPU 驱动 | `Ascend-hdk-{npu_type}-npu-driver_{version}_linux-{arch}.run` | 单击[获取链接](https://www.hiascend.com/developer/download/commercial/result?module=cann)，在左侧配套资源的"编辑资源选择"中进行配置，筛选配套的软件包，确认版本信息后获取所需软件包。 |
| 昇腾 NPU 固件 | `Ascend-hdk-{npu_type}-npu-firmware_{version}.run` | 同上 |
| CANN 软件包 | `Ascend-cann-toolkit_{version}_linux-{arch}.run` | 同上 |

> [!NOTE] 说明
>
> - `{npu_type}` 表示芯片名称。
> - `{version}` 表示软件版本号。
> - `{arch}` 表示 CPU 架构。

**安装 NPU 驱动固件和 CANN**

1. 参考《CANN 软件安装指南》中的"安装 NPU 驱动和固件"章节（商用版）或"安装 NPU 驱动和固件"章节（社区版）安装 NPU 驱动固件。
2. 参考《CANN 软件安装指南》的"安装 CANN"章节（商用版）或《CANN 软件安装指南》的"安装 CANN"章节（社区版）安装 CANN。

    > [!NOTE] 说明
    > - 安装 CANN（Toolkit），NPU 驱动固件和安装 Multimodal SDK 的用户需为同一用户，建议为普通用户。
    > - 安装 CANN 时，为确保 Multimodal SDK 正常使用，CANN 的相关依赖也需要一并安装。

### 安装 Python 软件包依赖

使用 Multimodal SDK 相关功能还需安装[表 4](#table-python-deps)中的所有依赖。若使用 [patcher](./api/patcher.md) 中的 patcher 环节，请安装 [patcher](./api/patcher.md) 中的镜像，无需额外安装[表 4](#table-python-deps)中的依赖。

<a id="table-python-deps"></a>

**表 4** Python 依赖名称对应版本

| 依赖名称 | 版本建议 | 获取建议 |
| -- | -- | -- |
| transformers | 4.51.3 | 建议通过 pip 获取。 |
| pillow | 11.2.1 | 建议通过 pip 获取。 |
| numpy | 1.26.4 | 建议通过 pip 获取。 |
| torch | 2.5.1 | 建议通过 pip 获取。 |

**注意事项**

如需安装 Multimodal SDK 软件包以外的第三方软件，请注意及时升级最新版本，关注并修补存在的漏洞。

## 安装 Multimodal SDK

**安装须知**

- 安装和运行 Multimodal SDK 的用户，需要满足：
    - 安装和运行 Multimodal SDK 的用户建议为普通用户。
    - 安装和运行 Multimodal SDK 的用户需为同一用户。
    - 安装 CANN（Toolkit），NPU 驱动固件和安装 Multimodal SDK 的用户需为同一用户，建议为普通用户。

- 软件包的安装、升级、卸载及版本查询相关的日志会保存至 `~/log/mindxsdk/deployment.log` 文件；完整性校验、提取文件、tar 命令访问相关的日志会保存至 `~/log/makeself/makeself.log` 文件。用户可查看相应文件，完成后续的日志跟踪及审计。

**安装准备**

确保安装环境中已执行 CANN 环境变量配置脚本，使环境变量生效。具体执行路径，请按照实际安装为准。

```bash
# 安装 toolkit 包
. /usr/local/Ascend/cann/set_env.sh # 此处为 CANN 默认安装路径，根据实际安装路径修改
```

**安装步骤**

1. 以软件包的安装用户登录安装环境。
2. 将 Multimodal SDK 软件包上传到安装环境的任意路径下并进入软件包所在路径。
3. 增加对软件包的可执行权限。

    ```bash
    chmod u+x Ascend-mindxsdk-multimodal_{version}_linux-{arch}.run
    ```

4. 执行如下命令，校验软件包的一致性和完整性。

    ```bash
    ./Ascend-mindxsdk-multimodal_{version}_linux-{arch}.run --check
    ```

    如果系统没有 shasum 或者 sha256sum 工具则会校验失败，此时需要自行安装 shasum 或者 sha256sum 工具。

    若显示如下信息，说明软件包已通过校验。

    ```bash
    Verifying archive integrity...  100%   SHA256 checksums are OK. All good.
    ```

5. 创建 Multimodal SDK 软件包的安装路径。不建议在 `/tmp` 路径下安装。
    - 若用户未指定安装路径，软件会默认安装到 Multimodal SDK 软件包所在的路径。
    - 若用户想指定安装路径，需要先创建安装路径。以安装路径 `/home/work/Mind_SDK` 为例：

        ```bash
        mkdir -p /home/work/Mind_SDK
        ```

6. 进入 Multimodal SDK 软件包所在路径，参考以下命令安装 Multimodal SDK（安装路径的相关约束请参见[表 5](#table-install-params)）。

    - 若用户指定了安装路径，将安装在指定的路径下。以安装路径 `/home/work/Mind_SDK` 为例：

        ```bash
        ./Ascend-mindxsdk-multimodal_{version}_linux-{arch}.run --install --install-path=/home/work/Mind_SDK
        ```

        或者

        ```bash
        echo y | ./Ascend-mindxsdk-multimodal_{version}_linux-{arch}.run --install --install-path=/home/work/Mind_SDK
        ```

    - 若用户未指定安装路径，将安装在当前路径下。

        ```bash
        ./Ascend-mindxsdk-multimodal_{version}_linux-{arch}.run --install
        ```

        或者

        ```bash
        echo y | ./Ascend-mindxsdk-multimodal_{version}_linux-{arch}.run --install
        ```

    > [!NOTE] 说明
    > `--install` 安装命令同时支持输入可选参数，如[表 5](#table-install-params)所示。

7. 进入 Multimodal SDK 安装路径下的 `script` 目录，执行以下命令使 Multimodal SDK 的环境变量生效。

    ```bash
    source set_env.sh
    ```

**相关参考**

<a id="table-install-params"></a>

**表 5** `--install` 安装命令可选参数表

| 输入参数 | 含义 |
| -- | -- |
| `--help` \| `-h` | 查询帮助信息。 |
| `--info` | 查询包构建信息。 |
| `--list` | 查询文件列表。 |
| `--check` | 查询包完整性。 |
| `--quiet` \| `-q` | 启用静默模式。需要和 `--install` 或 `--upgrade` 参数配合使用。 |
| `--nox11` | 废弃接口，无实际作用。 |
| `--noexec` | 不运行嵌入的脚本。 |
| `--extract=` | 直接提取到目标目录（绝对路径或相对路径）。通常与 `--noexec` 选项一起使用，仅用于提取文件而不运行它们。 |
| `--tar arg1 [arg2 ...]` | 通过 tar 命令访问归档文件的内容。 |
| `--install` | Multimodal SDK 软件包安装操作命令。约束：当前路径和安装路径不能存在非法字符，仅支持大小写字母、数字、`-_./` 特殊字符；安装路径下不能存在名为 multimodal 的文件或文件夹；若存在名为 multimodal 的软链接，则会被覆盖。 |
| `--install-path=` | （可选）自定义软件包安装根目录。如未设置，默认为当前命令执行所在目录。建议用户使用绝对路径；需和 `--install` 或 `--upgrade` 配合使用；与 `--upgrade` 配合使用时代表旧软件包的安装目录；路径不能存在非法字符。 |
| `--upgrade` | Multimodal SDK 软件包升级操作命令。升级需要确保已经安装过目录完整的 Multimodal SDK。 |
| `--version` | 查询软件包 Multimodal SDK 版本。 |

> [!NOTE] 说明
> 以下参数未展示在 `--help` 参数中，用户请勿直接使用。
>
> - `--xwin`：使用 xwin 模式运行。
> - `--phase2`：要求执行第二步动作。

# 卸载

**操作步骤**

1. 进入 Multimodal SDK 的安装路径，确认 Multimodal SDK 目录下 `script` 目录中的 `uninstall.sh` 脚本是否有可执行权限。

    ```bash
    cd multimodal/script
    ls -l uninstall.sh
    ```

    若脚本没有可执行权限，请执行如下命令，赋予 `uninstall.sh` 脚本可执行权限。

    ```bash
    chmod u+x uninstall.sh
    ```

2. 执行如下命令，开始执行卸载。在执行卸载脚本时，脚本会卸载已安装的 python whl 包并删除安装目录。

    ```bash
    ./uninstall.sh
    ```

    > [!NOTE] 说明
    > 使用 `uninstall.sh` 脚本进行卸载操作仅适用于正常安装途径，且安装后未对安装文件结构进行修改，如需解决安装异常等情况，请通过完全删除安装目录下任何有关 multimodal 的文件夹，以及 `pip uninstall mm` 卸载安装的 Python 包文件。
