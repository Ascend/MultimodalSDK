# 常见问题（FAQ）

## 安装与环境

### Q: 导入 `mm` 失败，提示 `ModuleNotFoundError: No module named 'mm'`？

**原因**：Python 未找到已安装的 `mm` 包（尚未安装 whl，或安装到了其他 Python 环境）。

**处理**：

```bash
pip3 show mm
python3 -c "import mm; print('mm import: OK')"
```

若 `pip3 show mm` 无输出，请先按[`run` 包安装](../03_installation_guide/installation_guide.md#方式一run-包安装)或[`Wheel` 包安装](../03_installation_guide/installation_guide.md#方式二wheel-包安装)完成安装。

### Q: 导入 `mm` 失败，提示 `libcore.so` 或 `libascendcl.so` 找不到？

**原因**：

- **Wheel 安装**：CANN 环境变量未加载，或 whl 版本不匹配/安装不完整。
- **`.run` 安装**：未执行 `source set_env.sh`，或安装目录不完整。

**处理（Wheel 安装）**：

```bash
source /usr/local/Ascend/cann/set_env.sh   # 默认CANN安装路径，请按实际安装路径修改
pip3 install --force-reinstall --no-deps /path/to/mm-*.whl
python3 -c "import mm; print('mm import: OK')"
```

Wheel 包无需设置 `MULTIMODAL_SDK_HOME`，原生库已捆绑在 whl 内并在 `import mm` 时自动加载。

**处理（`.run` 安装）**：

```bash
source ${MULTIMODAL_SDK_HOME}/script/set_env.sh
python3 -c "import mm; print('mm import: OK')"
```

若 `${MULTIMODAL_SDK_HOME}` 未设置，请按实际安装路径执行，例如：

```bash
source /home/work/Mind_SDK/multimodal/script/set_env.sh
```

### Q: `npu-smi info` 无输出或报错？

**原因**：NPU 驱动/固件未正确安装，或 CANN 环境变量未生效。

**处理**：

1. 确认已安装 Ascend HDK 26.0.RC1 及 CANN 9.0.0（或兼容补丁版本）。
2. 加载 CANN 环境变量：`source /usr/local/Ascend/cann/set_env.sh`（路径以实际安装为准）。
3. 重新执行 `npu-smi info`；若仍失败，重启宿主机后重试。

### Q: torch / transformers 版本冲突？

**原因**：第三方依赖版本与 Multimodal SDK 要求不一致。

**处理**：对照[安装部署 - 其他依赖](../03_installation_guide/installation_guide.md#其他依赖)固定版本安装：

```bash
pip3 install transformers==4.51.3 pillow==11.2.1 numpy==1.26.4
```

`torch` 与 `torch-npu` 需按 CANN 9.0.0 配套表安装，请参阅 [安装部署 - 其他依赖](../03_installation_guide/installation_guide.md#其他依赖)。

### Q: 已安装 lzma 模块，调用 torchvision 仍提示缺少 lzma？

安装 lzma 模块：

```shell
pip3 install backports.lzma
```

进入 Python 的库目录，以使用的 python3.11.4 为例：

```shell
cd /xx/xx/python-3.11.4/lib/python3.11
```

修改 `lzma.py`，将下面的内容：

```python
from _lzma import *
from _lzma import _encode_filter_properties, _decode_filter_properties
```

修改为：

```python
from backports.lzma import *
from backports.lzma import _encode_filter_properties, _decode_filter_properties
```

## Docker 与快速体验

### Q: Docker 容器内找不到宿主机上的测试图片？

**原因**：未将宿主机目录挂载进容器，或 `TEST_IMAGE` 使用了宿主机路径而非容器内路径。

**处理**：启动容器时添加卷挂载，并在容器内使用挂载路径：

```bash
docker run ... -v /path/to/testdata:/data ...
export TEST_IMAGE="/data/test.jpg"
```

详见 [快速入门 - 步骤 2](../02_quickstart/quickstart.md#步骤-2启动容器)。

### Q: 容器无法访问 NPU？

**原因**：`--device /dev/davinci*` 设备号与宿主机不一致，或驱动相关目录未挂载。

**处理**：

1. 在宿主机执行 `npu-smi info` 确认 NPU 可用。
2. 将 `--device /dev/davinci0` 中的编号改为实际设备（如 `davinci1`）。
3. 确认 quick_start 中列出的驱动挂载项均已包含在 `docker run` 命令中。

### Q: 读取文件时报文件权限错误（0x102003EE）？

**原因**：API 要求文件 owner 为当前用户，且权限不高于 640。

**处理**：

```bash
chmod 640 /path/to/your/file.jpg
```

## 运行与排障

- 遇到错误码？参见 [附录 - 错误码](./appendix.md#错误码)
- 环境变量问题？参见 [附录 - 环境变量说明](./appendix.md#环境变量说明)
- 安装步骤问题？参见 [安装部署 - 常见问题](../03_installation_guide/installation_guide.md#常见问题)
