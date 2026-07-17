# Installation Guide

## Installation Description

Multimodal SDK supports three installation methods: [offline installation](#offline-installation) (`run` package / `Wheel` package), [image installation](#image-installation), and [source code installation](#source-code-installation).

If you use offline installation or source code installation, please first [install related dependencies](#installing-dependencies). If you use image installation, skip this step.

**Precautions**

If you need to install third-party software other than the Multimodal SDK software package, upgrade the software to the latest version in a timely manner and fix existing vulnerabilities.

## Installing Dependencies

### Installing NPU Driver, Firmware, and CANN

Please refer to the [CANN Installation Guide](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/softwareinst/instg/instg_0000.html) to install NPU driver, firmware, and CANN 9.0.0 with the corresponding driver version.

### Other Dependencies

| Dependency | Recommended Version | Acquisition Suggestion |
| ------------ | ------------------------- | ---------------------------------------------------------------------- |
| CMake | 3.14 or later | Install using the package manager:<br>Ubuntu: `sudo apt-get install -y cmake`<br>OpenEuler: `sudo yum install -y cmake`<br>If the version does not meet the minimum requirement, install from source |
| Make | 4.1 or later | Install using the package manager:<br>Ubuntu: `sudo apt-get install -y make`<br>OpenEuler: `sudo yum install -y make`<br>If the version does not meet the minimum requirement, install from source |
| GCC | 9.4 or later | Install using the package manager:<br>Ubuntu: `sudo apt-get install -y build-essential`<br>OpenEuler: `sudo yum install -y gcc gcc-c++` |
| Python | Minimum 3.9; **Recommended 3.10 or 3.11** | Install using the package manager:<br>Ubuntu: `sudo apt-get install -y python3 python3-pip python3-dev`<br>OpenEuler: `sudo yum install -y python3 python3-pip python3-devel`<br>If the system version is too low, compile from source or install a higher version |
| transformers | 4.51.3 | Install using pip:<br>`pip3 install transformers==4.51.3` |
| einops | 0.8.2 | Install using pip:<br>`pip3 install einops==0.8.2` |
| pillow | 11.2.1 or later | Install using pip:<br>`pip3 install pillow==11.2.1` |
| numpy | 1.26.4 | Install using pip:<br>`pip3 install numpy==1.26.4` |
| torch | 2.5.1 | Install using pip:<br>`pip3 install torch==2.5.1`<br>Additional configuration required for ARM64 Ascend environment |
| torch-npu | 2.5.1.post1 | Install using pip:<br>`pip3 install torch-npu==2.5.1.post1`<br>Must be compatible with CANN 9.0.0 |

## Installation Methods

Multimodal SDK provides three installation methods: offline installation (`run` package / `Wheel` package), source code installation, and image installation. Choose the appropriate method based on your scenario.

### Offline Installation

#### Mode 1: `run` Package Installation

**Installation Precautions**

- It is recommended to use the **same common user** to complete the installation and running of CANN, NPU driver firmware, and Multimodal SDK.
- Logs related to package installation, upgrade, uninstallation, and version queries are saved to `~/log/mindxsdk/deployment.log`. Logs related to integrity verification, file extraction, and access through the `tar` command are saved to `~/log/makeself/makeself.log`. You can view the corresponding files to complete subsequent log tracing and audit.

**Installation Preparation**

Download the Multimodal SDK package from the [download link](https://gitcode.com/Ascend/MultimodalSDK/releases) (Ascend-mindxsdk-multimodal_{version}_linux-{arch}.run).

Ensure that the CANN environment variable configuration script has been executed in the installation environment:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh   # Default path, modify according to actual installation path
```

**Installation Procedure**

1. Log in to the installation environment as the package installation user.
2. Upload the Multimodal SDK package to any path in the installation environment and go to the package directory.
3. Add execute permissions to the package:

   ```bash
   chmod u+x ${MMSDK_PACKAGE}
   ```

4. Run the following command to verify the consistency and integrity of the package:

   ```bash
   ./${MMSDK_PACKAGE} --check
   ```

   If the system does not have the `sha256sum` tool, the verification fails. Install it using the following command:

   ```bash
   sudo apt install -y coreutils
   ```

   If `Verifying archive integrity... OK` is displayed, the package has passed verification.

5. Create the installation path for the Multimodal SDK package (optional).

   > [!NOTE]
   > It is not recommended to install in the `/tmp` path: after system restart, the `/tmp` content may be cleared, and the directory permissions and space are unstable, which is not suitable for persistent deployment.

   - If the user does not specify an installation path, the software is installed by default in the path where the Multimodal SDK package is located.
   - If the user wants to specify an installation path, create the installation path first. For example, if the installation path is `/home/work/Mind_SDK`:

   ```bash
   mkdir -p /home/work/Mind_SDK
   ```

6. Go to the path where the Multimodal SDK package is located and run the following command to install Multimodal SDK. For constraints on the installation path, see [Installation Command Parameters](#table-install-params).

   **Specify installation path** (using `/home/work/Mind_SDK` as an example):

   ```bash
   ./${MMSDK_PACKAGE} --install --install-path=/home/work/Mind_SDK
   ```

   **Install to current path by default**:

   ```bash
   ./${MMSDK_PACKAGE} --install
   ```

   Common optional parameters: `--quiet` (quiet mode, must be used with `--install`), `--version` (query version). For complete parameter descriptions, see [Installation Command Parameters](#table-install-params).

<a id="installation-verification"></a>

**Installation Verification**

After the `run` installation is complete, load the Multimodal SDK environment variables:

```bash
source ${MULTIMODAL_SDK_HOME}/script/set_env.sh
```

If `${MULTIMODAL_SDK_HOME}` is not set, go to the `script` directory in the installation directory first:

```bash
# Example: installation path is /home/work/Mind_SDK
source /home/work/Mind_SDK/multimodal/script/set_env.sh
```

To make it take effect automatically on each login, append the above command to `~/.bashrc`:

```bash
echo 'source ${MULTIMODAL_SDK_HOME}/script/set_env.sh' >> ~/.bashrc
source ~/.bashrc
```

For environment variable descriptions, see [Appendix - Environment Variables](./appendix.md#environment-variables).

Execute the following verification:

```bash
# You can view the installation path during .run installation; skip this line for Wheel installation
echo "MULTIMODAL_SDK_HOME=${MULTIMODAL_SDK_HOME}"
python3 -c "import mm; print('mm import: OK')"
```

Use a jpg/jpeg test image for functional verification (replace `$TEST_IMAGE` with the actual path):

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

If the following output is displayed, the SDK is ready:

```text
resize output shape: (500, 500, 3)
```

For more examples, see [Quick Start](./quickstart.md) and [Examples and Guidance](./user_guide.md).

**Installation Command Parameters**

<a id="table-install-params"></a>

| Input Parameter | Description |
| --- | --- |
| `--help`, `-h` | Query help information |
| `--info` | Query package build information |
| `--list` | Query file list |
| `--check` | Query package integrity |
| `--quiet`, `-q` | Enable quiet mode. Must be used with `--install` or `--upgrade` |
| `--noexec` | Do not run embedded scripts |
| `--extract=` | Extract directly to target directory (absolute or relative path). Usually used with `--noexec` to only extract files without running |
| `--tar arg1 [arg2 ...]` | Access archive file contents using the `tar` command |
| `--install` | Multimodal SDK installation operation. Constraints: current path and installation path must not contain invalid characters, only uppercase and lowercase letters, digits, `-_./` are supported; installation path must not contain a file or folder named `multimodal`; if a symbolic link named `multimodal` exists, it will be overwritten |
| `--install-path=` | (Optional) Customize the root directory for installation. If not set, defaults to the current command execution directory. It is recommended to use an absolute path; must be used with `--install` or `--upgrade`; when used with `--upgrade`, it represents the installation directory of the old package |
| `--upgrade` | Multimodal SDK upgrade operation. Must ensure that a complete old version installation exists |
| `--version` | Query package version |

**Internal Parameters**

> [!WARNING]
> The following parameters are for internal use only. Do not call them directly.

The following parameters are not displayed in `--help`:

- `--nox11`: Deprecated interface, no actual effect
- `--xwin`: Run in xwin mode
- `--phase2`: Require the second phase action to be executed

#### Mode 2: `Wheel` Package Installation

**Installation Precautions**

If you only need to use the `mm` package in the Python environment, you can skip the `run` installation package and directly install the `Wheel` package. The `Wheel` package bundles `libcore.so` and native dependencies such as FFmpeg, libjpeg-turbo, and soxr. **No need** to execute `source set_env.sh` or configure `MULTIMODAL_SDK_HOME`.

> [!NOTE]
>
> - The `Wheel` package **does not include** CANN (`libascendcl.so`, etc.) and Python third-party dependencies. Before use, you must complete the installation according to [Installing Dependencies](#installing-dependencies).
> - Before starting Python each time, you must load the CANN environment variables (path depends on actual installation):
>
>   ```bash
>   source /usr/local/Ascend/ascend-toolkit/set_env.sh
>   ```

**Installation Preparation**

You can obtain the `mm-*.whl` compatible with the SDK version through any of the following methods:

1. **Download from Release page**: Get the `Wheel` file of the same version as the `run` package from [Multimodal SDK Releases](https://gitcode.com/Ascend/MultimodalSDK/releases) (if provided with the release).

2. **Extract from `run` package**:

   ```bash
   ./${MMSDK_PACKAGE} --noexec --extract=/tmp/mmsdk_extract
   find /tmp/mmsdk_extract -name 'mm-*.whl'
   ```

3. **Build yourself**: Follow [CONTRIBUTING.md](../../CONTRIBUTING.md) to complete source code building, then get the generated `mm-*.whl` from the `MultimodalSDK/dist/` directory.

**Installation Procedure**

1. Confirm that the CANN environment variables have been loaded (see above).
2. Confirm that the Python dependencies listed in [Installing Dependencies](#installing-dependencies) have been installed.
3. Install the `Wheel` package (replace `mm-1.0.0-py3-none-any.whl` with the actual filename):

   ```bash
   pip3 install /path/to/mm-1.0.0-py3-none-any.whl
   ```

   If an old version already exists in the environment, you can force reinstall:

   ```bash
   pip3 install --force-reinstall --no-deps /path/to/mm-1.0.0-py3-none-any.whl
   ```

   > [!NOTE]
   > It is recommended to use `--no-deps` to avoid pip automatically upgrading or downgrading fixed dependency versions.

4. Follow the steps in [Installation Verification](#installation-verification) to confirm that the SDK is ready (Wheel package installation does not require additional `MULTIMODAL_SDK_HOME` environment variable settings).

### Source Code Installation

If you need to build Multimodal SDK from source, please refer to [CONTRIBUTING.md](../../CONTRIBUTING.md) to complete the compilation environment preparation and source code building. After building is complete:

- If `mm-*.whl` is generated, follow the steps in [Offline Installation: `Wheel` Package Installation](#mode-2-wheel-package-installation) to complete the installation.
- If a `run` installation package is generated, follow the steps in [Offline Installation: `run` Package Installation](#mode-1-run-package-installation) to complete the installation.

### Image Installation

Multimodal SDK supports containerized deployment. You can obtain images through the following two methods:

**Method 1: Pull Official Image**

Pull the pre-built image directly from the Ascend Community image repository. For details, see [Multimodal SDK Image Repository](https://www.hiascend.com/developer/ascendhub/detail/e0081aa3c4dd441dbd6a379bee8cc4c9).

**Method 2: Build Image Locally**

Use the Dockerfile provided in the `docker/` directory of the project to build it yourself. For detailed build instructions, see [docker/OVERVIEW.md](../../docker/OVERVIEW.md).

**Start Container**

After obtaining the image, execute the following command to start the container:

```bash
docker run -it \
    --name mmsdk_container \
    --device /dev/davinci1 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    ${image_name}:${tag} bash
```

After entering the container, you can use it directly:

```bash
python3 -c "import mm; print('mm import: OK')"
```

> [!NOTE]
>
> - CANN, NPU driver, and Multimodal SDK are pre-installed in the image. No additional environment variable configuration is required after entering the container.
> - The host must have NPU drivers compatible with the CANN version in the container installed (see [Installing NPU Driver, Firmware, and CANN](#installing-npu-driver-firmware-and-cann)).

## Upgrade

### `run` Installation Upgrade

Before upgrading, please confirm that a complete Multimodal SDK installation directory exists, and the new version is compatible with the CANN / HDK version.

1. Download the new version installation package and go to the package directory.
2. Load the CANN and Multimodal SDK environment variables.
3. Execute the upgrade command (`--install-path` must point to **the installation root directory of the old version**):

   ```bash
   ./Ascend-mindxsdk-multimodal_{version}_linux-{arch}.run --upgrade --install-path=/home/work/Mind_SDK
   ```

4. Reload the environment variables and execute the commands in [Installation Verification](#installation-verification).

> [!NOTE]
> During the upgrade process, the installation program will first uninstall the old version and then install the new version. If the upgrade fails, check `~/log/mindxsdk/deployment.log` to locate the cause.

### `Wheel` Package Upgrade

1. Get the new version `Wheel` package from [Multimodal SDK Releases](https://gitcode.com/Ascend/MultimodalSDK/releases).
2. Load the CANN environment variables.
3. Execute the upgrade command (replace `mm-x.x.x-py3-none-any.whl` with the actual filename):

   ```bash
   pip3 install --upgrade /path/to/mm-x.x.x-py3-none-any.whl
   ```

   If you need to force overwrite without updating dependencies:

   ```bash
   pip3 install --force-reinstall --no-deps /path/to/mm-x.x.x-py3-none-any.whl
   ```

4. Execute the commands in [Installation Verification](#installation-verification) to confirm the upgrade is successful.

> [!NOTE]
>
> - It is recommended to use `--no-deps` to avoid pip automatically upgrading or downgrading fixed dependency versions.
> - If you need to roll back, use `pip3 install --force-reinstall /path/to/mm-old-version.whl` to reinstall the old version.

## Uninstallation

### `run` Installation Uninstallation

The default installation directory structure of Multimodal SDK is `{install-path}/multimodal/`, and the uninstall script is located at `{install-path}/multimodal/script/uninstall.sh`.

**Procedure**

1. Go to the `script` directory in the Multimodal SDK installation path and check whether `uninstall.sh` has execute permissions:

   ```bash
   cd ${MULTIMODAL_SDK_HOME}/script
   ls -l uninstall.sh
   ```

   If the script does not have execute permissions:

   ```bash
   chmod u+x uninstall.sh
   ```

2. Execute the uninstall script. The script will uninstall the installed Python whl package and delete the installation directory:

   ```bash
   ./uninstall.sh
   ```

**Uninstallation Verification**

```bash
# Confirm the installation directory has been deleted
ls ${MULTIMODAL_SDK_HOME} 2>/dev/null && echo "Directory still exists" || echo "Directory has been deleted"

# Confirm the Python package has been uninstalled
pip3 show mm 2>/dev/null && echo "mm still exists" || echo "mm has been uninstalled"
```

> [!NOTE]
> `uninstall.sh` is only applicable to normal installation paths and installations whose directory structure has not been modified after installation. If the installation is abnormal, please manually delete all `multimodal` related folders in the installation directory and execute `pip uninstall mm -y` to uninstall the Python package.

### `Wheel` Package Uninstallation

```bash
pip3 uninstall -y mm
pip3 show mm 2>/dev/null && echo "mm still exists" || echo "mm has been uninstalled"
```

## FAQs

| Symptom | Solution |
| ------------------------- | ----------------------------------------------------------- |
| Failed to import `mm` (`run` installation) | Confirm that `source ${MULTIMODAL_SDK_HOME}/script/set_env.sh` has been executed |
| Failed to import `mm` (`Wheel` installation) | Confirm that CANN environment variables have been sourced; execute `pip3 show mm` to confirm whl is installed; if necessary, `pip3 install --force-reinstall --no-deps mm-*.whl` to reinstall |
| `libcore.so` / `libascendcl.so` not found | `Wheel` installation: confirm compatible version whl is used and CANN is sourced; `.run` installation: confirm `set_env.sh` has been sourced |
| `npu-smi info` has no output | Check if NPU driver/firmware is installed successfully, retry after restart if necessary |
| CANN environment variables not effective | Confirm CANN's `set_env.sh` has been sourced, path depends on actual installation |
| torch / transformers version conflict | Install fixed versions according to [table](#other-dependencies) |
| More issues | [FAQ - Installation and Environment](./faq.md#installation-and-environment), [Appendix - Error Codes](./appendix.md#error-codes) |
