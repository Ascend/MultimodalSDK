# Installation and Deployment

## Installation Overview

This document describes how to install the latest version of Multimodal SDK. It supports only the Atlas 800I A2 inference server running Ubuntu 22.04 or openEuler 24.03. It is recommended that you reserve at least 16GB of available disk space and 8GB of available memory. Multimodal SDK is currently an officially supported version and is compatible with CANN 9.1.0 and the corresponding supporting software.

Multimodal SDK supports three installation methods: [offline installation](#offline-installation) (`run` package / `Wheel` package), [source installation](#source-installation), and [image installation](#image-installation). The `run` package is a self-extracting installation script that includes all dependencies. The `Wheel` package is a Python binary distribution package. Image installation deploys Multimodal SDK using a container image.

If you use offline installation or source installation, first [install the required dependencies](#installing-dependencies). If you use image installation, skip this step.

**Precautions**

If you need to install third-party software other than the Multimodal SDK software package, promptly upgrade it to the latest version and address any existing vulnerabilities.

## Installing Dependencies

### Installing the NPU Driver, Firmware, and CANN

Refer to the [CANN Installation Guide](https://www.hiascend.com/cann/download), select CANN (Compute Architecture for Neural Networks) 9.1.0 and HDK (Hardware Developer Kit) 26.1.0, and install the NPU driver, firmware, and CANN software, including the Toolkit and ops packages. Then configure the required environment variables.

### Installing Other Dependencies

| Dependency | Recommended Version | Installation Method |
| ---------- | ------------------- | ------------------- |
| Python | Minimum 3.10; **recommended 3.12** | Install using the package manager:<br>Ubuntu: `sudo apt-get install -y python3 python3-pip python3-dev`<br>openEuler: `sudo yum install -y python3 python3-pip python3-devel`<br>If the system version is too low, compile from source or install a higher version. |
| transformers | 4.51.3 | Install using pip:<br>`pip3 install transformers==4.51.3` |
| einops | 0.8.2 | Install using pip:<br>`pip3 install einops==0.8.2` |
| pillow | 11.2.1 or later | Install using pip:<br>`pip3 install pillow==11.2.1` |
| numpy | 1.26.4 | Install using pip:<br>`pip3 install numpy==1.26.4` |
| torch | 2.6.0 | Install using pip:<br>`pip3 install torch==2.6.0` |
| torch-npu | 2.6.0.post5 | Install using pip:<br>`pip3 install torch-npu==2.6.0.post5` |

## Installing the SDK

Multimodal SDK provides three installation methods: offline installation (`run` package / `Wheel` package), source installation, and image installation. Choose an appropriate method based on your scenario.

### Offline Installation

#### Installing the `run` Package

**Prerequisites**

- Complete steps in [Installing Dependencies](#installing-dependencies).

**Installation Precautions**

- It is recommended to use the **same regular (non-root) user** to install and run CANN, the NPU driver and firmware, and Multimodal SDK.
- Logs related to package installation, upgrade, uninstallation, and version queries are saved to `~/log/mindxsdk/deployment.log`. Logs related to integrity verification, file extraction, and access through the `tar` command are saved to `~/log/makeself/makeself.log`. You can view the corresponding files for subsequent log tracing and auditing.

**Preparing for Installation**

Download the Multimodal SDK package from [Multimodal SDK Releases](https://gitcode.com/Ascend/MultimodalSDK/releases). The package name is `Ascend-mindxsdk-multimodal_{version}_linux-aarch64.run`, where `{version}` is the SDK version, such as `26.1.0`.

The following commands use `${MMSDK_PACKAGE}` to represent the name of the downloaded `.run` package. Set it to the actual package filename. For example:

```bash
export MMSDK_PACKAGE=Ascend-mindxsdk-multimodal_26.1.0_linux-aarch64.run
```

Make sure that the CANN environment variable configuration script has been executed in the installation environment:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh   # Default path; modify it according to the actual installation path.
```

**Installing the `run` Package**

1. Log in to the installation environment as the package installation user.
2. Upload the Multimodal SDK package to any path in the installation environment and go to the package directory.
3. Grant execute permission to the package:

   ```bash
   chmod u+x ${MMSDK_PACKAGE}
   ```

4. Verify the consistency and integrity of the package:

   ```bash
   ./${MMSDK_PACKAGE} --check
   ```

   If the system does not have the `sha256sum` tool, the verification fails. Install it using the following command:

   ```bash
   sudo apt-get install -y coreutils
   ```

   If `Verifying archive integrity... OK` is displayed, the package has passed verification.

5. Create the installation path for the Multimodal SDK package (optional).

   - If no installation path is specified, the software is installed in the directory where the Multimodal SDK package is located.
   - If you want to specify an installation path, create the path first. For example, if the installation path is `/home/work/Mind_SDK`:

   ```bash
   mkdir -p /home/work/Mind_SDK
   ```

6. Go to the directory where the Multimodal SDK package is located and install Multimodal SDK. For installation path constraints, see [Command-Line Options](#table-command-line-options).

   **Installing to the specified path** (using `/home/work/Mind_SDK` as an example):

   ```bash
   ./${MMSDK_PACKAGE} --install --install-path=/home/work/Mind_SDK
   ```

   **Installing to the current directory (default)**:

   ```bash
   ./${MMSDK_PACKAGE} --install
   ```

   Common optional parameters include `--quiet` (quiet mode; must be used with `--install`) and `--version` (query the package version). For complete parameter descriptions, see [Command-Line Options](#table-command-line-options).

<a id="installation-verification"></a>

**Verifying the Installation**

After the `run` package is installed, load the Multimodal SDK environment variables:

```bash
source ${MULTIMODAL_SDK_HOME}/script/set_env.sh
```

If `${MULTIMODAL_SDK_HOME}` is not set, go to the `script` directory under the installation directory and run the command. For example:

```bash
# Example: installation path is /home/work/Mind_SDK
source /home/work/Mind_SDK/multimodal/script/set_env.sh
```

To load the environment variables automatically each time you log in, append the command to `~/.bashrc`:

```bash
echo 'source ${MULTIMODAL_SDK_HOME}/script/set_env.sh' >> ~/.bashrc
source ~/.bashrc
```

For environment variable descriptions, see [Appendix > Environment Variables](../06_references/appendix.md#environment-variables).

Run the following commands to verify the installation:

```bash
# You can view the installation path during .run installation; skip this line for Wheel installation.
echo "MULTIMODAL_SDK_HOME=${MULTIMODAL_SDK_HOME}"
python3 -c "import mm; print('mm import: OK')"
```

Use a JPG/JPEG test image for functional verification. Replace `$TEST_IMAGE` with the actual path:

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

For more examples, see [Quick Start](../02_quickstart/quickstart.md) and [Examples and Guidance](../04_user_guide/user_guide.md).

**Command-Line Options**

<a id="table-command-line-options"></a>

| Option | Description |
| ------ | ----------- |
| `--help`, `-h` | Displays the help information. |
| `--info` | Displays the package build information. |
| `--list` | Displays the file list. |
| `--check` | Verifies the package integrity. |
| `--quiet`, `-q` | Enables quiet mode. Must be used with `--install` or `--upgrade`. |
| `--noexec` | Prevents embedded scripts from running. |
| `--extract=` | Extracts files directly to the target directory (absolute or relative path). Usually used with `--noexec` to extract files without running them. |
| `--tar arg1 [arg2 ...]` | Accesses the contents of the archive using the `tar` command. |
| `--install` | Installs Multimodal SDK. Constraints: the current path and installation path must not contain invalid characters. Only uppercase and lowercase letters, digits, `-`, `_`, `.`, and `/` are supported. The installation path must not contain a file or directory named `multimodal`. If a symbolic link named `multimodal` exists, it is overwritten. |
| `--install-path=` | Specifies the installation root directory. If not set, the directory from which the command is executed is used by default. An absolute path is recommended. Must be used with `--install` or `--upgrade`. When used with `--upgrade`, it specifies the installation directory of the old package. |
| `--upgrade` | Upgrades Multimodal SDK. A complete installation of the old version must already exist. |
| `--version` | Displays the package version. |

**Internal Parameters**

> [!NOTE]
> The following parameters are for internal use only. Do not invoke them directly.

The following parameters are not displayed in `--help`:

- `--nox11`: Deprecated interface with no actual effect.
- `--xwin`: Run in xwin mode.
- `--phase2`: Run the second phase of the operation.

#### Installing the `Wheel` Package

**Prerequisites**

- Complete [Installing Dependencies](#installing-dependencies).

**Installation Precautions**

If you only need to use the `mm` package in a Python environment, you can skip the `run` package and install the `Wheel` package directly. The `Wheel` package bundles `libcore.so` and native dependencies such as FFmpeg, libjpeg-turbo, and soxr. **You do not need to** execute `source ${MULTIMODAL_SDK_HOME}/script/set_env.sh` or configure `MULTIMODAL_SDK_HOME`.

> [!NOTE]
>
> - The `Wheel` package **does not include** CANN (`libascendcl.so`, etc.) or Python third-party dependencies. Before use, complete the installation according to [Installing Dependencies](#installing-dependencies).
> - Before starting Python, load the CANN environment variables each time. The path depends on the actual installation path:
>
>   ```bash
>   source /usr/local/Ascend/ascend-toolkit/set_env.sh
>   ```

**Preparing for Installation**

You can obtain the `mm-*.whl` compatible with the SDK version in any of the following ways:

1. **Downloading from the Release page**: Obtain the `Wheel` file corresponding to the same version as the `run` package from [Multimodal SDK Releases](https://gitcode.com/Ascend/MultimodalSDK/releases), if one is provided with the release.
2. **Extracting from the `run` package**:

   ```bash
   ./${MMSDK_PACKAGE} --noexec --extract=/tmp/mmsdk_extract
   find /tmp/mmsdk_extract -name 'mm-*.whl'
   ```

3. **Building from source**: Follow [CONTRIBUTING.md](../../../CONTRIBUTING.md) to prepare the build environment and build the source code. Then obtain the generated `mm-*.whl` from the `MultimodalSDK/dist/` directory.

**Installing the `Wheel` Package**

1. Confirm that the CANN environment variables have been loaded as described above.
2. Confirm that the Python dependencies listed in [Installing Dependencies](#installing-dependencies) have been installed.
3. Install the `Wheel` package. Replace `mm-1.0.0-py3-none-any.whl` with the actual filename:

   ```bash
   pip3 install /path/to/mm-1.0.0-py3-none-any.whl
   ```

   If an old version already exists in the environment, force a reinstall:

   ```bash
   pip3 install --force-reinstall --no-deps /path/to/mm-1.0.0-py3-none-any.whl
   ```

   > [!NOTE]
   > It is recommended to use `--no-deps` to prevent pip from automatically upgrading or downgrading the fixed dependency versions.

4. Follow the steps in [Verifying the Installation](#installation-verification) to verify that the SDK is ready. `Wheel` package installation does not require additional configuration of the `MULTIMODAL_SDK_HOME` environment variable.

### Source Installation

If you need to build Multimodal SDK from source, refer to [CONTRIBUTING.md](../../../CONTRIBUTING.md) to prepare the build environment and build the source code. After the build is complete:

- If `mm-*.whl` is generated, follow the steps in [Offline Installation: Installing the `Wheel` Package](#installing-the-wheel-package) to install it.
- If a `run` package is generated, follow the steps in [Offline Installation: Installing the `run` Package](#installing-the-run-package) to install it.

### Image Installation

Multimodal SDK supports containerized deployment. You can obtain an image in either of the following ways:

**Pulling the Official Image**

Pull the pre-built image directly from the Ascend Community image repository. For details, see [Multimodal SDK Image Repository](https://www.hiascend.com/developer/ascendhub/detail/e0081aa3c4dd441dbd6a379bee8cc4c9).

**Building an Image Locally**

Use the Dockerfile provided in the project's `docker/` directory to build the image locally. For detailed build instructions, see [docker/OVERVIEW.md](../../../docker/OVERVIEW.md).

**Starting the Container**

After obtaining the image, start the container using the following command:

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
    ${image_name}:${tag} bash
```

After entering the container, you can use Multimodal SDK directly:

```bash
python3 -c "import mm; print('mm import: OK')"
```

> [!NOTE]
>
> - CANN, the NPU driver, and Multimodal SDK are pre-installed in the image. No additional environment variable configuration is required after entering the container.
> - The host must have NPU drivers compatible with the CANN version in the container. For details, see [Installing the NPU Driver, Firmware, and CANN](#installing-the-npu-driver-firmware-and-cann).

## Upgrading

### Upgrading the `run` Installation

Before upgrading, make sure that a complete Multimodal SDK installation exists and that the new version is compatible with the corresponding CANN and HDK versions.

1. Download the new version package and go to the package directory.
2. Load the CANN and Multimodal SDK environment variables.
3. Run the upgrade command. `--install-path` must point to **the installation root directory of the old version**:

   ```bash
   ./Ascend-mindxsdk-multimodal_{version}_linux-{arch}.run --upgrade --install-path=/home/work/Mind_SDK
   ```

4. Reload the environment variables and follow the steps in [Verifying the Installation](#installation-verification).

> [!NOTE]
> During the upgrade, the installation program first uninstalls the old version and then installs the new version. If the upgrade fails, check `~/log/mindxsdk/deployment.log` to identify the cause.

### Upgrading the `Wheel` Package

1. Obtain the new `Wheel` package from [Multimodal SDK Releases](https://gitcode.com/Ascend/MultimodalSDK/releases).
2. Load the CANN environment variables.
3. Run the upgrade command. Replace `mm-x.x.x-py3-none-any.whl` with the actual filename:

   ```bash
   pip3 install --upgrade /path/to/mm-x.x.x-py3-none-any.whl
   ```

   If you need to force a reinstall without updating dependencies:

   ```bash
   pip3 install --force-reinstall --no-deps /path/to/mm-x.x.x-py3-none-any.whl
   ```

4. Follow the steps in [Verifying the Installation](#installation-verification) to verify that the upgrade was successful.

> [!NOTE]
>
> - It is recommended to use `--no-deps` to prevent pip from automatically upgrading or downgrading the fixed dependency versions.
> - To roll back, use `pip3 install --force-reinstall /path/to/mm-old-version.whl` to reinstall the old version.

## Uninstalling

### Uninstalling the `run` Installation

The default Multimodal SDK installation directory is `{install-path}/multimodal/`, and the uninstall script is located at `{install-path}/multimodal/script/uninstall.sh`.

**Uninstalling**

1. Go to the `script` directory under the Multimodal SDK installation path and check whether `uninstall.sh` has execute permission:

   ```bash
   cd ${MULTIMODAL_SDK_HOME}/script
   ls -l uninstall.sh
   ```

   If the script does not have execute permission:

   ```bash
   chmod u+x uninstall.sh
   ```

2. Run the uninstall script. The script uninstalls the installed Python `whl` package and deletes the installation directory:

   ```bash
   ./uninstall.sh
   ```

**Verifying the Uninstallation**

```bash
# Confirm that the installation directory has been deleted.
ls ${MULTIMODAL_SDK_HOME} 2>/dev/null && echo "Directory still exists" || echo "Directory has been deleted"

# Confirm that the Python package has been uninstalled.
pip3 show mm 2>/dev/null && echo "mm still exists" || echo "mm has been uninstalled"
```

> [!NOTE]
> `uninstall.sh` is applicable only to normal installations whose directory structure has not been modified after installation. If the installation is abnormal, manually delete all `multimodal`-related folders in the installation directory and run `pip3 uninstall mm -y` to uninstall the Python package.

### Uninstalling the `Wheel` Package

```bash
pip3 uninstall -y mm
pip3 show mm 2>/dev/null && echo "mm still exists" || echo "mm has been uninstalled"
```

## FAQ

| Symptom | Solution |
| ------- | -------- |
| Failed to import `mm` (`run` installation) | Make sure that `source ${MULTIMODAL_SDK_HOME}/script/set_env.sh` has been executed. |
| Failed to import `mm` (`Wheel` installation) | Make sure that the CANN environment variables have been sourced. Run `pip3 show mm` to confirm that the `whl` package is installed. If necessary, run `pip3 install --force-reinstall --no-deps mm-*.whl` to reinstall it. |
| `libcore.so`/`libascendcl.so` not found | For `Wheel` installation, make sure that a compatible version of the `whl` package is used and that CANN has been sourced. For `run` installation, make sure that `set_env.sh` has been sourced. |
| `npu-smi info` has no output | Check whether the NPU driver and firmware are installed correctly. Restart the host and try again if necessary. |
| CANN environment variables do not take effect | Make sure that the CANN `set_env.sh` script has been sourced. The path depends on the actual installation. |
| `torch`/`transformers` version conflict | Install the fixed versions listed in the [table](#installing-other-dependencies). |
| Other issues | See [FAQ > Installation and Environment](../06_references/faq.md#installation-and-environment). |
