# Installation and Deployment

## Installation Instructions

This document applies to the installation of the latest version of Multimodal SDK. Only the Atlas 800I A2 is supported as the hardware, and only Ubuntu 22.04 or openEuler 24.03 is supported as the operating system. You are advised to reserve at least 16 GB of available drive space and at least 8 GB of available memory. Multimodal SDK is currently an officially supported release, and is used together with CANN 9.2.0 and the related supporting components.

Multimodal SDK supports the following installation methods: [offline installation](#offline-installation) (`run` package/`Wheel` package), [source code installation](#source-code-installation), and [image installation](#image-installation). `run` package: a self-extracting installation script that contains all dependencies. `Wheel` package: a Python binary distribution package. Image installation: deployment based on a containerized image.

If you use offline installation or source code installation, first [install the dependencies](#installing-dependencies). If you use image installation, skip this step.

**Precautions**

If you need to install third-party software other than the Multimodal SDK package, upgrade it to the latest version in a timely manner, and track and fix existing vulnerabilities.

## Installing Dependencies

### Installing the NPU Driver, Firmware, and CANN

See [CANN Quick Installation](https://www.hiascend.com/cann/download). Select Compute Architecture for Neural Networks (CANN) 9.2.0 and Hardware Developer Kit (HDK) 26.2.0 to install the Neural Processing Unit (NPU) driver and firmware along with the CANN software (including the Toolkit and ops packages), and then configure the environment variables.

### Other Dependencies

| Dependency | Recommended Version | How to Obtain |
| ------------ | ------------------------- | ---------------------------------------------------------------------- |
| Python       | 3.10 minimum, **3.12 recommended** | You are advised to install it with a package manager:<br>Ubuntu: `sudo apt-get install -y python3 python3-pip python3-dev`<br>openEuler: `sudo yum install -y python3 python3-pip python3-devel`<br>If the version bundled with the OS is too old, compile a higher version from source or install a higher version. |
| transformers | 5.5.4                     | Install it with pip:<br>`pip3 install transformers==5.5.4`                       |
| einops       | 0.8.2                     | Install it with pip:<br>`pip3 install einops==0.8.2`                            |
| pillow       | 11.2.1 or later               | Install it with pip:<br>`pip3 install pillow==11.2.1`                           |
| numpy        | 1.26.4                    | Install it with pip:<br>`pip3 install numpy==1.26.4`                            |
| torch        | 2.10.0                     | Install it with pip:<br>`pip3 install torch==2.10.0` |
| torch-npu     | 2.10.0.post4               | Install it with pip:<br>`pip3 install torch-npu==2.10.0.post4`<br> |

## Installation Methods

Multimodal SDK provides the following installation methods: offline installation (`run` package/`Wheel` package), source code installation, and image installation. Choose the appropriate method based on your scenario.

### Offline Installation

#### Option 1: Installing the `run` Package

**Prerequisites**

- Complete [installing the dependencies](#installing-dependencies).

**Installation Notes**

- You are advised to use the **same common user** to install and run CANN, the NPU driver and firmware, and Multimodal SDK.
- Logs related to package installation, upgrade, uninstallation, and version query are saved to `~/log/mindxsdk/deployment.log`. Logs related to integrity verification, file extraction, and access through the `tar` command are saved to `~/log/makeself/makeself.log`. You can view these files for subsequent log tracking and auditing.

**Preparation**

Download the Multimodal SDK package (Ascend-mindxsdk-multimodal_{version}_linux-aarch64.run) from [Multimodal SDK Releases](https://atomgit.com/Ascend/MultimodalSDK/releases), where `{version}` is the SDK version number (for example, 26.2.0).

The following commands use `${MMSDK_PACKAGE}` to represent the file name of the downloaded `.run` package. Set it according to the actual file name, for example:

```bash
export MMSDK_PACKAGE=Ascend-mindxsdk-multimodal_26.2.0_linux-aarch64.run
```

Ensure that you have run the CANN environment variable configuration script in the installation environment:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh   # Default path, change it based on the actual installation path
```

**Installation Steps**

1. Log in to the installation environment as the user who installs the package.
2. Upload the Multimodal SDK package to any path in the installation environment and go to the path where the package resides.
3. Grant the executable permission on the package:

   ```bash
   chmod u+x ${MMSDK_PACKAGE}
   ```

4. Run the following command to verify the consistency and integrity of the package:

   ```bash
    ./${MMSDK_PACKAGE} --check
   ```

   If the system lacks the `sha256sum` tool, the verification fails. Run the following command to install it:

   ```bash
   sudo apt-get install -y coreutils
   ```

   If `Verifying archive integrity... OK` is displayed, the package has passed the verification.

5. Create the installation path for the Multimodal SDK package (optional).

   - If you do not specify an installation path, the software is installed to the path where the Multimodal SDK package resides by default.
   - If you want to specify an installation path, create the installation path first. The following example uses the installation path `/home/work/Mind_SDK`:

   ```bash
   mkdir -p /home/work/Mind_SDK
   ```

6. Go to the path where the Multimodal SDK package resides, and run the following commands to install Multimodal SDK (for installation path constraints, see [Appendix > Installation Command Parameters](#table-install-params)).

   **Specify the installation path** (using `/home/work/Mind_SDK` as an example):

   ```bash
   ./${MMSDK_PACKAGE} --install --install-path=/home/work/Mind_SDK
   ```

   **Install to the current path by default**:

   ```bash
   ./${MMSDK_PACKAGE} --install
   ```

   Common optional parameters: `--quiet` (quiet mode, used together with `--install`) and `--version` (query the version). For complete parameter details, see [Installation Command Parameters](#table-install-params).

<a id="installation-verification"></a>

**Installation Verification**

After the complete `run` installation is finished, load the Multimodal SDK environment variables:

```bash
source ${MULTIMODAL_SDK_HOME}/script/set_env.sh
```

If `${MULTIMODAL_SDK_HOME}` is not set, go to the `script` directory in the installation directory first, and then run:

```bash
# Example: the installation path is /home/work/Mind_SDK
source /home/work/Mind_SDK/multimodal/script/set_env.sh
```

If you want the settings to take effect automatically at every login, append the preceding command to `~/.bashrc`:

```bash
echo 'source ${MULTIMODAL_SDK_HOME}/script/set_env.sh' >> ~/.bashrc
source ~/.bashrc
```

For details about the environment variables, see [Appendix > Environment Variables](../06_references/appendix.md#environment-variables).

Run the following verification:

```bash
# For a .run installation, you can view the installation path. For a Wheel installation, you can skip this line
echo "MULTIMODAL_SDK_HOME=${MULTIMODAL_SDK_HOME}"
python3 -c "import mm; print('mm import: OK')"
```

Use a jpg/jpeg test image to verify the functionality (replace `$TEST_IMAGE` with the actual path):

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

**Installation Command Parameters**

<a id="table-install-params"></a>

| Input Parameter | Description |
| --- | --- |
| `--help`, `-h` | Query help information. |
| `--info` | Query the package build information. |
| `--list` | Query the file list. |
| `--check` | Query the package integrity. |
| `--quiet`, `-q` | Enable quiet mode. This parameter must be used together with `--install` or `--upgrade`. |
| `--noexec` | Do not run the embedded script. |
| `--extract=` | Extract files directly to the target directory (an absolute path or a relative path). This parameter is usually used together with `--noexec` to extract files only without running them. |
| `--tar arg1 [arg2 ...]` | Access the contents of the archive file through the `tar` command. |
| `--install` | Perform the Multimodal SDK installation. Constraints: The current path and the installation path cannot contain invalid characters. Only uppercase and lowercase letters, digits, `-`, `_`, `.`, and `/` are supported. The installation path cannot contain a file or folder named `multimodal`. If a soft link named `multimodal` exists, it is overwritten. |
| `--install-path=` | (Optional) Customize the installation root directory. If this parameter is not set, the directory where the current command is run is used by default. You are advised to use an absolute path. This parameter must be used together with `--install` or `--upgrade`. When it is used with `--upgrade`, it indicates the installation directory of the old package. |
| `--upgrade` | Perform the Multimodal SDK upgrade. Ensure that a complete old version installation directory exists. |
| `--version` | Query the package version. |

**Internal Parameters**

> [!NOTE]
> The following parameters are for internal use. Do not call them directly.

The following parameters are not displayed in `--help`:

- `--nox11`: A deprecated interface with no actual effect
- `--xwin`: Run in xwin mode.
- `--phase2`: Require the phase 2 action to be performed.

#### Option 2: Installing the `Wheel` Package

**Prerequisites**

- Complete [installing the dependencies](#installing-dependencies).

**Installation Notes**

If you only need to use the `mm` package in a Python environment, you can skip the `run` installation package and install the `Wheel` package directly. The `Wheel` package already bundles `libcore.so` and native dependencies such as FFmpeg, libjpeg-turbo, and soxr. You do **not** need to run `source ${MULTIMODAL_SDK_HOME}/script/set_env.sh` or configure `MULTIMODAL_SDK_HOME`.

> [!NOTE]
>
> - The `Wheel` package does **not** include CANN (such as `libascendcl.so`) or Python third-party dependencies. Before use, you must complete the installation according to [Installing Dependencies](#installing-dependencies).
> - Before each Python startup, you must load the CANN environment variables (the path depends on the actual installation):
>
>   ```bash
>   source /usr/local/Ascend/ascend-toolkit/set_env.sh
>   ```

**Preparation**

You can obtain the `multimodalsdk-*.whl` compatible with the SDK version in any of the following ways:

1. **Download from the Release page**: Obtain the `Wheel` file of the same version as the `run` package from [Multimodal SDK Releases](https://atomgit.com/Ascend/MultimodalSDK/releases) (if it is provided with the release).
2. **Extract from the `run` package**:

   ```bash
   ./${MMSDK_PACKAGE} --noexec --extract=/tmp/mmsdk_extract
   find /tmp/mmsdk_extract -name 'multimodalsdk-*.whl'
   ```

3. **Build it yourself**: After you complete the source code build according to [CONTRIBUTING.md](../../../CONTRIBUTING.md), obtain the generated `multimodalsdk-*.whl` in the `MultimodalSDK/dist/` directory.

**Installation Steps**

1. Confirm that the CANN environment variables are loaded (see the preceding description).
2. Confirm that the Python dependencies listed in [Installing Dependencies](#installing-dependencies) are installed.
3. Install the `Wheel` package (replace `multimodalsdk-1.0.0-py3-none-any.whl` with the actual file name):

   ```bash
   pip3 install /path/to/multimodalsdk-1.0.0-py3-none-any.whl
   ```

   If an old version already exists in the environment, you can force a reinstallation:

   ```bash
   pip3 install --force-reinstall --no-deps /path/to/multimodalsdk-1.0.0-py3-none-any.whl
   ```

   > [!NOTE]
   > You are advised to use `--no-deps` to prevent pip from automatically upgrading or downgrading to a fixed dependency version.

4. Perform the verification by referring to the steps in [Installation Verification](#installation-verification), and confirm that the SDK is ready (a `Wheel` package installation requires no additional configuration of the `MULTIMODAL_SDK_HOME` environment variable).

### Source Code Installation

To build Multimodal SDK from source, see [CONTRIBUTING.md](../../../CONTRIBUTING.md) to prepare the build environment and build the source code. After the build is complete:

- If `multimodalsdk-*.whl` is generated, you can complete the installation by following the steps in [Offline Installation: Installing the `Wheel` Package](#option-2-installing-the-wheel-package).
- If a `run` installation package is generated, you can complete the installation by following the steps in [Offline Installation: Installing the `run` Package](#option-1-installing-the-run-package).

### Image Installation

Multimodal SDK supports containerized deployment. You can obtain the image in either of the following ways:

**Option 1: Pulling the Official Image**

Pull the prebuilt image directly from the Ascend community image repository. For details, see the [Multimodal SDK image repository](https://www.hiascend.com/developer/ascendhub/detail/e0081aa3c4dd441dbd6a379bee8cc4c9).

**Option 2: Building the Image Locally**

Build the image yourself by using the Dockerfile provided in the `docker/` directory of the project. For detailed build instructions, see [docker/OVERVIEW.zh.md](../../../docker/OVERVIEW.zh.md).

**Starting the Container**

After you obtain the image, run the following command to start the container:

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

After you enter the container, you can use it directly:

```bash
python3 -c "import mm; print('mm import: OK')"
```

> [!NOTE]
>
> - The image has CANN, the NPU driver, and Multimodal SDK preinstalled. After you enter the container, no additional environment variable configuration is required.
> - The host must have an NPU driver installed that is compatible with the CANN version in the container (for details, see [Installing the NPU Driver, Firmware, and CANN](#installing-the-npu-driver-firmware-and-cann)).

## Upgrade

### Upgrading the `run` Installation

Before the upgrade, confirm that a complete Multimodal SDK installation exists and that the new version matches the CANN/Ascend HDK version.

1. Download the new version installation package and go to the path where the package resides.
2. Load the CANN and Multimodal SDK environment variables.
3. Run the upgrade command (`--install-path` must point to the **installation root directory of the old version**):

   ```bash
   ./Ascend-mindxsdk-multimodal_{version}_linux-{arch}.run --upgrade --install-path=/home/work/Mind_SDK
   ```

4. Reload the environment variables and run the commands in [Installation Verification](#installation-verification).

> [!NOTE]
> During the upgrade, the installation program uninstalls the old version first and then installs the new version. If the upgrade fails, check `~/log/mindxsdk/deployment.log` to identify the cause.

### Upgrading the `Wheel` Package

1. Obtain the new `Wheel` package ([Multimodal SDK Releases](https://atomgit.com/Ascend/MultimodalSDK/releases)).
2. Load the CANN environment variables.
3. Run the upgrade command (replace `multimodalsdk-x.x.x-py3-none-any.whl` with the actual file name):

   ```bash
   pip3 install --upgrade /path/to/multimodalsdk-x.x.x-py3-none-any.whl
   ```

   If you need to force an overwrite without updating dependencies:

   ```bash
   pip3 install --force-reinstall --no-deps /path/to/multimodalsdk-x.x.x-py3-none-any.whl
   ```

4. Run the commands in [Installation Verification](#installation-verification) to confirm that the upgrade is successful.

> [!NOTE]
>
> - You are advised to use `--no-deps` to prevent pip from automatically upgrading or downgrading the pinned dependency version.
> - To roll back, run `pip3 install --force-reinstall /path/to/multimodalsdk-old-version.whl` to reinstall the old version.

## Uninstall

### Uninstalling the `run` Installation

The default installation directory structure of Multimodal SDK is `{install-path}/multimodal/`, and the uninstallation script is located at `{install-path}/multimodal/script/uninstall.sh`.

**Procedure**

1. Go to the `script` directory in the Multimodal SDK installation path and check whether `uninstall.sh` has the executable permission:

   ```bash
   cd ${MULTIMODAL_SDK_HOME}/script
   ls -l uninstall.sh
   ```

      If the script does not have the executable permission:

   ```bash
   chmod u+x uninstall.sh
   ```

2. Run the uninstallation script. The script uninstalls the installed Python whl package and deletes the installation directory:

   ```bash
   ./uninstall.sh
   ```

**Uninstallation Verification**

```bash
# Confirm that the installation directory is deleted
ls ${MULTIMODAL_SDK_HOME} 2>/dev/null && echo "the directory still exists" || echo "the directory is deleted"

# Confirm that the Python package is uninstalled
pip3 show multimodalsdk 2>/dev/null && echo "multimodalsdk still exists" || echo "multimodalsdk is uninstalled"
```

> [!NOTE]
> `uninstall.sh` applies only to installations performed by the normal installation method where the directory structure has not been modified after installation. If the installation is abnormal, manually delete all `multimodal`-related folders in the installation directory, and run `pip3 uninstall multimodalsdk -y` to uninstall the Python package.

### Uninstalling the `Wheel` Package Installation

```bash
pip3 uninstall -y multimodalsdk
pip3 show multimodalsdk 2>/dev/null && echo "multimodalsdk still exists" || echo "multimodalsdk is uninstalled"
```

## Common Issues

| Symptom                        | Solution                                                        |
| ------------------------- | ----------------------------------------------------------- |
| Failure to import `mm` (`run` installation)      | Confirm that you have run `source ${MULTIMODAL_SDK_HOME}/script/set_env.sh`.     |
| Failure to import `mm` (`Wheel` installation)       | Confirm that the CANN environment variables have been sourced. Run `pip3 show multimodalsdk` to confirm that the whl is installed. If necessary, reinstall it with `pip3 install --force-reinstall --no-deps multimodalsdk-*.whl`. |
| `libcore.so`/`libascendcl.so` not found | For a `Wheel` installation: confirm that the matching version whl is used and that CANN is sourced. For a `.run` installation: confirm that `set_env.sh` has been sourced. |
| No output from `npu-smi info`        | Check whether the NPU driver and firmware are installed successfully, and restart the system and try again if necessary.                                 |
| CANN environment variables do not take effect              | Confirm that the CANN `set_env.sh` has been sourced, and use the path of the actual installation.                    |
| torch/transformers version conflict | Install the pinned versions by referring to the [table](#other-dependencies).                           |
| More issues                      | [FAQ > Installation and Environment](../06_references/faq.md#installation-and-environment) |
