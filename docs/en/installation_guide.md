# Installation and Deployment

## Obtaining the Installation Package

This section describes how to obtain the required software package and its corresponding digital signature file.

**Table 1**  Software packages

|Component|Package|
|--|--|
|Multimodal SDK|Multimodal software package|

**Verifying the Software Digital Signature**

To prevent the package from being maliciously tampered with during transmission or storage, download the digital signature file for integrity check while downloading the package.

After downloading the software package, verify its PGP digital signature according to the *OpenPGP Signature Verification Guide*. If the verification fails, do not use the software package, and contact Huawei technical support.

Before you use a software package for installation or upgrade, perform the preceding operations to verify its digital signature to ensure that the software package is not tampered with.

For carrier users, visit [https://support.huawei.com/carrier/digitalSignatureAction](https://support.huawei.com/carrier/digitalSignatureAction).

For enterprise users, visit [https://support.huawei.com/enterprise/en/tool/software-digital-signature-openpgp-validation-tool-TL1000000054](https://support.huawei.com/enterprise/en/tool/software-digital-signature-openpgp-validation-tool-TL1000000054)

## Installation Prerequisites

### Installing Ubuntu Dependencies

For the names, corresponding versions, and acquisition suggestions of dependencies required in the Ubuntu environment, see [Table 1 Ubuntu dependencies and corresponding versions](#table20540329125614).

**Table 1** Ubuntu dependencies and corresponding versions

<a id="table20540329125614"></a>

|Dependency|Recommended Version|Acquisition Suggestion|
|--|--|--|
|Python|3.9 or later|Obtain the source package from the Python official website and compile and install it.|
|CMake|3.14 or later|Install using a package manager. The installation command is as follows:<br>`sudo apt-get install -y cmake`<br>If the version in the package manager does not meet the minimum version requirement, you can install the package using the source code.|
|Make|4.1 or later|Install using a package manager. The installation command is as follows:<br>`sudo apt-get install -y make`<br>If the version in the package manager does not meet the minimum version requirement, you can install the package using the source code.|
|GCC|9.4 or later|Install using a package manager. The installation command is as follows:<br>`sudo apt-get install -y gcc`<br>If the version in the package manager does not meet the minimum version requirement, you can install the package using the source code.|
|G++|9.4 or later|Install using a package manager. The installation command is as follows:<br>`sudo apt-get install -y g++`<br>If the version in the package manager does not meet the minimum version requirement, you can install the package using the source code.|

Run the following commands to query the version information of dependency packages such as GCC, G++, Make, CMake, and Python to confirm whether they are installed:

```bash
gcc --version
g++ --version
make --version
cmake --version
python3 --version
```

If the information similar to the following is returned, the corresponding software has been installed:

```ColdFusion
gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
GNU Make 4.1
cmake version 3.14.2
Python 3.10.12
```

### Installing the NPU Driver, Firmware, and CANN

**Downloading Dependency Software Packages**

**Table 1** Package list

<table><thead align="left"><tr id="row724174542813"><th class="cellrowborder" valign="top" width="21.72%" id="mcps1.2.4.1.1"><p id="p112413454282">Software Type</p>
</th>
<th class="cellrowborder" valign="top" width="53.68000000000001%" id="mcps1.2.4.1.2"><p id="p725124572814">Package Name</p>
</th>
<th class="cellrowborder" valign="top" width="24.6%" id="mcps1.2.4.1.3"><p id="p919101122613">How to Obtain</p>
</th>
</tr>
</thead>
<tbody><tr id="row20251045102819"><td class="cellrowborder" valign="top" width="21.72%" headers="mcps1.2.4.1.1 "><p id="p73893235281">Ascend NPU driver</p>
</td>
<td class="cellrowborder" valign="top" width="53.68000000000001%" headers="mcps1.2.4.1.2 "><p id="p2444175718384">Ascend-hdk-{npu_type}-npu-driver_{version}_linux-{arch}.run</p>
</td>
<td class="cellrowborder" rowspan="3" valign="top" width="24.6%" headers="mcps1.2.4.1.3 "><p id="p1130893682616">Click the <a href="https://www.hiascend.com/developer/download/commercial/result?module=cann" target="_blank" rel="noopener noreferrer">download link</a>, configure the supporting resources in the "Edit Resource Selection" area on the left, filter the supporting software packages, and then obtain the required packages after you confirm the version information.</p>
</td>
</tr>
<tr id="row225114514289"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p339015237286">Ascend NPU firmware</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p14444165713389">Ascend-hdk-{npu_type}-npu-firmware_{version}.run</p>
</td>
</tr>
<tr id="row1838172311404"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p81248297404">CANN software package</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p91244293404">Ascend-cann-toolkit_{version}_linux-{arch}.run</p>
</td>
</tr>
</tbody>
</table>

>[!NOTE]NOTE
>
>- `{npu_type}` is the chip name.
>- `{version}` is the software version number.
>- `{arch}` is the CPU architecture.

**Installing the NPU Driver, Firmware, and CANN**

1. For details, see "Installing the NPU Driver and Firmware" (commercial edition) or "Installing the NPU Driver and Firmware" (community edition) in the *CANN Software Installation Guide*.
2. For details, see "Installing CANN" (commercial edition) or "Installing CANN" (community edition) in the *CANN Software Installation Guide*.

    >[!NOTE]NOTE
    >- The user who installs CANN (Toolkit), the NPU driver firmware, and Multimodal SDK must be the same user, preferably a common user.
    >- When CANN is installed, the related CANN dependencies must also be installed to ensure that Multimodal SDK can work properly.

### Installing Python Dependencies

To use Multimodal SDK features, all dependencies listed in [Table 1](#table20540329125613) must be installed. However, if you are using the patcher phase in patcher, please install the image specified in [patcher](./api/patcher.md). In this case, no additional installation of the dependencies in [Table 1](#table20540329125613) is required.

**Table 1** Dependency versions

<a name="table20540329125613"></a>

|Dependency|Recommended Version|Acquisition Suggestion|
|--|--|--|
|transformers|4.51.3|Install it using pip. The installation command is as follows:<br>`pip3 install transformers==4.51.3`|
|pillow|11.2.1|Install it using pip. The installation command is as follows:<br>`pip3 install pillow==11.2.1`|
|numpy|1.26.4|Install it using pip. The installation command is as follows:<br>`pip3 install numpy==1.26.4`|
|torch|2.5.1|Install it using pip. The installation command is as follows:<br>`pip3 install torch==2.5.1`|

**Precautions**

If you need to install third-party software other than the Multimodal SDK software package, upgrade the software to the latest version in a timely manner and fix existing vulnerabilities.

## Installing Multimodal SDK

**Installation Precautions**

- The user installing and running Multimodal SDK must meet the following requirements:
    - You are advised to install and run Multimodal SDK as a common user.
    - The user installing Multimodal SDK and the user running Multimodal SDK must be the same user.
    - The user installing CANN (Toolkit), the NPU driver firmware, and Multimodal SDK must be the same user, preferably a common user.

- Logs related to package installation, upgrade, uninstallation, and version queries are saved to `~/log/mindxsdk/deployment.log`. Logs related to integrity verification, file extraction, and access through the `tar` command are saved to `~/log/makeself/makeself.log`. You can view the corresponding files to complete subsequent log tracing and audit.

**Installation Preparation**

Ensure that the CANN environment variable configuration script has been run in the installation environment so that the environment variables take effect. Use the actual installation path to determine the script execution path.

```bash
# Install the toolkit package
. /usr/local/Ascend/cann/set_env.sh # This is the default CANN installation path. Modify it based on the actual installation path
```

**Installation Procedure**

1. Log in to the installation environment as the package installation user.
2. Upload the Multimodal SDK package to any path in the installation environment and go to the package directory.
3. Add execute permissions to the package.

    ```bash
    chmod u+x Ascend-mindxsdk-multimodal_{version}_linux-{arch}.run
    ```

4. Run the following command to verify the consistency and integrity of the package.

    ```bash
    ./Ascend-mindxsdk-multimodal_{version}_linux-{arch}.run --check
    ```

    If the system does not have the `shasum` or `sha256sum` tool, the verification fails. In that case, install `shasum` or `sha256sum` yourself.

    If the following information is displayed, the package has passed verification.

    ```bash
    Verifying archive integrity...  100%   SHA256 checksums are OK. All good.    
    ```

5. Create the installation path for the Multimodal SDK package. You are advised not to install it in the `/tmp` path.
    - If the user does not specify an installation path, the software is installed by default in the path where the Multimodal SDK package is located.
    - If the user wants to specify an installation path, create the installation path first. For example, if the installation path is `/home/work/Mind_SDK`:

        ```bash
        mkdir -p /home/work/Mind_SDK
        ```

6. Go to the path where the Multimodal SDK package is located and run the following command to install Multimodal SDK. For constraints on the installation path, see [Table 1](#table1361972315353).

    - If the user specifies an installation path, the software is installed in the specified path. For example, if the installation path is `/home/work/Mind_SDK`:

        ```bash
        ./Ascend-mindxsdk-multimodal_{version}_linux-{arch}.run --install --install-path=/home/work/Mind_SDK
        ```

        Or

        ```bash
        echo y | ./Ascend-mindxsdk-multimodal_{version}_linux-{arch}.run --install --install-path=/home/work/Mind_SDK
        ```

    - If the user does not specify an installation path, the software is installed in the current path.

        ```bash
        ./Ascend-mindxsdk-multimodal_{version}_linux-{arch}.run --install
        ```

        Or

        ```bash
        echo y | ./Ascend-mindxsdk-multimodal_{version}_linux-{arch}.run --install
        ```

    >[!NOTE]NOTE
    >`--install` also supports optional parameters, as shown in [Table 1](#table1361972315353).

7. Go to the `script` directory in the Multimodal SDK installation path and run the following command to make the Multimodal SDK environment variables take effect.

    ```bash
    source set_env.sh
    ```

**References**

**Table 1** Optional parameters for the --install command

<a id="table1361972315353"></a>

|Input Parameter|Description|
|--|--|
|--help \| -h|Queries help information.|
|--info|Queries package build information.|
|--list|Queries the file list.|
|--check|Checks package integrity.|
|--quiet \| -q|Enables the quiet mode. It must be used together with the `--install` or `--upgrade` parameter.|
|--nox11|Deprecated|
|--noexec|Does not run the embedded scripts.|
|--extract=&lt;path&gt;|Extracts files directly to the target directory (absolute path). It is usually used with the `--noexec` option to extract files without running them.|
|--tar arg1 [arg2 ...]|Accesses the contents of the archive file using the `tar` command.|
|--install|Multimodal SDK package installation command.<ul><li>The current path and the installation path must not contain invalid characters. Only uppercase and lowercase letters, digits, and special characters `-`, `_`, `.`, and `/` are supported.</li><li>The installation path must not contain a file or folder named `multimodal`. If a symbolic link named `multimodal` exists, it is overwritten.</li></ul>|
|--install-path=\<path>|(Optional) Customizes the root directory for package installation. If it is not set, the default is the directory where the command is executed.<ul><li>You are advised to specify an absolute path when installing Multimodal SDK. This parameter conflicts with the `--version` input parameter. You are not advised to install Multimodal SDK in `/tmp`. It must be used together with the `--install` or `--upgrade` parameter. When you use it with `--upgrade`, `--install-path` indicates the installation directory of the old package, and the upgrade runs in that directory. The input path cannot contain invalid characters. Only uppercase and lowercase letters, digits, and special characters `-`, `_`, `.`, and `/` are supported.</li></ul>|
|--upgrade|Upgrades Multimodal SDK software package. If an installation already exists, the system prompts you to choose whether to delete the previous installation and then reinstall Multimodal SDK.|
|--version|Queries the version information about Multimodal SDK software package. When this operation is performed, the system temporarily installs Multimodal SDK run package in `/tmp` and uninstalls it after the version number is queried.|

>[!NOTE]NOTE
>The following parameters are not displayed in the `--help` output. Do not use them directly.
>
>- `--xwin`: Runs in xwin mode.
>- `--phase2`: Requires the second phase to be executed.

# Uninstallation

**Procedure**

1. Go to the Multimodal SDK installation path and check whether the `uninstall.sh` script in the `script` directory of the Multimodal SDK installation directory has execute permissions.

    ```bash
    cd multimodal/script
    ls -l uninstall.sh
    ```

    If the script does not have execute permissions, run the following command to grant execute permissions to `uninstall.sh`.

    ```bash
    chmod u+x uninstall.sh
    ```

2. Run the following command to start uninstallation. When the uninstallation script runs, it uninstalls the installed Python wheel package and deletes the installation directory.

    ```bash
    ./uninstall.sh
    ```

    >[!NOTE]NOTE
    >Using the `uninstall.sh` script to uninstall applies only to a normal installation path and to installations whose file structure has not been modified after installation. If you need to handle an installation exception, delete any folder related to multimodal in the installation directory, and use `pip uninstall mm` to uninstall the installed Python package files.
