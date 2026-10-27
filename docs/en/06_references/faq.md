# Frequently Asked Questions (FAQ)

## Installation and Environment

### Q: The Import of `mm` Fails with `ModuleNotFoundError: No module named 'mm'`?

**Cause**: Python cannot find the installed `mm` package (the whl file has not been installed yet, or it was installed in another Python environment).

**Solution**:

```bash
pip3 show multimodalsdk
python3 -c "import mm; print('mm import: OK')"
```

If `pip3 show multimodalsdk` produces no output, complete the installation first by following [Installing the `run` Package](../03_installation_guide/installation_guide.md#option-1-installing-the-run-package) or [Installing the `Wheel` Package](../03_installation_guide/installation_guide.md#option-2-installing-the-wheel-package).

### Q: The Import of `mm` Fails Because `libcore.so` or `libascendcl.so` Cannot Be Found?

**Cause**:

- **Wheel installation**: The CANN environment variables are not loaded, or the whl version does not match or the installation is incomplete.
- **`.run` installation**: `source set_env.sh` has not been run, or the installation directory is incomplete.

**Solution (Wheel installation)**:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh   # Default CANN installation path, change it based on the actual installation path
pip3 install --force-reinstall --no-deps /path/to/multimodalsdk-*.whl
python3 -c "import mm; print('mm import: OK')"
```

The Wheel package does not require `MULTIMODAL_SDK_HOME` to be set. The native libraries are already bundled in the whl file and load automatically when you run `import mm`.

**Solution (`.run` installation)**:

```bash
source ${MULTIMODAL_SDK_HOME}/script/set_env.sh
python3 -c "import mm; print('mm import: OK')"
```

If `${MULTIMODAL_SDK_HOME}` is not set, run the command with the actual installation path, for example:

```bash
source /home/work/Mind_SDK/multimodal/script/set_env.sh
```

### Q: `npu-smi info` Produces No Output or Reports an Error?

**Cause**: The NPU driver or firmware is not installed correctly, or the CANN environment variables do not take effect.

**Solution**:

1. Confirm that Ascend HDK 26.2.0 and CANN 9.2.0 are installed (or a compatible patch version).
2. Load the CANN environment variables: `source /usr/local/Ascend/ascend-toolkit/set_env.sh` (use the actual installation path).
3. Run `npu-smi info` again. If it still fails, restart the host and try again.

### Q: Version Conflicts Between torch and transformers?

**Cause**: The versions of the third-party dependencies are inconsistent with those required by Multimodal SDK.

**Solution**: Install the pinned versions listed in [Installation and Deployment > Other Dependencies](../03_installation_guide/installation_guide.md#other-dependencies):

```bash
pip3 install transformers==5.5.4 pillow==11.2.1 numpy==1.26.4
```

Install `torch` and `torch-npu` according to the version mapping table for `vllm-ascend==v0.23.0`. For details, see [Installation and Deployment > Other Dependencies](../03_installation_guide/installation_guide.md#other-dependencies).

### Q: torchvision Still Reports a Missing lzma Module After lzma Is Installed?

Install the lzma module:

```shell
pip3 install backports.lzma
```

Go to the Python library directory. The following example uses python3.11.4:

```shell
cd /xx/xx/python-3.11.4/lib/python3.11
```

Open `lzma.py` and change the following content:

```python
from _lzma import *
from _lzma import _encode_filter_properties, _decode_filter_properties
```

to the following content:

```python
from backports.lzma import *
from backports.lzma import _encode_filter_properties, _decode_filter_properties
```

## Docker and Quick Start

### Q: The Test Images on the Host Cannot Be Found in the Docker Container?

**Cause**: The host directory is not mounted into the container, or `TEST_IMAGE` uses a host path instead of a container path.

**Solution**: Add a volume mount when you start the container, and use the mount path inside the container:

```bash
docker run ... -v /path/to/testdata:/data ...
export TEST_IMAGE="/data/test.jpg"
```

For details, see [Quick Start > Step 2: Start the Container](../02_quickstart/quickstart.md#step-2-starting-the-container).

### Q: The Container Cannot Access the NPU?

**Cause**: The device number in `--device /dev/davinci*` is inconsistent with that on the host, or the driver-related directories are not mounted.

**Solution**:

1. Run `npu-smi info` on the host to confirm that the NPU is available.
2. Change the number in `--device /dev/davinci0` to the actual device (for example, `davinci1`).
3. Confirm that all driver mount items listed in quick_start are included in the `docker run` command.

### Q: A File Permission Error (0x102003EE) Is Reported When Reading a File?

**Cause**: The API requires that the file owner is the current user and that the permissions are no higher than 640.

**Solution**:

```bash
chmod 640 /path/to/your/file.jpg
```

## Running and Troubleshooting

- For error codes, see [Appendix > Error Codes](./appendix.md#error-codes).
- For environment variable issues, see [Appendix > Environment Variables](./appendix.md#environment-variables).
- For installation step issues, see [Installation and Deployment > Common Issues](../03_installation_guide/installation_guide.md#common-issues).
