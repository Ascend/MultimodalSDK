# FAQ

## Installation and Environment

### Q: Why does `import mm` fail with the error `ModuleNotFoundError: No module named 'mm'`?

**Cause**: Python cannot find the installed `mm` package (the Wheel package has not been installed, or it was installed in a different Python environment).

**Solution**:

```bash
pip3 show mm
python3 -c "import mm; print('mm import: OK')"
```

If `pip3 show mm` produces no output, complete the installation by following the [`run` package installation](../03_installation_guide/installation_guide.md#installing-the-run-package) or [`Wheel` package installation](../03_installation_guide/installation_guide.md#installing-the-wheel-package) procedure.

### Q: Why does `import mm` fail when `libcore.so` or `libascendcl.so` cannot be found?

**Cause**:

- **Wheel installation**: CANN environment variables are not loaded, the Wheel package version does not match the required version, or the installation is incomplete.
- **`.run` installation**: `source set_env.sh` has not been executed, or the installation directory is not complete.

**Solution (Wheel installation)**:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh   # Default CANN installation path; modify it according to the actual installation path
pip3 install --force-reinstall --no-deps /path/to/mm-*.whl
python3 -c "import mm; print('mm import: OK')"
```

The Wheel package does not require `MULTIMODAL_SDK_HOME` to be set. Native libraries are bundled in the Wheel package and are automatically loaded when `mm` is imported.

**Solution (`.run` installation)**:

```bash
source ${MULTIMODAL_SDK_HOME}/script/set_env.sh
python3 -c "import mm; print('mm import: OK')"
```

If `${MULTIMODAL_SDK_HOME}` is not set, execute the command using the actual installation path. For example:

```bash
source /home/work/Mind_SDK/multimodal/script/set_env.sh
```

### Q: Why does `npu-smi info` produce no output or report an error?

**Cause**: The NPU driver or firmware is not properly installed, or the CANN environment variables have not taken effect.

**Solution**:

1. Ensure that Ascend HDK 26.1.0 and CANN 9.1.0 (or compatible patch versions) are installed.
2. Load the CANN environment variables: `source /usr/local/Ascend/ascend-toolkit/set_env.sh` (modify the path according to the actual installation).
3. Run `npu-smi info` again. If the issue persists, restart the host machine and retry.

### Q: Why is there a `torch`/`transformers` version conflict?

**Cause**: The versions of third-party dependencies do not match the Multimodal SDK requirements.

**Solution**: Install the required versions according to [Installation Guide > Other Dependencies](../03_installation_guide/installation_guide.md#installing-other-dependencies):

```bash
pip3 install transformers==4.51.3 pillow==11.2.1 numpy==1.26.4
```

Install `torch` and `TorchNPU` according to the compatibility table for `vllm-ascend==v0.8.5rc1`. For details, see [Installation Guide > Other Dependencies](../03_installation_guide/installation_guide.md#installing-other-dependencies).

### Q: Why does `torchvision` report that `lzma` is missing even though the `lzma` module is installed?

Install the `lzma` module:

```shell
pip3 install backports.lzma
```

Go to the Python library directory. Python 3.11.4 is used as an example below.

```shell
cd /xx/xx/python-3.11.4/lib/python3.11
```

Modify `lzma.py` by changing the following:

```python
from _lzma import *
from _lzma import _encode_filter_properties, _decode_filter_properties
```

to:

```python
from backports.lzma import *
from backports.lzma import _encode_filter_properties, _decode_filter_properties
```

## Docker and Quick Start

### Q: Why can't the Docker container find test images stored on the host?

**Cause**: The host directory has not been mounted into the container, or `TEST_IMAGE` uses the host path instead of the path inside the container.

**Solution**: Add a volume mount when starting the container and use the mounted path inside the container:

```bash
docker run ... -v /path/to/testdata:/data ...
export TEST_IMAGE="/data/test.jpg"
```

For details, see [Quick Start > Step 2: Starting the Container](../02_quickstart/quickstart.md#step-2-starting-the-container).

### Q: Why can't the container access the NPU?

**Cause**: The device number specified by `--device /dev/davinci*` does not match the host device number, or the driver-related directories have not been mounted.

**Solution**:

1. Run `npu-smi info` on the host to ensure that the NPU is available.
2. Change the number in `--device /dev/davinci0` to the actual device number (for example, `davinci1`).
3. Ensure that all driver mount items listed in `quick_start` are included in the `docker run` command.

### Q: Why does a file permission error (0x102003EE) occur when reading a file?

**Cause**: The API requires the file owner to be the current user and the permissions to be no more permissive than `640`.

**Solution**:

```bash
chmod 640 /path/to/your/file.jpg
```

## Running and Troubleshooting

- Encountered an error code? See [Appendix > Error Codes](./appendix.md#error-codes).
- Having environment variable issues? See [Appendix > Environment Variables](./appendix.md#environment-variables).
- Having issues with the installation steps? See [Installation Guide > FAQ](../03_installation_guide/installation_guide.md#faq).
