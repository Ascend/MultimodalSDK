# FAQ

## Installation and Environment

### Q: Failed to import `mm`, showing `ModuleNotFoundError: No module named 'mm'`?

**Cause**: Python cannot find the installed `mm` package (whl not installed, or installed in a different Python environment).

**Solution**:

```bash
pip3 show mm
python3 -c "import mm; print('mm import: OK')"
```

If `pip3 show mm` has no output, please first complete the installation according to [`run` package installation](./installation_guide.md#mode-1-run-package-installation) or [`Wheel` package installation](./installation_guide.md#mode-2-wheel-package-installation).

### Q: Failed to import `mm`, showing `libcore.so` or `libascendcl.so` not found?

**Cause**:

- **Wheel installation**: CANN environment variables not loaded, or whl version mismatch/incomplete installation.
- **`.run` installation**: `source set_env.sh` not executed, or installation directory incomplete.

**Solution (Wheel installation)**:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh   # Path depends on actual installation
pip3 install --force-reinstall --no-deps /path/to/mm-*.whl
python3 -c "import mm; print('mm import: OK')"
```

Wheel package does not require setting `MULTIMODAL_SDK_HOME`. Native libraries are bundled in the whl and automatically loaded during `import mm`.

**Solution (`.run` installation)**:

```bash
source ${MULTIMODAL_SDK_HOME}/script/set_env.sh
python3 -c "import mm; print('mm import: OK')"
```

If `${MULTIMODAL_SDK_HOME}` is not set, execute according to the actual installation path, for example:

```bash
source /home/work/Mind_SDK/multimodal/script/set_env.sh
```

### Q: `npu-smi info` has no output or reports an error?

**Cause**: NPU driver/firmware not correctly installed, or CANN environment variables not effective.

**Solution**:

1. Confirm Ascend HDK 26.0.RC1 and CANN 9.0.0 (or compatible patch version) are installed.
2. Load CANN environment variables: `source /usr/local/Ascend/ascend-toolkit/set_env.sh` (path depends on actual installation).
3. Re-execute `npu-smi info`; if it still fails, restart the host and retry.

### Q: torch / transformers version conflict?

**Cause**: Third-party dependency versions are inconsistent with Multimodal SDK requirements.

**Solution**: Install fixed versions according to [Other Dependencies](./installation_guide.md#other-dependencies):

```bash
pip3 install transformers==4.51.3 pillow==11.2.1 numpy==1.26.4
```

`torch` and `torch_npu` need to be installed according to the CANN 9.0.0 compatibility table. Please refer to [Other Dependencies](./installation_guide.md#other-dependencies).

### Q: lzma module is installed, but torchvision still reports missing lzma?

Install the lzma module:

```shell
pip install backports.lzma
```

Enter the Python library directory, using python3.11.4 as an example:

```shell
cd /xx/xx/python-3.11.4/lib/python3.11
```

Modify `lzma.py`, change the following content:

```python
from _lzma import *
from _lzma import _encode_filter_properties, _decode_filter_properties
```

To:

```python
from backports.lzma import *
from backports.lzma import _encode_filter_properties, _decode_filter_properties
```

## Docker and Quick Experience

### Q: Test image from host not found in Docker container?

**Cause**: Host directory not mounted into the container, or `TEST_IMAGE` uses the host path instead of the container path.

**Solution**: Add volume mount when starting the container, and use the mounted path inside the container:

```bash
docker run ... -v /path/to/testdata:/data ...
export TEST_IMAGE="/data/test.jpg"
```

For details, see [Quick Start - Step 2](./quickstart.md#step-2-start-the-container).

### Q: Container cannot access NPU?

**Cause**: `--device /dev/davinci*` device number is inconsistent with the host, or driver-related directories not mounted.

**Solution**:

1. Execute `npu-smi info` on the host to confirm NPU is available.
2. Change the number in `--device /dev/davinci0` to the actual device (e.g., `davinci1`).
3. Confirm that all driver mount items listed in quick_start are included in the `docker run` command.

### Q: File permission error when reading file (0x102003EE)?

**Cause**: API requires file owner to be the current user, and permissions not higher than 640.

**Solution**:

```bash
chmod 640 /path/to/your/file.jpg
```

## Running and Troubleshooting

- Encountered error codes? See [Appendix - Error Codes](./appendix.md#error-codes)
- Environment variable issues? See [Appendix - Environment Variables](./appendix.md#environment-variables)
- Installation step issues? See [FAQs](./installation_guide.md#faqs)
