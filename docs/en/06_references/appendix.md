# Appendix

## Public Network Addresses Included in the Software

The URLs in the Multimodal SDK installation package are removed after installation is complete. The SDK itself does not access the public URLs and email addresses listed in this manual. Therefore, they pose no risk.

For more public network addresses, see [Multimodal SDK Public Network Addresses.xlsx](../resource/Multimodal%20SDK%20public%20network%20addresses.xlsx) (provided with the Multimodal SDK release documentation package).

## Communication Matrix

Currently, the Multimodal SDK development kit package does not actively open or depend on any ports. Therefore, it does not involve a communication matrix.

## Error Codes

| Error Code (hex) | Error Message | Suggested Action |
| -- | -- | -- |
| 0x10100001 | Invalid parameter | Check whether the input parameter type, the value range, and the mandatory fields are consistent with the API documentation. |
| 0x10100002 | Unsupported type | Confirm that the input data type is in the API support list (for example, dtype and the image format). |
| 0x10100003 | Invalid pointer | Check whether the object passed in is correctly initialized, and avoid using released or null objects. |
| 0x10100004 | Value out of range | Verify whether the parameter constraints are met (for example, the image size is in the range [10, 8192] and the permissions are 640). |
| 0x102003E9 | Memory allocation failed | Check the available system memory, reduce the scale of concurrent or batch processing, and then try again. |
| 0x102003EA | Memory copy failed | Confirm that the source data memory is contiguous and valid, and check whether the data size exceeds the limit. |
| 0x102003EB | Memory release failed | Check whether a double free or memory corruption occurs, and restart the process if necessary. |
| 0x102003EC | Insufficient system memory | Free unnecessary memory usage, or increase the available system memory and then try again. |
| 0x102003ED | Failed to open the file | Confirm that the file path is valid, the file exists, and no other process locks the file. |
| 0x102003EE | Invalid file permissions | Ensure that the file owner is the current user and the permissions are not higher than 640. |
| 0x102003EF | Invalid file size | Check whether the file size is within the API limit (for example, an image does not exceed 1GB). |
| 0x103007D1 | Failed to initialize FFmpeg | Confirm that the FFmpeg dependency is correctly installed, and check whether the video file format is mp4. |
| 0x103007D2 | Failed to initialize Libjpeg | Confirm that the Libjpeg dependency is available, and check the system library path configuration. |
| 0x103007D3 | Failed to read the file with Libjpeg | Confirm that the image is in the jpg/jpeg format and the file is not damaged, and check the file permissions. |
| 0x103007D4 | FFmpeg execution failed | Check the integrity of the video file and whether the resolution is within the range [480, 4096]. |
| 0x10400BB9 | Internal operator failed | Check whether the input data format and size meet the operator requirements, and view the detailed logs to locate the issue. |
| 0x10400BBA | Internal function execution failed | View the SDK log output and confirm whether the preceding steps are completed successfully. |
| 0x10400BBB | Internal type conversion failed | Confirm that the source data is compatible with the target type (for example, dtype and the NCHW/NHWC layout format). |
| 0x10500FA1 | Timed out while waiting | Check the system load, and increase the timeout interval or reduce concurrent tasks as appropriate. |
| 0x10500FA2 | Invalid thread pool state | Restart the related process, and avoid repeatedly submitting tasks in an incorrect state. |

## Environment Variables

The program uses the following environment variables during installation or at runtime. Ensure that they are valid.

**Table 1** Environment variables

| Environment Variable | Description |
| -- | -- |
| PATH | Path to executable programs. |
| LD_LIBRARY_PATH | Path to dynamic link libraries. |
| PYTHONPATH | Default search path for Python module files. |
| HOME | Home directory of the current user. |
| PWD | Current working directory. |
| TMPDIR | Path to temporary files. |
| ASCEND_HOME_PATH | Path to CANN-related resources. Ensure that the path is valid. |
| ASCEND_HOME | CANN installation directory. Ensure that the path is valid. |
| ASCEND_VERSION | CANN version number. Do not change it arbitrarily. |
| ASCEND_CUSTOM_OPP_PATH | AscendC operator deployment path. Do not change it arbitrarily. |
| MULTIMODAL_SDK_HOME | Multimodal SDK installation directory (set by `set_env.sh` during a complete `.run` installation). You do not need to configure it when you install the wheel package. |
| HF_DATASETS_OFFLINE | Loads Hugging Face datasets offline. |
| HF_HUB_OFFLINE | Runs the Hugging Face library in offline mode. |

> [!NOTE]
> When you use the patcher provided by this software to run vLLM inference or other open-source libraries, you also use other environment variables related to those open-source libraries. This document does not describe them. Ensure that the environment variables are actually valid.
