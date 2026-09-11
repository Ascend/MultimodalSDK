# Appendix

## Public Network Addresses Included in the Software

The URLs in the Multimodal SDK installation package are removed once the installation is complete. The public URLs and email addresses listed in this manual are not accessed by the SDK itself and therefore do not pose any security risks.

For more public URLs and email addresses, see [Multimodal SDK 26.0.0 public network addresses.xlsx](../resource/Multimodal%20SDK%20public%20network%20addresses.xlsx) (provided with the Multimodal SDK documentation package).

## Communication Matrix

Currently, the Multimodal SDK does not actively open or depend on any ports, so there is no communication matrix to describe.

## Error Codes

| Error Code (Hexadecimal) | Error Message | Suggested Action |
| -- | -- | -- |
| 0x10100001 | Invalid parameter | Check that the input parameter types, value ranges, and required fields are consistent with the API documentation. |
| 0x10100002 | Unsupported type | Check that the input data type is included in the list of types supported by the API (for example, `dtype` and image formats). |
| 0x10100003 | Invalid pointer | Check that the input object has been properly initialized. Avoid using freed or null objects. |
| 0x10100004 | Value out of range | Check that the parameter constraints (for example, image size [10, 8192] and permissions no more permissive than `640`) are met. |
| 0x102003E9 | Memory allocation failed | Check the available system memory. Reduce concurrency or the batch size and retry. |
| 0x102003EA | Memory copy failed | Check that the source data memory is contiguous and valid, and that the data size does not exceed the limit. |
| 0x102003EB | Memory release failed | Check for duplicate memory releases or memory corruption. Restart the process if necessary. |
| 0x102003EC | Insufficient system memory | Release unnecessary memory usage, or increase the available system memory and retry. |
| 0x102003ED | Failed to open file | Check that the file path is valid, the file exists, and the file is not locked by another process. |
| 0x102003EE | Invalid file permissions | Ensure that the file owner is the current user and that the permissions are no more permissive than `640`. |
| 0x102003EF | Invalid file size | Check that the file size is within the API limits (for example, an image size not exceeding 1 GB). |
| 0x103007D1 | Failed to initialize FFmpeg | Check that the FFmpeg dependencies are properly installed and that the video file is in MP4 format. |
| 0x103007D2 | Failed to initialize Libjpeg | Check that the Libjpeg dependencies are available and that the system library path is properly configured. |
| 0x103007D3 | Failed to read file with Libjpeg | Check that the image is in JPG/JPEG format and is not corrupted, and check the file permissions. |
| 0x103007D4 | FFmpeg execution failed | Check the integrity of the video file and that its resolution is within the range [480, 4096]. |
| 0x10400BB9 | Internal operator failed | Check that the input data format and size meet the operator requirements, and check the detailed logs for troubleshooting. |
| 0x10400BBA | Internal function execution failed | Check the SDK log output and verify that the preceding steps completed successfully. |
| 0x10400BBB | Internal type conversion failed | Check that the source data is compatible with the target type (for example, `dtype` and layouts such as NCHW and NHWC). |
| 0x10500FA1 | Timeout while waiting | Check the system load. Increase the timeout or reduce the number of concurrent tasks as appropriate. |
| 0x10500FA2 | Invalid thread pool state | Restart the related processes and avoid repeatedly submitting tasks when the thread pool is in an incorrect state. |

## Environment Variables

The following environment variables are used during software installation or runtime. Ensure that they are correctly configured and valid.

**Table 1** Environment Variables

| Environment Variable | Description |
| -- | -- |
| PATH | Path to executable files. |
| LD_LIBRARY_PATH | Path to dynamic libraries. |
| PYTHONPATH | Default search path for Python module files. |
| HOME | Current user's home directory. |
| PWD | Current working directory. |
| TMPDIR | Path to temporary files. |
| ASCEND_HOME_PATH | Path to CANN-related resources. Ensure that the path is valid. |
| ASCEND_HOME | CANN installation directory. Ensure that the path is valid. |
| ASCEND_VERSION | CANN version number. Do not change it arbitrarily. |
| ASCEND_CUSTOM_OPP_PATH | AscendC operator deployment path. Do not change it arbitrarily. |
| MULTIMODAL_SDK_HOME | Multimodal SDK installation directory. This variable is set by `set_env.sh` during a full `.run` package installation. It does not need to be configured when installing the Wheel package. |
| HF_DATASETS_OFFLINE | Loads Hugging Face datasets in offline mode. |
| HF_HUB_OFFLINE | Runs the Hugging Face library in offline mode. |

> [!NOTE]
> When you use the patcher provided by this software to run vLLM inference or other open-source libraries, additional environment variables related to those open-source libraries are also used. These variables are not listed in this document. Ensure that the environment variables are correctly configured and valid.
