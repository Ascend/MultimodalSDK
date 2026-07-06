# Appendix

## Public Network Addresses Included in the Software

The URLs in the Multimodal SDK installation package are removed after installation. The SDK does not access them. Therefore, they pose no risk.

The public URLs and email addresses in this Multimodal SDK manual are not accessed by the SDK itself. Therefore, they pose no risk.

For more public URLs and email addresses, see [Multimodal SDK 26.0.0 public network addresses.xlsx](<./resource/Multimodal SDK 26.0.0 public network addresses.xlsx>).

## Communication Matrix

Currently, the Multimodal SDK package does not actively open or depend on any ports. Therefore, it does not involve a communication matrix.

## Error Codes

|Error Code (hex)|Error Message|Suggested Action|
|--|--|--|
|0x10100001|Invalid parameter|Check whether the input parameter type, value range, and required fields are consistent with the API documentation.|
|0x10100002|Unsupported type|Confirm that the input data type is in the API support list (such as dtype, image format, etc.).|
|0x10100003|Invalid pointer|Check whether the passed object has been correctly initialized and avoid using released or null objects.|
|0x10100004|Value out of range|Check whether parameter constraints (such as image size [10, 8192], permissions 640, etc.) are met.|
|0x102003E9|Memory allocation failed|Check available system memory, reduce concurrency or batch processing size and retry.|
|0x102003EA|Memory copy failed|Confirm that the source data memory is contiguous and valid, and check whether the data size exceeds the limit.|
|0x102003EB|Memory release failed|Check for duplicate releases or memory corruption, and restart the process if necessary.|
|0x102003EC|Insufficient system memory|Release unnecessary memory usage, or increase available system memory and retry.|
|0x102003ED|Failed to open file|Confirm that the file path is valid, the file exists, and is not locked by other processes.|
|0x102003EE|Invalid file permissions|Ensure the file owner is the current user and permissions do not exceed 640.|
|0x102003EF|Invalid file size|Check whether the file size is within the API limit (e.g., images not exceeding 1GB).|
|0x103007D1|Failed to initialize FFmpeg|Confirm FFmpeg dependencies are correctly installed and check whether the video file format is mp4.|
|0x103007D2|Failed to initialize Libjpeg|Confirm Libjpeg dependencies are available and check system library path configuration.|
|0x103007D3|Failed to read file with Libjpeg|Confirm the image is in jpg/jpeg format and not corrupted, and check file permissions.|
|0x103007D4|FFmpeg execution failed|Check video file integrity and whether resolution is within [480, 4096] range.|
|0x10400BB9|Internal operator failed|Check whether input data format and size meet operator requirements, and check detailed logs.|
|0x10400BBA|Internal function execution failed|Check SDK log output and confirm whether preceding steps completed successfully.|
|0x10400BBB|Internal type conversion failed|Confirm source data is compatible with target type (such as dtype, layout format NCHW/NHWC).|
|0x10500FA1|Timed out while waiting|Check system load, appropriately increase timeout or reduce concurrent tasks.|
|0x10500FA2|Invalid thread pool state|Restart related processes and avoid submitting tasks repeatedly in incorrect states.|

## Environment Variables

The following environment variables are used during installation or runtime. Ensure that they are valid.

**Table 1** environment variables

|Environment Variable|Description|
|--|--|
|PATH|Path to executables.|
|LD_LIBRARY_PATH|Path to dynamic libraries.|
|PYTHONPATH|Default search path for Python module files.|
|HOME|Current user's home directory.|
|PWD|Current working directory.|
|TMPDIR|Temporary directory.|
|ASCEND_HOME_PATH|Path to CANN-related resources. Ensure that the path is valid.|
|ASCEND_HOME|CANN installation directory. Ensure that the path is valid.|
|ASCEND_VERSION|CANN version number. Do not change it arbitrarily.|
|ASCEND_CUSTOM_OPP_PATH|AscendC operator deployment path. Do not change it arbitrarily.|
|MULTIMODAL_SDK_HOME|Multimodal SDK installation directory. Ensure that the path is valid.|
|HF_DATASETS_OFFLINE|Specifies to load Hugging Face datasets in offline mode.|
|HF_HUB_OFFLINE|Specifies to run the Hugging Face library in offline mode.|

>[!CAUTION] Note
>When you use the patcher provided by this software to run vLLM inference or other open-source libraries, it also uses additional environment variables related to those libraries. This document does not list them. Therefore, ensure that the environment variables are correct and valid.
