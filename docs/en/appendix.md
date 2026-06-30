# Appendix

## Public Network Addresses Included in the Software

The URLs in the Multimodal SDK installation package are removed after installation. The SDK does not access them. Therefore, they pose no risk.

The public URLs and email addresses in this Multimodal SDK manual are not accessed by the SDK itself. Therefore, they pose no risk.

For more public URLs and email addresses, see [Multimodal SDK 7.3.0 public network addresses.xlsx](<./resource/Multimodal SDK 7.3.0 public network addresses.xlsx>).

## Communication Matrix

Currently, the Multimodal SDK package does not actively open or depend on any ports. Therefore, it does not involve a communication matrix.

## Error Codes

|Error Code (hex)|Error Message|
|--|--|
|0x10100001|Invalid parameter|
|0x10100002|Unsupported type|
|0x10100003|Invalid pointer|
|0x10100004|Value out of range|
|0x102003E9|Memory allocation failed|
|0x102003EA|Memory copy failed|
|0x102003EB|Memory release failed|
|0x102003EC|Insufficient system memory|
|0x102003ED|Failed to open file|
|0x102003EE|Invalid file permissions|
|0x102003EF|Invalid file size|
|0x103007D1|Failed to initialize FFmpeg|
|0x103007D2|Failed to initialize Libjpeg|
|0x103007D3|Failed to read file with Libjpeg|
|0x103007D4|FFmpeg execution failed|
|0x10400BB9|Internal operator failed|
|0x10400BBA|Internal function execution failed|
|0x10400BBB|Internal type conversion failed|
|0x10500FA1|Timed out while waiting|
|0x10500FA2|Invalid thread pool state|

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
