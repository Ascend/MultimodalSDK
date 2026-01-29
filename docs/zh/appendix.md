# 附录<a name="ZH-CN_TOPIC_0000002423352012"></a>

## 软件中包含的公网地址<a name="ZH-CN_TOPIC_0000002457001025"></a>

Multimodal SDK的安装包中的网址安装结束后会被清除，并不会访问，不会造成风险。

Multimodal SDK本手册中存在的公开网址和邮箱地址，SDK本身不会访问，不会造成风险。

更多公网地址请参见[Multimodal SDK 7.3.0 公网地址.xlsx](./resource/Multimodal%20SDK%207.3.0%20公网地址.xlsx)。


## 软件通信矩阵<a name="ZH-CN_TOPIC_0000002434737330"></a>

目前Multimodal SDK开发套件包不会主动打开或者依赖任意端口，因此不涉及通信矩阵。


## 错误码<a name="ZH-CN_TOPIC_0000002469692481"></a>

|错误码（十六进制）|错误信息|
|--|--|
|0x10100001|参数无效|
|0x10100002|不支持的类型|
|0x10100003|指针无效|
|0x10100004|值超出范围|
|0x102003E9|内存分配失败|
|0x102003EA|内存拷贝失败|
|0x102003EB|内存释放失败|
|0x102003EC|系统内存不足|
|0x102003ED|打开文件失败|
|0x102003EE|文件权限无效|
|0x102003EF|文件大小无效|
|0x103007D1|FFmpeg初始化失败|
|0x103007D2|Libjpeg初始化失败|
|0x103007D3|Libjpeg读取文件失败|
|0x103007D4|FFmpeg执行失败|
|0x10400BB9|内部算子失败|
|0x10400BBA|内部函数执行失败|
|0x10400BBB|内部类型转换失败|
|0x10500FA1|等待超时|
|0x10500FA2|线程池状态无效|



## 环境变量说明<a name="ZH-CN_TOPIC_0000002470101636"></a>

以下环境变量会在程序安装或运行时使用，请确保有效。

**表 1**  环境变量

|环境变量名|说明|
|--|--|
|PATH|可执行程序的文件路径。|
|LD_LIBRARY_PATH|动态链接库路径。|
|PYTHONPATH|Python模块文件的默认搜索路径。|
|HOME|当前用户的家目录。|
|PWD|当前系统路径。|
|TMPDIR|临时文件路径。|
|ASCEND_HOME_PATH|CANN相关资源路径，请确保路径有效性。|
|ASCEND_HOME|CANN安装目录，请确保路径有效性。|
|ASCEND_VERSION|CANN版本号，请勿随意改动。|
|ASCEND_CUSTOM_OPP_PATH|AscendC算子部署路径，请勿随意改动。|
|MULTIMODAL_SDK_HOME|Multimodal SDK安装目录，请确保路径有效性。|
|HF_DATASETS_OFFLINE|离线加载Hugging Face数据集。|
|HF_HUB_OFFLINE|在离线模式下运行Hugging Face库。|


>[!CAUTION] 注意 
>使用本软件提供的patcher运行vllm推理或其他开源库时，还会使用到其他开源库相关环境变量，本文不做声明，请用户自行确保环境变量真实有效。


