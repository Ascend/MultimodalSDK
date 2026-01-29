# Python接口说明<a name="ZH-CN_TOPIC_0000002456910817"></a>

>[!NOTE] 说明 
>-   对于资料中注明的类、接口为公开的，可供用户调用。对于其他类，接口均为内部使用，不建议直接调用。如有特殊需要可以通过源码查看。
>-   引入多模态SDK时会显式设置环境变量“HF\_DATASETS\_OFFLINE“和“HF\_HUB\_OFFLINE“为1，即显式启用huggingface的离线模式，不会联网获取数据。

**源码查看方式<a name="section12411139493"></a>**

可以通过以下示例代码查看源码所在安装地址，并进入安装地址查看源码文件。

```
import mm
print(mm.__file__)
```

打印的内容即为源码文件所在的地址。

## 数据类型<a name="ZH-CN_TOPIC_0000002423192124"></a>

### DataType<a name="ZH-CN_TOPIC_0000002423192160"></a>

数据类型枚举类

|属性名|说明|
|--|--|
|DataType.INT8|int8类型。|
|DataType.UINT8|uint8类型。|
|DataType.FLOAT32|float32类型。|



### TensorFormat<a name="ZH-CN_TOPIC_0000002423352008"></a>

Tensor数据排布格式枚举类

|属性名|说明|
|--|--|
|TensorFormat.ND|ND格式。表示通用N维数组。|
|TensorFormat.NHWC|NHWC格式。表示张量排布为Batch, Height, Width, Channel。|
|TensorFormat.NCHW|NCHW格式。表示张量排布为Batch, Channel, Height, Width。|



### ImageFormat<a name="ZH-CN_TOPIC_0000002456790953"></a>

ImageFormat枚举类

|属性名|说明|
|--|--|
|ImageFormat.RGB|RGB类型。|
|ImageFormat.BGR|BGR类型。|
|ImageFormat.RGB_PLANAR|RGB_PLANAR类型。|
|ImageFormat.BGR_PLANAR|BGR_PLANAR类型。|



### LogLevel<a name="ZH-CN_TOPIC_0000002423192156"></a>

日志级别枚举类

|属性名|说明|
|--|--|
|LogLevel.DEBUG|调试级别。|
|LogLevel.INFO|提示级别。|
|LogLevel.WARN|警告级别。|
|LogLevel.ERROR|错误级别。|
|LogLevel.FATAL|致命错误级别。|



### DeviceMode<a name="ZH-CN_TOPIC_0000002456910801"></a>

操作运行的模式

|属性名|说明|
|--|--|
|DeviceMode.CPU|当前操作运行在CPU模式下。|



### Interpolation<a name="ZH-CN_TOPIC_0000002423192132"></a>

resize操作中使用的插值算法

|属性名|说明|
|--|--|
|Interpolation.BICUBIC|双立方插值算法。|

