# 快速入门<a name="ZH-CN_TOPIC_0000002456790933"></a>

**简介<a name="section142771553125211"></a>**

Multimodal SDK提供了一系列CPU高性能和易用性的接口，本章节通过介绍部分接口的使用，帮助用户熟悉使用本软件。

**环境准备<a name="section543617275526"></a>**

-   准备Atlas A2 推理系列产品的服务器，并安装对应的驱动和固件，具体安装过程请参见《CANN 软件安装指南》中的“安装NPU驱动和固件”章节（商用版）或“安装NPU驱动和固件”章节（社区版）。
-   安装CANN Toolkit，具体安装过程请参见《CANN 软件安装指南》的“安装CANN”章节（商用版）或《CANN 软件安装指南》的“安装CANN”章节（社区版）。
-   安装Multimodal SDK以及相关依赖，具体安装过程请参见[安装部署](./installation_guide.md)。

**使用流程<a name="section167395353541"></a>**

-   使用高性能接口

    Multimodal SDK提供了一系列CPU和NPU高性能接口，用户可以根据这些接口自行选用集成到自己的业务流程中，本章节提供几个高性能接口的示例作为参考。

    1.  高性能图像解码接口：

        ```
        from mm import Image
        img= Image.open("/home/test.jpg", "cpu")  # 样例代码的图片需要开发者自行替换
        ```

    2.  高性能图像数据处理resize接口：

        ```
        from mm import Image, DeviceMode, Interpolation
        img = Image.open("/home/test.jpg", "cpu") # 样例代码的图片需要开发者自行替换
        img_resize = img.resize((500,500), Interpolation.BICUBIC, DeviceMode.CPU)
        ```

    3.  高性能图像数据处理crop接口：

        ```
        from mm import Image, DeviceMode
        img = Image.open("/home/test.jpg", "cpu") # 样例代码的图片需要开发者自行替换
        img_crop = img.crop(10, 10, 10, 10, DeviceMode.CPU)
        ```

-   使用开源推理框架对接接口

    基于Multimodal SDK提供的高性能接口，Multimodal SDK也提供了对接开源推理框架的vllm的适配方案，基于vllm以及vllm-ascend的patch机制，用户可以根据[patcher](./api/patcher.md)中的操作指导将Multimodal SDK的加速效果应用在自己的程序之中。

