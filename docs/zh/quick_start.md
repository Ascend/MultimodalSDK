# 快速入门

**简介**

Multimodal SDK 提供了一系列 CPU 高性能和易用性的接口，本章节通过介绍部分接口的使用，帮助用户熟悉使用本软件。

**环境准备**

- 准备 Atlas A2 推理系列产品的服务器，并安装对应的驱动和固件，具体安装过程请参见《CANN 软件安装指南》中的"安装 NPU 驱动和固件"章节（商用版）或"安装 NPU 驱动和固件"章节（社区版）。
- 安装 CANN Toolkit，具体安装过程请参见《CANN 软件安装指南》的"安装 CANN"章节（商用版）或《CANN 软件安装指南》的"安装 CANN"章节（社区版）。
- 安装 Multimodal SDK 以及相关依赖，具体安装过程请参见[安装部署](./installation_guide.md)。

**使用流程**

- 使用高性能接口

    Multimodal SDK 提供了一系列 CPU 高性能接口（当前 Python API 的 `DeviceMode` 仅支持 CPU）。在 Ascend 环境下如需 NPU 加速，可通过 [patcher](./api/patcher.md) 集成 vLLM 补丁实现。用户可以根据这些接口自行选用集成到自己的业务流程中，本章节提供几个高性能接口的示例作为参考。

    1. 高性能图像解码接口：

        ```python
        from mm import Image
        img = Image.open("/home/test.jpg", "cpu")  # 样例代码的图片需要开发者自行替换
        ```

    2. 高性能图像数据处理 resize 接口：

        ```python
        from mm import Image, DeviceMode, Interpolation
        img = Image.open("/home/test.jpg", "cpu")  # 样例代码的图片需要开发者自行替换
        img_resize = img.resize((500, 500), Interpolation.BICUBIC, DeviceMode.CPU)
        ```

    3. 高性能图像数据处理 crop 接口：

        ```python
        from mm import Image, DeviceMode
        img = Image.open("/home/test.jpg", "cpu")  # 样例代码的图片需要开发者自行替换
        img_crop = img.crop(10, 10, 10, 10, DeviceMode.CPU)
        ```

- 端到端验证示例

    完成[安装部署](./installation_guide.md)后，可按以下步骤验证 SDK 是否正常工作（请将 `/home/test.jpg` 替换为实际 jpg/jpeg 文件路径，并确保文件权限不高于 640）：

    ```python
    # 1. 安装完成后在 shell 中执行：source ${MULTIMODAL_SDK_HOME}/script/set_env.sh
    from mm import Image, DeviceMode, Interpolation

    img = Image.open("/home/test.jpg", "cpu")
    img_resize = img.resize((500, 500), Interpolation.BICUBIC, DeviceMode.CPU)
    arr = img_resize.numpy()
    print(f"resize output shape: {arr.shape}")  # 预期输出 (500, 500, 3)
    ```

- 使用开源推理框架对接接口

    基于 Multimodal SDK 提供的高性能接口，Multimodal SDK 也提供了对接开源推理框架 vLLM 的适配方案，基于 vLLM 以及 vLLM-Ascend 的 patch 机制，用户可以根据 [patcher](./api/patcher.md) 中的操作指导将 Multimodal SDK 的加速效果应用在自己的程序之中。
