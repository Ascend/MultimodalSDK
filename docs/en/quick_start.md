# Quick Start

**Introduction**

Multimodal SDK provides a set of high-performance, easy-to-use CPU interfaces. This chapter introduces several of these interfaces to help you become familiar with the software.

**Environment Preparation**

- Prepare a server for Atlas A2 inference products and install the corresponding driver and firmware. For details about the installation procedure, see "Installing NPU Drivers and Firmware" in the CANN Software Installation Guide for the commercial edition or the "Installing NPU Drivers and Firmware" section in the CANN Software Installation Guide for the community edition.
- Install CANN Toolkit. For details about the installation procedure, see "Installing the NPU Driver and Firmware" (commercial edition) or "Installing the NPU Driver and Firmware" (community edition) in the *CANN Software Installation Guide*.
- Install Multimodal SDK and the related dependencies. For details about the installation procedure, see [Installation and Deployment](./installation_guide.md).

**Usage Process**

- Use the high-performance interfaces.

    Multimodal SDK provides a set of high-performance CPU and NPU interfaces. You can select the interfaces as needed and integrate them into your service process. This chapter provides several high-performance interface examples for reference.

    1. High-performance image decoding interface:

        ```python
        from mm import Image
        img = Image.open("/home/test.jpg", "cpu")  # You must replace the image in the sample code
        ```

    2. High-performance image resizing interface:

        ```python
        from mm import Image, DeviceMode, Interpolation
        img = Image.open("/home/test.jpg", "cpu")  # You must replace the image in the sample code
        img_resize = img.resize((500,500), Interpolation.BICUBIC, DeviceMode.CPU)
        ```

    3. High-performance image cropping interface:

        ```python
        from mm import Image, DeviceMode
        img = Image.open("/home/test.jpg", "cpu")  # You must replace the image in the sample code
        img_crop = img.crop(10, 10, 10, 10, DeviceMode.CPU)
        ```

- Use the integration interfaces for open-source inference frameworks.

    Based on the high-performance interfaces provided by Multimodal SDK, Multimodal SDK also provides an adaptation solution for the open-source vLLM inference framework. Using the patch mechanism of vLLM and vLLM-Ascend, you can follow the instructions in [patcher](./api/patcher.md) to apply the acceleration effects of Multimodal SDK to your own program.
