/*
* -------------------------------------------------------------------------
*  This file is part of the MultimodalSDK project.
* Copyright (c) 2025 Huawei Technologies Co.,Ltd.
*
* MultimodalSDK is licensed under Mulan PSL v2.
* You can use this software according to the terms and conditions of the Mulan PSL v2.
* You may obtain a copy of Mulan PSL v2 at:
*
*           http://license.coscl.org.cn/MulanPSL2
*
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
* EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
* MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
* See the Mulan PSL v2 for more details.
* -------------------------------------------------------------------------
 * Description: ImageTest Cpp file.
 * Author: ACC SDK
 * Create: 2025
 * History: NA
 */
#include <sys/stat.h>
#include <unistd.h>
#include <vector>
#include <cstdint>
#include <fstream>
#include <cstring>
#include <future>
#include <filesystem>
#include <gtest/gtest.h>
#include "securec.h"
#include "acc/utils/LogImpl.h"
#include "acc/ErrorCode.h"
#include "acc/image/Image.h"

static const size_t IM_WIDTH = 320;
static const size_t IM_HEIGHT = 240;
static const size_t IM_WIDTH_EXCEED = 10240;
static const size_t IM_HEIGHT_EXCEED = 10240;
static const size_t IM_WIDTH_BELOW = 1;
static const size_t IM_HEIGHT_BELOW = 1;
static const size_t ONE_CHANNEL = 1;
static const size_t THREE_CHANNEL = 3;
static const size_t FOUR_CHANNEL = 4;
static const size_t ONE_BATCH = 1;

using namespace Acc;

class ImageTest : public testing::Test {
protected:
    template<typename T>
    std::vector<T> Create_Image_Data(size_t nElem)
    {
        std::vector<T> data(nElem, 0);
        int baseValue = 100;
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = static_cast<T>(i) / baseValue;
        }
        return data;
    }
    std::filesystem::path validImPath = std::filesystem::path(__FILE__).parent_path() / "assets" / "dog_1920_1080.jpg";
    std::filesystem::path invalidImPath =
        std::filesystem::path(__FILE__).parent_path() / "assets" / "dog_1920_1080.png";

    const size_t picWidth = 1920;
    const size_t picHeight = 1080;
    const int numDevice = 4;
    const int numThread = 16;
};

TEST_F(ImageTest, Test_Create_RGB_Image_Should_Success)
{
    size_t nElem = IM_WIDTH * IM_HEIGHT * THREE_CHANNEL;
    std::vector<uint8_t> imData = Create_Image_Data<uint8_t>(nElem);
    Image img(imData.data(), {IM_WIDTH, IM_HEIGHT}, ImageFormat::RGB, DataType::UINT8, "cpu");
    ASSERT_EQ(img.Width(), IM_WIDTH);
    ASSERT_EQ(img.Height(), IM_HEIGHT);
    ASSERT_EQ(img.Size(), (std::vector<size_t>{IM_WIDTH, IM_HEIGHT}));
    ASSERT_EQ(img.Format(), ImageFormat::RGB);
    ASSERT_EQ(img.DType(), DataType::UINT8);
    ASSERT_EQ(img.NumBytes(), imData.size() * sizeof(uint8_t));
}

TEST_F(ImageTest, Test_Create_BGR_Image_Should_Success)
{
    size_t nElem = IM_WIDTH * IM_HEIGHT * THREE_CHANNEL;
    std::vector<uint8_t> imData = Create_Image_Data<uint8_t>(nElem);
    Image img(imData.data(), {IM_WIDTH, IM_HEIGHT}, ImageFormat::BGR, DataType::UINT8, "cpu");
    ASSERT_EQ(img.Width(), IM_WIDTH);
    ASSERT_EQ(img.Height(), IM_HEIGHT);
    ASSERT_EQ(img.Size(), (std::vector<size_t>{IM_WIDTH, IM_HEIGHT}));
    ASSERT_EQ(img.Format(), ImageFormat::BGR);
    ASSERT_EQ(img.DType(), DataType::UINT8);
    ASSERT_EQ(img.NumBytes(), imData.size() * sizeof(uint8_t));
}

TEST_F(ImageTest, Test_Create_BGR_PLANAR_Image_Should_Success)
{
    size_t nElem = IM_WIDTH * IM_HEIGHT * THREE_CHANNEL;
    std::vector<uint8_t> imData = Create_Image_Data<uint8_t>(nElem);
    Image img(imData.data(), {IM_WIDTH, IM_HEIGHT}, ImageFormat::BGR_PLANAR, DataType::UINT8, "cpu");
    ASSERT_EQ(img.Width(), IM_WIDTH);
    ASSERT_EQ(img.Height(), IM_HEIGHT);
    ASSERT_EQ(img.Size(), (std::vector<size_t>{IM_WIDTH, IM_HEIGHT}));
    ASSERT_EQ(img.Format(), ImageFormat::BGR_PLANAR);
    ASSERT_EQ(img.DType(), DataType::UINT8);
    ASSERT_EQ(img.NumBytes(), imData.size() * sizeof(uint8_t));
}

TEST_F(ImageTest, Test_Create_RGB_PLANAR_Image_Should_Success)
{
    size_t nElem = IM_WIDTH * IM_HEIGHT * THREE_CHANNEL;
    std::vector<uint8_t> imData = Create_Image_Data<uint8_t>(nElem);
    Image img(imData.data(), {IM_WIDTH, IM_HEIGHT}, ImageFormat::RGB_PLANAR, DataType::UINT8, "cpu");
    ASSERT_EQ(img.Width(), IM_WIDTH);
    ASSERT_EQ(img.Height(), IM_HEIGHT);
    ASSERT_EQ(img.Size(), (std::vector<size_t>{IM_WIDTH, IM_HEIGHT}));
    ASSERT_EQ(img.Format(), ImageFormat::RGB_PLANAR);
    ASSERT_EQ(img.DType(), DataType::UINT8);
    ASSERT_EQ(img.NumBytes(), imData.size() * sizeof(uint8_t));
}

TEST_F(ImageTest, Test_Create_Image_Clone_Should_Success)
{
    size_t nElem = IM_WIDTH * IM_HEIGHT * THREE_CHANNEL;
    std::vector<uint8_t> imData = Create_Image_Data<uint8_t>(nElem);
    Image img(imData.data(), {IM_WIDTH, IM_HEIGHT}, ImageFormat::RGB, DataType::UINT8, "cpu");
    ASSERT_EQ(img.Width(), IM_WIDTH);
    ASSERT_EQ(img.Height(), IM_HEIGHT);
    ASSERT_EQ(img.Size(), (std::vector<size_t>{IM_WIDTH, IM_HEIGHT}));
    ASSERT_EQ(img.Format(), ImageFormat::RGB);
    ASSERT_EQ(img.DType(), DataType::UINT8);
    ASSERT_EQ(img.NumBytes(), imData.size() * sizeof(uint8_t));
    Image cloneImg;
    ErrorCode ret = img.Clone(cloneImg);
    ASSERT_EQ(ret, SUCCESS);
    ASSERT_EQ(cloneImg.Width(), IM_WIDTH);
    ASSERT_EQ(cloneImg.Height(), IM_HEIGHT);
    ASSERT_EQ(cloneImg.Size(), (std::vector<size_t>{IM_WIDTH, IM_HEIGHT}));
    ASSERT_EQ(cloneImg.Format(), ImageFormat::RGB);
    ASSERT_EQ(cloneImg.DType(), DataType::UINT8);
    ASSERT_EQ(cloneImg.NumBytes(), imData.size() * sizeof(uint8_t));
}

TEST_F(ImageTest, Test_Create_Image_Clone_Should_Fail)
{
    size_t nElem = IM_WIDTH * IM_HEIGHT * THREE_CHANNEL;
    std::vector<uint8_t> imData = Create_Image_Data<uint8_t>(nElem);
    Image img(imData.data(), {IM_WIDTH, IM_HEIGHT}, ImageFormat::RGB, DataType::UINT8, "cpu");
    ASSERT_EQ(img.Width(), IM_WIDTH);
    ASSERT_EQ(img.Height(), IM_HEIGHT);
    ASSERT_EQ(img.Size(), (std::vector<size_t>{IM_WIDTH, IM_HEIGHT}));
    ASSERT_EQ(img.Format(), ImageFormat::RGB);
    ASSERT_EQ(img.DType(), DataType::UINT8);
    ASSERT_EQ(img.NumBytes(), imData.size() * sizeof(uint8_t));
    ErrorCode ret = img.Clone(img);
    ASSERT_EQ(ret, ERR_INVALID_PARAM);
}

TEST_F(ImageTest, Test_Create_Image_On_CPU_Should_Success)
{
    size_t nElem = IM_WIDTH * IM_HEIGHT * THREE_CHANNEL;
    std::vector<uint8_t> imData = Create_Image_Data<uint8_t>(nElem);
    Image img(imData.data(), {IM_WIDTH, IM_HEIGHT}, ImageFormat::RGB, DataType::UINT8, "cpu");
    ASSERT_EQ(img.Width(), IM_WIDTH);
    ASSERT_EQ(img.Height(), IM_HEIGHT);
    ASSERT_EQ(img.Size(), (std::vector<size_t>{IM_WIDTH, IM_HEIGHT}));
    ASSERT_EQ(img.Format(), ImageFormat::RGB);
    ASSERT_EQ(img.DType(), DataType::UINT8);
    ASSERT_EQ(img.NumBytes(), imData.size() * sizeof(uint8_t));
}

TEST_F(ImageTest, Test_Create_Image_From_Share_Ptr_On_CPU_Should_Success)
{
    size_t nElem = IM_WIDTH * IM_HEIGHT * THREE_CHANNEL;
    std::vector<uint8_t> imData = Create_Image_Data<uint8_t>(nElem);
    std::shared_ptr<void> dataPtr(imData.data(), [](void*) {});
    Image img(dataPtr, {IM_WIDTH, IM_HEIGHT}, ImageFormat::RGB, DataType::UINT8, "cpu");
    ASSERT_EQ(img.Width(), IM_WIDTH);
    ASSERT_EQ(img.Height(), IM_HEIGHT);
    ASSERT_EQ(img.Size(), (std::vector<size_t>{IM_WIDTH, IM_HEIGHT}));
    ASSERT_EQ(img.Format(), ImageFormat::RGB);
    ASSERT_EQ(img.DType(), DataType::UINT8);
    ASSERT_EQ(img.NumBytes(), imData.size() * sizeof(uint8_t));
}

TEST_F(ImageTest, Test_Create_Image_With_Float32_Should_Failed)
{
    size_t nElem = IM_WIDTH * IM_HEIGHT * THREE_CHANNEL;
    std::vector<uint8_t> imData = Create_Image_Data<uint8_t>(nElem);
    bool isFailed = false;
    try {
        Image img(imData.data(), {IM_WIDTH, IM_HEIGHT}, ImageFormat::RGB, DataType::FLOAT32, "cpu");
    } catch (...) {
        isFailed = true;
    }
    ASSERT_TRUE(isFailed);
}

TEST_F(ImageTest, Test_Create_Image_With_Int8_Should_Failed)
{
    size_t nElem = IM_WIDTH * IM_HEIGHT * THREE_CHANNEL;
    std::vector<uint8_t> imData = Create_Image_Data<uint8_t>(nElem);
    bool isFailed = false;
    try {
        Image img(imData.data(), {IM_WIDTH, IM_HEIGHT}, ImageFormat::RGB, DataType::INT8, "cpu");
    } catch (...) {
        isFailed = true;
    }
    ASSERT_TRUE(isFailed);
}

TEST_F(ImageTest, Test_Create_Image_With_Beyond_Height_Should_Failed)
{
    size_t nElem = IM_WIDTH * IM_HEIGHT_EXCEED * THREE_CHANNEL;
    std::vector<uint8_t> imData = Create_Image_Data<uint8_t>(nElem);
    bool isFailed = false;
    try {
        Image img(imData.data(), {IM_WIDTH, IM_HEIGHT_EXCEED}, ImageFormat::RGB, DataType::UINT8, "cpu");
    } catch (...) {
        isFailed = true;
    }
    ASSERT_TRUE(isFailed);
}

TEST_F(ImageTest, Test_Create_Image_With_Beyond_Width_Should_Failed)
{
    size_t nElem = IM_WIDTH_EXCEED * IM_HEIGHT * THREE_CHANNEL;
    std::vector<uint8_t> imData = Create_Image_Data<uint8_t>(nElem);
    bool isFailed = false;
    try {
        Image img(imData.data(), {IM_WIDTH_EXCEED, IM_HEIGHT}, ImageFormat::RGB, DataType::UINT8, "cpu");
    } catch (...) {
        isFailed = true;
    }
    ASSERT_TRUE(isFailed);
}

TEST_F(ImageTest, Test_Create_Image_From_Share_Ptr_With_Float32_Should_Failed)
{
    size_t nElem = IM_WIDTH_EXCEED * IM_HEIGHT * THREE_CHANNEL;
    std::vector<uint8_t> imData = Create_Image_Data<uint8_t>(nElem);
    std::shared_ptr<void> dataPtr(imData.data(), [](void*) {});
    bool isFailed = false;
    try {
        Image img(dataPtr, {IM_WIDTH, IM_HEIGHT}, ImageFormat::RGB, DataType::FLOAT32, "cpu");
    } catch (...) {
        isFailed = true;
    }
    ASSERT_TRUE(isFailed);
}

TEST_F(ImageTest, Test_Create_Image_From_Share_Ptr_With_Int8_Should_Failed)
{
    size_t nElem = IM_WIDTH * IM_HEIGHT * THREE_CHANNEL;
    std::vector<uint8_t> imData = Create_Image_Data<uint8_t>(nElem);
    std::shared_ptr<void> dataPtr(imData.data(), [](void*) {});
    bool isFailed = false;
    try {
        Image img(dataPtr, {IM_WIDTH, IM_HEIGHT}, ImageFormat::RGB, DataType::INT8, "cpu");
    } catch (...) {
        isFailed = true;
    }
    ASSERT_TRUE(isFailed);
}

TEST_F(ImageTest, Test_Create_Image_From_Share_Ptr_With_Beyond_Height_Should_Failed)
{
    size_t nElem = IM_WIDTH * IM_HEIGHT_EXCEED * THREE_CHANNEL;
    std::vector<uint8_t> imData = Create_Image_Data<uint8_t>(nElem);
    std::shared_ptr<void> dataPtr(imData.data(), [](void*) {});
    bool isFailed = false;
    try {
        Image img(dataPtr, {IM_WIDTH, IM_HEIGHT_EXCEED}, ImageFormat::RGB, DataType::UINT8, "cpu");
    } catch (...) {
        isFailed = true;
    }
    ASSERT_TRUE(isFailed);
}

TEST_F(ImageTest, Test_Create_Image_From_Share_Ptr_With_Beyond_Width_Should_Failed)
{
    size_t nElem = IM_WIDTH_EXCEED * IM_HEIGHT * THREE_CHANNEL;
    std::vector<uint8_t> imData = Create_Image_Data<uint8_t>(nElem);
    std::shared_ptr<void> dataPtr(imData.data(), [](void*) {});
    bool isFailed = false;
    try {
        Image img(dataPtr, {IM_WIDTH_EXCEED, IM_HEIGHT}, ImageFormat::RGB, DataType::UINT8, "cpu");
    } catch (...) {
        isFailed = true;
    }
    ASSERT_TRUE(isFailed);
}

TEST_F(ImageTest, Test_Create_Image_On_NPU_Should_Failed)
{
    size_t nElem = IM_WIDTH * IM_HEIGHT * THREE_CHANNEL;
    std::vector<uint8_t> imData = Create_Image_Data<uint8_t>(nElem);
    bool isFailed = false;
    try {
        Image img(imData.data(), {IM_WIDTH, IM_HEIGHT}, ImageFormat::RGB, DataType::UINT8, "npu");
    } catch (...) {
        isFailed = true;
    }
    ASSERT_TRUE(isFailed);
}

TEST_F(ImageTest, Test_Create_Image_With_Wrong_Format_Should_Fail)
{
    size_t nElem = IM_WIDTH * IM_HEIGHT * THREE_CHANNEL;
    std::vector<uint8_t> imData = Create_Image_Data<uint8_t>(nElem);
    bool isFailed = false;
    try {
        Image img(imData.data(), {IM_WIDTH, IM_HEIGHT}, ImageFormat::UNDEFINED, DataType::UINT8, "cpu");
    } catch (...) {
        isFailed = true;
    }
    ASSERT_TRUE(isFailed);
}

TEST_F(ImageTest, Test_Create_Image_With_Wrong_Size_Length_Should_Fail)
{
    size_t nElem = IM_WIDTH * IM_HEIGHT * THREE_CHANNEL;
    std::vector<uint8_t> imData = Create_Image_Data<uint8_t>(nElem);
    bool isFailed = false;
    try {
        Image img(imData.data(), {IM_WIDTH, IM_HEIGHT, THREE_CHANNEL}, ImageFormat::RGB, DataType::UINT8, "cpu");
    } catch (...) {
        isFailed = true;
    }
    ASSERT_TRUE(isFailed);
}

TEST_F(ImageTest, Test_Create_Image_With_Wrong_Size_Value_Should_Fail)
{
    size_t nElem = IM_WIDTH * IM_HEIGHT * THREE_CHANNEL;
    std::vector<uint8_t> imData = Create_Image_Data<uint8_t>(nElem);
    bool isFailed = false;
    try {
        Image img(imData.data(), {IM_WIDTH * IM_HEIGHT, 0}, ImageFormat::RGB, DataType::UINT8, "cpu");
    } catch (...) {
        isFailed = true;
    }
    ASSERT_TRUE(isFailed);
}

TEST_F(ImageTest, Test_Create_Image_From_Path_With_Invalid_Permission_On_CPU_Should_Fail)
{
    const char* device = "cpu";
    bool isFailed = false;
    try {
        std::string pathStr = validImPath.string();
        chmod(pathStr.c_str(), 0644);
        Image img = Image(pathStr.c_str(), device);
    } catch (...) {
        isFailed = true;
    }
    ASSERT_TRUE(isFailed);
}

TEST_F(ImageTest, Test_Create_Image_From_Path_On_CPU_Should_Success)
{
    const char* device = "cpu";

    try {
        std::string pathStr = validImPath.string();
        chmod(pathStr.c_str(), 0640);
        Image img = Image(pathStr.c_str(), device);
        ASSERT_EQ(img.Width(), picWidth);
        ASSERT_EQ(img.Height(), picHeight);
        ASSERT_EQ(img.Size(), (std::vector<size_t>{picWidth, picHeight}));
        ASSERT_EQ(img.Format(), ImageFormat::RGB);
        ASSERT_EQ(img.DType(), DataType::UINT8);
        size_t expectedNumBytes = picWidth * picHeight * THREE_CHANNEL * sizeof(uint8_t);
        ASSERT_EQ(img.NumBytes(), expectedNumBytes);
    } catch (...) {
        FAIL() << "Image creation from path on CPU failed unexpectedly.";
    }
}

TEST_F(ImageTest, Test_Create_Image_From_Invalid_Postfix_On_CPU_Should_Fail)
{
    const char* device = "cpu";
    bool isFailed = false;
    try {
        std::string pathStr = invalidImPath.string();
        chmod(pathStr.c_str(), 0640);
        Image img = Image(pathStr.c_str(), device);
    } catch (...) {
        isFailed = true;
    }
    ASSERT_TRUE(isFailed);
}

TEST_F(ImageTest, Test_Create_Image_From_Valid_Path_On_NPU_Should_Fail)
{
    const char* device = "npu";
    bool isFailed = false;
    try {
        std::string pathStr = validImPath.string();
        chmod(pathStr.c_str(), 0640);
        Image img = Image(pathStr.c_str(), device);
    } catch (...) {
        isFailed = true;
    }
    ASSERT_TRUE(isFailed);
}

TEST_F(ImageTest, Test_Create_Image_From_Path_Nullptr_Should_Fail)
{
    const char* device = "cpu";
    EXPECT_THROW(Image(nullptr, device), std::runtime_error);
}

int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    return ret;
}