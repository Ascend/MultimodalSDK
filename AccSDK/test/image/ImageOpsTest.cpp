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
 * Description: Test file of the Tensor Ops.
 * Author: ACC SDK
 * Create: 2025
 * History: NA
 */
#include <gtest/gtest.h>
#include "acc/image/Image.h"
#include "acc/image/ImageOps.h"
using namespace Acc;
namespace {
constexpr size_t BATCH_SIZE_ONE = 1;
constexpr size_t SHAPE_1920 = 1920;
constexpr size_t SHAPE_1080 = 1080;
constexpr size_t SHAPE_960 = 960;
constexpr size_t SHAPE_540 = 540;
constexpr size_t SHAPE_11 = 11;
constexpr uint32_t CROP_HEIGHT = 10;
constexpr uint32_t CROP_WIDTH = 11;
constexpr size_t CHANNEL_THREE = 3;
constexpr uint8_t VALID_VALUE = 100;
constexpr char* CPU = "cpu";
std::vector<uint8_t> g_vector1080PUint8Value100(SHAPE_1920* SHAPE_1080* CHANNEL_THREE, VALID_VALUE);
std::vector<uint8_t> g_vector1080PHalfUint8Value100(SHAPE_960* SHAPE_540* CHANNEL_THREE, VALID_VALUE);
class ImageOpsTest : public testing::Test {
};

TEST_F(ImageOpsTest, Test_ImageResize_Success_With_Premalloced_Dst)
{
    Image src(g_vector1080PUint8Value100.data(), {SHAPE_1920, SHAPE_1080}, ImageFormat::RGB, DataType::UINT8, CPU);
    Image dst(g_vector1080PHalfUint8Value100.data(), {SHAPE_960, SHAPE_540}, ImageFormat::RGB, DataType::UINT8, CPU);
    auto ret = ImageResize(src, dst, SHAPE_960, SHAPE_540, Interpolation::BICUBIC, DeviceMode::CPU);
    EXPECT_EQ(ret, SUCCESS);
}

TEST_F(ImageOpsTest, Test_ImageResize_Success_With_ImplicitMalloc)
{
    Image src(g_vector1080PUint8Value100.data(), {SHAPE_1920, SHAPE_1080}, ImageFormat::RGB, DataType::UINT8, CPU);
    Image dst;
    auto ret = ImageResize(src, dst, SHAPE_960, SHAPE_540, Interpolation::BICUBIC, DeviceMode::CPU);
    EXPECT_EQ(ret, SUCCESS);
}

TEST_F(ImageOpsTest, Test_ImageResize_Failed_With_Invalid_Params)
{
    Image src(g_vector1080PUint8Value100.data(), {SHAPE_1920, SHAPE_1080}, ImageFormat::RGB_PLANAR, DataType::UINT8,
              CPU);
    Image dst;
    auto ret = ImageResize(src, dst, SHAPE_960, SHAPE_540, Interpolation::BICUBIC, DeviceMode::CPU);
    EXPECT_EQ(ret, ERR_INVALID_PARAM);
}

TEST_F(ImageOpsTest, Test_ImageCrop_Success_With_Premalloced_Dst)
{
    Image src(g_vector1080PUint8Value100.data(), {SHAPE_11, SHAPE_11}, ImageFormat::RGB, DataType::UINT8, CPU);
    Image dst(g_vector1080PHalfUint8Value100.data(), {CROP_WIDTH, CROP_HEIGHT}, ImageFormat::RGB, DataType::UINT8, CPU);
    auto ret = ImageCrop(src, dst, 0, 0, CROP_HEIGHT, CROP_WIDTH, DeviceMode::CPU);
    EXPECT_EQ(ret, SUCCESS);
}

TEST_F(ImageOpsTest, Test_ImageCrop_Success_With_ImplicitMalloc)
{
    Image src(g_vector1080PUint8Value100.data(), {SHAPE_11, SHAPE_11}, ImageFormat::RGB, DataType::UINT8, CPU);
    Image dst;
    auto ret = ImageCrop(src, dst, 0, 0, CROP_HEIGHT, CROP_WIDTH, DeviceMode::CPU);
    EXPECT_EQ(ret, SUCCESS);
}

TEST_F(ImageOpsTest, Test_ImageCrop_Failed_With_Invalid_Params)
{
    Image src(g_vector1080PUint8Value100.data(), {SHAPE_11, SHAPE_11}, ImageFormat::RGB_PLANAR, DataType::UINT8, CPU);
    Image dst;
    auto ret = ImageCrop(src, dst, 0, 0, CROP_HEIGHT, CROP_WIDTH, DeviceMode::CPU);
    EXPECT_EQ(ret, ERR_INVALID_PARAM);
}
} // namespace
int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    return ret;
}