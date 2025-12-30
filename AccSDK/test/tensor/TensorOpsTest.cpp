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
#include "acc/tensor/Tensor.h"
#include "acc/tensor/TensorOps.h"
using namespace Acc;
namespace {
constexpr size_t BATCH_SIZE_ONE = 1;
constexpr size_t BATCH_SIZE_TWO = 2;
constexpr size_t SHAPE_1920 = 1920;
constexpr size_t SHAPE_1080 = 1080;
constexpr size_t SHAPE_960 = 960;
constexpr size_t SHAPE_540 = 540;
constexpr uint32_t CROP_HEIGHT = 10;
constexpr uint32_t CROP_WIDTH = 10;
constexpr size_t CHANNEL_THREE = 3;
constexpr size_t CHANNEL_ONE = 1;
const std::vector<size_t> SHAPE1_NHWC = {1, 11, 11, 3};
constexpr uint8_t VALID_VALUE = 100;
constexpr float VALID_VALUE_FLOAT = 100.0f;
constexpr float VALID_VALUE_FLOAT_NEW = 99.0f;
constexpr char* CPU = "cpu";
std::vector<uint8_t> g_vector1080PUint8Value100(SHAPE_1920* SHAPE_1080* CHANNEL_THREE, VALID_VALUE);
std::vector<uint8_t> g_vector1080PHalfUint8Value100(SHAPE_960* SHAPE_540* CHANNEL_THREE, VALID_VALUE);
std::vector<float> g_vector1080PFloatValue100(SHAPE_1920* SHAPE_1080* CHANNEL_THREE, VALID_VALUE_FLOAT);
std::vector<float> g_vector1080PFloatValue100New(SHAPE_1920* SHAPE_1080* CHANNEL_THREE, VALID_VALUE_FLOAT_NEW);

class TensorOpsTest : public testing::Test {
};

TEST_F(TensorOpsTest, Test_TensorCrop_Success_With_ImplicitMalloc)
{
    Tensor src(g_vector1080PUint8Value100.data(), SHAPE1_NHWC, DataType::UINT8, TensorFormat::NHWC, CPU);
    Tensor dst;
    auto ret = TensorCrop(src, dst, 0, 0, CROP_HEIGHT, CROP_WIDTH, DeviceMode::CPU);
    EXPECT_EQ(ret, SUCCESS);
}

TEST_F(TensorOpsTest, Test_TensorCrop_Failed_With_Premalloced_Dst_Is_The_Same_As_Src)
{
    Tensor src(g_vector1080PUint8Value100.data(), SHAPE1_NHWC, DataType::UINT8, TensorFormat::NHWC, CPU);
    Tensor dst(g_vector1080PUint8Value100.data(), {BATCH_SIZE_ONE, CROP_HEIGHT, CROP_WIDTH, CHANNEL_THREE},
               DataType::UINT8, TensorFormat::NHWC, CPU);
    auto ret = TensorCrop(src, dst, 0, 0, CROP_HEIGHT, CROP_WIDTH, DeviceMode::CPU);
    EXPECT_EQ(ret, ERR_BAD_COPY);
}

TEST_F(TensorOpsTest, Test_TensorCrop_Success_With_Premalloced_Dst)
{
    Tensor src(g_vector1080PUint8Value100.data(), SHAPE1_NHWC, DataType::UINT8, TensorFormat::NHWC, CPU);
    Tensor dst(g_vector1080PHalfUint8Value100.data(), {BATCH_SIZE_ONE, CROP_HEIGHT, CROP_WIDTH, CHANNEL_THREE},
               DataType::UINT8, TensorFormat::NHWC, CPU);
    auto ret = TensorCrop(src, dst, 0, 0, CROP_HEIGHT, CROP_WIDTH, DeviceMode::CPU);
    EXPECT_EQ(ret, SUCCESS);
}

TEST_F(TensorOpsTest, Test_TensorCrop_Failed_With_Invalid_Params)
{
    Tensor src(g_vector1080PUint8Value100.data(), SHAPE1_NHWC, DataType::UINT8, TensorFormat::NCHW, CPU);
    Tensor dst(g_vector1080PUint8Value100.data(), {BATCH_SIZE_ONE, CROP_HEIGHT, CROP_WIDTH, CHANNEL_THREE},
               DataType::UINT8, TensorFormat::NCHW, CPU);
    // device mode invalid
    auto ret = TensorCrop(src, dst, 0, 0, CROP_HEIGHT, CROP_WIDTH, static_cast<DeviceMode>(1));
    EXPECT_EQ(ret, ERR_INVALID_PARAM);
    // invalid input tensor format
    src.SetFormat(TensorFormat::NCHW);
    ret = TensorCrop(src, dst, 0, 0, CROP_HEIGHT, CROP_WIDTH, DeviceMode::CPU);
    EXPECT_EQ(ret, ERR_INVALID_PARAM);
    // invalid input tensor shape
    Tensor src1(g_vector1080PUint8Value100.data(), {BATCH_SIZE_ONE, BATCH_SIZE_ONE, SHAPE_1920, CHANNEL_THREE},
                DataType::UINT8, TensorFormat::NHWC, CPU);
    ret = TensorCrop(src1, dst, 0, 0, CROP_HEIGHT, CROP_WIDTH, DeviceMode::CPU);
    EXPECT_EQ(ret, ERR_INVALID_PARAM);
    // invalid input tensor dtype
    Tensor src2(g_vector1080PUint8Value100.data(), SHAPE1_NHWC, DataType::INT8, TensorFormat::NHWC, CPU);
    ret = TensorCrop(src2, dst, 0, 0, CROP_HEIGHT, CROP_WIDTH, DeviceMode::CPU);
    EXPECT_EQ(ret, ERR_INVALID_PARAM);
    // crop shape mismatch
    src.SetFormat(TensorFormat::NHWC);
    ret = TensorCrop(src, dst, 0, 0, SHAPE_1080, SHAPE_1080, DeviceMode::CPU);
    EXPECT_EQ(ret, ERR_INVALID_PARAM);
    // dst format invalid
    dst.SetFormat(TensorFormat::NCHW);
    ret = TensorCrop(src, dst, 0, 0, SHAPE_1080, SHAPE_1080, DeviceMode::CPU);
    EXPECT_EQ(ret, ERR_INVALID_PARAM);
}

TEST_F(TensorOpsTest, Test_TensorResize_Success_Use_Cpu)
{
    Tensor src(g_vector1080PUint8Value100.data(), {BATCH_SIZE_ONE, SHAPE_1080, SHAPE_1920, CHANNEL_THREE},
               DataType::UINT8, TensorFormat::NHWC, CPU);
    Tensor dst(g_vector1080PHalfUint8Value100.data(), {BATCH_SIZE_ONE, SHAPE_540, SHAPE_960, CHANNEL_THREE},
               DataType::UINT8, TensorFormat::NHWC, CPU);
    auto ret = TensorResize(src, dst, SHAPE_540, SHAPE_960, Interpolation::BICUBIC, DeviceMode::CPU);
    EXPECT_EQ(ret, SUCCESS);
}

TEST_F(TensorOpsTest, Test_TensorResize_Success_Use_Cpu_With_ImplicitMalloc)
{
    Tensor src(g_vector1080PUint8Value100.data(), {BATCH_SIZE_ONE, SHAPE_1080, SHAPE_1920, CHANNEL_THREE},
               DataType::UINT8, TensorFormat::NHWC, CPU);
    Tensor dst;
    auto ret = TensorResize(src, dst, SHAPE_540, SHAPE_960, Interpolation::BICUBIC, DeviceMode::CPU);
    EXPECT_EQ(ret, SUCCESS);
}

TEST_F(TensorOpsTest, Test_TensorResize_Should_Return_Failed_Use_Cpu_With_Invalid_Params)
{
    Tensor src(g_vector1080PUint8Value100.data(), {BATCH_SIZE_ONE, SHAPE_1080, SHAPE_1920, CHANNEL_THREE},
               DataType::UINT8, TensorFormat::NHWC, CPU);
    Tensor dst(g_vector1080PHalfUint8Value100.data(), {BATCH_SIZE_ONE, SHAPE_540, SHAPE_960, CHANNEL_THREE},
               DataType::UINT8, TensorFormat::NHWC, CPU);
    // interpolation invalid
    auto ret = TensorResize(src, dst, SHAPE_540, SHAPE_960, static_cast<Interpolation>(1), DeviceMode::CPU);
    EXPECT_EQ(ret, ERR_UNSUPPORTED_TYPE);
    // device mode invalid
    ret = TensorResize(src, dst, SHAPE_540, SHAPE_960, Interpolation::BICUBIC, static_cast<DeviceMode>(1));
    EXPECT_EQ(ret, ERR_UNSUPPORTED_TYPE);
    // invalid input tensor format
    src.SetFormat(TensorFormat::NCHW);
    ret = TensorResize(src, dst, SHAPE_540, SHAPE_960, Interpolation::BICUBIC, DeviceMode::CPU);
    EXPECT_EQ(ret, ERR_INVALID_PARAM);
    // invalid input tensor shape
    Tensor src1(g_vector1080PUint8Value100.data(), {BATCH_SIZE_ONE, BATCH_SIZE_ONE, SHAPE_1920, CHANNEL_THREE},
                DataType::UINT8, TensorFormat::NHWC, CPU);
    ret = TensorResize(src1, dst, SHAPE_540, SHAPE_960, Interpolation::BICUBIC, DeviceMode::CPU);
    EXPECT_EQ(ret, ERR_INVALID_PARAM);
    // invalid input tensor dtype
    Tensor src2(g_vector1080PUint8Value100.data(), {BATCH_SIZE_ONE, SHAPE_1080, SHAPE_1920, CHANNEL_THREE},
                DataType::INT8, TensorFormat::NHWC, CPU);
    ret = TensorResize(src2, dst, SHAPE_540, SHAPE_960, Interpolation::BICUBIC, DeviceMode::CPU);
    EXPECT_EQ(ret, ERR_INVALID_PARAM);
    // resize shape mismatch
    src.SetFormat(TensorFormat::NHWC);
    ret = TensorResize(src, dst, SHAPE_540, SHAPE_540, Interpolation::BICUBIC, DeviceMode::CPU);
    EXPECT_EQ(ret, ERR_INVALID_PARAM);
    // dst format invalid
    dst.SetFormat(TensorFormat::NCHW);
    ret = TensorResize(src, dst, SHAPE_540, SHAPE_960, Interpolation::BICUBIC, DeviceMode::CPU);
    EXPECT_EQ(ret, ERR_INVALID_PARAM);
}

TEST_F(TensorOpsTest, Test_TensorNormalize_Should_Return_Success_With_NHWC)
{
    Tensor src(g_vector1080PFloatValue100.data(), {BATCH_SIZE_ONE, SHAPE_1080, SHAPE_1920, CHANNEL_THREE},
               DataType::FLOAT32, TensorFormat::NHWC, CPU);
    Tensor dst(g_vector1080PFloatValue100New.data(), {BATCH_SIZE_ONE, SHAPE_1080, SHAPE_1920, CHANNEL_THREE},
               DataType::FLOAT32, TensorFormat::NHWC, CPU);
    std::vector<float> mean = {0.1f, 0.1f, 0.1f};
    std::vector<float> std = {0.1f, 0.1f, 0.1f};
    auto ret = TensorNormalize(src, dst, mean, std, DeviceMode::CPU);
    EXPECT_EQ(ret, SUCCESS);
}

TEST_F(TensorOpsTest, Test_TensorNormalize_Should_Return_Success_With_NCHW)
{
    Tensor src(g_vector1080PFloatValue100.data(), {BATCH_SIZE_ONE, CHANNEL_THREE, SHAPE_1080, SHAPE_1920},
               DataType::FLOAT32, TensorFormat::NCHW, CPU);
    Tensor dst(g_vector1080PFloatValue100New.data(), {BATCH_SIZE_ONE, CHANNEL_THREE, SHAPE_1080, SHAPE_1920},
               DataType::FLOAT32, TensorFormat::NCHW, CPU);
    std::vector<float> mean = {0.1f, 0.1f, 0.1f};
    std::vector<float> std = {0.1f, 0.1f, 0.1f};
    auto ret = TensorNormalize(src, dst, mean, std, DeviceMode::CPU);
    EXPECT_EQ(ret, SUCCESS);
}

TEST_F(TensorOpsTest, Test_TensorNormalize_Should_Return_Success_With_ImplicitMalloc)
{
    Tensor src(g_vector1080PFloatValue100.data(), {BATCH_SIZE_ONE, SHAPE_1080, SHAPE_1920, CHANNEL_THREE},
               DataType::FLOAT32, TensorFormat::NHWC, CPU);
    Tensor dst;
    std::vector<float> mean = {0.1f, 0.1f, 0.1f};
    std::vector<float> std = {0.1f, 0.1f, 0.1f};
    auto ret = TensorNormalize(src, dst, mean, std, DeviceMode::CPU);
    EXPECT_EQ(ret, SUCCESS);
}

TEST_F(TensorOpsTest, Test_TensorNormalize_Should_Return_Failed_With_Invalid_Param)
{
    std::vector<float> mean = {0.1f, 0.1f, 0.1f};
    std::vector<float> std = {0.1f, 0.1f, 0.1f};
    Tensor dst;
    Tensor src(g_vector1080PFloatValue100.data(), {BATCH_SIZE_ONE, SHAPE_1080, SHAPE_1920, CHANNEL_THREE},
               DataType::FLOAT32, TensorFormat::NCHW, CPU);

    // invalid batch
    Tensor invalid_batch(g_vector1080PFloatValue100.data(), {BATCH_SIZE_TWO, SHAPE_1080, SHAPE_1920, CHANNEL_THREE},
                         DataType::FLOAT32, TensorFormat::NHWC, CPU);
    auto ret = TensorNormalize(invalid_batch, dst, mean, std, DeviceMode::CPU);
    EXPECT_EQ(ret, ERR_INVALID_PARAM);

    // invalid channel
    Tensor invalid_channel(g_vector1080PFloatValue100.data(), {BATCH_SIZE_TWO, SHAPE_1080, SHAPE_1920, CHANNEL_ONE},
                           DataType::FLOAT32, TensorFormat::NHWC, CPU);
    ret = TensorNormalize(invalid_channel, dst, mean, std, DeviceMode::CPU);
    EXPECT_EQ(ret, ERR_INVALID_PARAM);

    // invalid mean and std
    std::vector<float> invalid_mean = {0.1f, 0.1f};
    std::vector<float> invalid_std = {0.1f, 0.1f};
    ret = TensorNormalize(src, dst, invalid_mean, invalid_std, DeviceMode::CPU);
    EXPECT_EQ(ret, ERR_INVALID_PARAM);

    // mean and std vector's length is not equal
    ret = TensorNormalize(src, dst, mean, invalid_std, DeviceMode::CPU);
    EXPECT_EQ(ret, ERR_INVALID_PARAM);
}

TEST_F(TensorOpsTest, Test_TensorNormalize_Should_Return_Failed_With_Invalid_Dtype)
{
    Tensor src(g_vector1080PUint8Value100.data(), {BATCH_SIZE_TWO, SHAPE_1080, SHAPE_1920, CHANNEL_THREE},
               DataType::UINT8, TensorFormat::NHWC, CPU);
    Tensor dst;
    std::vector<float> mean = {0.1f, 0.1f, 0.1f};
    std::vector<float> std = {0.1f, 0.1f, 0.1f};
    auto ret = TensorNormalize(src, dst, mean, std, DeviceMode::CPU);
    EXPECT_EQ(ret, ERR_INVALID_PARAM);
}

TEST_F(TensorOpsTest, Test_TensorToTensor_Should_Return_Success)
{
    Tensor src(g_vector1080PUint8Value100.data(), {BATCH_SIZE_ONE, SHAPE_1080, SHAPE_1920, CHANNEL_THREE},
               DataType::UINT8, TensorFormat::NHWC, CPU);
    Tensor dst;
    auto ret = TensorToTensor(src, dst, TensorFormat::NHWC, DeviceMode::CPU);
    EXPECT_EQ(ret, SUCCESS);

    Tensor dst2;
    ret = TensorToTensor(src, dst2, TensorFormat::NCHW, DeviceMode::CPU);
    EXPECT_EQ(ret, SUCCESS);
}

TEST_F(TensorOpsTest, Test_TensorToTensor_Should_Return_Failed_With_Invalid_Target_format)
{
    Tensor src(g_vector1080PUint8Value100.data(), {BATCH_SIZE_ONE, SHAPE_1080, SHAPE_1920, CHANNEL_THREE},
               DataType::UINT8, TensorFormat::NHWC, CPU);
    Tensor dst;
    auto ret = TensorToTensor(src, dst, TensorFormat::ND, DeviceMode::CPU);
    EXPECT_EQ(ret, ERR_INVALID_PARAM);
}

TEST_F(TensorOpsTest, Test_TensorToTensor_Should_Return_Success_With_NCHW_Input)
{
    Tensor src(g_vector1080PUint8Value100.data(), {BATCH_SIZE_ONE, CHANNEL_THREE, SHAPE_1080, SHAPE_1920},
               DataType::UINT8, TensorFormat::NCHW, CPU);
    Tensor dst;
    auto ret = TensorToTensor(src, dst, TensorFormat::NCHW, DeviceMode::CPU);
    EXPECT_EQ(ret, SUCCESS);

    Tensor dst2;
    ret = TensorToTensor(src, dst2, TensorFormat::NHWC, DeviceMode::CPU);
    EXPECT_EQ(ret, SUCCESS);
}

TEST_F(TensorOpsTest, Test_TensorToTensor_Should_Return_Failed_With_Invalid_Input)
{
    Tensor src(g_vector1080PUint8Value100.data(), {BATCH_SIZE_ONE, SHAPE_1080, SHAPE_1920, CHANNEL_THREE},
               DataType::UINT8, TensorFormat::ND, CPU);
    Tensor dst;
    auto ret = TensorToTensor(src, dst, TensorFormat::NHWC, DeviceMode::CPU);
    EXPECT_EQ(ret, ERR_INVALID_PARAM);

    Tensor src1(g_vector1080PFloatValue100.data(), {BATCH_SIZE_ONE, SHAPE_1080, SHAPE_1920, CHANNEL_THREE},
               DataType::FLOAT32, TensorFormat::NHWC, CPU);
    ret = TensorToTensor(src1, dst, TensorFormat::NHWC, DeviceMode::CPU);
    EXPECT_EQ(ret, ERR_INVALID_PARAM);
}

TEST_F(TensorOpsTest, Test_TensorToTensor_Should_Return_Failed_With_Invalid_Output)
{
    Tensor src(g_vector1080PUint8Value100.data(), {BATCH_SIZE_ONE, SHAPE_1080, SHAPE_1920, CHANNEL_THREE},
               DataType::UINT8, TensorFormat::NHWC, CPU);
    Tensor dst(g_vector1080PUint8Value100.data(), {BATCH_SIZE_ONE, SHAPE_1080, SHAPE_1920, CHANNEL_THREE},
               DataType::UINT8, TensorFormat::NHWC, CPU);
    auto ret = TensorToTensor(src, dst, TensorFormat::NHWC, DeviceMode::CPU);
    EXPECT_EQ(ret, ERR_INVALID_PARAM);
}
} // namespace

int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    return ret;
}