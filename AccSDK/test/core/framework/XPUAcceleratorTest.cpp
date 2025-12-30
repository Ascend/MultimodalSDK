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
 * Description: test XPUAccelerator api.
 * Author: ACC SDK
 * Create: 2025
 * History: NA
 */
#include <cstdint>
#include <gtest/gtest.h>
#include "acc/core/framework/XPUAccelerator.h"
#include "acc/core/framework/OperatorContext.h"
#include "acc/tensor/TensorDataType.h"
#include "acc/ErrorCode.h"
#include "acc/utils/LogImpl.h"

using namespace Acc;
namespace {
constexpr size_t BATCH_SIZE_ONE = 1;
constexpr size_t SHAPE_1920 = 1920;
constexpr size_t SHAPE_1080 = 1080;
constexpr uint32_t CROP_HEIGHT = 10;
constexpr uint32_t CROP_WIDTH = 10;
constexpr size_t CHANNEL_THREE = 3;
constexpr uint8_t VALID_VALUE = 100;
const std::vector<size_t> SHAPE1_NHWC = {1, 11, 11, 3};
const std::vector<size_t> SHAPE2_NHWC = {1, 10, 10, 3};
std::vector<uint8_t> g_vector1080PUint8Value100(SHAPE_1920 * SHAPE_1080* CHANNEL_THREE, VALID_VALUE);
std::vector<uint8_t> g_vector1080PUint8Value100New(CROP_HEIGHT * CROP_HEIGHT* CHANNEL_THREE, VALID_VALUE);

class XPUAcceleratorTest : public testing::Test {
    void SetUp() override
    {
        RegisterLogConf(LogLevel::WARN, nullptr);
    }
};

TEST_F(XPUAcceleratorTest, Test_Get_XPU_Accelerator_Should_Success)
{
    EXPECT_NO_THROW({
        auto accelerator = Acc::GetAccelerator(DeviceMode::CPU);
    });
}

TEST_F(XPUAcceleratorTest, Test_Get_XPU_Accelerator_Execute_Should_Success)
{
    Tensor src(g_vector1080PUint8Value100.data(), SHAPE1_NHWC, DataType::UINT8, TensorFormat::NHWC, "cpu");
    Tensor dst(g_vector1080PUint8Value100New.data(), {BATCH_SIZE_ONE, CROP_HEIGHT, CROP_WIDTH, CHANNEL_THREE},
               DataType::UINT8, TensorFormat::NHWC, "cpu");
    auto accelerator = Acc::GetAccelerator(DeviceMode::CPU);
    CropContext opCtx{{std::cref(src)}, {std::ref(dst)}, 0, 0, CROP_HEIGHT, CROP_WIDTH, DeviceMode::CPU};
    ErrorCode ret = accelerator.ExecuteOperator(OperatorId::CROP, opCtx);
    EXPECT_EQ(ret, SUCCESS);
}

TEST_F(XPUAcceleratorTest, Test_Get_XPU_Accelerator_Execute_Should_Failed_With_Invalid_Operator)
{
    Tensor src(g_vector1080PUint8Value100.data(), SHAPE1_NHWC, DataType::UINT8, TensorFormat::NHWC, "cpu");
    Tensor dst(g_vector1080PUint8Value100New.data(), {BATCH_SIZE_ONE, CROP_HEIGHT, CROP_WIDTH, CHANNEL_THREE},
               DataType::UINT8, TensorFormat::NHWC, "cpu");
    auto accelerator = Acc::GetAccelerator(DeviceMode::CPU);
    ResizeContext opCtx{{std::cref(src)}, {std::ref(dst)}, 0, 0, Interpolation::BICUBIC, DeviceMode::CPU};
    ErrorCode ret = accelerator.ExecuteOperator(OperatorId::OTHER, opCtx);
    EXPECT_EQ(ret, ERR_INVALID_PARAM);
}
}

int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    return ret;
}