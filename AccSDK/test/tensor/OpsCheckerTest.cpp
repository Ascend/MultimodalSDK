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
 * Description: test tensor.
 * Author: ACC SDK
 * Create: 2025
 * History: NA
 */

#include <gtest/gtest.h>
#include <cstdint>
#include <iostream>
#include <climits>
#include <string>
#include "acc/tensor/Tensor.h"
#include "acc/tensor/OpsBaseChecker.h"
#include "acc/tensor/OpsCustomChecker.h"
#include "acc/core/framework/OperatorIndex.h"
#include "acc/core/framework/OperatorContext.h"
using namespace Acc;
namespace {
constexpr char* CPU = "cpu";
constexpr int TOTAL_BYTES = 100;
constexpr size_t SHAPE_H = 10;
constexpr size_t SHAPE_W = 11;
constexpr size_t SHAPE_C = 3;
constexpr size_t SMALL_H = 5;
constexpr size_t SMALL_W = 6;
constexpr size_t LARGE_H = 8193;
constexpr size_t LARGE_W = 8194;
class OpsCheckerTest : public testing::Test {
};
TEST_F(OpsCheckerTest, Test_OpsBaseChecker_Constructor_Fail_With_Invalid_OpName)
{
    EXPECT_THROW(OpsBaseChecker checker{OperatorId::OTHER}, std::runtime_error);
}

TEST_F(OpsCheckerTest, Test_OpsBaseChecker_CheckEachTensorValid_Fail_With_Invalid_Input)
{
    std::int8_t* data = new std::int8_t[TOTAL_BYTES];
    std::shared_ptr<std::int8_t> arr(data, std::default_delete<std::int8_t[]>());
    std::vector<size_t> tensorShape = {SHAPE_H, SHAPE_H, SHAPE_H, SHAPE_H};
    Tensor tensor(arr, tensorShape, DataType::INT8, TensorFormat::NHWC, CPU);
    // input tensor size invalid
    OperatorContext ctx{{std::cref(tensor), std::cref(tensor)}, {std::ref(tensor)}};
    OpsBaseChecker checker = OpsBaseChecker(OperatorId::RESIZE);
    EXPECT_NE(checker.CheckAndImplicitMalloc(ctx), 0);

    // output tensor size invalid
    OperatorContext ctx1{{std::cref(tensor)}, {std::ref(tensor), std::ref(tensor)}};
    EXPECT_NE(checker.CheckAndImplicitMalloc(ctx1), 0);

    // input tensor dtype invalid
    OperatorContext ctx2{{std::cref(tensor)}, {std::ref(tensor)}};
    EXPECT_NE(checker.CheckAndImplicitMalloc(ctx2), 0);

    // input tensor format invalid
    Tensor tensor1(arr, tensorShape, DataType::UINT8, TensorFormat::NCHW, CPU);
    OperatorContext ctx3{{std::cref(tensor1)}, {std::ref(tensor)}};
    EXPECT_NE(checker.CheckAndImplicitMalloc(ctx3), 0);

    // input tensor batch invalid
    Tensor tensor2(arr, tensorShape, DataType::UINT8, TensorFormat::NHWC, CPU);
    OperatorContext ctx6{{std::cref(tensor2)}, {std::ref(tensor)}};
    EXPECT_NE(checker.CheckAndImplicitMalloc(ctx6), 0);

    // input tensor batch invalid
    std::vector<size_t> tensorShape1 = {SHAPE_C, SHAPE_H, SHAPE_W, SHAPE_C};
    Tensor tensor3(arr, tensorShape1, DataType::UINT8, TensorFormat::NHWC, CPU);
    OperatorContext ctx4{{std::cref(tensor3)}, {std::ref(tensor)}};
    EXPECT_NE(checker.CheckAndImplicitMalloc(ctx4), 0);

    // input tensor channel invalid
    std::vector<size_t> tensorShape2 = {1, SHAPE_H, SHAPE_W, 1};
    Tensor tensor4(arr, tensorShape2, DataType::UINT8, TensorFormat::NHWC, CPU);
    OperatorContext ctx5{{std::cref(tensor4)}, {std::ref(tensor)}};
    EXPECT_NE(checker.CheckAndImplicitMalloc(ctx5), 0);

    // input tensor height invalid
    std::vector<size_t> tensorShape3 = {1, SMALL_H, SHAPE_W, SHAPE_C};
    Tensor tensor5(arr, tensorShape3, DataType::UINT8, TensorFormat::NHWC, CPU);
    OperatorContext ctx10{{std::cref(tensor5)}, {std::ref(tensor)}};
    EXPECT_NE(checker.CheckAndImplicitMalloc(ctx10), 0);
    std::vector<size_t> tensorShape4 = {1, LARGE_H, SHAPE_W, SHAPE_C};
    Tensor tensor6(arr, tensorShape4, DataType::UINT8, TensorFormat::NHWC, CPU);
    OperatorContext ctx7{{std::cref(tensor6)}, {std::ref(tensor)}};
    EXPECT_NE(checker.CheckAndImplicitMalloc(ctx7), 0);

    // input tensor weight invalid
    std::vector<size_t> tensorShape5 = {1, SHAPE_H, SMALL_W, SHAPE_C};
    Tensor tensor7(arr, tensorShape5, DataType::UINT8, TensorFormat::NHWC, CPU);
    OperatorContext ctx8{{std::cref(tensor7)}, {std::ref(tensor)}};
    EXPECT_NE(checker.CheckAndImplicitMalloc(ctx8), 0);
    std::vector<size_t> tensorShape6 = {1, SHAPE_H, LARGE_W, SHAPE_C};
    Tensor tensor8(arr, tensorShape6, DataType::UINT8, TensorFormat::NHWC, CPU);
    OperatorContext ctx9{{std::cref(tensor8)}, {std::ref(tensor)}};
    EXPECT_NE(checker.CheckAndImplicitMalloc(ctx9), 0);
}

TEST_F(OpsCheckerTest, Test_OpsBaseChecker_Success_With_Vaild_Input_Output)
{
    std::int8_t* data = new std::int8_t[TOTAL_BYTES];
    std::shared_ptr<std::int8_t> arr(data, std::default_delete<std::int8_t[]>());
    std::vector<size_t> tensorShape = {1, SHAPE_H, SHAPE_W, SHAPE_C};
    Tensor tensor(arr, tensorShape, DataType::UINT8, TensorFormat::NHWC, CPU);
    OpsBaseChecker checker = OpsBaseChecker(OperatorId::RESIZE);
    // pre-malloc output
    std::vector<size_t> tensorShape1 = {1, SHAPE_H, SHAPE_W, SHAPE_C};
    Tensor tensor1(arr, tensorShape1, DataType::UINT8, TensorFormat::NHWC, CPU);
    OperatorContext ctx{{std::cref(tensor)}, {std::ref(tensor1)}};
    EXPECT_EQ(checker.CheckAndImplicitMalloc(ctx), 0);

    // implicit malloc
    Tensor tensor2;
    OperatorContext ctx1{{std::cref(tensor)}, {std::ref(tensor2)}};
    EXPECT_EQ(checker.CheckAndImplicitMalloc(ctx1), 0);
}

TEST_F(OpsCheckerTest, Test_ResizeChecker_CheckCustomRules_Fail_With_Invalid_Input)
{
    std::int8_t* data = new std::int8_t[TOTAL_BYTES];
    std::shared_ptr<std::int8_t> arr(data, std::default_delete<std::int8_t[]>());
    std::vector<size_t> tensorShape = {1, SHAPE_H, SHAPE_W, SHAPE_C};
    std::vector<size_t> tensorShape1 = {1, SHAPE_H, SHAPE_W, SHAPE_C};
    Tensor tensor(arr, tensorShape, DataType::UINT8, TensorFormat::NHWC, CPU);
    Tensor tensor1(arr, tensorShape1, DataType::UINT8, TensorFormat::NHWC, CPU);
    ResizeChecker checker = ResizeChecker(OperatorId::RESIZE);
    // interpolation type invalid
    ResizeContext ctx{{std::cref(tensor)},           {std::ref(tensor1)}, 0, 0,
                           static_cast<Interpolation>(1), DeviceMode::CPU};
    EXPECT_NE(checker.CheckAndImplicitMalloc(ctx), 0);

    // reized shape invalid
    ResizeContext ctx1{{std::cref(tensor)}, {std::ref(tensor1)}, 0, 0, Interpolation::BICUBIC, DeviceMode::CPU};
    EXPECT_NE(checker.CheckAndImplicitMalloc(ctx1), 0);

    // output tensor shape invalid
    ResizeContext ctx2{{std::cref(tensor)},    {std::ref(tensor1)}, SHAPE_H, SHAPE_H,
                            Interpolation::BICUBIC, DeviceMode::CPU};
    EXPECT_NE(checker.CheckAndImplicitMalloc(ctx2), 0);
    ResizeContext ctx3{{std::cref(tensor)},    {std::ref(tensor1)}, SHAPE_W, SHAPE_W,
                            Interpolation::BICUBIC, DeviceMode::CPU};
    EXPECT_NE(checker.CheckAndImplicitMalloc(ctx3), 0);
}

TEST_F(OpsCheckerTest, Test_ResizeChecker_Success_With_Vaild_Input_Output)
{
    std::int8_t* data = new std::int8_t[TOTAL_BYTES];
    std::shared_ptr<std::int8_t> arr(data, std::default_delete<std::int8_t[]>());
    std::vector<size_t> tensorShape = {1, SHAPE_H, SHAPE_W, SHAPE_C};
    std::vector<size_t> tensorShape1 = {1, SHAPE_H, SHAPE_W, SHAPE_C};
    Tensor tensor(arr, tensorShape, DataType::UINT8, TensorFormat::NHWC, CPU);
    Tensor tensor1(arr, tensorShape1, DataType::UINT8, TensorFormat::NHWC, CPU);
    ResizeChecker checker = ResizeChecker(OperatorId::RESIZE);
    ResizeContext ctx{{std::cref(tensor)},    {std::ref(tensor1)}, SHAPE_H, SHAPE_W,
                           Interpolation::BICUBIC, DeviceMode::CPU};
    EXPECT_EQ(checker.CheckAndImplicitMalloc(ctx), 0);
}

TEST_F(OpsCheckerTest, Test_CropChecker_CheckCustomRules_Fail_With_Invalid_Input)
{
    std::int8_t* data = new std::int8_t[TOTAL_BYTES];
    std::shared_ptr<std::int8_t> arr(data, std::default_delete<std::int8_t[]>());
    std::vector<size_t> tensorShape = {1, SHAPE_H, SHAPE_W, SHAPE_C};
    std::vector<size_t> tensorShape1 = {1, SHAPE_H, SHAPE_W, SHAPE_C};
    Tensor tensor(arr, tensorShape, DataType::UINT8, TensorFormat::NHWC, CPU);
    Tensor tensor1(arr, tensorShape1, DataType::UINT8, TensorFormat::NHWC, CPU);
    CropChecker checker = CropChecker(OperatorId::CROP);

    // crop shape invalid
    CropContext ctx1{{std::cref(tensor)}, {std::ref(tensor1)}, 1, 1, SHAPE_H, SHAPE_H, DeviceMode::CPU};
    EXPECT_NE(checker.CheckAndImplicitMalloc(ctx1), 0);
    CropContext ctx2{{std::cref(tensor)}, {std::ref(tensor1)}, 1, 1, 1, SHAPE_W, DeviceMode::CPU};
    EXPECT_NE(checker.CheckAndImplicitMalloc(ctx2), 0);
    CropContext ctx3{{std::cref(tensor)}, {std::ref(tensor1)}, 1, 1, 1, 1, DeviceMode::CPU};
    EXPECT_NE(checker.CheckAndImplicitMalloc(ctx3), 0);

    // output shape invalid
    CropContext ctx4{{std::cref(tensor)}, {std::ref(tensor1)}, 0, 0, SHAPE_H, SHAPE_H, DeviceMode::CPU};
    EXPECT_NE(checker.CheckAndImplicitMalloc(ctx4), 0);
}

TEST_F(OpsCheckerTest, Test_CropChecker_Success_With_Vaild_Input_Output)
{
    std::int8_t* data = new std::int8_t[TOTAL_BYTES];
    std::shared_ptr<std::int8_t> arr(data, std::default_delete<std::int8_t[]>());
    std::vector<size_t> tensorShape = {1, SHAPE_H, SHAPE_W, SHAPE_C};
    std::vector<size_t> tensorShape1 = {1, SHAPE_H, SHAPE_H, SHAPE_C};
    Tensor tensor(arr, tensorShape, DataType::UINT8, TensorFormat::NHWC, CPU);
    Tensor tensor1(arr, tensorShape1, DataType::UINT8, TensorFormat::NHWC, CPU);
    CropChecker checker = CropChecker(OperatorId::CROP);
    CropContext ctx{{std::cref(tensor)}, {std::ref(tensor1)}, 0, 0, SHAPE_H, SHAPE_H, DeviceMode::CPU};
    EXPECT_EQ(checker.CheckAndImplicitMalloc(ctx), 0);
}
} // namespace

int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    return ret;
}