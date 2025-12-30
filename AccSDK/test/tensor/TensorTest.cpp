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
#include "acc/tensor/Tensor.h"

using namespace Acc;
namespace {
constexpr char* INVALID_DEVICE = "npu:0";
constexpr char* CPU = "cpu";
constexpr int TOTAL_BYTES = 10;
constexpr int SHAPE_H = 2;
constexpr int SHAPE_W = 5;
class TensorTest : public testing::Test {
};

TEST_F(TensorTest, Test_Tensor_Construct_With_SharedPtr_Should_Return_Success_When_Everything_Is_Ok)
{
    std::int8_t* data = new std::int8_t[TOTAL_BYTES];
    std::shared_ptr<std::int8_t> arr(data, std::default_delete<std::int8_t[]>());
    std::vector<size_t> tensorShape = {SHAPE_H, SHAPE_W};
    Tensor tensor(arr, tensorShape, DataType::INT8, TensorFormat::ND, CPU);
    EXPECT_EQ(tensor.Shape(), tensorShape);
    EXPECT_EQ(tensor.DType(), DataType::INT8);
    EXPECT_STREQ(tensor.Device().get(), CPU);
    EXPECT_EQ(tensor.Format(), TensorFormat::ND);
    EXPECT_EQ(tensor.NumBytes(), tensorShape[0] * tensorShape[1]);
    EXPECT_EQ(tensor.Ptr(), arr.get());
    EXPECT_EQ(tensor.AuxInfo().elementNums, TOTAL_BYTES);
    EXPECT_EQ(tensor.AuxInfo().perElementBytes, 1);
    EXPECT_EQ(tensor.AuxInfo().totalBytes, TOTAL_BYTES);
    EXPECT_EQ(tensor.AuxInfo().logicalStrides[0], SHAPE_W);
    EXPECT_EQ(tensor.AuxInfo().memoryStrides[0], SHAPE_W);
    EXPECT_EQ(tensor.AuxInfo().logicalStrides[1], 1);
    EXPECT_EQ(tensor.AuxInfo().memoryStrides[1], 1);
}

TEST_F(TensorTest, Test_Tensor_Construct_With_SharedPtr_Should_Return_Failure_When_Input_Is_Invalid)
{
    std::int8_t* data = new std::int8_t[TOTAL_BYTES];
    std::shared_ptr<std::int8_t> arr(data, std::default_delete<std::int8_t[]>());
    std::vector<size_t> tensorShape = {SHAPE_H, SHAPE_W};
    EXPECT_THROW(Tensor(nullptr, tensorShape, DataType::INT8, TensorFormat::ND, CPU), std::runtime_error);
    EXPECT_THROW(Tensor(arr, std::vector<size_t>(), DataType::INT8, TensorFormat::ND, CPU), std::runtime_error);
    EXPECT_THROW(Tensor(arr, tensorShape, DataType::INT8, TensorFormat::ND, INVALID_DEVICE), std::runtime_error);
}

TEST_F(TensorTest, Test_Tensor_Construct_With_Ptr_Should_Return_Success_When_Everything_Is_Ok)
{
    int8_t* arr = new int8_t[TOTAL_BYTES];
    std::vector<size_t> tensorShape = {SHAPE_H, SHAPE_W};
    Tensor tensor(arr, tensorShape, DataType::INT8, TensorFormat::ND, CPU);
    EXPECT_EQ(tensor.Shape(), tensorShape);
    EXPECT_EQ(tensor.DType(), DataType::INT8);
    EXPECT_STREQ(tensor.Device().get(), CPU);
    EXPECT_EQ(tensor.Format(), TensorFormat::ND);
    EXPECT_EQ(tensor.NumBytes(), tensorShape[0] * tensorShape[1]);
    EXPECT_EQ(tensor.Ptr(), arr);
    EXPECT_EQ(tensor.AuxInfo().elementNums, TOTAL_BYTES);
    EXPECT_EQ(tensor.AuxInfo().perElementBytes, 1);
    EXPECT_EQ(tensor.AuxInfo().totalBytes, TOTAL_BYTES);
    EXPECT_EQ(tensor.AuxInfo().logicalStrides[0], SHAPE_W);
    EXPECT_EQ(tensor.AuxInfo().memoryStrides[0], SHAPE_W);
    EXPECT_EQ(tensor.AuxInfo().logicalStrides[1], 1);
    EXPECT_EQ(tensor.AuxInfo().memoryStrides[1], 1);
    delete arr;
}

TEST_F(TensorTest, Test_Tensor_Construct_With_Ptr_Should_Return_Failure_When_Input_Is_Invalid)
{
    int8_t* arr = new int8_t[TOTAL_BYTES];
    std::vector<size_t> tensorShape = {SHAPE_H, SHAPE_W};
    EXPECT_THROW(Tensor(nullptr, tensorShape, DataType::INT8, TensorFormat::ND, CPU), std::runtime_error);
    EXPECT_THROW(Tensor(arr, std::vector<size_t>(), DataType::INT8, TensorFormat::ND, CPU), std::runtime_error);
    EXPECT_THROW(Tensor(arr, tensorShape, DataType::INT8, TensorFormat::ND, INVALID_DEVICE), std::runtime_error);
    tensorShape = {SHAPE_H, SHAPE_W, UINT_MAX};
    EXPECT_THROW(Tensor(arr, tensorShape, DataType::INT8, TensorFormat::ND, CPU), std::runtime_error);
}

TEST_F(TensorTest, Test_Tensor_Clone_Return_Success_When_Everything_Is_Ok)
{
    std::int8_t* data = new std::int8_t[TOTAL_BYTES];
    std::shared_ptr<std::int8_t> arr(data, std::default_delete<std::int8_t[]>());
    std::vector<size_t> tensorShape = {SHAPE_H, SHAPE_W};
    Tensor tensor(arr, tensorShape, DataType::INT8, TensorFormat::ND, CPU);
    Tensor anotherTensor;
    auto ret = tensor.Clone(anotherTensor);
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(anotherTensor.Shape(), tensorShape);
    EXPECT_EQ(anotherTensor.DType(), DataType::INT8);
    EXPECT_STREQ(anotherTensor.Device().get(), CPU);
    EXPECT_EQ(anotherTensor.Format(), TensorFormat::ND);
    EXPECT_EQ(anotherTensor.NumBytes(), tensorShape[0] * tensorShape[1]);
}

TEST_F(TensorTest, Test_Tensor_GetProperties_Return_Success_When_Default_Construct)
{
    Tensor tensor;
    EXPECT_EQ(tensor.Shape().size(), 0);
    EXPECT_EQ(tensor.DType(), DataType::FLOAT32);
    EXPECT_STREQ(tensor.Device().get(), CPU);
    EXPECT_EQ(tensor.Format(), TensorFormat::ND);
    EXPECT_EQ(tensor.NumBytes(), 0);
    EXPECT_EQ(tensor.Ptr(), nullptr);
    EXPECT_EQ(tensor.AuxInfo().elementNums, 0);
    EXPECT_EQ(tensor.AuxInfo().perElementBytes, 0);
    EXPECT_EQ(tensor.AuxInfo().totalBytes, 0);
    EXPECT_EQ(tensor.AuxInfo().logicalStrides.size(), 0);
    EXPECT_EQ(tensor.AuxInfo().memoryStrides.size(), 0);
}

TEST_F(TensorTest, Test_Tensor_SetFormat_Return_Success_With_Correct_Format)
{
    std::int8_t* data = new std::int8_t[TOTAL_BYTES];
    std::shared_ptr<std::int8_t> arr(data, std::default_delete<std::int8_t[]>());
    std::vector<size_t> tensorShape = {SHAPE_H, SHAPE_H, SHAPE_H, SHAPE_H};
    Tensor tensor(arr, tensorShape, DataType::INT8, TensorFormat::NHWC, CPU);
    ErrorCode ret = tensor.SetFormat(TensorFormat::NHWC);
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(tensor.Format(), TensorFormat::NHWC);
    ret = tensor.SetFormat(TensorFormat::ND);
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(tensor.Format(), TensorFormat::ND);
}

TEST_F(TensorTest, Test_Tensor_SetFormat_Return_Fail_With_Wrong_Format)
{
    std::int8_t* data = new std::int8_t[TOTAL_BYTES];
    std::shared_ptr<std::int8_t> arr(data, std::default_delete<std::int8_t[]>());
    std::vector<size_t> tensorShape = {SHAPE_H, SHAPE_W};
    Tensor tensor(arr, tensorShape, DataType::INT8, TensorFormat::ND, CPU);
    ErrorCode ret = tensor.SetFormat(TensorFormat::NCHW);
    EXPECT_NE(ret, 0);
}

TEST_F(TensorTest, Test_GetByteSize_Success_With_Correct_DataType)
{
    size_t bSize = GetByteSize(DataType::INT8);
    EXPECT_EQ(bSize, ONE_BYTE);
    bSize = GetByteSize(DataType::UINT8);
    EXPECT_EQ(bSize, ONE_BYTE);
    bSize = GetByteSize(DataType::FLOAT32);
    EXPECT_EQ(bSize, FOUR_BYTE);
}

TEST_F(TensorTest, Test_Construct_Tensor_Failed_With_Device_IsNullptr)
{
    std::int8_t* data = new std::int8_t[TOTAL_BYTES];
    std::shared_ptr<std::int8_t> arr(data, std::default_delete<std::int8_t[]>());
    std::vector<size_t> tensorShape = {SHAPE_H, SHAPE_W};
    EXPECT_THROW(Tensor tensor(arr, tensorShape, DataType::INT8, TensorFormat::ND, nullptr), std::runtime_error);
}

TEST_F(TensorTest, Test_GetByteSize_Wrong_With_Correct_DataType)
{
    const int tmp = -1;
    EXPECT_THROW(GetByteSize(static_cast<DataType>(tmp)), std::runtime_error);
}
} // namespace

int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    return ret;
}