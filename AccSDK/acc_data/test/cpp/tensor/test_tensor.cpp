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
 * @Description:
 * @Version: 1.0
 * @Date: 2025-4-7 9:00:00
 * @LastEditors: dev
 * @LastEditTime: 2025-4-7 9:00:00
 */
#include <gtest/gtest.h>
#include <thread>

#include "tensor/tensor.h"

namespace {

using namespace acclib::accdata;

class TestTensor : public ::testing::Test {
public:
    void SetUp() {}
    void TearDown() {}
    size_t mHeight360p{640};   // 640 360pH
    size_t mHeight720p{1280};  // 1280 720pH
    size_t mWidth360p{360};   // 360 360pW
    size_t mWidth720p{720};   // 720 720pW
    size_t mChannel{3};   // 3 is channel count
    std::vector<size_t> mShapeNhwc360p{1, mHeight360p, mWidth360p, mChannel};
    std::vector<size_t> mShapeNchw360p{1, mChannel, mHeight360p, mWidth360p};
    std::vector<size_t> mShapeNhwc720p{1, mHeight720p, mWidth720p, mChannel};
    std::vector<size_t> mShapeNchw720p{1, mChannel, mHeight720p, mWidth720p};
    size_t mBytes360p = mHeight360p * mWidth360p * mChannel * sizeof(float);
    size_t mBytes720p = mHeight720p * mWidth720p * mChannel * sizeof(float);
};

TEST_F(TestTensor, TestTensorBasic)
{
    Tensor t = std::move(Tensor());
    t = Tensor();

    t.SetLayout(TensorLayout::NCHW);
    ASSERT_EQ(t.Layout(), TensorLayout::NCHW);

    t.SetLayout(TensorLayout::NHWC);
    ASSERT_EQ(t.Layout(), TensorLayout::NHWC);
    t.Shape();

    std::stringstream ss;
    ss << TensorLayout::PLAIN << TensorLayout::NCHW << TensorLayout::NHWC << TensorLayout::LAST;
    ss << TensorDataType::CHAR << TensorDataType::FP32 << TensorDataType::UINT8 << TensorDataType::LAST;

    ASSERT_EQ(sizeof(char), TensorDataTypeSize(TensorDataType::CHAR));
    ASSERT_EQ(sizeof(float), TensorDataTypeSize(TensorDataType::FP32));
    ASSERT_EQ(sizeof(uint8_t), TensorDataTypeSize(TensorDataType::UINT8));
    ASSERT_EQ(0, TensorDataTypeSize(TensorDataType::LAST));
}

TEST_F(TestTensor, TestResize)
{
    Tensor originTensor;
    originTensor.SetLayout(TensorLayout::NCHW);
    originTensor.Resize(mShapeNchw360p, TensorDataType::FP32);
    ASSERT_EQ(originTensor.GetSize(), mBytes360p);
    auto originPtr = originTensor.RawDataPtr();

    originTensor.Resize(mShapeNchw360p, TensorDataType::FP32);
    ASSERT_EQ(originTensor.GetSize(), mBytes360p);
    ASSERT_EQ(originPtr, originTensor.RawDataPtr());

    originTensor.Resize(mShapeNhwc720p, TensorDataType::FP32);
    ASSERT_EQ(originTensor.GetSize(), mBytes720p);
    ASSERT_NE(originPtr, originTensor.RawDataPtr());

    originTensor.Resize<float>(mShapeNhwc720p);
    ASSERT_EQ(originTensor.GetSize(), mBytes720p);
}

TEST_F(TestTensor, TestCopy)
{
    Tensor originTensor;
    originTensor.SetLayout(TensorLayout::NCHW);
    originTensor.Resize(mShapeNchw360p, TensorDataType::FP32);

    Tensor copyTensor;
    AccDataErrorCode errCode = copyTensor.Copy(originTensor);
    ASSERT_EQ(errCode, H_OK);
    ASSERT_EQ(copyTensor.Layout(), originTensor.Layout());
    ASSERT_EQ(copyTensor.GetSize(), originTensor.GetSize());
    ASSERT_EQ(copyTensor.DataType(), originTensor.DataType());
    ASSERT_NE(copyTensor.RawDataPtr(), originTensor.RawDataPtr());
    uint8_t *src = reinterpret_cast<uint8_t *>(originTensor.RawDataPtr().get());
    uint8_t *dst = reinterpret_cast<uint8_t *>(copyTensor.RawDataPtr().get());
    for (int i = 0; i < mBytes360p; i++) {
        ASSERT_EQ(src[i], dst[i]);
    }
}

TEST_F(TestTensor, TestCopyWithDataIsNullptr) // 源数据指针为空指针，copy失败
{
    Tensor copyTensor;
    TensorShape tensorShape = {1, 3, 1080, 1920};
    EXPECT_EQ(copyTensor.Copy<float>(nullptr, tensorShape), AccDataErrorCode::H_TENSOR_ERROR);
}

TEST_F(TestTensor, TestCopyWithErrorType) // 源数据Datatype异常
{
    Tensor originTensor;
    originTensor.SetLayout(TensorLayout::NCHW);
    originTensor.Resize(mShapeNchw360p, TensorDataType::FP32);

    Tensor copyTensor;
    AccDataErrorCode errCode = copyTensor.Copy(originTensor.RawDataPtr().get(),
        {1, mChannel, mHeight360p, mWidth360p}, TensorDataType::LAST);
    EXPECT_EQ(errCode, AccDataErrorCode::H_TENSOR_ERROR);
}

TEST_F(TestTensor, TestShareData)
{
    Tensor originTensor;
    originTensor.SetLayout(TensorLayout::NCHW);
    originTensor.Resize(mShapeNchw360p, TensorDataType::FP32);

    Tensor copyTensor;
    copyTensor.ShareData(originTensor);
    ASSERT_EQ(copyTensor.Layout(), originTensor.Layout());
    ASSERT_EQ(copyTensor.GetSize(), originTensor.GetSize());
    ASSERT_EQ(copyTensor.DataType(), originTensor.DataType());
    ASSERT_EQ(copyTensor.RawDataPtr(), originTensor.RawDataPtr());
}

TEST_F(TestTensor, TestShareDataIsNullptr) // 源数据指针为空指针，share失败
{
    Tensor shareTensor;
    TensorShape tensorShape = {1, 3, 1080, 1920};
    EXPECT_EQ(shareTensor.ShareData<float>(nullptr, tensorShape), AccDataErrorCode::H_TENSOR_ERROR);
}

TEST_F(TestTensor, TestShareDataErrorType) // 源数据Datatype异常
{
    auto rawPtr = std::make_shared<float>(0.0f);
    Tensor shareTensor;
    AccDataErrorCode errCode = shareTensor.ShareData(rawPtr, {1}, TensorDataType::LAST);
    EXPECT_EQ(errCode, AccDataErrorCode::H_TENSOR_ERROR);
}

TEST_F(TestTensor, TestIsDataType)
{
    Tensor originTensor;
    originTensor.SetLayout(TensorLayout::NCHW);
    originTensor.Resize(mShapeNchw360p, TensorDataType::FP32);

    bool isFp32 = originTensor.IsDataType<float>();
    ASSERT_EQ(isFp32, true);
    bool isUint8 = originTensor.IsDataType<uint8_t>();
    ASSERT_EQ(isUint8, false);
    bool isChar = originTensor.IsDataType<char>();
    ASSERT_EQ(isChar, false);
    bool isDouble = originTensor.IsDataType<double>();
    ASSERT_EQ(isDouble, false);
}

TEST_F(TestTensor, TestShareDataErrorShape) // 源数据shape存在0
{
    auto rawPtr = std::make_shared<float>(0.0f);

    Tensor shareTensor;
    AccDataErrorCode errCode = shareTensor.ShareData(rawPtr, {1, 0, 0}, TensorDataType::FP32);
    EXPECT_EQ(errCode, AccDataErrorCode::H_TENSOR_ERROR);
    errCode = shareTensor.ShareData(rawPtr, {1, 0, 0}, TensorDataType::FP32);
    EXPECT_EQ(errCode, AccDataErrorCode::H_TENSOR_ERROR);
}

TEST_F(TestTensor, TestCopyWithErrorShape) // 源数据shape存在0
{
    Tensor originTensor;
    originTensor.SetLayout(TensorLayout::NCHW);
    originTensor.Resize(mShapeNchw360p, TensorDataType::FP32);
 
    Tensor copyTensor;
    AccDataErrorCode errCode = copyTensor.Copy(originTensor.RawDataPtr().get(),
        {1, 0, 0}, TensorDataType::FP32);
    EXPECT_EQ(errCode, AccDataErrorCode::H_TENSOR_ERROR);
}

}  // namespace
