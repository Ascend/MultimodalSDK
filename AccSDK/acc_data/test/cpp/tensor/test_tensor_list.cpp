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

#include "tensor/tensor_list.h"

namespace {

using namespace acclib::accdata;

class TestTensorList : public ::testing::Test {
public:
    void SetUp() {}
    void TearDown() {}
    uint64_t mBatchSize{10};
};

TEST_F(TestTensorList, TestTensorListBasic)
{
    TensorList tl = TensorList();
    tl = std::move(tl);
    tl = TensorList(mBatchSize);
    ASSERT_EQ(tl.NumTensors(), mBatchSize);

    tl.SetLayout(TensorLayout::NCHW);
    for (int i = 0; i < mBatchSize; i++) {
        ASSERT_EQ(tl[i].Layout(), TensorLayout::NCHW);
    }
}

TEST_F(TestTensorList, TestSetLayoutFailed) // TensorList中Tensor数量与TensorLayoutView的Size不相等，设置layout失败
{
    TensorList tl(mBatchSize);
    TensorLayoutView tensorLayoutView(1, TensorLayout::NCHW);
    EXPECT_EQ(tl.SetLayout(tensorLayoutView), AccDataErrorCode::H_TENSOR_ERROR);
}

TEST_F(TestTensorList, TestTensorListCopy)
{
    TensorList tl(mBatchSize);

    tl.SetLayout(TensorLayout::NCHW);
    TensorListShape tlShape(2, {2, 2});  // 2个2*2的Tensor
    tl.Resize<float>(tlShape);

    TensorList copyTl;
    copyTl.Copy(tl);

    ASSERT_EQ(tl.NumTensors(), copyTl.NumTensors());
    for (int i = 0; i < tl.NumTensors(); i++) {
        ASSERT_EQ(tl[i].Layout(), copyTl[i].Layout());
        ASSERT_EQ(tl[i].DataType(), copyTl[i].DataType());
        ASSERT_EQ(tl[i].GetSize(), copyTl[i].GetSize());
    }
}

TEST_F(TestTensorList, TestTensorListShare)
{
    TensorList tl(mBatchSize);

    tl.SetLayout(TensorLayout::NCHW);
    TensorListShape tlShape(2, {2, 2});  // 2个2*2的Tensor
    tl.Resize<float>(tlShape);

    TensorList shareTl;
    shareTl.ShareData(tl);

    ASSERT_EQ(tl.NumTensors(), shareTl.NumTensors());
    for (int i = 0; i < tl.NumTensors(); i++) {
        ASSERT_EQ(tl[i].Layout(), shareTl[i].Layout());
        ASSERT_EQ(tl[i].DataType(), shareTl[i].DataType());
        ASSERT_EQ(tl[i].RawDataPtr(), shareTl[i].RawDataPtr());
        ASSERT_EQ(tl[i].GetSize(), shareTl[i].GetSize());
    }
}

TEST_F(TestTensorList, TestResizeFailed) // TensorListShape的Size与TensorDataTypeView的Size不相等，Resize失败
{
    TensorList tl(mBatchSize);
    TensorDataTypeView tensorDataTypeView(1, TensorDataType::FP32);
    TensorListShape tlShape(2, {2, 2});  // 2个2*2的Tensor
    EXPECT_EQ(tl.Resize(tlShape, tensorDataTypeView), AccDataErrorCode::H_TENSOR_ERROR);
}

}  // namespace
