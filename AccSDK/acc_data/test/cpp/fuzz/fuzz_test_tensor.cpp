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
 * Description:
 * Author: DEV
 * Create: 2025-04-08
 */

#include <gtest/gtest.h>
#include "secodeFuzz.h"

#include "accdata_tensor.h"

#include "random_utils.h"

namespace acclib {
namespace accdata {
class FuzzTestTensor : public testing::Test {
public:
    void SetUp()
    {
        Logger::SetLogLevelStr("error");
        DT_Set_Running_Time_Second(EXEC_SECOND);
        DT_Enable_Leak_Check(0, 0);
    }

    std::vector<TensorDataType> typeOptions = { TensorDataType::UINT8, TensorDataType::FP32, TensorDataType::CHAR,
        TensorDataType::LAST };
    std::vector<TensorLayout> layoutOptions = { TensorLayout::NCHW, TensorLayout::NHWC, TensorLayout::PLAIN,
        TensorLayout::LAST };
    uint64_t dataSize = 1ULL * 3ULL * 10ULL * 10ULL;
};

TEST_F(FuzzTestTensor, CopyDump)
{
    std::string caseName = "AccDataTensor::Copy";
    DT_FUZZ_START(0, TEST_COUNT, caseName.data(), 0)
    {
        auto tensorList = AccDataTensorList::Create(1);
        TensorShape shape;
        shape.emplace_back(GenerateOneData<int64_t>(-1, 2));
        shape.emplace_back(GenerateOneData<int64_t>(-1, 2));
        shape.emplace_back(GenerateOneData<int64_t>(-1, 2));
        shape.emplace_back(GenerateOneData<int64_t>(-1, 2));
        auto type = RandomSelectOne(typeOptions);
        std::cout << type << std::endl;
        std::vector<uint64_t> src;
        int64_t dataCount = 1;
        for (uint32_t i = 0; i < shape.size(); ++i) {
            dataCount *= shape[i];
        }
        if (dataCount > 0) {
            src.resize(dataCount);
            (*tensorList)[0].Copy(src.data(), shape, type);
        } else {
            (*tensorList)[0].Copy(nullptr, shape, type);
        }
    }
    DT_FUZZ_END()
}

TEST_F(FuzzTestTensor, ShareData)
{
    auto dataPtr = std::make_shared<std::vector<uint8_t>>();
    dataPtr->resize(dataSize);
    std::string caseName = "AccDataTensor::ShareData";
    DT_FUZZ_START(0, TEST_COUNT, caseName.data(), 0)
    {
        auto tensorList = AccDataTensorList::Create(1);
        TensorShape shape;
        shape.emplace_back(GenerateOneData<int64_t>(-1, 2));
        shape.emplace_back(GenerateOneData<int64_t>(-1, 4));
        shape.emplace_back(GenerateOneData<int64_t>(-1, 15));
        shape.emplace_back(GenerateOneData<int64_t>(-1, 15));
        auto type = RandomSelectOne(typeOptions);
        std::cout << type << std::endl;
        float flag = GenerateOneData<float>(-1, 9);
        if (flag > 0) {
            (*tensorList)[0].ShareData(dataPtr, shape, type);
        } else {
            (*tensorList)[0].ShareData(nullptr, shape, type);
        }
    }
    DT_FUZZ_END()
}

TEST_F(FuzzTestTensor, RawDataPtr)
{
    std::string caseName = "AccDataTensor::RawDataPtr";
    DT_FUZZ_START(0, TEST_COUNT, caseName.data(), 0)
    {
        auto tensorList = AccDataTensorList::Create(1);
        (*tensorList)[0].RawDataPtr();
    }
    DT_FUZZ_END()
}

TEST_F(FuzzTestTensor, SetAndGetLayout)
{
    std::string caseName = "AccDataTensor::SetLayout&&Layout";
    DT_FUZZ_START(0, TEST_COUNT, caseName.data(), 0)
    {
        auto tensorList = AccDataTensorList::Create(1);
        auto layout = RandomSelectOne(layoutOptions);
        (*tensorList)[0].SetLayout(layout);
        std::cout << (*tensorList)[0].Layout() << std::endl;
    }
    DT_FUZZ_END()
}

TEST_F(FuzzTestTensor, DataType)
{
    std::string caseName = "AccDataTensor::DataType";
    DT_FUZZ_START(0, TEST_COUNT, caseName.data(), 0)
    {
        auto tensorList = AccDataTensorList::Create(1);
        (*tensorList)[0].DataType();
    }
    DT_FUZZ_END()
}

TEST_F(FuzzTestTensor, Shape)
{
    std::string caseName = "AccDataTensor::Shape";
    DT_FUZZ_START(0, TEST_COUNT, caseName.data(), 0)
    {
        auto tensorList = AccDataTensorList::Create(1);
        (*tensorList)[0].Shape();
    }
    DT_FUZZ_END()
}

class FuzzTestTensorList : public testing::Test {
public:
    void SetUp()
    {
        Logger::SetLogLevelStr("error");
        DT_Set_Running_Time_Second(EXEC_SECOND);
        DT_Enable_Leak_Check(0, 0);
    }
};

TEST_F(FuzzTestTensorList, NumTensors)
{
    int numTensors = 0;
    std::string caseName = "AccDataTensorList::Create&&NumTensors";
    DT_FUZZ_START(0, TEST_COUNT, caseName.data(), 0)
    {
        ++numTensors;
        auto tl = AccDataTensorList::Create(numTensors);
        tl->NumTensors();
    }
    DT_FUZZ_END()
}

TEST_F(FuzzTestTensorList, GetTensorByIdx)
{
    int tensorIdx = 0;
    std::string caseName = "AccDataTensorList::operator[]";
    DT_FUZZ_START(0, TEST_COUNT, caseName.data(), 0)
    {
        ++tensorIdx;
        auto tl = AccDataTensorList::Create(TEST_COUNT);
        tl->operator[](tensorIdx);
    }
    DT_FUZZ_END()
}
}
}