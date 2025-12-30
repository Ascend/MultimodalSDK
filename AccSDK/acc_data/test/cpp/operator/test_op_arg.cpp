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
 * @Date: 2025-3-27 17:00:00
 * @LastEditors: dev
 * @LastEditTime: 2025-3-27 17:00:00
 */

#include <gtest/gtest.h>

#include "operator/op_arg.h"

namespace {
using namespace acclib::accdata;

class TestOpArg : public ::testing::Test {
public:
    void SetUp()
    {
        opArgVar = new OpArgVar<float>(name, value);
    }

    void TearDown()
    {
        OpArgVar<float>* deleter = opArgVar;
        delete deleter;
        opArgVar = nullptr;
    }

    std::string name = "mean";
    float value = 1.0;
    OpArgVar<float> *opArgVar = nullptr;
};

TEST_F(TestOpArg, GetValueSuccess)
{
    OpArg &opArg = *opArgVar;
    float value = 0.0;
    EXPECT_EQ(opArg.Value(value), AccDataErrorCode::H_OK);
}

TEST_F(TestOpArg, GetValueError)
{
    OpArg &opArg = *opArgVar;
    int64_t value = 0;
    EXPECT_EQ(opArg.Value(value), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);
}

TEST_F(TestOpArg, CheckIsType)
{
    OpArg &opArg = *opArgVar;
    EXPECT_TRUE(opArg.IsType<float>());
}

TEST_F(TestOpArg, CheckIsNotType)
{
    OpArg &opArg = *opArgVar;
    EXPECT_FALSE(opArg.IsType<int>());
}

}