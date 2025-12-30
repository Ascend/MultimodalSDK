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
 * @Date: 2025-3-28 15:00:00
 * @LastEditors: dev
 * @LastEditTime: 2025-3-28 15:00:00
 */

#include <gtest/gtest.h>

#include "operator/op_factory.h"
#include "operator/op_spec.h"

namespace {
using namespace acclib::accdata;

class TestOpFactory : public ::testing::Test {
public:
    void SetUp()
    {
        opSpec = new OpSpec(name_1);
    }

    void TearDown()
    {
        OpSpec* opSpecDeleter = opSpec;
        delete opSpecDeleter;
        opSpec = nullptr;
    }

    std::string name_1 = "Normalize";
    std::string name_2 = "Resize";
    OpSpec *opSpec = nullptr;
    std::unique_ptr<Operator> result;
};

TEST_F(TestOpFactory, CreateSuccess)
{
    OpFactory &opFactory = OpFactory::Instance();
    EXPECT_EQ(opFactory.Create(name_1, *opSpec, result), AccDataErrorCode::H_OK);
}

TEST_F(TestOpFactory, CreateFailed)
{
    OpFactory &opFactory = OpFactory::Instance();
    EXPECT_EQ(opFactory.Create(name_2, *opSpec, result), AccDataErrorCode::H_COMMON_OPERATOR_ERROR);
}

}