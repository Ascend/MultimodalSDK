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
 * Author: Dev
 * Create: 2025-04-07
 */

#include <gtest/gtest.h>
#include "secodeFuzz.h"

#include "interface/accdata_op_spec.h"

#include "random_utils.h"

namespace acclib {
namespace accdata {
class FuzzTestOpSpec : public testing::Test {
public:
    void SetUp()
    {
        Logger::SetLogLevelStr("error");
        DT_Set_Running_Time_Second(EXEC_SECOND);
        DT_Enable_Leak_Check(0, 0);
    }
};

TEST_F(FuzzTestOpSpec, CreateOpSpec)
{
    std::string caseName = "AccDataOpSpec::Create";
    DT_FUZZ_START(0, TEST_COUNT, caseName.data(), 0)
    {
        auto opSpec = AccDataOpSpec::Create("testSpec");
    }
    DT_FUZZ_END()
}

TEST_F(FuzzTestOpSpec, AddInput)
{
    std::string caseName = "AccDataOpSpec::AddInput";
    DT_FUZZ_START(0, TEST_COUNT, caseName.data(), 0)
    {
        auto opSpec = AccDataOpSpec::Create("testSpec");
        opSpec->AddInput("testInput", "cpu");
    }
    DT_FUZZ_END()
}

TEST_F(FuzzTestOpSpec, AddOutput)
{
    std::string caseName = "AccDataOpSpec::AddOutput";
    DT_FUZZ_START(0, TEST_COUNT, caseName.data(), 0)
    {
        auto opSpec = AccDataOpSpec::Create("testSpec");
        opSpec->AddOutput("testOutput", "cpu");
    }
    DT_FUZZ_END()
}

TEST_F(FuzzTestOpSpec, AddArg)
{
    std::string caseName = "AccDataOpSpec::AddArg";
    DT_FUZZ_START(0, TEST_COUNT, caseName.data(), 0)
    {
        bool overwrite = GenerateOneData<bool>(0, 1);
        auto opSpec = AccDataOpSpec::Create("testSpec");
        std::vector<std::string> argNameOptions = {"argA", "argB", "argC", "argD", "argE", "argF", "argG", "argH"};
        opSpec->AddArg<int64_t>(RandomSelectOne(argNameOptions), 1LL, overwrite);
        opSpec->AddArg<std::string>(RandomSelectOne(argNameOptions), "node", overwrite);
        opSpec->AddArg<bool>(RandomSelectOne(argNameOptions), overwrite, overwrite);
        opSpec->AddArg<float>(RandomSelectOne(argNameOptions), 1.0F, overwrite);
        opSpec->AddArg<std::vector<int64_t>>(RandomSelectOne(argNameOptions), {1LL}, overwrite);
        opSpec->AddArg<std::vector<std::string>>(RandomSelectOne(argNameOptions), {"node"}, overwrite);
        opSpec->AddArg<std::vector<bool>>(RandomSelectOne(argNameOptions), {overwrite}, overwrite);
        opSpec->AddArg<std::vector<float>>(RandomSelectOne(argNameOptions), {1.0F}, overwrite);
    }
    DT_FUZZ_END()
}
}
}  // namespace
