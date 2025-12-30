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
 * Description: Errorcode utils test file.
 * Author: ACC SDK
 * Create: 2025
 * History: NA
 */
#include "acc/utils/ErrorCodeUtils.h"
#include <gtest/gtest.h>
#include <iostream>
using namespace Acc;
class ErrorCodeUtilsTest : public testing::Test {
};
TEST_F(ErrorCodeUtilsTest, Test_GetErrorInfo_Success_With_Data_Check_Code_Info)
{
    ModuleID pId = ModuleID::ACC;
    ErrorType errorType = ErrorType::DATA_CHECK_ERROR;
    const std::string tmp = "Undefined error code";
    for (uint32_t i = static_cast<uint32_t>(SubErrorCode::DATA_CHECK_ERROR_BEGIN) + 1;
         i < static_cast<uint32_t>(SubErrorCode::DATA_CHECK_ERROR_END); i++) {
        ErrorCode code = MakeErrorCode(pId, errorType, static_cast<SubErrorCode>(i));
        std::string errorInfo = GetErrorInfo(code);
        EXPECT_EQ(errorInfo.find(tmp), std::string::npos);
    }
}

TEST_F(ErrorCodeUtilsTest, Test_GetErrorInfo_Success_With_Resource_Code_Info)
{
    ModuleID pId = ModuleID::ACC;
    ErrorType errorType = ErrorType::RESOURCE_ERROR;
    const std::string tmp = "Undefined error code";
    for (uint32_t i = static_cast<uint32_t>(SubErrorCode::RESOURCE_ERROR_BEGIN) + 1;
         i < static_cast<uint32_t>(SubErrorCode::RESOURCE_ERROR_END); i++) {
        ErrorCode code = MakeErrorCode(pId, errorType, static_cast<SubErrorCode>(i));
        std::string errorInfo = GetErrorInfo(code);
        EXPECT_EQ(errorInfo.find(tmp), std::string::npos);
    }
}

TEST_F(ErrorCodeUtilsTest, Test_GetErrorInfo_Success_With_OPENSOURCE_Code_Info)
{
    ModuleID pId = ModuleID::ACC;
    ErrorType errorType = ErrorType::OPENSOURCE_ERROR;
    const std::string tmp = "Undefined error code";
    for (uint32_t i = static_cast<uint32_t>(SubErrorCode::OPENSOURCE_ERROR_BEGIN) + 1;
         i < static_cast<uint32_t>(SubErrorCode::OPENSOURCE_ERROR_END); i++) {
        ErrorCode code = MakeErrorCode(pId, errorType, static_cast<SubErrorCode>(i));
        std::string errorInfo = GetErrorInfo(code);
        EXPECT_EQ(errorInfo.find(tmp), std::string::npos);
    }
}

TEST_F(ErrorCodeUtilsTest, Test_GetErrorInfo_Success_With_THIRD_PARTY_Code_Info)
{
    ModuleID pId = ModuleID::ACC;
    ErrorType errorType = ErrorType::THIRD_PARTY_ERROR;
    const std::string tmp = "Undefined error code";
    for (uint32_t i = static_cast<uint32_t>(SubErrorCode::THIRD_PARTY_ERROR_BEGIN) + 1;
         i < static_cast<uint32_t>(SubErrorCode::THIRD_PARTY_ERROR_END); i++) {
        ErrorCode code = MakeErrorCode(pId, errorType, static_cast<SubErrorCode>(i));
        std::string errorInfo = GetErrorInfo(code);
        EXPECT_EQ(errorInfo.find(tmp), std::string::npos);
    }
}

TEST_F(ErrorCodeUtilsTest, Test_GetErrorInfo_Success_With_BUSINESS_Code_Info)
{
    ModuleID pId = ModuleID::ACC;
    ErrorType errorType = ErrorType::RUNTIME_ERROR;
    const std::string tmp = "Undefined error code";
    for (uint32_t i = static_cast<uint32_t>(SubErrorCode::RUNTIME_ERROR_BEGIN) + 1;
         i < static_cast<uint32_t>(SubErrorCode::RUNTIME_ERROR_END); i++) {
        ErrorCode code = MakeErrorCode(pId, errorType, static_cast<SubErrorCode>(i));
        std::string errorInfo = GetErrorInfo(code);
        EXPECT_EQ(errorInfo.find(tmp), std::string::npos);
    }
}

int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    return ret;
}