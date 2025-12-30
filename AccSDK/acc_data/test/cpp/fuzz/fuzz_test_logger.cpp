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
 * Create: 2025-04-08
 */
#include <gtest/gtest.h>

#include "secodeFuzz.h"

#include "logger.h"
#include "accdata_pipeline.h"

#include "random_utils.h"

namespace acclib {
namespace accdata {

void TestLogFunction(LogLevel level, const char *msg, const char *file, int line, const char *function)
{
    std::string levelStr;
    switch (level) {
        case LogLevel::DEBUG:
            levelStr = "debug";
            break;
        case LogLevel::INFO:
            levelStr = "info";
            break;
        case LogLevel::WARN:
            levelStr = "warn";
            break;
        case LogLevel::ERROR:
            levelStr = "error";
            break;
        case LogLevel::FATAL:
            levelStr = "fatal";
            break;
        default:
            levelStr = "INVALID LOG LEVEL";
            break;
    }

    std::cout << "[" << levelStr << "][" << file << "][" << line << "][" << function << "]" << msg << std::endl;
}

class FuzzTestLogger : public testing::Test {
public:
    void SetUp()
    {
        DT_Set_Running_Time_Second(EXEC_SECOND);
        DT_Enable_Leak_Check(0, 0);
    }
};

TEST_F(FuzzTestLogger, SetLogLevelStr)
{
    std::string caseName = "Logger::SetLogLevelStr";
    DT_FUZZ_START(0, TEST_COUNT, caseName.data(), 0)
    {
        std::vector<std::string> levelOptions = { "debug", "warn", "info", "error", "wrong" };
        std::string logLevel = RandomSelectOne(levelOptions);
        Logger::SetLogLevelStr(logLevel);
    }
    DT_FUZZ_END()
}

TEST_F(FuzzTestLogger, PrintDefaultLog)
{
    std::string caseName = "Logger(print_default_log)";
    DT_FUZZ_START(0, TEST_COUNT, caseName.data(), 0)
    {
        Logger::SetLogLevelStr("info");
        (void)AccDataPipeline::Create();
    }
    DT_FUZZ_END()
}

TEST_F(FuzzTestLogger, SetLogFunction)
{
    std::string caseName = "Logger::SetLogFunction";
    DT_FUZZ_START(0, TEST_COUNT, caseName.data(), 0)
    {
        float flag = GenerateOneData<float>(-1, 1);
        if (flag > 0) {
            Logger::SetLogFunction(TestLogFunction);
        } else {
            Logger::SetLogFunction(nullptr);
        }
    }
    DT_FUZZ_END()
}
}
}