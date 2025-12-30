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
 * @Date: 2025-7-9 9:00:00
 * @LastEditors: dev
 * @LastEditTime: 2025-7-9 9:00:00
 */

#include <gtest/gtest.h>
#include <chrono>
#include <sstream>
#include <iomanip>
#include "logger.h"
#include "accdata_pipeline.h"
#include "common/accdata_logger.h"

namespace {
using namespace acclib::accdata;
const std::string ANSI_COLOR_RESET = "\x1b[0m";
const std::string ANSI_COLOR_YELLOW = "\x1b[33m";
const std::string ANSI_COLOR_RED = "\x1b[31m";
const std::string ANSI_COLOR_MAGENTA = "\x1b[35m";
void TestLogFunction(LogLevel level, const char *msg, const char *file, int line, const char *function)
{
    std::string color = "";
    std::string levelStr;
    switch (level) {
        case LogLevel::DEBUG:
            levelStr = "debug";
            break;
        case LogLevel::INFO:
            levelStr = "info";
            break;
        case LogLevel::WARN:
            color = ANSI_COLOR_YELLOW;
            levelStr = "warn";
            break;
        case LogLevel::ERROR:
            color = ANSI_COLOR_RED;
            levelStr = "error";
            break;
        case LogLevel::FATAL:
            color = ANSI_COLOR_MAGENTA;
            levelStr = "fatal";
            break;
        default:
            levelStr = "INVALID LOG LEVEL";
            break;
    }

    std::cout << color << "[TestLogFunction]"
              << "[" << levelStr << "][" << file << "][" << function << "][" << line << "]" << msg <<
        ANSI_COLOR_RESET << std::endl;
}

class TestAccDataLogger : public ::testing::Test {
public:
    void SetUp()
    {
        buffer.str(std::string()); // clears the buffer.
        sbuf = std::cout.rdbuf();
        std::cout.rdbuf(buffer.rdbuf());
    }
    void TearDown()
    {
        std::cout.rdbuf(sbuf);
        std::cout << buffer.str() << std::endl;
    }

    std::stringstream buffer;
    std::streambuf *sbuf;
    ExternalLog logFunc = nullptr;
};

TEST_F(TestAccDataLogger, UseDefaultLogFunction)
{
    auto pipe = AccDataPipeline::Create();
    EXPECT_NE(pipe, nullptr);
    EXPECT_NE(pipe->Build({}, {}), AccDataErrorCode::H_OK);
    auto logger_string = buffer.str();
    EXPECT_NE(logger_string.find("Failed to build graph."), std::string::npos);
    ACCDATA_DEBUG("TestDefaultLog");
    ACCDATA_INFO("TestDefaultLog");
    ACCDATA_WARN("TestDefaultLog");
    ACCDATA_ERROR("TestDefaultLog");
}

TEST_F(TestAccDataLogger, LogWhileMLogFunctionIsNullptr)
{
    auto logger = AccdataLogger::Instance();
    logger->Log(LogLevel::DEBUG, "test log", "test_accdata_logger.cpp", 92, "log");
    logger->Log(LogLevel::INFO, "test log", "test_accdata_logger.cpp", 93, "log");
    logger->Log(LogLevel::WARN, "test log", "test_accdata_logger.cpp", 94, "log");
    logger->Log(LogLevel::ERROR, "test log", "test_accdata_logger.cpp", 95, "log");
    logger->Log(LogLevel::FATAL, "test log", "test_accdata_logger.cpp", 96, "log");
    auto logger_string = buffer.str();
    EXPECT_NE(logger_string.find("DEBUG"), std::string::npos);
    EXPECT_NE(logger_string.find("INFO"), std::string::npos);
    EXPECT_NE(logger_string.find("WARN"), std::string::npos);
    EXPECT_NE(logger_string.find("ERROR"), std::string::npos);
    EXPECT_NE(logger_string.find("FATAL"), std::string::npos);
}

TEST_F(TestAccDataLogger, SetLogFunctionNullptr)
{
    auto errorCode = Logger::SetLogFunction(logFunc);
    EXPECT_EQ(errorCode, -1);
}

TEST_F(TestAccDataLogger, SetLogFunctionSuccess)
{
    ExternalLog logFunc = TestLogFunction;
    auto errorCode = Logger::SetLogFunction(logFunc);
    EXPECT_EQ(errorCode, 0);
    auto pipe = AccDataPipeline::Create(1, 1, 2, false);
    EXPECT_NE(pipe, nullptr);
    errorCode = pipe->Build({}, {});
    EXPECT_NE(errorCode, 0);
    auto logger_string = buffer.str();
    EXPECT_NE(logger_string.find("TestLogFunction"), std::string::npos);
    TestLogFunction(LogLevel::DEBUG, "Test log function for debug log.", "test_accdata_logger.cpp", 112,
        "TestLogFunction");
    TestLogFunction(LogLevel::INFO, "Test log function for info log.", "test_accdata_logger.cpp", 114,
        "TestLogFunction");
    TestLogFunction(LogLevel::WARN, "Test log function for warn log.", "test_accdata_logger.cpp", 116,
        "TestLogFunction");
    TestLogFunction(LogLevel::ERROR, "Test log function for error log.", "test_accdata_logger.cpp", 118,
        "TestLogFunction");
    TestLogFunction(LogLevel::FATAL, "Test log function for fatal log.", "test_accdata_logger.cpp", 120,
        "TestLogFunction");
}
}
