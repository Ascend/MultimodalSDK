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
 * Description: LogTest Cpp file.
 * Author: ACC SDK
 * Create: 2025
 * History: NA
 */

#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <thread>
#include <mutex>
#include <gtest/gtest.h>
#include "acc/utils/LogImpl.h"
#define CustomDebug(msg) CustomLogFn(LogLevel::DEBUG, msg, __FILE__, __LINE__, __FUNCTION__)
#define CustomInfo(msg)  CustomLogFn(LogLevel::INFO, msg, __FILE__, __LINE__, __FUNCTION__)
#define CustomWarn(msg)  CustomLogFn(LogLevel::WARN, msg, __FILE__, __LINE__, __FUNCTION__)
#define CustomError(msg) CustomLogFn(LogLevel::ERROR, msg, __FILE__, __LINE__, __FUNCTION__)
#define CustomFatal(msg) CustomLogFn(LogLevel::FATAL, msg, __FILE__, __LINE__, __FUNCTION__)

const char* TEMP_FILE = "temp_log.txt";
const char* CUSTOM_FILE = "custom_log.txt";
using namespace Acc;
class LogTest : public testing::Test {
protected:
    void SetUp() override
    {
        // Create a temporary file, back up and redirect stdout
        FILE* original = freopen(TEMP_FILE, "w", stdout);
        // Set default log and min log level
        RegisterLogConf(LogLevel::INFO, nullptr);
    }

    void TearDown() override
    {
        remove(TEMP_FILE);
    }

    static std::string GetConsoleInfo()
    {
        // Restoring stdout (Linux/Mac)
        freopen("/dev/tty", "w", stdout);
        // read tmp file contents
        std::ifstream in(TEMP_FILE);
        std::string output((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        return output;
    }

    static std::string GetCustomConsoleInfo()
    {
        freopen("/dev/tty", "w", stdout);
        std::ifstream in(CUSTOM_FILE);
        std::string output((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        return output;
    }

    static void DefaultThreadTask(int id)
    {
        LogInfo << "Thread-" << id << ": This is a test message for default Log";
    }

    static void CustomLogFn(LogLevel level, const char* message, const char* file, int line, const char* function)
    {
        static std::mutex testMutex;
        std::lock_guard<std::mutex> lock(testMutex);
        static std::ofstream logFile(CUSTOM_FILE, std::ios::app);
        if (logFile.is_open()) {
            auto now = std::chrono::system_clock::now();
            auto inTime = std::chrono::system_clock::to_time_t(now);

            const char* levelStr = "";
            switch (level) {
                case LogLevel::DEBUG:
                    levelStr = "DEBUG";
                    break;
                case LogLevel::INFO:
                    levelStr = "INFO";
                    break;
                case LogLevel::WARN:
                    levelStr = "WARNING";
                    break;
                case LogLevel::ERROR:
                    levelStr = "ERROR";
                    break;
                case LogLevel::FATAL:
                    levelStr = "FATAL";
                    break;
            }
            logFile << "[Custom] " << std::put_time(std::localtime(&inTime), "%Y-%m-%d %X") << " "
                    << "[" << levelStr << "] "
                    << "[" << file << ":" << line << ":" << function << "] " << message << std::endl;
        }
    }
};

TEST_F(LogTest, Test_DefaultLog)
{
    LogInfo << "This is a test message for default Log";
    std::string output = LogTest::GetConsoleInfo();
    EXPECT_NE(output.find("This is a test message for default Log"), std::string::npos);
}

TEST_F(LogTest, Test_DefaultLog_With_WrongMinLevel)
{
    RegisterLogConf(LogLevel::WARN, nullptr);
    LogInfo << "This is a test message for default Log";
    std::string output = LogTest::GetConsoleInfo();
    EXPECT_EQ(output.find("This is a test message for default Log"), std::string::npos);
}

TEST_F(LogTest, Test_DefaultLog_On_MultiThreads)
{
    const int nThreads = 5;
    std::vector<std::thread> threads;
    for (int i = 0; i < nThreads; ++i) {
        threads.emplace_back(LogTest::DefaultThreadTask, i);
    }
    for (auto& t : threads) {
        t.join();
    }
    std::string output = LogTest::GetConsoleInfo();
    for (int i = 0; i < nThreads; ++i) {
        std::string target = "Thread-" + std::to_string(i) + ": This is a test message for default Log";
        EXPECT_NE(output.find(target), std::string::npos);
    }
}

TEST_F(LogTest, Test_DefaultLog_Without_Message)
{
    LogInfo << "";
    std::string output = LogTest::GetConsoleInfo();
    int count = std::count(output.begin(), output.end(), '\n');
    EXPECT_EQ(count, 1);
}

TEST_F(LogTest, Test_DefaultLog_Filter_InvalidChar)
{
    LogInfo << "This\nis\fa\rtest\b中\t\v文\u000Dmessage\u000Afor\u000C\u000Bdefault\u0009\u0008Log\u0007";
    std::string output = LogTest::GetConsoleInfo();
    EXPECT_NE(output.find("This is a test 中 文 message for default Log "), std::string::npos);
}

TEST_F(LogTest, Test_DefaultLog_Filter_LongLog_With_InvalidChar)
{
    const size_t msgLen = 1500;
    char msg[msgLen + 1];
    const char tmpMsg[] = "123456789\n";
    const size_t tmpLen = strlen(tmpMsg);
    for (size_t i = 0; i < msgLen; ++i) {
        msg[i] = tmpMsg[i % tmpLen];
    }
    msg[msgLen] = '\0';
    LogInfo << msg;
    std::string output = LogTest::GetConsoleInfo();
    const char tmpTarget[] = "123456789 ";
    const size_t tmpTargetLen = strlen(tmpTarget);
    const size_t maxLen = 1024;
    std::string target;
    target.reserve(maxLen);
    for (size_t i = 0; i < maxLen; ++i) {
        target[i] = tmpTarget[i % tmpTargetLen];
    }
    EXPECT_NE(output.find(target), std::string::npos);
}

TEST_F(LogTest, Test_CustomLog)
{
    RegisterLogConf(LogLevel::INFO, LogTest::CustomLogFn);
    LogError << "ERR msg.";
    LogWarn << "WARN msg.";
    LogInfo << "INFO msg.";
    LogDebug << "DEBUG msg.";
    std::string output = LogTest::GetCustomConsoleInfo();
    EXPECT_NE(output.find("[Custom]"), std::string::npos);
    EXPECT_NE(output.find("ERR msg"), std::string::npos);
    EXPECT_NE(output.find("WARN msg"), std::string::npos);
    EXPECT_NE(output.find("INFO msg"), std::string::npos);
    EXPECT_EQ(output.find("DEBUG msg"), std::string::npos);
}

int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    return ret;
}