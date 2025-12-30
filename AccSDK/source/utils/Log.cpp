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
 * Description: Log Cpp file.
 * Author: ACC SDK
 * Create: 2025
 * History: NA
 */

#include <iostream>
#include <mutex>
#include <iomanip>
#include <ctime>
#include <map>
#include <string>
#include <cstring>
#include <algorithm>
#include "acc/utils/LogImpl.h"
#include "logger.h"
namespace Acc {
namespace {
const size_t TIMESTAMP_WIDTH = 3;
const size_t MAX_LOG_LENGTH = 1024;
constexpr unsigned char ASCII_END = 0x7F;
std::mutex g_logMutex;

char* FilterInvalidChar(const char* msg)
{
    // 256-size static lookup table, O(1) time complexity
    static bool invalidCharsLookup[256] = {false};
    static bool isInitialized = false;
    if (!isInitialized) {
        std::lock_guard<std::mutex> lock(g_logMutex);
        if (!isInitialized) {
            const char* invalidChars[] = {"\n",     "\f",     "\r",     "\b",     "\t",     "\v",    "\u000D",
                                          "\u000A", "\u000C", "\u000B", "\u0009", "\u0008", "\u0007"};
            for (const char* ch : invalidChars) {
                invalidCharsLookup[static_cast<unsigned char>(ch[0])] = true;
            }
            isInitialized = true;
        }
    }

    size_t msgLen = strlen(msg);
    size_t maxLen = std::min(msgLen, MAX_LOG_LENGTH);
    char* filteredMsg = new(std::nothrow) char[maxLen + 1];
    if (filteredMsg == nullptr) {
        throw std::runtime_error("Malloc space for log message faild, please check system is functioning properly.");
    }

    // the current valid character position
    size_t curPos = 0;
    bool preCharIsInvalid = false;
    for (size_t i = 0; i < msgLen && curPos < maxLen; ++i) {
        unsigned char uc = static_cast<unsigned char>(msg[i]);
        if (uc <= ASCII_END && invalidCharsLookup[uc]) {
            // If consecutive invalid characters are used, replace them with a space
            if (!preCharIsInvalid) {
                filteredMsg[curPos++] = ' ';
                preCharIsInvalid = true;
            }
        } else {
            // Merge non-ASCII characters directly
            filteredMsg[curPos++] = msg[i];
            preCharIsInvalid = false;
        }
    }

    filteredMsg[curPos] = '\0';
    return filteredMsg;
}

// Default Log Processing Function
void LogFnx(LogLevel level, const char* message, const char* file, int line, const char* function)
{
    static const char* levelStr[] = {"DEBUG", "INFO", "WARN", "ERROR", "FATAL"};
    // Filter invalid characters.
    char* filteredMessage = FilterInvalidChar(message);

    // A lock must be added to prevent logs from being disordered due to multiple thread.
    std::lock_guard<std::mutex> lock(g_logMutex);
    auto now = std::chrono::system_clock::now();
    auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch());
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(microseconds);
    std::time_t time = seconds.count();
    std::tm tm = *std::gmtime(&time);
    auto millis = (microseconds.count() % 1'000'000) / 1000;
    auto micros = microseconds.count() % 1000;

    std::ostringstream oss;
    oss << "[" << levelStr[static_cast<int>(level)] << "] " << std::put_time(&tm, "%Y-%m-%d-%H:%M:%S") << "."
        << std::setfill('0') << std::setw(TIMESTAMP_WIDTH) << millis << "." << std::setfill('0')
        << std::setw(TIMESTAMP_WIDTH) << micros << " "
        << "[" << file << ":" << line << "] " << function << ": " << filteredMessage << "\n";
    std::cout << oss.str();

    delete[] filteredMessage;
}

// Global log processing function
LogFn g_LogFn = LogFnx;

// Global min log level
LogLevel g_MinLogLevel = LogLevel::INFO;

void Log(LogLevel level, const char* message, const char* file, int line, const char* function)
{
    if (message == nullptr || message[0] == '\0') {
        return;
    }
    if (level < LogLevel::DEBUG || level > LogLevel::FATAL) {
        return;
    }
    if (level >= g_MinLogLevel) {
        g_LogFn(level, message, file, line, function);
    }
}

void AccDataLogHook(acclib::accdata::LogLevel, const char* msg, const char* file, int line, const char* function)
{
    Log(Acc::LogLevel::DEBUG, msg, file, line, function);
}
} // namespace

LogMessage::LogMessage(LogLevel level, const char* file, int line, const char* function)
    : level_(level), file_(file), line_(line), function_(function)
{
}

LogMessage::~LogMessage()
{
    std::string msg = sstream_.str();
    Log(level_, msg.c_str(), file_, line_, function_);
}

void RegisterLogConf(LogLevel minLevel, LogFn fn)
{
    // Avoid deadlocks
    {
        std::lock_guard<std::mutex> lock(g_logMutex);
        if (fn) {
            g_LogFn = fn;
        } else {
            g_LogFn = LogFnx;
        }
        g_MinLogLevel = minLevel;
        acclib::accdata::Logger::SetLogFunction(AccDataLogHook);
    }
    if (!fn) {
        LogWarn << "An empty function is input. Use the default log function.";
    }
}
} // namespace Acc