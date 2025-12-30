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
 * @Date: 2025-1-22 14:00:00
 * @LastEditors: dev
 * @LastEditTime: 2025-1-22 14:00:00
 */

#include "accdata_logger.h"

namespace acclib {
namespace accdata {

void DefaultLogFunction(LogLevel level, const char *msg, const char *file, int line, const char *function)
{
    std::string levelStr;
    std::unordered_map<LogLevel, std::string> levelMap = {
        {LogLevel::DEBUG, "DEBUG"},
        {LogLevel::INFO, "INFO"},
        {LogLevel::WARN, "WARN"},
        {LogLevel::ERROR, "ERROR"},
        {LogLevel::FATAL, "FATAL"}
    };

    if (levelMap.find(level) != levelMap.end()) {
        levelStr = levelMap[level];
    } else {
        levelStr = "INVALID LOG LEVEL";
    }

    auto nowTm = GetCurrentTime();
    std::cout << "[" << (levelStr) << "][" << getpid() << "][" << syscall(SYS_gettid) << "][" <<
        (nowTm.tm_year + 1900U) << "-" << (nowTm.tm_mon + 1U) << "-" << nowTm.tm_mday << " " << nowTm.tm_hour << ":" <<
        nowTm.tm_min << ":" << nowTm.tm_sec << "." << nowTm.microseconds << "][" << file << "(" << function << "):" <<
        line << "] " << msg << std::endl;
}

void AccdataLogger::SetLogLevel(LogLevel level)
{
    mLogLevel = level;
}

void AccdataLogger::SetLogFunction(const ExternalLog logFunc)
{
    mLogFunc = logFunc;
}

LogLevel AccdataLogger::GetLogLevel()
{
    return mLogLevel;
}

void AccdataLogger::Log(LogLevel level, const char *msg, const char *file, int line, const char *function)
{
    if (mLogFunc != nullptr) {
        mLogFunc(level, msg, file, line, function);
    } else {
        DefaultLogFunction(level, msg, file, line, function);
    }
}

void Logger::SetLogLevelStr(const std::string &level)
{
    static constexpr std::pair<const char*, LogLevel> logLevels[] = {
        {"debug", LogLevel::DEBUG},
        {"info", LogLevel::INFO},
        {"warn", LogLevel::WARN},
        {"error", LogLevel::ERROR},
    };

    for (auto logLevel : logLevels) {
        if (!strcasecmp(level.c_str(), logLevel.first)) {
            AccdataLogger::Instance()->SetLogLevel(logLevel.second);
            return;
        }
    }
    std::cerr << "[WARN] Invalid log level, which should be debug, info, warn or error" << std::endl;
    return;
}

int Logger::SetLogFunction(const ExternalLog logFunc)
{
    if (logFunc != nullptr) {
        AccdataLogger::Instance()->SetLogFunction(logFunc);
        return 0;
    }
    std::cerr << "[ERROR] Log function cannot be nullptr" << std::endl;
    return -1;
}

} // namespace accdata
} // namespace acclib
