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

#ifndef ACCDATA_SRC_CPP_COMMON_ACCDATA_LOGGER_H_
#define ACCDATA_SRC_CPP_COMMON_ACCDATA_LOGGER_H_

#include <mutex>
#include <iostream>
#include <sstream>
#include <functional>
#include <unistd.h>
#include <sys/syscall.h>

#include <cstring>

#include "singleton.h"
#include "utility.h"
#include "logger.h"

namespace acclib {
namespace accdata {

class AccdataLogger : public Logger {
public:
    AccdataLogger() = default;

    static AccdataLogger *Instance()
    {
        return Singleton<AccdataLogger>::GetInstance();
    }

    void SetLogLevel(LogLevel level);

    void SetLogFunction(const ExternalLog logFunc);

    LogLevel GetLogLevel();

    std::mutex &GetMutex()
    {
        return gMutex;
    }

    void Log(LogLevel level, const char *msg, const char *file, int line, const char *function);

    AccdataLogger(const AccdataLogger &) = delete;

    AccdataLogger &operator = (const AccdataLogger &) = delete;

    AccdataLogger(AccdataLogger &&) = delete;

    AccdataLogger &operator = (const AccdataLogger &&) = delete;

private:
    std::mutex gMutex;
    LogLevel mLogLevel{ LogLevel::INFO };
    ExternalLog mLogFunc = nullptr;
};

#define FILE_NAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

#define ACCDATA_LOG(level, msg)                                                       \
    do {                                                                              \
        auto logger = AccdataLogger::Instance();                                      \
        if (logger != nullptr && (level) >= logger->GetLogLevel()) {                  \
            std::ostringstream oss;                                                   \
            oss << msg;                                                               \
            std::lock_guard<std::mutex> guard(logger->GetMutex());                    \
            logger->Log(level, oss.str().c_str(), FILE_NAME, __LINE__, __FUNCTION__); \
        }                                                                             \
    } while (0)

#define ACCDATA_ERROR(msg) ACCDATA_LOG(LogLevel::ERROR, msg)
#define ACCDATA_WARN(msg) ACCDATA_LOG(LogLevel::WARN, msg)
#define ACCDATA_INFO(msg) ACCDATA_LOG(LogLevel::INFO, msg)
#define ACCDATA_DEBUG(msg) ACCDATA_LOG(LogLevel::DEBUG, msg)
} // namespace accdata
} // namespace acclib

#endif // ACCDATA_SRC_CPP_COMMON_ACCDATA_LOGGER_H_
