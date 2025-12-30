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
 * Description: Internal Log header file.
 * Author: ACC SDK
 * Create: 2025
 * History: NA
 */

#ifndef LOGIMPL_H
#define LOGIMPL_H

#include <sstream>
#include "acc/Log.h"
namespace Acc {
class LogMessage {
public:
    /**
     * @brief Construct a new Log Message object
     *
     * @param level log level, range is DEBUG, INFO, WARN, ERROR, FATAL
     * @param file the file currently running
     * @param line  the line currently running
     * @param function the function currently running
     */
    LogMessage(LogLevel level, const char* file, int line, const char* function);
    /**
     * @brief Destroy the Log Message object
     *
     */
    ~LogMessage();
    /**
     * @brief stream operator
     *
     * @tparam T
     * @param message log message
     * @return std::ostringstream&
     */
    template<typename T>
    std::ostringstream& operator<<(const T& message)
    {
        sstream_ << message;
        return sstream_;
    }

private:
    // string stream for receiving log information
    std::ostringstream sstream_{};
    LogLevel level_ = LogLevel::DEBUG;
    const char* file_ = nullptr;
    int line_ = -1;
    const char* function_ = nullptr;
};

// log macro
#define LogDebug LogMessage(Acc::LogLevel::DEBUG, __FILE__, __LINE__, __FUNCTION__)
#define LogInfo  LogMessage(Acc::LogLevel::INFO, __FILE__, __LINE__, __FUNCTION__)
#define LogWarn  LogMessage(Acc::LogLevel::WARN, __FILE__, __LINE__, __FUNCTION__)
#define LogError LogMessage(Acc::LogLevel::ERROR, __FILE__, __LINE__, __FUNCTION__)
#define LogFatal LogMessage(Acc::LogLevel::FATAL, __FILE__, __LINE__, __FUNCTION__)
} // namespace Acc
#endif // LOGIMPL_H