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
 * Description: log file for python.
 * Author: ACC SDK
 * Create: 2025
 * History: NA
 */
#ifndef PYLOG_H
#define PYLOG_H
#include "acc/Log.h"
namespace PyAcc {
class LogCallBacker {
public:
    /**
     * @brief Construct a new Log Call Backer object
     *
     */
    LogCallBacker() = default;
    /**
     * @brief Destroy the Log Call Backer object
     *
     */
    virtual ~LogCallBacker() = default;
    /**
     * @brief Pure virtual interface, Log callback
     *
     * @param level Current log level
     * @param message Current log message
     * @param file Current log file
     * @param line Current log line
     * @param function Current log function
     */
    virtual void log(Acc::LogLevel level, const char* message, const char* file, int line,
                     const char* function) const = 0;
    /**
     * @brief Register log config
     *
     * @param min_level Minimum log level. Logs whose levels are higher than this value are output.
     * @param callbacker Log callback function
     */
    static void register_log_conf(Acc::LogLevel min_level, const LogCallBacker* callbacker);
};
} // namespace PyAcc
#endif // PYLOG_H