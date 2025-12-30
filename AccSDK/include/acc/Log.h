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
 * Description: Log header file.
 * Author: ACC SDK
 * Create: 2025
 * History: NA
 */
#ifndef LOG_H
#define LOG_H
namespace Acc {
enum class LogLevel { DEBUG = 0, INFO = 1, WARN = 2, ERROR = 3, FATAL = 4 };

/**
 * @brief Log callback prototype
 * @param level Current log level
 * @param message Current log message
 * @param file Current log file
 * @param line Current log line
 * @param function Current log function
 */
using LogFn = void (*)(Acc::LogLevel level, const char* message, const char* file, int line, const char* function);

/**
 * @brief Register log config
 *
 * @param minLevel Minimum log level. Logs whose levels are higher than this value are output
 * @param fn Log callback function
 */
void RegisterLogConf(LogLevel minLevel, LogFn fn);
} // namespace Acc
#endif // LOG_H
