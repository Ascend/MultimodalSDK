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
 * @Date: 2025-3-22 17:00:00
 * @LastEditors: dev
 * @LastEditTime: 2025-3-22 17:00:00
 */

#ifndef ACCDATA_SRC_CPP_INTERFACE_LOGGER_H_
#define ACCDATA_SRC_CPP_INTERFACE_LOGGER_H_

namespace acclib {
namespace accdata {

enum class LogLevel {
    DEBUG = 0,
    INFO = 1,
    WARN = 2,
    ERROR = 3,
    FATAL = 4,
};

using ExternalLog = void (*)(LogLevel level, const char *msg, const char *file, int line, const char *function);

/**
 * @class Logger
 * @brief AccData内部日志系统。
 *
 * Logger为AccData内部日志系统，提供设置日志级别接口。
 */
class Logger {
public:
    /**
     * @brief 根据传入的字符串设置日志级别。
     *
     * @note 该函数根据提供的字符串设置内部日志系统的日志级别。
     * @note 字符串应表示有效的日志级别，如 "debug"、"info"、"warn"、"error" 等。
     * @note 默认日志级别为"info"。
     *
     * @param level 需要设置的日志级别的字符串表示（非大小写敏感）。
     *        该参数应为有效的日志级别字符串（例如 "debug"、"info"、"warn"、"error"）。
     *        如果提供了无效的级别，会打印WARN日志，"Invalid log level, which should be debug, info, warn or error"
     */
    static void SetLogLevelStr(const std::string &level);

    /**
     * @brief 设置日志打印函数。
     *
     * @note 日志打印函数由调用者自定义，根据ExternalLog函数定义，日志打印函数仅接受两个参数，日志级别level和打印信息msg。
     *
     * @param logFunc 需要设置的日志打印函数指针，该指针不能为空，日志打印函数的安全性由用户保证。
     */
    static int SetLogFunction(const ExternalLog logFunc);
};

}
}

#endif  // ACCDATA_SRC_CPP_INTERFACE_LOGGER_H_
