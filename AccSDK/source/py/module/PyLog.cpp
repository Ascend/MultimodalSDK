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
#include "PyLog.h"
#include <mutex>
namespace PyAcc {
namespace {
const LogCallBacker* g_logCb = nullptr;
std::mutex g_cbMutex;
void LogHook(Acc::LogLevel level, const char* message, const char* file, int line, const char* function)
{
    if (g_logCb) {
        g_logCb->log(level, message, file, line, function);
    }
}
} // namespace
void LogCallBacker::register_log_conf(Acc::LogLevel min_level, const LogCallBacker* callbacker)
{
    std::lock_guard<std::mutex> lock(g_cbMutex);
    g_logCb = callbacker;
    if (callbacker) {
        RegisterLogConf(min_level, &LogHook);
    } else {
        RegisterLogConf(min_level, nullptr);
    }
}
} // namespace PyAcc