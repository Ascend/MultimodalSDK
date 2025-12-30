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
 * Description: Error code utils.
 * Author: ACC SDK
 * Create: 2025
 * History: NA
 */

#ifndef ERROR_CODE_UTILS_H
#define ERROR_CODE_UTILS_H
#include <string>
#include "acc/ErrorCode.h"
namespace Acc {
std::string GetErrorInfo(ErrorCode err, std::string callingFuncName = "");
}
#endif // ERROR_CODE_UTILS_H