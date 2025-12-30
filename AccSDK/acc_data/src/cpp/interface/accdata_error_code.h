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
 * @Date: 2025-3-22 9:00:00
 * @LastEditors: dev
 * @LastEditTime: 2025-3-22 9:00:00
 */

#ifndef ACCDATA_SRC_CPP_INTERFACE_ACCDATAERRORCODE_H_
#define ACCDATA_SRC_CPP_INTERFACE_ACCDATAERRORCODE_H_

namespace acclib {
namespace accdata {

using AccDataResult = int;

// 错误码
enum AccDataErrorCode : int {
    H_OK = 0,
    H_COMMON_ERROR = 1,
    H_COMMON_UNKNOWN_ERROR = 2,
    H_COMMON_LOGGER_ERROR = 3,
    H_COMMON_INVALID_PARAM = 4,
    H_COMMON_OPERATOR_ERROR = 5,
    H_COMMON_NULLPTR = 6,
    H_SINGLEOP_ERROR = 7,
    H_FUSIONOP_ERROR = 8,
    H_USEROP_ERROR = 9,
    H_PIPELINE_ERROR = 10,
    H_PIPELINE_BUILD_ERROR = 11,
    H_PIPELINE_STATE_ERROR = 12,
    H_TENSOR_ERROR = 13,
    H_THREADPOOL_ERROR = 14,
};

} // namespace accdata
} // namespace acclib

#endif  // ACCDATA_SRC_CPP_INTERFACE_ACCDATAERRORCODE_H_
