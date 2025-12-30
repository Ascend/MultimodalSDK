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
 * Description:
 * Create: 2025-03-24
 */

#ifndef ACCDATA_SRC_CPP_COMMON_OPERATOR_PARAM_H_
#define ACCDATA_SRC_CPP_COMMON_OPERATOR_PARAM_H_

namespace acclib {
namespace accdata {

struct OperatorParam {
    uint64_t height{ 0 };
    uint64_t width{ 0 };
    uint64_t channel{ 3 };
    /* Task range */
    uint64_t begin{ 0 };
    uint64_t end{ 0 };
    int64_t cropOffsetX{ 0 };
    int64_t cropOffsetY{ 0 };
};

} // namespace accdata
} // namespace acclib

#endif  // ACCDATA_SRC_CPP_COMMON_OPERATOR_PARAM_H_
