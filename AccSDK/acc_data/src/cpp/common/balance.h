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
 * @Date: 2025-1-23 14:00:00
 * @LastEditors: dev
 * @LastEditTime: 2025-1-23 14:00:00
 */

#ifndef ACCDATA_SRC_CPP_COMMON_BALANCE_H_
#define ACCDATA_SRC_CPP_COMMON_BALANCE_H_

#include "interface/accdata_error_code.h"
#include "common/check.h"

namespace acclib {
namespace accdata {

class Balance {
public:
    struct Task {
        int64_t begin;
        int64_t end;
    };

public:
    /**
     * @brief Evenly distribute tasks to each member
     *
     * @param [in] numTasks     Number of tasks
     * @param [in] numMembers   Number of members executing tasks
     * @param [in] mid          member id
     */
    static AccDataErrorCode Assign(int64_t numTasks, int32_t numMembers, int32_t mid, Task &range)
    {
        int64_t begin;
        int64_t end;

        if (numMembers <= 0) {
            ACCDATA_ERROR("The numMembers should be greater than zero");
            return AccDataErrorCode::H_COMMON_ERROR;
        }

        if (numMembers == 1 || numTasks == 0) {
            begin = 0;
            end = numTasks;
        } else {
            int64_t avg = numTasks / numMembers;
            int64_t mod = numTasks % numMembers;
            int64_t num = avg + (mid < mod ? 1 : 0);
            begin = mid * avg + (mid < mod ? mid : mod);
            end = begin + num;
        }
        range = {begin, end};

        return AccDataErrorCode::H_OK;
    }
};

} // namespace accdata
} // namespace acclib

#endif // ACCDATA_SRC_CPP_COMMON_BALANCE_H_
