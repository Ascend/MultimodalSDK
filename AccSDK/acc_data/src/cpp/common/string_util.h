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
 * @Date: 2025-1-23 09:00:00
 * @LastEditors: dev
 * @LastEditTime: 2025-1-23 09:00:00
 */

#ifndef ACCDATA_SRC_CPP_COMMON_STRING_UTIL_H_
#define ACCDATA_SRC_CPP_COMMON_STRING_UTIL_H_

#include <iostream>
#include <vector>
#include <string>
#include <unordered_set>

namespace acclib {
namespace accdata {

/**
 * @brief find a minimum vector of strings that cover the target string
 *
 * @param target the target string need to cover
 * @param candidateStrs the subset of strings which can be used
 * @return a minimum vector of strings chosen from candidateStrs which can cover the target string
 */
std::vector<std::string> FindMinSubStrSet(const std::string &target, const std::vector<std::string> &candidateStrs);

inline bool EndWith(const std::string &src, const std::string &end)
{
    if (src.size() < end.size()) {
        return false;
    }
    return src.compare(src.size() - end.size(), end.size(), end) == 0;
}

} // namespace accdata
} // namespace acclib
#endif // ACCDATA_SRC_CPP_COMMON_STRING_UTIL_H_
