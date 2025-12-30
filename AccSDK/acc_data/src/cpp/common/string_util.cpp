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
#include <algorithm>
#include <climits>

#include "string_util.h"

namespace acclib {
namespace accdata {

void BuildDpTable(const std::string &candidate, const size_t &i, const std::string &target, std::vector<int> &dp,
    std::vector<int> &path)
{
    size_t len = candidate.size();
    if (i < len || target.compare(i - len, len, candidate) != 0) {
        return;
    }
    if (dp[i - len] == INT_MAX || dp[i - len] + 1 >= dp[i]) {
        return;
    }
    dp[i] = dp[i - len] + 1;
    path[i] = static_cast<int>(i - len);
}

std::vector<std::string> FindMinSubStrSet(const std::string &target, const std::vector<std::string> &candidateStrs)
{
    size_t n = target.size();
    // dp[i] 表示前 i 个字符组成的最小子串集合的大小
    std::vector<int> dp(n + 1, INT_MAX);
    dp[0] = 0; // 空字符串不需要任何子串

    // path[i] 记录了到达 i 位置时使用的最后一个子串的起始位置
    std::vector<int> path(n + 1, -1);

    for (size_t i = 1ULL; i <= n; ++i) {
        for (const auto &candidate : candidateStrs) {
            BuildDpTable(candidate, i, target, dp, path);
        }
    }

    // 如果 dp[n] 仍然是 INT_MAX，说明无法组成字符串 s
    if (dp[n] == INT_MAX) {
        return {};
    }

    // 回溯找到最小子串集合
    std::vector<std::string> result;
    int pos = static_cast<int>(n);
    while (pos > 0) {
        int start = path[pos];
        if (start == -1) {
            break;
        }
        result.push_back(target.substr(start, pos - start));
        pos = start;
    }

    std::reverse(result.begin(), result.end());
    return result;
}

} // namespace accdata
} // namespace acclib
