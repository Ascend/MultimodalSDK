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

#ifndef ACCDATA_SRC_CPP_COMMON_UTILITY_H_
#define ACCDATA_SRC_CPP_COMMON_UTILITY_H_

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>

namespace acclib {
namespace accdata {

#define ACCDATA_STR_IMPL(a) #a
#define ACCDATA_STR(a) ACCDATA_STR_IMPL(a)
#define ACCDATA_FILE_AND_LINE __FILE__ ":" ACCDATA_STR(__LINE__)

#define ACCDATA_CONCAT_IMPL(var1, var2) var1##var2
#define ACCDATA_CONCAT(var1, var2) ACCDATA_CONCAT_IMPL(var1, var2)
#define ACCDATA_UNIQUE_NAME(name) ACCDATA_CONCAT(name, __LINE__)

#define ACCDATA_LIKELY(X) (__builtin_expect(!!(X), 1) != 0)
#define ACCDATA_UNLIKELY(X) (__builtin_expect(!!(X), 0) != 0)

constexpr int64_t ACCDATA_ALIGN_SIZE = 64;

constexpr int RGB_CHANNELS = 3;
constexpr int RGB_CHANNEL_RED = 0;
constexpr int RGB_CHANNEL_GREEN = 1;
constexpr int RGB_CHANNEL_BLUE = 2;

constexpr int64_t RESIZE_HEIGHT_MIN = 10;
constexpr int64_t RESIZE_HEIGHT_MAX = 8192;
constexpr int64_t RESIZE_WIDTH_MIN = 10;
constexpr int64_t RESIZE_WIDTH_MAX = 8192;
constexpr int64_t CROP_HEIGHT_MIN = 10;
constexpr int64_t CROP_HEIGHT_MAX = 8192;
constexpr int64_t CROP_WIDTH_MIN = 10;
constexpr int64_t CROP_WIDTH_MAX = 8192;

/**
 * @brief Align up the input to the nearest multiple of ACCDATA_ALIGN_SIZE
 */
inline uint64_t AlignUp(int64_t input, uint64_t size)
{
    return ((static_cast<uint64_t>(input) * size) / ACCDATA_ALIGN_SIZE + 1ULL) * ACCDATA_ALIGN_SIZE;
}


/**
 * @brief Trim the left leading space of the string.
 */
inline std::string &TrimLeft(std::string &s)
{
    auto nonSpace = [](int c) -> bool { return !std::isspace(c); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), nonSpace));
    return s;
}

/**
 * @brief Trim the right trailing space of the string.
 */
inline std::string &TrimRight(std::string &s)
{
    auto nonSpace = [](int c) -> bool { return !std::isspace(c); };
    s.erase(std::find_if(s.rbegin(), s.rend(), nonSpace).base(), s.end());
    return s;
}

/**
 * @brief Trim the leading and trailing space of the string.
 */
inline std::string &Trim(std::string &s)
{
    return TrimLeft(TrimRight(s));
}

/**
 * @brief Print all elements in the vector.
 */
template <typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &v)
{
    os << "[";
    for (auto it = v.begin(); it != v.end(); ++it) {
        os << *it << ((it + 1) == v.end() ? "]" : ", ");
    }
    return os;
}

struct AccdataTime : public std::tm {
public:
    static const int million = 1000000;
    long long microseconds;
};

inline struct AccdataTime GetCurrentTime()
{
    AccdataTime nowTm;
    auto now = std::chrono::system_clock::now();
    std::time_t nowTimeT = std::chrono::system_clock::to_time_t(now);
    auto nowUs = std::chrono::time_point_cast<std::chrono::microseconds>(now);
    localtime_r(&nowTimeT, &nowTm);
    nowTm.microseconds = nowUs.time_since_epoch().count() % AccdataTime::million;
    return nowTm;
}

/**
 * simulate python round method
 * Python: Round Half to Even, also known as Bankers' Rounding.
 * C++: std::round rounds away from zero.
 * This function implements the Python behavior.
 * @param f: The floating-point number to round.
 * @return: The rounded integer value.
 */
inline int64_t PyRound(double f)
{
    auto i = static_cast<int64_t>(f);
    if (ACCDATA_UNLIKELY(i % 2 == 0 && (f - i) == 0.5)) {
        return i;
    }
    return static_cast<int64_t>(std::round(f));
}

}  // namespace accdata
}  // namespace acclib

#endif  // ACCDATA_SRC_CPP_COMMON_UTILITY_H_
