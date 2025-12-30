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
 * Author: Dev
 * Create: 2025-07-26
 */

#ifndef ACCDATA_COMMON_H
#define ACCDATA_COMMON_H

#include <climits>
#include <random>

#include "accdata_pipeline.h"
#include "logger.h"

namespace acclib {
namespace accdata {
constexpr int64_t TEST_COUNT = 3000;
constexpr int64_t EXEC_SECOND = 30;

template <typename T> T GenerateOneData(int start, int end)
{
    static std::default_random_engine engine(std::random_device{}());
    std::uniform_int_distribution<int> distribution(start, end);
    return static_cast<T>(distribution(engine));
}

template <typename T> T RandomSelectOne(std::vector<T> &options)
{
    uint32_t selectedPos = GenerateOneData<uint32_t>(0, options.size() - 1);
    return options[selectedPos];
}

template <typename T> T *NullPtrByChance(T *input)
{
    if (GenerateOneData<float>(-1, 9) < 0) {
        return nullptr;
    }
    return input;
}
}
}
#endif // ACCDATA_COMMON_H
