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
 * Description: Tensor.
 * Author: ACC SDK
 * Create: 2025
 * History: NA
 */
#ifndef TENSOR_DATA_TYPE_H
#define TENSOR_DATA_TYPE_H
#include <cstddef>   // size_t
#include <cstdint>   // uint32_t
#include <vector>
namespace Acc {
enum class DataType { INT8 = 2, UINT8 = 4, FLOAT32 = 0 };

enum class TensorFormat {
    ND = 2,
    NCHW = 0,
    NHWC = 1,
};

// Tensor auxiliary information
struct TensorAuxInfo {
    size_t elementNums;
    size_t perElementBytes;
    size_t totalBytes;
    std::vector<uint32_t> memoryStrides;
    std::vector<uint32_t> logicalStrides;
};

enum class Interpolation {
    BICUBIC = 2,
};

enum class DeviceMode {
    CPU = 0,
};
} // namespace Acc
#endif