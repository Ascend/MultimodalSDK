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
 * @Date: 2025-3-21 9:00:00
 * @LastEditors: dev
 * @LastEditTime: 2025-3-21 9:00:00
 */

#include "accdata_tensor.h"
#include "tensor_list.h"

namespace acclib {
namespace accdata {

std::ostream &operator << (std::ostream &os, TensorDataType layout)
{
    switch (layout) {
        case TensorDataType::UINT8:
            os << "UINT8";
            break;
        case TensorDataType::FP32:
            os << "FP32";
            break;
        case TensorDataType::CHAR:
            os << "CHAR";
            break;
        default:
            os << "Unknown";
            break;
    }
    return os;
}

int32_t TensorDataTypeSize(TensorDataType dataType)
{
    switch (dataType) {
        case TensorDataType::UINT8:
            return sizeof(uint8_t);
        case TensorDataType::FP32:
            return sizeof(float);
        case TensorDataType::CHAR:
            return sizeof(char);
        default:
            ACCDATA_ERROR("Unknown data type.");
            return 0;
    }
}

std::ostream &operator << (std::ostream &os, TensorLayout layout)
{
    switch (layout) {
        case TensorLayout::PLAIN:
            os << "PLAIN";
            break;
        case TensorLayout::NHWC:
            os << "NHWC";
            break;
        case TensorLayout::NCHW:
            os << "NCHW";
            break;
        default:
            os << "Unknown";
            break;
    }
    return os;
}


std::shared_ptr<AccDataTensorList> AccDataTensorList::Create(uint64_t batchSize)
{
    return std::make_shared<TensorList>(batchSize);
}

}
}