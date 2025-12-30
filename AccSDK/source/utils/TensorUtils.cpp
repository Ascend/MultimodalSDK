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
 * Description: TensorUtils cpp file.
 * Author: ACC SDK
 * Create: 2025
 * History: NA
 */

#include "acc/utils/TensorUtils.h"

#include <map>
#include "accdata_tensor.h"
#include "acc/tensor/TensorDataType.h"

using namespace acclib::accdata;
namespace {
    using namespace Acc;
    // Map TensorFormat to TensorLayout
    const std::map<TensorFormat, TensorLayout> tensorFormatToTensorLayout = {
        {TensorFormat::NCHW, TensorLayout::NCHW},
        {TensorFormat::NHWC, TensorLayout::NHWC},
        {TensorFormat::ND,   TensorLayout::PLAIN}
    };
    // Map TensorLayout to TensorFormat
    const std::map<TensorLayout, TensorFormat> tensorLayoutToTensorFormat = {
        {TensorLayout::NCHW,  TensorFormat::NCHW},
        {TensorLayout::NHWC,  TensorFormat::NHWC},
        {TensorLayout::PLAIN, TensorFormat::ND}
    };

    // Map DataType to TensorDataType
    const std::map<DataType, TensorDataType> dataTypeToTensorDataType = {
        {DataType::UINT8, TensorDataType::UINT8},
        {DataType::FLOAT32, TensorDataType::FP32}
    };

    // Map TensorDataType to DataType
    const std::map<TensorDataType, DataType> tensorDataTypeToDataType = {
        {TensorDataType::UINT8, DataType::UINT8},
        {TensorDataType::FP32, DataType::FLOAT32}
    };
}

namespace Acc {
    TensorLayout ToTensorLayout(TensorFormat format)
    {
        auto it = tensorFormatToTensorLayout.find(format);
        if (it != tensorFormatToTensorLayout.end()) {
            return it->second;
        }
        throw std::invalid_argument("Convert Multimodal SDK tensor format to acc_data tensor layout failed, "
                                    "unsupported tensor format.");
    }

    TensorFormat ToTensorFormat(TensorLayout layout)
    {
        auto it = tensorLayoutToTensorFormat.find(layout);
        if (it != tensorLayoutToTensorFormat.end()) {
            return it->second;
        }
        throw std::invalid_argument("Convert acc_data TensorLayout to Multimodal SDK TensorFormat failed, "
                                    "unsupported tensor layout.");
    }


    TensorDataType ToTensorDataType(DataType dataType)
    {
        auto it = dataTypeToTensorDataType.find(dataType);
        if (it != dataTypeToTensorDataType.end()) {
            return it->second;
        }
        throw std::invalid_argument("Convert Multimodal SDK DataType to acc_data TensorDataType failed, "
                                    "unsupported data type.");
    }

    DataType ToDataType(TensorDataType tensorDataType)
    {
        auto it = tensorDataTypeToDataType.find(tensorDataType);
        if (it != tensorDataTypeToDataType.end()) {
            return it->second;
        }
        throw std::invalid_argument("Convert acc_data TensorDataType to Multimodal SDK DataType failed, "
                                    "unsupported tensor data type.");
    }
}