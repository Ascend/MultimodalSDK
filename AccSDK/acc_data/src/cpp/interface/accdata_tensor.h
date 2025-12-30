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
 * @Date: 2025-3-17 9:00:00
 * @LastEditors: dev
 * @LastEditTime: 2025-3-17 9:00:00
 */

#ifndef ACCDATA_SRC_CPP_INTERFACE_OCKACCDATATENSOR_H_
#define ACCDATA_SRC_CPP_INTERFACE_OCKACCDATATENSOR_H_

#include <memory>
#include <vector>
#include <iostream>
#include <string>
#include <sys/types.h>

#include "accdata_error_code.h"

namespace acclib {
namespace accdata {

/**
 * @brief Tensor layout, value is same with APP_ACC
 */
enum class TensorLayout {
    NCHW = 0,
    NHWC = 1,
    PLAIN = 2,
    LAST = -1, // Invalid
};

std::ostream &operator << (std::ostream &os, TensorLayout layout);

/**
 * @brief Tensor data type, value is same with CANN
 */
enum class TensorDataType {
    FP32 = 0,
    UINT8 = 4,
    CHAR = 13,
    LAST = -1, // Invalid
};

int32_t TensorDataTypeSize(TensorDataType dataType);

/**
 * @brief Tensor shape
 */
using TensorShape = std::vector<size_t>;

std::ostream &operator << (std::ostream &os, TensorDataType layout);

/**
 * @class AccDataTensor
 * @brief AccData数据格式
 */
class AccDataTensor {
public:
    /**
     * @brief 复制数据到当前Tensor对象
     *
     * @note 如果data为空，将返回错误码并终止复制操作
     *
     * @param data 指向源数据的指针
     * @param shape 输入数据的形状
     * @param dataType 输入数据的数据类型
     *
     * @return AccData错误码：
     * - H_OK 成功复制Tensor
     * - H_TENSOR_ERROR data为空或者Tensor拷贝失败
     */
    virtual AccDataErrorCode Copy(const void *data, const TensorShape &shape, TensorDataType dataType) = 0;

    /**
     * @brief 共享数据到当前Tensor对象
     *
     * @param data 指向源数据的共享指针
     * @param shape 输入数据的形状
     * @param dataType 输入数据的数据类型
     */
    virtual AccDataErrorCode ShareData(const std::shared_ptr<void> &data, const TensorShape &shape,
                                       TensorDataType dataType) = 0;

    /**
     * @brief 获取当前Tensor的数据指针
     *
     * @return 数据指针
     */
    virtual std::shared_ptr<void> RawDataPtr() const = 0;

    /**
     * @brief 获取当前Tensor的数据布局
     *
     * @return TensorLayout：
     * - NHWC
     * - NCHW
     * - LAST 不支持的布局
     */
    virtual TensorLayout Layout() const = 0;

    /**
     * @brief 获取当前Tensor的数据类型格式
     *
     * @return TensorDataType：
     * - UINT8 1字节无符号整数
     * - FP32 单精度浮点数
     * - CHAR 字符
     * - LAST 不支持的数据类型
     */
    virtual TensorDataType DataType() const = 0;

    /**
     * @brief 获取当前Tensor的数据形状
     *
     * @return 当前Tensor的数据形状
     */
    virtual const TensorShape &Shape() const = 0;

    /**
     * @brief 设置当前Tensor的数据布局
     *
     * @param layout 当前Tensor的布局
     */
    virtual void SetLayout(TensorLayout layout) = 0;
};

/**
 * @class AccDataTensorList
 * @brief 非均匀共享的Tensor列表
 *
 * AccDataTensorList管理一个非均匀共享的Tensor列表，列表中的Tensor可以有不同的形状、数据类型、布局
 */
class AccDataTensorList {
public:
    /**
     * @brief 创建一个TensorList，包含batchSize个Tensor对象
     *
     * @param batchSize TensorList中包含的Tensor数量
     *
     * @return 返回一个std::shared_ptr<AccDataTensorList>，指向新创建的对象
     */
    static std::shared_ptr<AccDataTensorList> Create(uint64_t batchSize);

    /**
     * @brief 获取当前TensorList包含的Tensor数量
     *
     * @return Tensor数量
     */
    virtual uint64_t NumTensors() const = 0;

    /**
     * @brief 获取指定索引的AccDataTensor引用
     *
     * 通过下标访问TensorList中的元素。
     * 返回指定索引idx位置上的AccDataTensor的引用，使得可以修改该位置上的 Tensor。
     *
     * @param idx 索引值，表示TensorList中的元素位置。
     *
     * @return AccDataTensor& 指定索引位置的AccDataTensor引用。
     */
    virtual AccDataTensor &operator[](uint64_t idx) = 0;

    /**
     * @brief 获取指定索引的AccDataTensor常量引用。
     *
     * 通过下标访问TensorList中的元素，返回一个常量引用。
     *
     * @param idx 索引值，表示TensorList中的元素位置。
     *
     * @return const AccDataTensor& 指定索引位置的AccDataTensor常量引用。
     */
    virtual const AccDataTensor &operator[](uint64_t idx) const = 0;
};

}
}

#endif  // ACCDATA_SRC_CPP_INTERFACE_OCKACCDATATENSOR_H_
