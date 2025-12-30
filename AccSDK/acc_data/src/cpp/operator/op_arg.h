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
 * @Date: 2025-2-11 16:00:00
 * @LastEditors: dev
 * @LastEditTime: 2025-2-11 16:00:00
 */

#ifndef ACCDATA_SRC_CPP_OPERATOR_OP_ARG_H_
#define ACCDATA_SRC_CPP_OPERATOR_OP_ARG_H_

#include <memory>

#include "common/check.h"
#include "interface/accdata_error_code.h"

namespace acclib {
namespace accdata {

/**
 * @brief Operator Argument
 */
class OpArg {
public:
    template <typename T>
    static std::shared_ptr<OpArg> Create(const std::string& name, const T& value);

public:
    explicit OpArg(const std::string& name) : mName(name)
    {
    }

    virtual ~OpArg() = default;

    std::string Name() const
    {
        return mName;
    }

    template <typename T>
    AccDataErrorCode Value(T &value);

    template <typename T>
    bool IsType();

private:
    std::string mName;
};

/**
 * @brief Operator Argument Variant
 *
 * @tparam T Type of value
 */
template <typename T>
class OpArgVar : public OpArg {
public:
    OpArgVar(const std::string& name, const T& value) : OpArg(name), mValue(value)
    {
    }

    ~OpArgVar() override = default;

    T Value()
    {
        return mValue;
    }

private:
    T mValue;
};

template <typename T>
std::shared_ptr<OpArg> OpArg::Create(const std::string& name, const T& value)
{
    return std::make_shared<OpArgVar<T>>(name, value);
}

template <typename T>
AccDataErrorCode OpArg::Value(T &value)
{
    auto* arg = dynamic_cast<OpArgVar<T>*>(this);
    if (arg == nullptr) {
        ACCDATA_ERROR("Invalid value type of argument.");
        return AccDataErrorCode::H_COMMON_OPERATOR_ERROR;
    }
    value = arg->Value();
    return AccDataErrorCode::H_OK;
}

template <typename T>
bool OpArg::IsType()
{
    return dynamic_cast<OpArgVar<T>*>(this) != nullptr;
}

}  // namespace accdata
}  // namespace acclib

#endif  // ACCDATA_SRC_CPP_OPERATOR_OP_ARG_H_
