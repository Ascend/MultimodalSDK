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
 * @Date: 2025-2-11 17:00:00
 * @LastEditors: dev
 * @LastEditTime: 2025-2-11 17:00:00
 */

#ifndef ACCDATA_SRC_CPP_OPERATOR_OP_FACTORY_H_
#define ACCDATA_SRC_CPP_OPERATOR_OP_FACTORY_H_

#include <unordered_map>
#include <memory>
#include <functional>

#include "op_spec.h"
#include "common/check.h"
#include "operator.h"

namespace acclib {
namespace accdata {

class OpFactory {
public:
    using Creator = std::function<std::unique_ptr<Operator>(const OpSpec&)>;

public:
    static OpFactory& Instance()
    {
        static OpFactory factor;
        return factor;
    }

public:
    ~OpFactory() = default;

    AccDataErrorCode Create(const std::string& name, const OpSpec& spec, std::unique_ptr<Operator> &result)
    {
        auto it = mCreators.find(name);
        if (it == mCreators.end()) {
            ACCDATA_ERROR("Operator not registered.");
            return AccDataErrorCode::H_COMMON_OPERATOR_ERROR;
        }
        result = it->second(spec);
        return AccDataErrorCode::H_OK;
    }

    AccDataErrorCode Register(const std::string& name, const Creator& creator, bool isFuseOps = false)
    {
        if (mCreators.find(name) != mCreators.end()) {
            ACCDATA_ERROR("Operator '" + name + "' already registered.");
            return AccDataErrorCode::H_COMMON_OPERATOR_ERROR;
        }
        mCreators[name] = creator;
        if (isFuseOps) {
            mFuseOps.emplace_back(name);
        }
        return AccDataErrorCode::H_OK;
    }

    const auto& GetFuseOpsNames()
    {
        return mFuseOps;
    }

private:
    OpFactory() = default;

private:
    std::unordered_map<std::string, Creator> mCreators;
    std::vector<std::string> mFuseOps;
};

template <typename OpType>
class Registerer {
public:
    static std::unique_ptr<Operator> Create(const OpSpec& spec)
    {
        return std::make_unique<OpType>(spec);
    }

public:
    Registerer(const std::string& name, bool isFuseOps = false)
    {
        OpFactory::Instance().Register(name, Registerer::Create, isFuseOps);
    }
};

#define ACCDATA_REGISTER_OPERATOR(name, opType) \
    static acclib::accdata::Registerer<opType> ACCDATA_UNIQUE_NAME(name)(#name)

#define ACCDATA_REGISTER_FUSION_OPERATOR(name, opType) \
    static acclib::accdata::Registerer<opType> ACCDATA_UNIQUE_NAME(name)(#name, true)
} // namespace accdata
} // namespace acclib

#endif  // ACCDATA_SRC_CPP_OPERATOR_OP_FACTORY_H_
