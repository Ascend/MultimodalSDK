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
 * @Date: 2025-3-18 14:00:00
 * @LastEditors: dev
 * @LastEditTime: 2025-3-18 14:00:00
 */

#include "op_spec.h"

#include <vector>

#include "common/utility.h"

namespace acclib {
namespace accdata {

constexpr uint64_t MAX_INPUT_STRING_LENGTH = 1024ULL;

std::shared_ptr<AccDataOpSpec> AccDataOpSpec::Create(const std::string& name)
{
    if (name.size() > MAX_INPUT_STRING_LENGTH) {
        ACCDATA_ERROR("The size of input string(" << name.size() << ") can't be larger than " <<
            MAX_INPUT_STRING_LENGTH);
        return nullptr;
    }
    return std::make_shared<OpSpec>(name);
}

bool CheckStringLength(const std::string &str)
{
    if (str.size() > MAX_INPUT_STRING_LENGTH) {
        ACCDATA_ERROR("The size of input string(" << str.size() << ") can't be larger than " <<
            MAX_INPUT_STRING_LENGTH);
        return false;
    }
    return true;
}

AccDataOpSpec& OpSpec::AddInput(const std::string &name, const std::string &device)
{
    if (!CheckStringLength(name) || !CheckStringLength(device)) {
        return *this;
    }
    mOpInputs.push_back({name, device});
    return *this;
}

AccDataOpSpec& OpSpec::AddOutput(const std::string &name, const std::string &device)
{
    if (!CheckStringLength(name) || !CheckStringLength(device)) {
        return *this;
    }
    mOpOutputs.push_back({name, device});
    return *this;
}

template <typename T>
AccDataOpSpec& OpSpec::AddArg(const std::string& name, const T& value, bool overwrite)
{
    static_assert(IsValidArgType<T>());
    if (name.size() > MAX_INPUT_STRING_LENGTH) {
        ACCDATA_ERROR("The size of input string(" << name.size() << ") can't be larger than " <<
            MAX_INPUT_STRING_LENGTH);
        return *this;
    }
    auto it = mOpArgIdxs.find(name);
    if (it == mOpArgIdxs.end()) {
        mOpArgIdxs.insert({name, mOpArgs.size()});
        mOpArgs.push_back(OpArg::Create<T>(name, value));
    } else if (overwrite) {
        mOpArgs[it->second] = OpArg::Create<T>(name, value);
    }
    return *this;
}

AccDataOpSpec& OpSpec::AddArgInner(const std::string& name, const bool& value, bool overwrite)
{
    return AddArg<bool>(name, value, overwrite);
}

AccDataOpSpec& OpSpec::AddArgInner(const std::string& name, const std::vector<bool>& value, bool overwrite)
{
    return AddArg<std::vector<bool>>(name, value, overwrite);
}

AccDataOpSpec& OpSpec::AddArgInner(const std::string& name, const int64_t& value, bool overwrite)
{
    return AddArg<int64_t>(name, value, overwrite);
}

AccDataOpSpec& OpSpec::AddArgInner(const std::string& name, const std::vector<int64_t>& value, bool overwrite)
{
    return AddArg<std::vector<int64_t>>(name, value, overwrite);
}

AccDataOpSpec& OpSpec::AddArgInner(const std::string& name, const float& value, bool overwrite)
{
    return AddArg<float>(name, value, overwrite);
}

AccDataOpSpec& OpSpec::AddArgInner(const std::string& name, const std::vector<float>& value, bool overwrite)
{
    return AddArg<std::vector<float>>(name, value, overwrite);
}

AccDataOpSpec& OpSpec::AddArgInner(const std::string& name, const std::string& value, bool overwrite)
{
    return AddArg<std::string>(name, value, overwrite);
}

AccDataOpSpec& OpSpec::AddArgInner(const std::string& name, const std::vector<std::string>& value, bool overwrite)
{
    return AddArg<std::vector<std::string>>(name, value, overwrite);
}

}
}