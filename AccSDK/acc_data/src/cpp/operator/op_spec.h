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

#ifndef ACCDATA_SRC_CPP_OPERATOR_OP_SPEC_H_
#define ACCDATA_SRC_CPP_OPERATOR_OP_SPEC_H_

#include <vector>
#include <string>
#include <unordered_map>

#include "common/accdata_logger.h"
#include "common/traits.h"
#include "common/check.h"
#include "pipeline/workspace/workspace.h"

#include "op_arg.h"
#include "accdata_op_spec.h"

namespace acclib {
namespace accdata {

/**
 * @brief Operator Specification
 *
 * Defines the construction argument and runtime argument of an Operator.
 */
class OpSpec : public AccDataOpSpec {
public:
    struct InOutDesc {
        std::string name;
        std::string device;
    };

public:
    /**
     * @brief Check whether the argument type is supported.
     *
     * To ensure correct type conversion and simplify implementation, only std::vector are supported.
     */
    template <typename T>
    static constexpr bool IsValidArgType()
    {
        return std::is_same_v<T, bool> || std::is_same_v<T, std::vector<bool>> || std::is_same_v<T, int64_t> ||
               std::is_same_v<T, std::vector<int64_t>> || std::is_same_v<T, float> ||
               std::is_same_v<T, std::vector<float>> || std::is_same_v<T, std::string> ||
               std::is_same_v<T, std::vector<std::string>>;
    }

public:
    /**
     * @brief Construct a new OpSpec object
     *
     * @param [in] name     Operator name.
     */
    explicit OpSpec(const std::string& name) : mOpName(name)
    {
    }

    /**
     * @brief Add Operator's input.
     *
     * @note The adding sequence must meet operator requirements.
     * @param [in] name     Input name uniquely identifies an input
     * @param [in] device   Device where the input is located.
     * @return OpSpec&
     */
    AccDataOpSpec& AddInput(const std::string& name, const std::string& device);

    /**
     * @brief Add Operator's output.
     *
     * @note The adding sequence must meet operator requirements.
     * @param [in] name     Output name uniquely identifies an output.
     * @param [in] device   Device where the output is located.
     * @return OpSpec&
     */
    AccDataOpSpec& AddOutput(const std::string& name, const std::string& device);

    /**
     * @brief Add Operator's argument whose value is specified by the input.
     *
     * @note Argument with the same name can be added through this routine and AddArg().
     * @note The argument added through AddArgInput() have a higher priority.
     * @param [in] name         Argument name.
     * @param [in] input        Input name.
     * @param [in] overwrite    Whether to overwrite the input of existed argument.
     * @return OpSpec&
     */
    OpSpec& AddArgInput(const std::string& name, const std::string& input, bool overwrite = true)
    {
        auto it = mOpArgInputIdxs.find(name);
        if (it == mOpArgInputIdxs.end()) {
            mOpArgInputIdxs.insert({name, mOpArgInputs.size()});
            mOpArgInputs.push_back(input);
        } else if (overwrite) {
            mOpArgInputs[it->second] = input;
        }
        return *this;
    }

    /**
     * @brief Add Operator's argument
     *
     * @note Argument with the same name can be added through this routine and AddArgInput().
     * @note The argument added through AddArgInput() have a higher priority.
     * @param [in] name         Argument name.
     * @param [in] value        Argument value.
     * @param [in] overwrite    Whether to overwrite the value of existed argument.
     * @return OpSpec&
     */
    template <typename T>
    AccDataOpSpec& AddArg(const std::string& name, const T& value, bool overwrite = true);

    OpSpec& AddArg(const std::string& name, std::shared_ptr<OpArg>& value, bool overwrite = true)
    {
        auto it = mOpArgIdxs.find(name);
        if (it == mOpArgIdxs.end()) {
            mOpArgIdxs.insert({name, mOpArgs.size()});
            mOpArgs.push_back(std::move(value));
        } else if (overwrite) {
            mOpArgs[it->second] = std::move(value);
        }
        return *this;
    }

public:
    uint64_t NumInput() const
    {
        return mOpInputs.size();
    }

    uint64_t NumOutput() const
    {
        return mOpOutputs.size();
    }

    uint64_t NumArgInput() const
    {
        return mOpArgInputs.size();
    }

    uint64_t NumArg() const
    {
        return mOpArgs.size();
    }

    /**
     * @brief Has argument input or not.
     *
     * @param [in] name     Argument name.
     */
    bool HasArgInput(const std::string& name) const
    {
        return mOpArgInputIdxs.count(name) != 0;
    }

    /**
     * @brief Has argument
     *
     * @param [in] name     Argument name.
     */
    bool HasArg(const std::string& name) const
    {
        return mOpArgIdxs.count(name) != 0;
    }

    /**
     * @brief Has argument or argument input.
     *
     * @param [in] name     Argument name.
     */
    bool HasArgOrArgInput(const std::string& name) const
    {
        return HasArg(name) || HasArgInput(name);
    }

    std::string Name() const
    {
        return mOpName;
    }

    AccDataErrorCode GetInput(uint64_t idx, InOutDesc &input) const
    {
        if (idx >= NumInput()) {
            ACCDATA_ERROR("Out of range.");
            return AccDataErrorCode::H_COMMON_OPERATOR_ERROR;
        }
        input = mOpInputs[idx];
        return AccDataErrorCode::H_OK;
    }

    AccDataErrorCode GetOutput(uint64_t idx, InOutDesc &output) const
    {
        if (idx >= NumOutput()) {
            ACCDATA_ERROR("Out of range.");
            return AccDataErrorCode::H_COMMON_OPERATOR_ERROR;
        }
        output = mOpOutputs[idx];
        return AccDataErrorCode::H_OK;
    }

    /**
     * @brief Get input name.
     */
    AccDataErrorCode GetArgInput(uint64_t idx, std::string &argInput) const
    {
        if (idx >= NumArgInput()) {
            ACCDATA_ERROR("Out of range.");
            return AccDataErrorCode::H_COMMON_OPERATOR_ERROR;
        }
        argInput = mOpArgInputs[idx];
        return AccDataErrorCode::H_OK;
    }

    auto& GetOpArg(uint64_t idx, AccDataErrorCode &errCode)
    {
        errCode = AccDataErrorCode::H_COMMON_OPERATOR_ERROR;
        if (idx < NumArg()) {
            errCode = AccDataErrorCode::H_OK;
            return mOpArgs[idx];
        }
        return opArg;
    }

    /**
     * @brief Get input name corresponding to the argument name.
     *
     * @param [in] name     Argument name.
     * @return std::string  Input name.
     */
    AccDataErrorCode GetArgInput(const std::string& name, std::string &argInput) const
    {
        if (!HasArgInput(name)) {
            ACCDATA_ERROR("There's no argument named '" << name << "'.");
            return AccDataErrorCode::H_COMMON_OPERATOR_ERROR;
        }
        auto it = mOpArgInputIdxs.find(name);
        argInput = mOpArgInputs[it->second];
        return AccDataErrorCode::H_OK;
    }

    /**
     * @brief Try to get input name corresponding to the argument name.
     *
     * @param [in]  name    Argument name.
     * @param [out] input   Input name.
     */
    bool TryGetArgInput(const std::string& name, std::string& input) const
    {
        if (!HasArgInput(name)) {
            return false;
        }
        auto it = mOpArgInputIdxs.find(name);
        input = mOpArgInputs[it->second];
        return true;
    }

    const auto& GetArgInputIdxs() const
    {
        return mOpArgInputIdxs;
    }

    const auto& GetArgIdxs() const
    {
        return mOpArgIdxs;
    }

    /**
     * @brief Get the argument value.
     *
     * @param [in] name     Argument name.
     * @return T            Argument value.
     */
    template <typename T>
    AccDataErrorCode GetArg(const std::string& name, T &value) const
    {
        if (!HasArg(name)) {
            ACCDATA_ERROR("There's no argument named '" << name << "'.");
            return AccDataErrorCode::H_COMMON_OPERATOR_ERROR;
        }
        auto it = mOpArgIdxs.find(name);
        auto arg = mOpArgs[it->second];
        return arg->Value<T>(value);
    }

    /**
     * @brief Get the argument value from OpSpec or Workspace.
     *
     * @param [in] name     Argument name.
     * @param [in] ws       Workspace.
     * @return T            Argument value.
     */
    template <typename T>
    AccDataErrorCode GetArg(const std::string& name, Workspace& ws, T &value) const
    {
        if (!HasArgInput(name)) {
            return GetArg<T>(name, value);
        }
        if constexpr (IsVector<T>::value) {
            return GetVectorArg<T>(name, ws, value);
        } else {
            return GetOneArg<T>(name, ws, value);
        }
    }

    /**
     * @brief Try to get the argument value.
     *
     * @param [in]  name    Argument name.
     * @param [out] value   Argument value.
     */
    template <typename T>
    bool TryGetArg(const std::string& name, T& value) const
    {
        if (!HasArg(name)) {
            return false;
        }
        auto it = mOpArgIdxs.find(name);
        auto arg = mOpArgs[it->second];
        if (!arg->IsType<T>()) {
            return false;
        }
        AccDataErrorCode errCode = arg->Value<T>(value);
        if (errCode != AccDataErrorCode::H_OK) {
            return false;
        }
        return true;
    }

private:
    AccDataOpSpec& AddArgInner(const std::string& name, const bool& value, bool overwrite);
    AccDataOpSpec& AddArgInner(const std::string& name, const std::vector<bool>& value, bool overwrite);
    AccDataOpSpec& AddArgInner(const std::string& name, const int64_t& value, bool overwrite);
    AccDataOpSpec& AddArgInner(const std::string& name, const std::vector<int64_t>& value, bool overwrite);
    AccDataOpSpec& AddArgInner(const std::string& name, const float& value, bool overwrite);
    AccDataOpSpec& AddArgInner(const std::string& name, const std::vector<float>& value, bool overwrite);
    AccDataOpSpec& AddArgInner(const std::string& name, const std::string& value, bool overwrite);
    AccDataOpSpec& AddArgInner(const std::string& name, const std::vector<std::string>& value, bool overwrite);

    template <typename T>
    AccDataErrorCode GetVectorArg(const std::string& name, Workspace& ws, T &result) const
    {
        static_assert(IsVector<T>::value && IsValidArgType<T>());
        auto errCode = AccDataErrorCode::H_OK;
        auto& arg = ws.GetArgInput(name, errCode);
        ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Argument '" << name << "' not found.",
                                       errCode);

        for (uint64_t i = 0; i < arg.NumTensors(); ++i) {
            void* ptr = arg[i].RawDataPtr().get();
            uint64_t numElements = NumElements(arg[i].Shape());
            if (numElements < 1) {
                ACCDATA_ERROR("Argument tensor should not be empty.");
                return AccDataErrorCode::H_COMMON_OPERATOR_ERROR;
            }
            if constexpr (std::is_same_v<typename T::value_type, std::string>) {
                if (!arg[i].IsDataType<char>()) {
                    ACCDATA_ERROR("Unexpected datatype.");
                    return AccDataErrorCode::H_COMMON_OPERATOR_ERROR;
                }
                result.push_back({static_cast<char*>(ptr), size_t(numElements)});
            } else {
                if (!arg[i].IsDataType<typename T::value_type>()) {
                    ACCDATA_ERROR("Unexpected datatype.");
                    return AccDataErrorCode::H_COMMON_OPERATOR_ERROR;
                }
                auto* values = static_cast<typename T::value_type*>(ptr);
                for (uint64_t j = 0; j < numElements; ++j) {
                    result.push_back(values[j]);
                }
            }
        }
        return AccDataErrorCode::H_OK;
    }

    template <typename T>
    AccDataErrorCode GetOneArg(const std::string& name, Workspace& ws, T &result) const
    {
        static_assert(!IsVector<T>::value && IsValidArgType<T>());
        auto errCode = AccDataErrorCode::H_OK;
        auto& arg = ws.GetArgInput(name, errCode);
        ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Argument '" << name << "' not found.",
                                       errCode);

        if (arg.NumTensors() != 1) {
            ACCDATA_ERROR("TensorList should contains one argument tensor.");
            return AccDataErrorCode::H_COMMON_OPERATOR_ERROR;
        }
        void* ptr = arg[0].RawDataPtr().get();
        uint64_t numElements = NumElements(arg[0].Shape());
        if (numElements != 1) {
            ACCDATA_ERROR("A Tensor should contain one argument value.");
            return AccDataErrorCode::H_COMMON_OPERATOR_ERROR;
        }
        if constexpr (std::is_same_v<T, std::string>) {
            if (!arg[0].IsDataType<char>()) {
                ACCDATA_ERROR("Unexpected datatype.");
                return AccDataErrorCode::H_COMMON_OPERATOR_ERROR;
            }
            result = {static_cast<char*>(ptr), numElements};
        } else {
            if (!arg[0].IsDataType<T>()) {
                ACCDATA_ERROR("Unexpected datatype.");
                return AccDataErrorCode::H_COMMON_OPERATOR_ERROR;
            }
            result = static_cast<T*>(ptr)[0];
        }
        return AccDataErrorCode::H_OK;
    }

private:
    std::string mOpName;
    /* Argument */
    std::vector<std::shared_ptr<OpArg>> mOpArgs{};
    std::unordered_map<std::string, int> mOpArgIdxs{};
    /* Argument initialized by input */
    std::vector<std::string> mOpArgInputs{};
    std::unordered_map<std::string, int> mOpArgInputIdxs{};
    /* Input and Output */
    std::vector<InOutDesc> mOpInputs{};
    std::vector<InOutDesc> mOpOutputs{};
    std::shared_ptr<OpArg> opArg{ nullptr };
};

} // namespace accdata
} // namespace acclib

#endif  // ACCDATA_SRC_CPP_OPERATOR_OP_SPEC_H_
