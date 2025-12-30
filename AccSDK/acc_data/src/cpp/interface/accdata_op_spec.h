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
 * @Date: 2025-3-18 9:00:00
 * @LastEditors: dev
 * @LastEditTime: 2025-3-18 9:00:00
 */

#ifndef ACCDATA_SRC_CPP_INTERFACE_OCKACCDATAOPSPEC_H_
#define ACCDATA_SRC_CPP_INTERFACE_OCKACCDATAOPSPEC_H_

namespace acclib {
namespace accdata {

/**
 * @class AccDataOpSpec
 * @brief 算子描述信息（输入输出、参数）。
 *
 * AccDataOpSpec为算子描述信息接口类，提供添加输入、添加输出、添加参数等对外接口。
 */
class AccDataOpSpec {
public:
    /**
     * @brief 创建一个指定名称的 `AccDataOpSpec` 类型对象。
     *
     * @note 该函数根据传入的名称参数创建并返回一个指向 `AccDataOpSpec` 类型的智能指针。
     * @note 创建成功，返回一个指向 `AccDataOpSpec` 类型的 `std::shared_ptr`。
     *
     * @param name 要创建算子描述的名称。
     *
     * @return 返回一个 `std::shared_ptr<AccDataOpSpec>`，指向新创建的对象。
     */
    static std::shared_ptr<AccDataOpSpec> Create(const std::string& name);

    /**
     * @brief 向算子描述中添加输入项。
     *
     * @note 该函数用于向当前算子描述添加一个输入数据源。输入项由名称和设备标识符组成，
     *
     * @param name 输入项的名称，用于标识输入数据。
     * @param device 设备标识符，指定输入项所在的设备（例如 CPU、NPU 等）。
     *
     * @return 返回当前操作对象的引用，以便进行链式调用。
     */
    virtual AccDataOpSpec& AddInput(const std::string& name, const std::string& device) = 0;

    /**
     * @brief 向算子描述中添加输出项。
     *
     * @note 该函数用于向当前算子描述添加一个输出。输出项由名称和设备标识符组成，
     *
     * @param name 输出项的名称，用于标识输出数据。
     * @param device 设备标识符，指定输出项所在的设备（例如 CPU、NPU 等）。
     *
     * @return 返回当前操作对象的引用，以便进行链式调用。
     */
    virtual AccDataOpSpec& AddOutput(const std::string& name, const std::string& device) = 0;

    /**
     * @brief 向算子描述中添加参数。
     *
     * @note 该函数用于将一个名为 `name` 的参数添加到当前算子描述中，并为其指定一个值 `value`。
     * @note 如果存在具有相同名称的参数，可以通过 `overwrite` 参数来控制是否覆盖已有的参数值。
     * @note 支持的类型包括bool、int64_t、float、string以及以上类型的vector
     *
     * @tparam T 参数值的类型
     *
     * @param name 参数的名称，用于标识该参数。
     * @param value 参数的值，具体类型由模板参数 T 决定。
     * @param overwrite 如果为 `true`，则在参数名重复时覆盖已有参数的值；如果为 `false`，则保持原有值不变，默认为 `true`。
     *
     * @return 返回当前操作对象的引用，以便进行链式调用。
     */
    template <typename T>
    AccDataOpSpec& AddArg(const std::string& name, const T& value, bool overwrite = true)
    {
        return AddArgInner(name, value, overwrite);
    }

private:
    virtual AccDataOpSpec& AddArgInner(const std::string& name,
                                          const bool& value, bool overwrite) = 0;
    virtual AccDataOpSpec& AddArgInner(const std::string& name,
                                          const std::vector<bool>& value, bool overwrite) = 0;
    virtual AccDataOpSpec& AddArgInner(const std::string& name,
                                          const int64_t& value, bool overwrite) = 0;
    virtual AccDataOpSpec& AddArgInner(const std::string& name,
                                          const std::vector<int64_t>& value, bool overwrite) = 0;
    virtual AccDataOpSpec& AddArgInner(const std::string& name,
                                          const float& value, bool overwrite) = 0;
    virtual AccDataOpSpec& AddArgInner(const std::string& name,
                                          const std::vector<float>& value, bool overwrite) = 0;
    virtual AccDataOpSpec& AddArgInner(const std::string& name,
                                          const std::string& value, bool overwrite) = 0;
    virtual AccDataOpSpec& AddArgInner(const std::string& name,
                                          const std::vector<std::string>& value, bool overwrite) = 0;
};

}
}

#endif  // ACCDATA_SRC_CPP_INTERFACE_OCKACCDATAOPSPEC_H_
