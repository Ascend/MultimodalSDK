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
 * @Date: 2025-1-22 14:00:00
 * @LastEditors: dev
 * @LastEditTime: 2025-1-22 14:00:00
 */

#ifndef ACCDATA_SRC_CPP_COMMON_SINGLETON_H_
#define ACCDATA_SRC_CPP_COMMON_SINGLETON_H_

namespace acclib {
namespace accdata {

template <typename T> class Singleton {
public:
    Singleton() = delete;

    Singleton(const Singleton &singleton) = delete;

    Singleton &operator = (const Singleton &singleton) = delete;

    static T *GetInstance()
    {
        try {
            static T instance;
            return &instance;
        } catch (std::exception &e) {
            std::cerr << " create singleton error" << std::endl;
            return nullptr;
        }
    }
};

} // namespace accdata
} // namespace acclib

#endif // ACCDATA_SRC_CPP_COMMON_SINGLETON_H_
