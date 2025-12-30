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
 * @Date: 2025-1-24 10:00:00
 * @LastEditors: dev
 * @LastEditTime: 2025-1-24 10:00:00
 */

#ifndef ACCDATA_SRC_CPP_COMMON_CACHE_LIST_H_
#define ACCDATA_SRC_CPP_COMMON_CACHE_LIST_H_

#include <list>

#include "check.h"

namespace acclib {
namespace accdata {
/**
 * @brief Cache list
 *
 * Free list can be reclaimed and reused to reduce memory allocation overhead.
 */
template <typename T> class CacheList {
public:
    std::list<T> GetFree()
    {
        std::list<T> tmp;
        if (mFree.empty()) {
            tmp.emplace_back();
        } else {
            tmp.splice(tmp.begin(), mFree, mFree.begin());
        }
        return tmp;
    }

    void Recycle(std::list<T> &elm)
    {
        mFree.splice(mFree.end(), elm, elm.begin());
        return;
    }

    bool Empty()
    {
        return mInUse.empty();
    }

    void PushBack(std::list<T> &elm)
    {
        mInUse.splice(mInUse.end(), elm, elm.begin());
    }

    AccDataErrorCode PopFront(std::list<T> &tmp)
    {
        if (mInUse.empty()) {
            ACCDATA_ERROR("List is empty.");
            return AccDataErrorCode::H_COMMON_ERROR;
        }
        tmp.splice(tmp.begin(), mInUse, mInUse.begin());
        return AccDataErrorCode::H_OK;
    }

private:
    std::list<T> mInUse{};
    std::list<T> mFree{};
};
} // namespace accdata
} // namespace acclib

#endif // ACCDATA_SRC_CPP_COMMON_CACHE_LIST_H_
