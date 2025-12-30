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
 * @Date: 2025-10-13 17:00:00
 * @LastEditors: dev
 * @LastEditTime: 2025-3-27 17:00:00
 */

#include "common/string_util.h"

#include "gtest/gtest.h"

namespace acclib {
namespace accdata {
TEST(TestStringUtils, EndWith)
{
    std::string a = "CDE";
    std::string b = "ABCDE";
    EXPECT_EQ(EndWith(a, b), false);
    EXPECT_EQ(EndWith(b, a), true);
}
}
}
