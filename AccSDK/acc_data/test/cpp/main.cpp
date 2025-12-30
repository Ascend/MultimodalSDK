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
 * @Date: 2025-2-20 9:00:00
 * @LastEditors: dev
 * @LastEditTime: 2025-2-20 9:00:00
 */
#include <gtest/gtest.h>
#include "secodeFuzz.h"

int main(int argc, char **argv)
{
    DT_Set_Report_Path(const_cast<char *>("../build/report"));
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
