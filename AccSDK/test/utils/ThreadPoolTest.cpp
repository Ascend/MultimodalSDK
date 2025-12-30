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
 * Description: Test for ThreadPool.
 * Author: ACC SDK
 * Create: 2025
 * History: NA
 */

#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <thread>
#include "acc/utils/LogImpl.h"
#include "acc/utils/ThreadPool.h"

using namespace Acc;

namespace {
constexpr int ONE = 1;
constexpr int TWO = 2;
constexpr int THREE = 3;
constexpr int FOUR = 4;

class ThreadPoolTest : public testing::Test {};

void SimpleThreadFunc1(int x)
{
    LogInfo << "ThreadFunc1 Working: " << x << " in thread " << std::this_thread::get_id();
}

void SimpleThreadFunc2(int x)
{
    LogInfo << "ThreadFunc2 Working: " << x << " in thread " << std::this_thread::get_id();
}

TEST_F(ThreadPoolTest, Test_Thread_Pool_With_Two_Func)
{
    std::vector<std::future<void>> futures;
    futures.push_back(ThreadPool::GetInstance().Submit(SimpleThreadFunc1, ONE));
    futures.push_back(ThreadPool::GetInstance().Submit(SimpleThreadFunc2, TWO));
    futures.push_back(ThreadPool::GetInstance().Submit(SimpleThreadFunc2, THREE));
    futures.push_back(ThreadPool::GetInstance().Submit(SimpleThreadFunc1, FOUR));
    ASSERT_EQ(futures.size(), 4);
    ThreadPool::GetInstance().WaitAll(futures);
}

TEST_F(ThreadPoolTest, Test_Thread_Pool_Submit_After_Shutdown_Should_Return_Failed)
{
    std::vector<std::future<void>> futures;
    futures.push_back(ThreadPool::GetInstance().Submit(SimpleThreadFunc1, ONE));
    ThreadPool::GetInstance().WaitAll(futures);
    ThreadPool::GetInstance().Shutdown();
    try {
        ThreadPool::GetInstance().Submit(SimpleThreadFunc2, TWO);
    } catch (const std::runtime_error& e) {
        EXPECT_EQ(e.what(), std::string("ThreadPool has been shut down, can not submit new task please restart."));
    }
}

} // namespace

int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    return ret;
}