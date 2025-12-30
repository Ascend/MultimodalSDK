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
 * Description: File for thread pool.
 * Author: ACC SDK
 * Create: 2025
 * History: NA
 */

#include "acc/utils/ThreadPool.h"
#include <iostream>

namespace {
constexpr size_t THREAD_POOL_DEFAULT_THREAD_NUMS = 128;
}

namespace Acc {

ThreadPool::ThreadPool(size_t numThreads) : stop_(false)
{
    for (size_t i = 0; i < numThreads; ++i) {
        workers_.emplace_back([this]() { this->WorkerLoop(); });
    }
}

ThreadPool::~ThreadPool()
{
    Shutdown();
}

ThreadPool& ThreadPool::GetInstance()
{
    static ThreadPool instance(THREAD_POOL_DEFAULT_THREAD_NUMS);
    return instance;
}

void ThreadPool::Shutdown()
{
    {
        std::unique_lock<std::mutex> lock(mutex_);
        stop_ = true;
    }
    condition_.notify_all();
    for (std::thread& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

void ThreadPool::WorkerLoop()
{
    while (true) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            condition_.wait(lock, [this]() { return stop_ || !tasks_.empty(); });

            if (stop_ && tasks_.empty()) {
                return;
            }

            task = std::move(tasks_.front());
            tasks_.pop();
        }
        task();
    }
}
} // namespace Acc