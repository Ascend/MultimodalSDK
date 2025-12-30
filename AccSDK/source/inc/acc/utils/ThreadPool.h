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
 * Description: Head file for thread pool.
 * Author: ACC SDK
 * Create: 2025
 * History: NA
 */

#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace Acc {

class ThreadPool {
public:
    /**
     * @brief Get the thread pool instance
     */
    static ThreadPool& GetInstance();

    /**
     * @brief submit the thread func to the thread pool
     *
     * @param f function name
     * @param args function param
     */
    template<class F, class... Args>
    auto Submit(F&& f, Args&&... args) -> std::future<decltype(f(args...))>
    {
        using returnType = decltype(f(args...));
        auto task = std::make_shared<std::packaged_task<returnType()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...));
        std::future<returnType> result = task->get_future();
        {
            std::unique_lock<std::mutex> lock(mutex_);
            if (stop_) {
                throw std::runtime_error("ThreadPool has been shut down, can not submit new task please restart.");
            }
            tasks_.emplace([task]() { (*task)(); });
        }
        condition_.notify_one();
        return result;
    }

    /**
     * @brief Wait for all tasks submitted to the thread pool to complete.
     *
     * @param futures A list of futures obtained from submitting tasks
     */
    template<typename T>
    void WaitAll(std::vector<std::future<T>>& futures) const
    {
        for (auto& fut : futures) {
            fut.get();
        }
    }

    /**
     * @brief Shut down the thread pool.
     */
    void Shutdown();

private:
    explicit ThreadPool(size_t numThreads = std::thread::hardware_concurrency());
    ~ThreadPool();
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    void WorkerLoop();
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mutex_; // lock when submitting tasks and worker threads retrieve tasks.
    std::condition_variable condition_; // wake up the waiting worker thread and block wait when task queue is empty.
    std::atomic<bool> stop_;
};
} // namespace Acc
#endif // THREAD_POOL_H
