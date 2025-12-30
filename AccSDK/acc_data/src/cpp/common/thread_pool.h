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
#ifndef ACCDATA_SRC_CPP_COMMON_THREAD_POOL_H_
#define ACCDATA_SRC_CPP_COMMON_THREAD_POOL_H_

#include <functional>
#include <thread>
#include <vector>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <string>

#include "interface/accdata_error_code.h"

namespace acclib {
namespace accdata {

class ThreadPool {
public:
    using Task = std::function<void(int threadId, AccDataErrorCode &errCode)>;

public:
    /**
     * @brief Construct a new Thread Pool object
     *
     * @param [in] numThreads   Number of threads.
     * @param [in] setAffinity  Wether to bind to cores.
     * @param [in] name         Thread name whose length is restricted to 16 characters.
     */
    ThreadPool(int numThreads, bool setAffinity, const std::string &name);

    ~ThreadPool();

    /**
     * @brief Add task to pool.
     *
     * @param [in] task     Task handler.
     */
    void AddTask(Task task);

    /**
     * @brief Run all tasks
     *
     * @param [in] wait     Wether to wait for the tasks to complete.
     */
    AccDataErrorCode RunAll(bool wait = true);

    /**
     * @brief Wait until all tasks are finished.
     *
     * @param [in] throwErrors      Indicates whether to throw exceptions during task execution.
     */
    AccDataErrorCode WaitAll(bool ifThrowErrors = true);

    uint64_t NumThreads()
    {
        return mThreads.size();
    }

private:
    void Work(int id, const std::string &name);

private:
    std::vector<std::thread> mThreads;
    std::mutex mMutex;
    bool mRunning = false;
    bool mTaskDone = true;
    int mActiveThreads = 0;
    std::queue<Task> mTaskQueue;
    std::condition_variable mWakeupCond;
    std::condition_variable mTaskDoneCond;
    std::vector<AccDataErrorCode> mErrors;
};

} // namespace accdata
} // namespace acclib

#endif // ACCDATA_SRC_CPP_COMMON_THREAD_POOL_H_
