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
#include "thread_pool.h"

#include <pthread.h>
#include <stdexcept>

#include "check.h"

namespace acclib {
namespace accdata {

ThreadPool::ThreadPool(int numThreads, bool setAffinity, const std::string &name)
{
    mRunning = true;
    mErrors.resize(numThreads);
    mThreads.resize(numThreads);
    for (int i = 0; i < numThreads; ++i) {
        mThreads[i] = std::thread(std::bind(&ThreadPool::Work, this, i, name));
    }
}

ThreadPool::~ThreadPool()
{
    WaitAll(false);
    {
        std::unique_lock<std::mutex> lock(mMutex);
        mRunning = false;
        mWakeupCond.notify_all();
    }
    for (auto &thread : mThreads) {
        thread.join();
    }
}

void ThreadPool::AddTask(Task task)
{
    std::unique_lock<std::mutex> lock(mMutex);
    mTaskQueue.push(std::move(task));
    mTaskDone = false;
    return;
}

AccDataErrorCode ThreadPool::RunAll(bool wait)
{
    mWakeupCond.notify_all();
    if (wait) {
        auto errCode = WaitAll();
        ACCDATA_CHECK_ERRORCODE_RETURN(errCode == AccDataErrorCode::H_OK, "Failed to run task",
                                       errCode);
    }
    return AccDataErrorCode::H_OK;
}

AccDataErrorCode ThreadPool::WaitAll(bool ifThrowErrors)
{
    {
        std::unique_lock<std::mutex> lock(mMutex);
        mTaskDoneCond.wait(lock, [this] {return mTaskDone;});
    }
    if (!ifThrowErrors) {
        return AccDataErrorCode::H_OK;
    }
    AccDataErrorCode err = AccDataErrorCode::H_OK;
    for (uint32_t i = 0; i < mErrors.size(); ++i) {
        if (mErrors[i] != AccDataErrorCode::H_OK) {
            ACCDATA_ERROR("Task " << i << "run op error: " << mErrors[i]);
            err = mErrors[i]; // Return last task error to User.
        }
    }
    return err;
}

void ThreadPool::Work(int id, const std::string &name)
{
    pthread_setname_np(pthread_self(), name.c_str());
    Task task;
    auto errCode = AccDataErrorCode::H_OK;
    while (mRunning) {
        /* Get task */
        {
            std::unique_lock<std::mutex> lock(mMutex);
            mWakeupCond.wait(lock, [this] {return !mTaskQueue.empty() || !mRunning;});
            if (!mRunning) {
                break;
            }
            task = std::move(mTaskQueue.front());
            mTaskQueue.pop();
            ++mActiveThreads;
        }
        /* Do task */
        try {
            task(id, errCode);
            std::unique_lock<std::mutex> lock(mMutex);
            mErrors[id] = errCode;
        } catch (...) {
            std::unique_lock<std::mutex> lock(mMutex);
            mErrors[id] = H_THREADPOOL_ERROR;
        }
        /* Check wether all task are done and all threads are idle. */
        {
            std::unique_lock<std::mutex> lock(mMutex);
            --mActiveThreads;
            if (mTaskQueue.empty() && mActiveThreads == 0) {
                mTaskDone = true;
            }
        }
        /* Notify that all tasks are complete. */
        if (mTaskDone) {
            mTaskDoneCond.notify_one();
        }
    }
    return;
}

} // namespace accdata
} // namespace acclib
