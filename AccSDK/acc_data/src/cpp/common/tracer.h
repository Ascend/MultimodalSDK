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
 * @Date: 2025-2-11 16:00:00
 * @LastEditors: dev
 * @LastEditTime: 2025-2-11 16:00:00
 */

#ifndef ACCDATA_SRC_CPP_COMMON_TRACER_H_
#define ACCDATA_SRC_CPP_COMMON_TRACER_H_

namespace acclib {
namespace accdata {

static __always_inline uint64_t GetTimeNs()
{
    struct timespec tpDelay = {0, 0};
    clock_gettime(CLOCK_MONOTONIC, &tpDelay);
    return tpDelay.tv_sec * 1000000000ULL + tpDelay.tv_nsec;
}

#ifdef ENABLE_TRACER
#define TRACE_BEGIN(TP_ID) \
uint64_t tpBegin##TP_ID = GetTimeNs()

#define TRACE_END(TP_ID) \
std::cout << "AccData timecost " <<  #TP_ID << " " << (double)(GetTimeNs()-tpBegin##TP_ID)/1000000 << " ms" << \
std::endl

#else
#define TRACE_BEGIN(TP_ID)
#define TRACE_END(TP_ID)
#endif

} // namespace accdata
} // namespace acclib

#endif // ACCDATA_SRC_CPP_COMMON_TRACER_H_
