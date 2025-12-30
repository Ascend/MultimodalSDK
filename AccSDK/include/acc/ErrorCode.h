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
 * Description: Definition of Error Code.
 * Author: ACC SDK
 * Create: 2025
 * History: NA
 */

#ifndef ERROR_CODE_H
#define ERROR_CODE_H
#include <cstdint>
namespace Acc {
using ErrorCode = uint32_t;
// 31            28 27                  20 19                     0
// +---------------+----------------------+------------------------+
// |   ModuleID   |      ErrorType       |      SubErrorCode      |
// |    (4 bit)    |      (8 bit)         |        (20 bit)        |
// +---------------+----------------------+------------------------+
enum class ModuleID : uint32_t {
    ACC = 1,
};

enum class ErrorType : uint32_t {
    DATA_CHECK_ERROR = 1,
    RESOURCE_ERROR = 2,
    OPENSOURCE_ERROR = 3,
    THIRD_PARTY_ERROR = 4,
    RUNTIME_ERROR = 5,
};

constexpr int RANGE_SIZE = 1000;

enum class SubErrorCode : uint32_t {
    // DATA_CHECK_ERROR 0~999
    DATA_CHECK_ERROR_BEGIN = 0,
    INVALID_PARAM = 1,
    UNSUPPORTED_TYPE = 2,
    INVALID_POINTER = 3,
    OUT_OF_RANGE = 4,
    DATA_CHECK_ERROR_END,

    // RESOURCE_ERROR 1000~1999
    RESOURCE_ERROR_BEGIN = DATA_CHECK_ERROR_BEGIN + RANGE_SIZE,
    BAD_ALLOC = RESOURCE_ERROR_BEGIN + 1,
    BAD_COPY = RESOURCE_ERROR_BEGIN + 2,
    BAD_FREE = RESOURCE_ERROR_BEGIN + 3,
    OUT_OF_MEM = RESOURCE_ERROR_BEGIN + 4,
    OPEN_FILE_FAILURE = RESOURCE_ERROR_BEGIN + 5,
    INVALID_FILE_PERMISSION = RESOURCE_ERROR_BEGIN + 6,
    INVALID_FILE_SIZE = RESOURCE_ERROR_BEGIN + 7,
    RESOURCE_ERROR_END,

    // OPENSOURCE_ERROR 2000~2999
    OPENSOURCE_ERROR_BEGIN =
        DATA_CHECK_ERROR_BEGIN + RANGE_SIZE * (static_cast<uint32_t>(ErrorType::OPENSOURCE_ERROR) - 1),
    FFMPEG_INIT_FAILURE = OPENSOURCE_ERROR_BEGIN + 1,
    LIBJPEG_INIT_FAILURE = OPENSOURCE_ERROR_BEGIN + 2,
    LIBJPEG_READ_FILE_FAILURE = OPENSOURCE_ERROR_BEGIN + 3,
    FFMPEG_COMMON_FAILURE = OPENSOURCE_ERROR_BEGIN + 4,
    OPENSOURCE_ERROR_END,

    // THIRD_PARTY_ERROR 3000~3999
    THIRD_PARTY_ERROR_BEGIN =
        DATA_CHECK_ERROR_BEGIN + RANGE_SIZE * (static_cast<uint32_t>(ErrorType::THIRD_PARTY_ERROR) - 1),
    ACC_DATA_INIT_FAILURE = THIRD_PARTY_ERROR_BEGIN + 1,
    ACC_DATA_EXECUTE_FAILURE = THIRD_PARTY_ERROR_BEGIN + 2,
    ACC_DATA_PROPERTY_CONVERT_FAILURE = THIRD_PARTY_ERROR_BEGIN + 3,
    THIRD_PARTY_ERROR_END,

    // RUNTIME_ERROR 4000~4999
    RUNTIME_ERROR_BEGIN = DATA_CHECK_ERROR_BEGIN + RANGE_SIZE * (static_cast<uint32_t>(ErrorType::RUNTIME_ERROR) - 1),
    WAIT_TIME_OUT = RUNTIME_ERROR_BEGIN + 1,
    INVALID_THREAD_POOL_STATUS = RUNTIME_ERROR_BEGIN + 2,
    RUNTIME_ERROR_END,
};

constexpr ErrorCode MakeErrorCode(ModuleID product, ErrorType type, SubErrorCode code)
{
    return (static_cast<ErrorCode>(product) << 28) | (static_cast<ErrorCode>(type) << 20) |
           (static_cast<ErrorCode>(code));
}

constexpr ErrorCode SUCCESS = 0;
constexpr ErrorCode ERR_INVALID_PARAM =
    MakeErrorCode(ModuleID::ACC, ErrorType::DATA_CHECK_ERROR, SubErrorCode::INVALID_PARAM);
constexpr ErrorCode ERR_UNSUPPORTED_TYPE =
    MakeErrorCode(ModuleID::ACC, ErrorType::DATA_CHECK_ERROR, SubErrorCode::UNSUPPORTED_TYPE);
constexpr ErrorCode ERR_INVALID_POINTER =
    MakeErrorCode(ModuleID::ACC, ErrorType::DATA_CHECK_ERROR, SubErrorCode::INVALID_POINTER);
constexpr ErrorCode ERR_OUT_OF_RANGE =
    MakeErrorCode(ModuleID::ACC, ErrorType::DATA_CHECK_ERROR, SubErrorCode::OUT_OF_RANGE);
constexpr ErrorCode ERR_BAD_ALLOC = MakeErrorCode(ModuleID::ACC, ErrorType::RESOURCE_ERROR, SubErrorCode::BAD_ALLOC);
constexpr ErrorCode ERR_BAD_COPY = MakeErrorCode(ModuleID::ACC, ErrorType::RESOURCE_ERROR, SubErrorCode::BAD_COPY);
constexpr ErrorCode ERR_BAD_FREE = MakeErrorCode(ModuleID::ACC, ErrorType::RESOURCE_ERROR, SubErrorCode::BAD_FREE);
constexpr ErrorCode ERR_OUT_OF_MEM = MakeErrorCode(ModuleID::ACC, ErrorType::RESOURCE_ERROR, SubErrorCode::OUT_OF_MEM);
constexpr ErrorCode ERR_OPEN_FILE_FAILURE =
    MakeErrorCode(ModuleID::ACC, ErrorType::RESOURCE_ERROR, SubErrorCode::OPEN_FILE_FAILURE);
constexpr ErrorCode ERR_INVALID_FILE_PERMISSION =
    MakeErrorCode(ModuleID::ACC, ErrorType::RESOURCE_ERROR, SubErrorCode::INVALID_FILE_PERMISSION);
constexpr ErrorCode ERR_INVALID_FILE_SIZE =
    MakeErrorCode(ModuleID::ACC, ErrorType::RESOURCE_ERROR, SubErrorCode::INVALID_FILE_SIZE);
constexpr ErrorCode ERR_FFMPEG_INIT_FAILURE =
    MakeErrorCode(ModuleID::ACC, ErrorType::OPENSOURCE_ERROR, SubErrorCode::FFMPEG_INIT_FAILURE);
constexpr ErrorCode ERR_ACC_DATA_INIT_FAILURE =
    MakeErrorCode(ModuleID::ACC, ErrorType::THIRD_PARTY_ERROR, SubErrorCode::ACC_DATA_INIT_FAILURE);
constexpr ErrorCode ERR_ACC_DATA_EXECUTE_FAILURE =
    MakeErrorCode(ModuleID::ACC, ErrorType::THIRD_PARTY_ERROR, SubErrorCode::ACC_DATA_EXECUTE_FAILURE);
constexpr ErrorCode ERR_ACC_DATA_PROPERTY_CONVERT_FAILURE =
    MakeErrorCode(ModuleID::ACC, ErrorType::THIRD_PARTY_ERROR, SubErrorCode::ACC_DATA_PROPERTY_CONVERT_FAILURE);
constexpr ErrorCode ERR_WAIT_TIME_OUT =
    MakeErrorCode(ModuleID::ACC, ErrorType::RUNTIME_ERROR, SubErrorCode::WAIT_TIME_OUT);
constexpr ErrorCode ERR_INVALID_THREAD_POOL_STATUST =
    MakeErrorCode(ModuleID::ACC, ErrorType::RUNTIME_ERROR, SubErrorCode::INVALID_THREAD_POOL_STATUS);
constexpr ErrorCode ERR_LIBJPEG_INIT_FAILURE =
    MakeErrorCode(ModuleID::ACC, ErrorType::OPENSOURCE_ERROR, SubErrorCode::LIBJPEG_INIT_FAILURE);
constexpr ErrorCode ERR_LIBJPEG_READ_FILE_FAILURE =
    MakeErrorCode(ModuleID::ACC, ErrorType::OPENSOURCE_ERROR, SubErrorCode::LIBJPEG_READ_FILE_FAILURE);
constexpr ErrorCode ERR_FFMPEG_COMMON_FAILURE =
    MakeErrorCode(ModuleID::ACC, ErrorType::OPENSOURCE_ERROR, SubErrorCode::FFMPEG_COMMON_FAILURE);
} // namespace Acc
#endif // ERROR_CODE_H