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
 * @Date: 2025-3-15 10:00:00
 * @LastEditors: dev
 * @LastEditTime: 2025-3-15 10:00:00
 */

#ifndef ACCDATA_SRC_CPP_INTERFACE_OCKACCDATAPIPELINE_H_
#define ACCDATA_SRC_CPP_INTERFACE_OCKACCDATAPIPELINE_H_

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

#include "accdata_op_spec.h"
#include "accdata_tensor.h"
#include "accdata_error_code.h"

namespace acclib {
namespace accdata {

class AccDataPipeline {
public:
    /**
     * @brief 创建一个 `AccDataPipeline` 类型对象。
     *
     * @param batchSize 加载样本的个数，取值范围在[1， 1024]之间，默认为1
     * @param numThreads 使用的线程数量，取值范围在[1， 当前系统可用cpu核数]之间，默认为1
     * @param depth 预取的队列长度，取值范围在[2, 128]之间，默认为2
     * @param enableFusion 是否开启融合，默认开启
     *
     * @return 返回一个 `std::shared_ptr<AccDataPipeline>`，指向新创建的对象。
     */
    static std::shared_ptr<AccDataPipeline> Create(int batchSize = 1, int numThreads = 1, int depth = 2,
        bool enableFusion = true);

    virtual ~AccDataPipeline() noexcept = default;

    /**
     * @brief 根据输出算子信息，构建Pipeline。
     *
     * @param specs 需要添加的算子描述
     * @param outputs 算子输出节点
     *
     * @return AccData错误码:
     * - H_OK 成功构建
     * - H_PIPELINE_ERROR Pipeline系统错误
     * - H_PIPELINE_BUILD_ERROR Pipeline构建错误
     */
    virtual AccDataErrorCode Build(const std::vector<std::shared_ptr<AccDataOpSpec>> &specs,
        const std::vector<std::string> &outputs) = 0;

    /**
     * @brief 运行Pipeline
     *
     * @note 调用Pipeline运行接口前，需要调用Pipeline Build接口构建计算图。
     *
     * @param inputs 输入名称和数据
     * @param outputs Pipeline输出结果
     * @param copy 是否拷贝
     *
     * @return AccData错误码：
     * - H_OK 成功运行
     * - H_PIPELINE_STATE_ERROR Pipeline状态不符合预期
     * - H_PIPELINE_ERROR Pipeline系统错误
     */
    virtual AccDataErrorCode Run(std::unordered_map<std::string, std::shared_ptr<AccDataTensorList>> inputs,
        std::vector<std::shared_ptr<AccDataTensorList>> &outputs, bool copy) = 0;
};

}
}

#endif  // ACCDATA_SRC_CPP_INTERFACE_OCKACCDATAPIPELINE_H_
