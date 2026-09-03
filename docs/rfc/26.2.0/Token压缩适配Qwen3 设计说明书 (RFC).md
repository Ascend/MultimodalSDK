# 多模态SDK token压缩适配Qwen3.5、Qwen3.6模型 设计说明书 (RFC)

**状态 (Status):** Draft
**作者 (Authors):** @damttty
**创建日期 (Created):** 2026-08-26
**更新日期 (Updated):** 2026-08-26
**相关 Issue/PR:** [#38](https://gitcode.com/Ascend/MultimodalSDK/issues/38)

---

# 1. 概述

## 1.1 简介

*多模态SDK当前仅支持Qwen2.5-VL系列模型的token压缩能力，而行业主流已逐步切换到Qwen3、Qwen3.5、Qwen3.6视觉模型，需要升级适配以支持新一代视觉模型。*

## 1.2 动机

*当前SDK的token压缩能力只适配了Qwen2.5-VL模型，导致：*

1. *客户模型选择受限，无法使用最新的Qwen3系列模型*
2. *长上下文场景下无法复用SDK的token压缩优化能力*
3. *与上游开源社区的兼容性不足*

*通过升级适配Qwen3-VL系列模型，可扩展SDK的模型兼容范围，提升客户体验。*

## 1.3 目标

*在指定数据集、场景和模型的前提下，达成：*

1. *支持Qwen3-VL系列模型（Qwen3-VL-8B-Instruct、Qwen3.5-35B-A3B、Qwen3.6-27B）的token压缩*
2. *保持Qwen2.5-VL模型的向后兼容*
3. *精度变化：token压缩后的精度相对原始模型下降不超过1个点*
4. *速度提升：token压缩后推理速度提升不低于20%*

*非指定数据集和场景下的精度和性能不做保证。*

# 2. 用例分析

*适配模型*

1. Qwen/Qwen3-VL-8B-Instruct
2. Qwen/Qwen3.5-35B-A3B
3. Qwen/Qwen3.6-27B

*测试框架与数据集*

1. 使用lmms-eval开源多模态评测框架
2. 配套VideoMME数据集
3. 评测任务：VideoMME多项选择题（MCQ）
4. 评测指标：准确率（Accuracy）、子任务准确率
5. 评估场景：原始模型 vs token压缩后模型精度对比

*设备*

1. 硬件：Atlas 800I A2
2. OS：Ubuntu 22.04
3. Arch：aarch64

# 3. 方案设计

## 3.1 总体方案

*算子增强*

1. 增加Resize算子支持NCHW格式，放开N为1的限制
2. 在原有NHWC格式基础上，新增NCHW格式支持，扩展算子适用范围

*预处理加速适配*

1. 图片预处理加速适配vllm-ascend 0.23.0版本
2. 视频预处理加速适配vllm-ascend 0.23.0版本
3. 对齐vllm-ascend 0.23.0的预处理接口

*Token压缩*

1. 新增SCC（Self-Clustering Compression）token压缩代码
2. 新增patch入口，搭配vllm的插件机制激活
3. 通过环境变量控制SCC压缩的开启和参数配置
4. 支持Qwen3-VL系列模型（Qwen3-VL-8B-Instruct、Qwen3.5-35B-A3B、Qwen3.6-27B）

*兼容性范围与限制*

1. 暂不支持video_decode和load_audio加速音视频解码，相关代码改动不在本设计说明书范围内
2. master分支只维护适配vllm-ascend 0.23.0版本的patcher和adapter
3. 旧版本（vllm-ascend 0.8.5rc1）的patcher和adapter不在master分支维护

## 3.2 技术选型

*基于升级vllm-ascend 0.23.0 + Qwen3-VL的方案*

1. 支持Qwen3-VL-8B-Instruct、Qwen3.5-35B-A3B、Qwen3.6-27B等多款模型
2. 兼容上游vllm-ascend升级节奏
3. 保留对Qwen2.5-VL的向后兼容

## 3.3 功能与性能设计

*关键适配点*

1. Resize算子扩展：在原有NHWC格式基础上，增加NCHW格式支持，放开N为1的限制
2. 预处理加速：图片和视频预处理流程对齐vllm-ascend 0.23.0版本接口
3. SCC Token压缩：新增SCC（Self-Clustering Compression）token压缩代码和patch入口
4. 插件机制激活：通过vllm插件机制和环境变量激活SCC压缩能力
5. 推理接口：保证与vllm-ascend 0.23.0的推理接口对齐
6. 精度验收：在原Qwen2.5-VL推理精度基础上，新增Qwen3-VL系列的精度验收

## 3.4 安全隐私与DFX设计

*无*

## 3.5 编程与调用设计

*若本设计说明书相关特性/功能的组件/模块等支持被开发者集成调用（二次开发），则需要提供便捷易用的编程与调用能力。要站在开发者如何进行编程开发、接口调用及系统集成的使用方式上，给出相应的**编程模型定义和设计**，包括各要素的可获取方式和途径。*

### 3.5.1 编程模型基本设计

***开发环境设计：**明确好开发者使用的软/硬件环境、开发&调试工具链、编程框架、要提供的加速库或算子等。*

***开发约束：**开发者使用过程中的约束和限制说明，如硬件平台、编程语言限制等。*

1. 硬件需求：Atlas 800I A2
2. OS：Ubuntu 22.04
3. Arch：aarch64
4. 编程语言：Python
5. 模型依赖：Qwen/Qwen3-VL-8B-Instruct、Qwen/Qwen3.5-35B-A3B、Qwen/Qwen3.6-27B、vllm-ascend 0.23.0、Processor等

***可验收设计：**提供相应功能、性能指标等的验收环境、标准或用例设计，保证最终的实现可达成既定目标。*

*精度验收*

1. 基于lmms-eval测试框架，覆盖Qwen3-VL-8B-Instruct、Qwen3.5-35B-A3B、Qwen3.6-27B三款模型在VideoMME数据集上的精度评估
2. 对比token压缩前后的精度变化，确保压缩后精度相对原始模型下降不超过1个点
3. 评测指标：准确率（Accuracy）、子任务准确率

*性能验收*

1. 对比token压缩前后的推理速度，确保压缩后推理速度提升不低于20%
2. 性能指标：首token延迟（TTFT）、端到端延迟
3. 在Atlas 800I A2硬件平台、Ubuntu 22.04、aarch64环境下进行基准测试

### 3.5.2 接口定义与设计

*给出相关组件/模块被集成调用的API定义或变更、对接上下游主流生态技术栈的适配方案、提供功能被使用或集成的参考代码或方法等。*

*接口变更说明*

1. 删除了原 `mm/adaptor` 目录下的两个 processor 接口：
   - 删除 `MultimodalQwen2VLImageProcessor`
   - 删除 `InternVL2PreProcessor`
2. 新增 `mm/core/processor.py` 目录下的统一 `resize_and_normalize` 接口，统一处理 Qwen2.5-VL 和 Qwen3-VL 模型的图像/视频预处理

*变更理由*

1. 旧 adaptor 目录下的 processor 接口与模型紧耦合，扩展性差
2. 新 core/processor.py 下的 `resize_and_normalize` 接口作为统一的预处理入口，简化调用方逻辑
3. 通过 vllm patcher 机制激活，支持多模型复用

#### 3.5.2.1 CLI命令

#### 3.5.2.1.1 serve

* *接口描述：复用vllm serve命令启动多模态推理服务*
* *接口原型：vllm serve <model_path>*
* *输入/输出参数：参考vllm serve官方文档*
* *支持的模型列表：*

| 模型                      | 说明                 |
| ------------------------- | -------------------- |
| Qwen/Qwen3-VL-8B-Instruct | Qwen3-VL-8B-Instruct |
| Qwen/Qwen3.5-35B-A3B           | Qwen3.5-35B-A3B           |
| Qwen/Qwen3.6-27B          | Qwen3.6-27B          |
| Qwen2-VL-7B-Instruct      | Qwen2.5-VL（兼容）   |

* *特性激活方式：环境变量配置*

服务启动前通过设置环境变量激活对应 patcher 应用能力，无需修改代码。`resize_and_normalize` 等接口定位为内部接口，最终通过环境变量激活。

| 环境变量                   | 类型  | 默认值 | 说明                                            |
| -------------------------- | ----- | ------ | ----------------------------------------------- |
| MM_SCC_RATE                | float | 1.0    | 视觉token压缩比例（如0.3表示压缩到30%）         |
| MM_SCC_TAU                 | float | 0.95   | SCC聚类余弦相似度阈值，范围(0, 1]               |
| MM_SCC_EPSILON             | float | 0.05   | Union-Find采样误差容忍度，越小越精确但越慢      |
| MM_SCC_MAX_TOKENS_PER_ITEM | int   | 8192   | 单个视觉输入超过该token数则跳过压缩             |
| MM_PREPROCESSOR            | bool  | false  | 是否启用SDK加速预处理器（resize_and_normalize） |

* *使用示例*

```bash
# 设置SDK侧环境变量
export MM_SCC_RATE=0.3
export MM_SCC_TAU=0.95
export MM_SCC_EPSILON=0.05
export MM_PREPROCESSOR=true

# 启动对应模型服务
vllm serve /path/to/Qwen3-VL-8B-Instruct
```

启动后，对应模型的 patcher 自动加载，应用上述特性。

#### 3.5.2.2 resize_and_normalize（统一预处理器）

* *接口描述：Qwen3-VL/Qwen2.5-VL系列模型统一图像/视频预处理接口*
* *接口原型：mm.core.processor.resize_and_normalize*
* *输入/输出参数：*

| 参数名称   | 输入/输出 | 类型         | 描述                      | 取值范围                                    |
| ---------- | --------- | ------------ | ------------------------- | ------------------------------------------- |
| frames     | 输入      | torch.Tensor | 待处理的图像/视频帧tensor | shape必须为4D (N, C, H, W)或者（N, H, W, C) |
| height     | 输入      | int          | 目标resize高度            | > 0                                         |
| width      | 输入      | int          | 目标resize宽度            | > 0                                         |
| image_mean | 输入      | List[float]  | 归一化均值                | 长度为3，值范围[0.0, 1.0]                   |
| image_std  | 输入      | List[float]  | 归一化标准差              | 长度为3，值> 0                              |

* *返回参数：torch.Tensor*

| 参数名称 | 类型         | 描述             | 取值范围                  |
| -------- | ------------ | ---------------- | ------------------------- |
| output   | torch.Tensor | 预处理后的tensor | shape和输入frames保持一致 |

* *参数校验：通过 mm.comm.log._Logger 记录校验日志，包括 error/warn/info 三个级别*
* *异常处理：参数非法时通过 _Log.error 记录并抛出 ValueError*

### 3.5.3 编程手册设计

为了帮助开发者能快速上手开发，要设计好本设计说明书相关特性/功能的《编程手册》，要包含哪些内容和章节，单独输出还是共用，在已有的手册中更新还是新输出等。确保最后输出的《编程手册》中有相关变更内容。*

# 4. 缺点和风险

*说明潜在风险（Breaking Change、性能回退、复杂度提升、引入的安全问题）、负面影响（对现有功能/用户的冲击）、实现成本（代码量/维护成本/人力投入）、是否有API或版本兼容性、旧版本迁移方案问题等，给出应对措施。*

*潜在风险*

1. 精度退化风险：token压缩后可能存在精度不一致
2. 功能限制：暂不支持video_decode和load_audio加速音视频解码，对音视频处理性能有依赖的场景可能受影响
3. 维护成本：master分支只维护适配vllm-ascend 0.23.0版本的patcher和adapter，旧版本不再维护，需要升级到新版本的客户将获得技术支持

*应对措施*

1. API兼容性：尽量复用现有API，必要时通过可选参数方式兼容新旧版本

# 5. 现有技术

*参考其他项目/社区的类似设计，说明借鉴与差异。*

*参考来源*

1. vllm-ascend 0.23.0上游对Qwen3-VL模型的官方支持
2. HuggingFace Transformers对Qwen3-VL系列模型的Processor实现
3. Qwen3-VL官方仓库的示例代码和最佳实践

*借鉴与差异*

1. 借鉴：vllm-ascend 0.23.0中对Qwen3-VL的接口设计
2. 差异：多模态SDK需要在vllm-ascend基础上叠加token压缩能力，需要在保持压缩能力的同时兼容新模型

# 6. 未解决问题

*待社区讨论/决策的开放问题，如硬件适配范围、参数默认值等（需在RFC通过前解决）。*

---

附录

* **参考资料链接。**
* **术语表。**
* **文档更新计划**
