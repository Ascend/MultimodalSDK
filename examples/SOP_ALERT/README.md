# 基于 MultimodalSDK 的工厂操作视频 SOP 违规自动预警最佳实践

## 1 概述

工业生产场景下设备操作通常存在 SOP（标准作业流程）。本实践在 MultimodalSDK 能力基础上，对工人操作视频提取多模态信息（视频帧、音频、关键帧），理解视频中的实际操作步骤，将其与标准 SOP 进行对比分析，自动输出违规告警报告，实现"操作视频 + SOP 输入 → JSON/Markdown 分析报告输出"的端到端流水线。

支持的告警能力：

| 告警类型 | 级别 | 说明 |
|---------|------|------|
| `STEP_MISSING` | ERROR（关键步骤）/ WARNING | 步骤丢失：SOP 规定的步骤未在视频中检测到 |
| `STEP_ORDER_VIOLATION` | WARNING | 步骤顺序异常：实际执行时序与 SOP 规定顺序不符 |
| `STEP_DURATION_ANOMALY` | WARNING | 步骤耗时异常：耗时超出 SOP 规定的时长范围 |
| `AUDIO_SILENCE_ANOMALY` | WARNING | 声音缺失异常：规程要求伴随设备声音的步骤未检测到声音活动 |
| `UNKNOWN_OPERATION` | INFO | 计划外操作：观测到不属于 SOP 的操作行为 |

## 2 目录结构

```text
SOP_ALERT/
|-- README.md                 // 本文档
|-- test_e2e.sh               // 容器化端到端测试脚本（双容器：vLLM + SDK 镜像）
|-- run.sh                    // SDK 容器内的便捷启动脚本：封装 sop_alert_pipeline.py 的调用参数
|-- sop_alert_pipeline.py     // Pipeline 主入口
|-- media_processor.py        // 媒体处理层：MultimodalSDK 接口唯一调用入口
|-- step_extractor.py         // 步骤理解层：关键帧检索 + VLM 确认，产出实际步骤序列
|-- sop_comparator.py         // 比对分析层：实际步骤 vs SOP，生成违规告警
|-- report_generator.py       // 报告层：JSON / Markdown 双格式报告
|-- sop_loader.py             // SOP 文件加载与校验
|-- common.py                 // 公共工具（图像编码、VLM 调用等）
`-- data/
    `-- cases/                // 验证用例（每个用例包含 SOP + 视频）
        |-- case1/                // 用例 1（数控机床零件加工标准作业流程）
        |   |-- sop.json         // SOP 定义
        |   `-- video/           // 操作视频为ai生成
        |       `-- sop.mp4
        |-- case2/                // 用例 2（注塑机塑胶成型取料标准作业流程）
        |   |-- sop.json
        |   `-- video/
        |       `-- sop.mp4
        `-- case3/                // 用例 3（粉体配料车间投料标准作业流程）
            |-- sop.json
            `-- video/
                `-- sop.mp4
```

## 3 实践流程阐述

### 3.1 整体架构

```text
输入: 操作视频(mp4) + 标准SOP(json)
  │
  ▼
┌────────────── 媒体处理层 media_processor.py ──────────────┐
│                （MultimodalSDK 接口调用层）                │
│  ① mm.video_decode      按采样率解码视频 → RGB 帧序列      │
│  ② mm.load_audio        加载分离音轨 → 逐秒声音能量曲线    │
│  ③ mm.KRangFrameSelector 按步骤视觉描述定位帧区间+关键帧   │
└──────────────────────────┬───────────────────────────────┘
                           ▼
┌────────────── 步骤理解层 step_extractor.py ───────────────┐
│  对每个SOP步骤: 区间关键帧 + 步骤描述 → VLM(Qwen2.5-VL)    │
│  确认步骤是否真实发生，输出置信度与画面证据；               │
│  结合音频能量曲线判定区间内有无设备声音                     │
│  → 按时间排序的实际步骤观测序列                            │
└──────────────────────────┬───────────────────────────────┘
                           ▼
┌────────────── 比对分析层 sop_comparator.py ───────────────┐
│  步骤丢失检测 / 顺序异常检测(最长非降子序列) /              │
│  耗时异常检测 / 声音缺失检测 / 计划外操作检测               │
│  → 告警列表 + 合规评分(0-100)                             │
└──────────────────────────┬───────────────────────────────┘
                           ▼
┌────────────── 报告层 report_generator.py ─────────────────┐
│  结构化报告 → sop_report.json + sop_report.md             │
└───────────────────────────────────────────────────────────┘
```

### 3.2 MultimodalSDK 在 Pipeline 中做了什么

MultimodalSDK 的调用全部集中在 (media_processor.py)，承担整条流水线的多模态信息提取：

| SDK 接口 | Pipeline 环节 | 具体工作 |
|---------|--------------|---------|
| `mm.video_decode` | 视频解码 | 对输入 mp4 按目标采样帧率等间隔解码，输出 RGB `Image` 对象列表，供后续关键帧检索与 VLM 理解使用（`decode_video`） |
| `mm.load_audio` | 音频加载 | 加载从视频分离出的 wav 音轨并重采样至 16kHz、自动转单声道，输出音频 Tensor；Pipeline 基于其计算逐秒 RMS 能量，实现声音活动检测，辅助判定"启动设备/开启通风"类步骤是否真实发生（`load_audio_activity`） |
| `mm.KRangFrameSelector` | 关键帧定位 | 以每个 SOP 步骤的 `visual_query` 为查询文本，通过 CLIP 文本-图像相似度在帧序列中定位语义相关的连续场景区间，并在区间内自适应重采样关键帧，同时给出步骤发生的起止时刻（`StepRangeLocator.locate`） |

在此基础上，步骤理解层只需将 SDK 定位的少量关键帧送入 VLM 做二次确认，避免对全视频逐帧调用大模型，显著降低推理开销。

### 3.3 步骤提取与比对算法

1. **实际步骤提取**：对 SOP 的每个步骤，`KRangFrameSelector` 返回的关键帧区间即为候选时间段；VLM 根据关键帧输出 `{performed, confidence, evidence}`，置信度达标才计入观测序列，最终按开始时间排序得到"实际操作步骤序列"。
2. **步骤丢失**：SOP 中定义但未出现在观测序列中的步骤即判定丢失；`critical: true` 的关键步骤告 ERROR，其余告 WARNING。
3. **顺序异常**：将观测序列映射为 SOP 步骤下标序列，求**最长非降子序列（LIS）**；不在 LIS 中的步骤即为破坏全局顺序的乱序步骤，逐个告警并给出发生时间段。
4. **耗时异常**：观测耗时超出步骤 `[min_duration_s, max_duration_s]` 范围时告警。
5. **声音缺失**：`requires_sound: true` 的步骤在其时间段内未检测到声音活动（基于 `mm.load_audio` 的能量曲线）时告警。
6. **合规评分**：从 100 分按告警类型/级别扣分（关键步骤丢失 -25、普通步骤丢失 -15、顺序异常 -10、耗时/声音异常 -5、计划外操作 -2），直观呈现整体合规程度。

## 4 SOP 文件格式

```json
{
  "name": "数控机床零件加工标准作业流程",
  "description": "SOP-CNC-001",
  "steps": [
    {
      "id": "S1",
      "name": "佩戴防护装备",
      "description": "工人佩戴安全帽、护目镜和防护手套（供 VLM 判定）",
      "visual_query": "工人正在佩戴安全帽护目镜和防护手套（供 CLIP 关键帧检索）",
      "critical": true,
      "min_duration_s": 5,
      "max_duration_s": 120,
      "requires_sound": false
    }
  ]
}
```

`id/name/description/visual_query` 为必选字段，其余可选。完整示例见 (https://gitcode.com/Ascend/MultimodalSDK/issues/49 的 data.zip)目录。

## 5 运行方式

### 5.1 环境准备（完整流水线，推荐容器化部署）

完整流水线需要：昇腾 NPU 环境（Atlas 800I A2）+ Multimodal SDK + CLIP 模型 + VLM 推理服务 + ffmpeg。推荐容器化部署（已在 8x Ascend 910B4、驱动 24.1.0.3、Docker 26.1.3 环境实测验证）：

**双容器架构**：

- **vLLM 容器**：运行 Qwen2.5-VL-7B 推理服务（镜像：`quay.io/ascend/vllm-ascend:v0.21.0rc1`）
- **SDK 容器**：运行 Pipeline（镜像：官方 MultimodalSDK 镜像）

`test_e2e.sh` 会自动完成镜像检查、双容器创建、vLLM 服务拉起、ffmpeg 安装、依赖安装、文件权限修复并执行端到端测试。

**方式一：一键容器化部署（推荐）**

```bash
cd examples/SOP_ALERT

# 运行单个用例的端到端测试（默认 vLLM 用卡 6,7，Pipeline 用卡 4）
bash test_e2e.sh data/cases/case1          # 用例 1
bash test_e2e.sh data/cases/case2          # 用例 2
bash test_e2e.sh data/cases/case3          # 用例 3

# 自定义 NPU 卡号：通过环境变量传参
VISIBLE_DEVICES=0,1 DEVICE_ID=2 bash test_e2e.sh data/cases/case1
```

环境变量说明：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `VISIBLE_DEVICES` | `6,7` | vLLM 容器挂载的 NPU 卡号（逗号分隔），如 `0,1` 表示挂载 davinci0、davinci1 |
| `DEVICE_ID` | `2` | Pipeline 容器使用的 NPU 卡号，如 `2` 表示挂载 davinci2 |

每个用例目录（`data/cases/caseX/`）下应包含：

- `sop.json`：SOP 定义文件
- `video/sop.mp4`：操作视频文件

测试完成后，报告输出到 `output/<case_name>_<diagnosis>_<timestamp>/` 目录，例如：

- `output/case1_missing_20260824_100000/` - case1 运行，诊断结果：步骤缺失

**说明**：

- `<case_name>`：用例名称（如 `case1`、`case2`、`case3`）
- `<diagnosis>`：诊断结果（`normal` 正常 / `missing` 步骤缺失 / `wrong_order` 顺序错误 / `unknown` 未知）
- `<timestamp>`：运行时间戳（格式 `YYYYMMDD_HHMMSS`），避免多次运行互相覆盖

**方式二：容器内手动准备**

以下示例以 vLLM 用卡 6,7、Pipeline 用卡 4 为例，实际使用时请根据可用 NPU 卡号调整 `--device /dev/davinciX` 中的数字。

```bash
# 1. 启动 vLLM 容器（运行 Qwen2.5-VL-7B 推理服务）
#    示例挂载 davinci6、davinci7 给 vLLM 使用
docker run -itd --name vllm-sop-alert \
  --network=host \
  --device /dev/davinci6 --device /dev/davinci7 \
  --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /data1/sop_alert_models:/models \
  quay.io/ascend/vllm-ascend:v0.21.0rc1 bash

# 在 vLLM 容器内启动推理服务
docker exec -it vllm-sop-alert bash
export ASCEND_RT_VISIBLE_DEVICES=0,1  # 容器内视角：davinci6→0，davinci7→1
vllm serve /models/Qwen2.5-VL-7B-Instruct --served-model-name=qwen2.5-vl-7b \
    -tp=2 --max-model-len=32768 --enforce-eager --port 18002

# 2. 启动 SDK 容器（运行 Pipeline）
#    示例挂载 davinci4 给 Pipeline 使用
docker run -itd --name test-sop-alert \
  --network=host \
  --device /dev/davinci4 \
  --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /data1/sop_alert_models:/models \
  -v /root/MultimodalSDK/examples/SOP_ALERT:/workspace \
  swr.cn-south-1.myhuaweicloud.com/ascendhub/multimodalsdk:26.1.0-cann9.1.0-torch_npu2.6.0.post5-910b-ubuntu22.04-py3.12-aarch64 bash

# 在 SDK 容器内安装 ffmpeg 并设置环境
docker exec -it test-sop-alert bash
apt-get update && apt-get install -y ffmpeg
source /usr/local/multimodal/script/set_env.sh
exit

# 3. 在 SDK 容器内运行 Pipeline
docker exec -it test-sop-alert bash
cd /workspace
python3 sop_alert_pipeline.py \
    --video data/cases/case1/video/sop.mp4 \
    --sop data/cases/case1/sop.json \
    --clip-model-path /models/chinese-clip-vit-large-patch14-336px \
    --device-id 0 \
    --clip-model-type cn_clip \
    --vlm-url http://127.0.0.1:18002/v1 \
    --vlm-model-name qwen2.5-vl-7b \
    --sample-num 12 \
    --output-dir ./output \
    --print-markdown
```

> **NPU 卡号说明**：上述示例中 vLLM 容器挂载宿主机的 davinci6、davinci7，SDK 容器挂载 davinci4。如果你的环境卡号不同，请修改 `--device /dev/davinciX` 中的数字。容器内视角下，挂载的设备会重新编号（如只挂载 davinci6、davinci7，则容器内看到 davinci0、davinci1），`ASCEND_RT_VISIBLE_DEVICES` 和 `--device-id` 参数需使用容器内的编号。
> **文件权限注意**：SDK 对输入文件做安全校验，要求 other 组无任何权限（目录 ≤750、文件 ≤640），否则 `mm.video_decode`/`mm.load_audio`/CLIP 模型加载会报 `CheckFilePermission` 类错误。`test_e2e.sh` 已自动在宿主机上修复权限；手动运行时需对视频/SOP 执行 `chmod 640`，对 CLIP 模型目录执行 `find <模型目录> -type d -exec chmod 750 {} \;` 与 `find <模型目录> -type f -exec chmod 640 {} \;`。

### 5.2 一键运行（完整流水线）

```bash
cd examples/SOP_ALERT

CLIP_MODEL_PATH=/models/chinese-clip-vit-large-patch14-336px \
DEVICE_ID=4 \
VLM_URL=http://127.0.0.1:18002/v1 \
VLM_MODEL_NAME=qwen2.5-vl-7b \
bash run.sh --video data/cases/case1/video/sop.mp4 --sop data/cases/case1/sop.json
```

也可直接调用主入口自定义参数：

```bash
python3 sop_alert_pipeline.py \
    --video data/cases/case1/video/sop.mp4 \
    --sop data/cases/case1/sop.json \
    --clip-model-path /models/chinese-clip-vit-large-patch14-336px \
    --device-id 4 \
    --vlm-url http://127.0.0.1:18002/v1 \
    --vlm-model-name qwen2.5-vl-7b \
    --sample-fps 1.0 --sample-num 12 \
    --output-dir ./output
```

### 5.3 实测结果

使用 `data/cases/` 下的工业场景视频，在 8x Ascend 910B4 + Qwen2.5-VL-7B 环境下实测：

| 用例 |  实测结果 |
|------|--------|
| `case1` | output/case1_missing_20260824_033636/ |
| `case2` | output/case2_missing_20260825_023255/ |
| `case3` | output/case3_missing_20260824_034106/ |

## 6 结果呈现

Pipeline 端到端输出两种格式的分析报告（默认写入 `./output`）：

- `sop_report.json`：结构化报告，含输入信息、SOP 摘要、实际观测步骤序列（时间段/置信度/证据/声音检测）、合规评分与告警明细；
- `sop_report.md`：人类可读报告，含总体结论、SOP 步骤执行情况表、实际步骤序列表和违规告警明细表。

终端输出示例（case1 步骤丢失）：

```text
[1/4] mm.video_decode 解码视频: /cases/case3/video/sop.mp4 (采样率 1.0 fps)
      解码得到 35 帧，视频时长 35.5s
[2/4] mm.load_audio 加载音轨并进行声音活动检测
      音轨加载成功
[3/4] mm.KRangFrameSelector 定位步骤区间 + VLM 逐步骤确认（共 6 个 SOP 步骤）
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
        - Avoid using `tokenizers` before the fork if possible
        - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
      提取到 5 个实际操作步骤
[4/4] SOP 比对分析与报告生成

合规评分: 60/100  结论: 存在违规  告警数: 3
  [WARNING] STEP_ORDER_VIOLATION: 步骤“检查投料设备与除尘系统”(S2) 执行顺序异常：SOP 规定为第 2 步，实际执行时序与规程不符。
  [WARNING] STEP_DURATION_ANOMALY: 步骤“现场清洁与记录”(S6) 耗时 10.1s，超过规程允许的最长 5.0s，疑似操作卡滞或异常。
  [ERROR] STEP_MISSING: 关键步骤“原料核对与拆包”(S3) 未在视频中检测到，疑似漏做。

JSON 报告:     /workspace/output/temp_analysis/sop_report.json
Markdown 报告: /workspace/output/temp_analysis/sop_report.md
```
