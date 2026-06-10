# Qwen2.5-VL SCC 视觉 Token 压缩使用说明

本文档介绍如何在 vllm-ascend v0.18.0 中应用 SCC 视觉 token 压缩补丁、启动 Qwen2.5-VL 服务，并完成基础功能验证和 VideoMME 性能评估。

## SCC 实现介绍

SCC（Semantic Connected Components）用于压缩 Qwen2.5-VL 的视觉 token，目标是在尽量保持模型效果的前提下降低视觉 token 数量，从而提升多模态推理吞吐。

该补丁的核心改动包括：

1. 在 `vllm_ascend/ops/scc_compressor.py` 中新增 SCC 压缩算法，对视觉 embedding 按语义相似度构建连通分量，并将每个分量聚合为代表 token。
2. 在 Processor 侧按 `K = ceil(N * ratio)` 缩短 image/video placeholder 数量，确保 prompt 中的视觉占位符数量与压缩后的视觉 embedding 数量一致。
3. 在 Model 侧包装 Qwen2.5-VL 的 `_process_image_input` 和 `_process_video_input`，对 ViT 输出做 SCC 压缩。
4. 适配 Qwen2.5-VL 的 mROPE 位置计算，避免压缩后视觉 embedding 数量和位置数量不一致。
5. 在 EngineCore/worker 子进程中重新应用 monkey patch，确保服务端推理链路中 SCC 生效。

SCC 完全在服务端执行，客户端请求格式无需修改。

## 前置准备

1、 准备配备昇腾 NPU 的硬件服务器，例如 Atlas 800 A2 系列。

2、 准备可正常工作的 vllm-ascend v0.18.0 环境。安装方法可参考 [vllm-ascend 安装指南](https://docs.vllm.ai/projects/ascend/zh-cn/v0.18.0/installation.html)，建议使用官方 [Docker 镜像](https://quay.io/repository/ascend/vllm-ascend?tab=tags)：

```bash
docker pull quay.io/ascend/vllm-ascend:v0.18.0
```

3、 下载 Qwen2.5-VL 模型权重，例如 `Qwen2.5-VL-7B-Instruct`。可参考 [Qwen2.5-VL-7B-Instruct ModelScope](https://modelscope.cn/models/Qwen/Qwen2.5-VL-7B-Instruct)。

4、 如需进行 VideoMME 性能评估，准备 `lmms-eval` 测试框架：

```bash
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval.git
cd lmms-eval
pip install qwen_vl_utils accelerate decord2 sacrebleu evaluate pytz tenacity pytablewriter pytest
```

`lmms-eval` 默认会从 Hugging Face 下载数据集。离线或网络受限环境可提前下载 [VideoMME 数据集 ModelScope](https://modelscope.cn/datasets/lmms-lab/Video-MME)。

## 快速开始

### 1. 应用补丁

SCC 补丁已按目标文件拆分到 `examples/SCC/patches` 目录下，文件名中的两位数字表示推荐应用顺序。请在 vllm-ascend 仓库根目录按顺序应用这些 patch。

```bash
cd /path/to/vllm-ascend
git apply --check /path/to/MultimodalSDK/examples/SCC/patches/*.patch
git apply /path/to/MultimodalSDK/examples/SCC/patches/*.patch
```

如果当前就在 `MultimodalSDK` 仓库旁边，也可以使用相对路径，例如：

```bash
cd /path/to/vllm-ascend
git apply --check ../MultimodalSDK/examples/SCC/patches/*.patch
git apply ../MultimodalSDK/examples/SCC/patches/*.patch
```

也可以只应用原始完整补丁：

```bash
cd /path/to/vllm-ascend
git apply /path/to/MultimodalSDK/examples/SCC/vllm-ascend-v0.18.0-scc.patch
```

拆分后的 patch 与原始完整 patch 内容一致，只是为了便于审阅和分阶段应用。

如果曾经用循环方式应用过并出现失败，请先在 vllm-ascend 仓库中恢复到应用补丁前的干净状态，再重新执行上面的 `git apply --check` 和 `git apply`。否则已经应用成功的前半部分 patch 会导致重复应用或上下文不匹配。

### 2. 启动服务

补丁会新增 `start_vllm_server.sh`。启动前请确认脚本中的 `MODEL_PATH` 指向可用的 Qwen2.5-VL 模型目录，也可以通过环境变量覆盖。

基线模式默认关闭 SCC：

```bash
cd /path/to/vllm-ascend
MODEL_PATH=/path/to/Qwen2.5-VL-7B-Instruct bash start_vllm_server.sh
```

启用 SCC：

```bash
cd /path/to/vllm-ascend
MODEL_PATH=/path/to/Qwen2.5-VL-7B-Instruct VLLM_ASCEND_ENABLE_SCC=1 bash start_vllm_server.sh
```

服务启动日志中会打印 SCC 状态：

```text
=== Starting vLLM server ===
  Model:    /path/to/Qwen2.5-VL-7B-Instruct
  Endpoint: http://0.0.0.0:8000/v1
  TP size:  2
  HF_HOME:  /root/.cache/huggingface
  SCC:      ON (ratio=0.5, tau=0.98, epsilon=0.05)
============================
```

`start_vllm_server.sh` 默认在 `http://0.0.0.0:8000/v1` 暴露 OpenAI 兼容 API，并使用 `--tensor-parallel-size 2`。如需调整 `HOST`、`PORT`、`TP_SIZE`，可通过环境变量覆盖。

### 3. 配置 SCC 参数

SCC 通过环境变量控制，需在启动 vLLM 服务前设置。

| 变量 | vllm-ascend 默认值 | 范围 | 描述 |
|---|---|---|---|
| `VLLM_ASCEND_ENABLE_SCC` | `0` | `0` 或 `1` | SCC 主开关，`1` 表示启用。 |
| `VLLM_ASCEND_SCC_RATIO` | `0.3` | `(0, 1]` | 保留的视觉 token 比例，`K = ceil(N * ratio)`。 |
| `VLLM_ASCEND_SCC_TAU` | `0.95` | `(0, 1]` | 余弦相似度阈值，值越高，聚类越稀疏。 |
| `VLLM_ASCEND_SCC_EPSILON` | `0.05` | `(0, 1)` | CPU fallback 近似连通分量采样误差容限，值越小越精确但越慢。 |
| `VLLM_ASCEND_SCC_MAX_TOKENS_PER_ITEM` | `8192` | `>= 0` | 单个图片或视频项目超过该 token 数时跳过压缩，`0` 表示不限制。 |

示例：

```bash
# 启用 SCC，使用启动脚本默认参数
VLLM_ASCEND_ENABLE_SCC=1 bash start_vllm_server.sh

# 启用 SCC，并自定义压缩参数
VLLM_ASCEND_ENABLE_SCC=1 \
VLLM_ASCEND_SCC_RATIO=0.5 \
VLLM_ASCEND_SCC_TAU=0.98 \
VLLM_ASCEND_SCC_EPSILON=0.05 \
bash start_vllm_server.sh
```

## 功能验证

### 1. 单元测试

单元测试用于验证 SCC 算法和 patch 辅助逻辑，不依赖真实模型推理。

```bash
cd /path/to/vllm-ascend

# SCC 算法
pytest tests/ut/ops/test_scc_compressor.py

# SCC patch 辅助逻辑
pytest tests/ut/patch/platform/test_patch_engine_core_scc.py
```

### 2. 分层冒烟测试

补丁中新增了 `tests/scc` 目录，用于验证 SCC 集成链路。

```bash
cd /path/to/vllm-ascend

# Layer 1：仅验证算法，CPU 可运行，无需启动 vLLM 服务
python tests/scc/test_layer1_algorithm.py

# Layer 2：验证补丁注册和启用路径
VLLM_ASCEND_ENABLE_SCC=1 python tests/scc/test_layer2_patch.py

# Layer 2：验证补丁禁用路径
VLLM_ASCEND_ENABLE_SCC=0 python tests/scc/test_layer2_patch.py
```

端到端冒烟测试需要先启动服务，再发送请求：

```bash
# 终端 1：启动带 SCC 的服务
cd /path/to/vllm-ascend
VLLM_ASCEND_ENABLE_SCC=1 bash start_vllm_server.sh

# 终端 2：发送测试请求
cd /path/to/vllm-ascend
bash tests/scc/curl_smoke.sh
```

请求完成后，检查客户端日志中出现以下日志，则表示用例通过：

```text
=== Request completed. Now check the server log for [SCC] lines. ===
```

### 3. 简单 OpenAI API 请求

可以使用如下脚本验证 vLLM 服务是否可正常响应。该脚本只验证服务可用性；SCC 是否生效仍以服务端 `[SCC]` 日志为准。

```python
import json
import urllib.request

BASE_URL = "http://localhost:8000"
MODEL = "/path/to/Qwen2.5-VL-7B-Instruct"

messages = [
    {
        "role": "user",
        "content": "请用三句话介绍量子计算",
    }
]

payload = json.dumps({
    "model": MODEL,
    "messages": messages,
    "max_tokens": 128,
}).encode("utf-8")

req = urllib.request.Request(
    f"{BASE_URL}/v1/chat/completions",
    data=payload,
    headers={"Content-Type": "application/json"},
)

with urllib.request.urlopen(req, timeout=120) as resp:
    result = json.loads(resp.read())

print(result["choices"][0]["message"]["content"])
print(result.get("usage", {}))
```

## 性能验证

性能评估建议使用同一套模型、同一份数据集，分别在基线模式和 SCC 模式下运行 `lmms-eval`，对比 `value` 和 `avg_speed`。

### 1. 启动服务端

基线模式：

```bash
cd /path/to/vllm-ascend
MODEL_PATH=/path/to/Qwen2.5-VL-7B-Instruct bash start_vllm_server.sh
```

SCC 模式：

```bash
cd /path/to/vllm-ascend
MODEL_PATH=/path/to/Qwen2.5-VL-7B-Instruct \
VLLM_ASCEND_ENABLE_SCC=1 \
VLLM_ASCEND_SCC_RATIO=0.5 \
VLLM_ASCEND_SCC_TAU=0.98 \
VLLM_ASCEND_SCC_EPSILON=0.05 \
bash start_vllm_server.sh
```

### 2. 运行 lmms-eval

在 `lmms-eval` 仓库中创建 `qwen25vl_vllm.sh`：

```bash
#!/bin/bash
set -euo pipefail

TASK=videomme
MODEL_NAME=/path/to/Qwen2.5-VL-7B-Instruct
VLLM_HOST=localhost
VLLM_PORT=8000
RUN_TAG="${RUN_TAG:-baseline}"
LIMIT="${LIMIT:-200}"

export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export DECORD_NUM_THREADS=1
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false

OUTPUT_DIR=./logs/qwen25vl_vllm_${TASK}_${RUN_TAG}

python3 -m lmms_eval \
    --model openai \
    --model_args=model_version=${MODEL_NAME},base_url=http://${VLLM_HOST}:${VLLM_PORT}/v1,api_key=EMPTY,max_frames_num=16,timeout=300,num_concurrent=1 \
    --tasks ${TASK} \
    --batch_size 1 \
    --limit ${LIMIT} \
    --log_samples \
    --output_path ${OUTPUT_DIR}
```

运行：

```bash
cd /path/to/lmms-eval

# 基线结果
RUN_TAG=baseline bash qwen25vl_vllm.sh

# SCC 结果
RUN_TAG=scc_ratio0.5_tau0.98_eps0.05 bash qwen25vl_vllm.sh
```

建议多轮运行后取平均值。通常关注：

1. `value`：精度指标，建议确认相对基线无明显下降。
2. `avg_speed`：速度指标，观察 SCC 启用后的提升幅度。

### 3. 离线数据集说明

如果无法在线下载 VideoMME，可先下载数据集到本地，并通过环境变量指定本地路径：

```bash
export LMMS_LOCAL_DATASET_PATH=/path/to/Video-MME
export LMMS_LOCAL_VIDEOMME_PATH=/path/to/Video-MME
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

不同版本的 `lmms-eval` 对本地数据集路径的支持可能不同。如果环境变量无法生效，需要按当前 `lmms-eval` 版本调整 `lmms_eval/api/task.py` 的数据集加载逻辑，使其从本地 parquet 或缓存目录加载 VideoMME，以下给个修改示例：

```text
vim lmms-eval/lmms_eval/api/task.py

# 第一处，找到以下代码：
dataset_kwargs["From_YouTube"] = True
cache_path = snapshot_download(repo_id=self.DATASET_PATH, repo_type="dataset")  # download_parquet
# 修改成：
dataset_kwargs["From_YouTube"] = True
local_dataset_path = os.environ.get("LMMS_LOCAL_DATASET_PATH")
if local_dataset_path:
    cache_path = local_dataset_path
    print(f"[LMMS] Use local dataset YouTube cache path: {local_dataset_path}")
else:
    cache_path = snapshot_download(repo_id=self.DATASET_PATH, repo_type="dataset")  # download_parquet
# 第二处，找到以下代码：
if not os.path.exists(cache_dir) or (create_link and os.path.islink(cache_dir)):
    cache_path = snapshot_download(repo_id=self.DATASET_PATH, revision=revision, repo_type="dataset", force_download=force_download, etag_timeout=60)
# 修改成：
if not os.path.exists(cache_dir) or (create_link and os.path.islink(cache_dir)):
    local_dataset_path = os.environ.get("LMMS_LOCAL_DATASET_PATH")
    if local_dataset_path:
        cache_path = local_dataset_path
        print(f"[LMMS] Use local dataset cache path: {local_dataset_path}")
    else:
        cache_path = snapshot_download(repo_id=self.DATASET_PATH, revision=revision, repo_type="dataset", force_download=force_download, etag_timeout=60)
# 第三处，找到以下代码：
load_dataset_cache_dir = load_dataset_kwargs.pop("cache_dir", resolved_dataset_cache_dir)
self.dataset = datasets.load_dataset(
    path=self.DATASET_PATH,
    name=self.DATASET_NAME,
    cache_dir=load_dataset_cache_dir,
    download_mode=datasets.DownloadMode.REUSE_DATASET_IF_EXISTS,
    download_config=download_config,
    num_proc=1,
    **load_dataset_kwargs,
)
# 修改成：
load_dataset_cache_dir = load_dataset_kwargs.pop("cache_dir", resolved_dataset_cache_dir)
local_videomme_path = os.environ.get("LMMS_LOCAL_VIDEOMME_PATH")

if local_videomme_path and self.DATASET_PATH == "lmms-lab/Video-MME":
    parquet_file = os.path.join(
        local_videomme_path,
        "videomme",
        "test-00000-of-00001.parquet",
    )
    print(f"[LMMS] Loading local Video-MME parquet: {parquet_file}")

    self.dataset = datasets.load_dataset(
        "parquet",
            data_files={"test": parquet_file},
    )
else:
    self.dataset = datasets.load_dataset(
        path=self.DATASET_PATH,
        name=self.DATASET_NAME,
        cache_dir=load_dataset_cache_dir,
        download_mode=datasets.DownloadMode.REUSE_DATASET_IF_EXISTS,
        download_config=download_config,
        num_proc=1,
        **load_dataset_kwargs,
    )
```

## 常见问题

### 看不到任何 `[SCC]` 日志

通常说明 SCC patch 没有加载。请确认：

1. 补丁已成功应用。
2. 启动服务时设置了 `VLLM_ASCEND_ENABLE_SCC=1`。
3. 服务端日志中出现了 SCC 启动 banner。

### 只看到 SCC 启动日志，看不到压缩日志

通常说明请求没有触发图片或视频路径。请使用 `tests/scc/curl_smoke.sh` 或包含 `image_url` 的请求进行验证。

### 服务启动时内存压力过大

`start_vllm_server.sh` 已限制 OMP/BLAS/tokenizer 线程数，并设置了 `--max-num-seqs 1`、`--max-model-len 12288`、`--limit-mm-per-prompt '{"image":16,"video":0}'`。如果仍然 OOM，可继续降低 `TP_SIZE`、`max_model_len` 或并发设置。
