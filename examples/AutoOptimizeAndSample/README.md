# AutoOptimizeAndSample — Video RAG 参考设计

## 1. 概述

本参考设计实现了一个基于 BentoML 的 Video RAG（检索增强生成）系统，用于视频理解与问答。系统通过多模态处理流水线对视频进行帧提取、语音识别、OCR、目标检测和语义检索，将检索结果作为上下文增强 LLM 的生成质量。

系统采用**双容器部署架构**：

- **vLLM 容器**：运行 Qwen2.5-VL 多模态大模型推理服务
- **VRAG 容器**：运行 BentoML 微服务集群，负责视频处理、检索和编排

两个容器通过 OpenAI 兼容 API 通信。

## 2. 系统架构

### 2.1 整体架构

```text
视频输入
  │
  ▼
┌─────────────────────── VRAG 容器 ───────────────────────┐
│                                                         │
│  VideoProcessService ──► 帧提取 + 音频提取               │
│       │                                                 │
│       ▼                                                 │
│  VideoTranscribeService                                 │
│       ├── WhisperService ──► ASR 语音转文字              │
│       └── MineruService ──► OCR 文字识别                 │
│       │                                                 │
│       ▼                                                 │
│  DetectionService                                       │
│       ├── AksBlipService ──► 自适应关键帧选择            │
│       │     └── BlipService ──► 图文相似度计算           │
│       └── MMDINOService ──► 目标检测                     │
│       │                                                 │
│       ▼                                                 │
│  FaissService ──► 语义向量检索                           │
│       └── QwenEmbeddingService ──► 文本嵌入              │
│       │                                                 │
│       ▼                                                 │
│  QwenRerankerService ──► 检索结果重排序                  │
│       │                                                 │
│       ▼                                                 │
│  VideoRagService ──► Prompt 组装 + LLM 调用 ──────────┐ │
│                                                     │ │
└─────────────────────────────────────────────────────┼─┘
                                                      │
                                          OpenAI API  │
                                                      │
┌─────────────────────────────────────────────────────┼─┐
│                                                     ▼ │
│               vLLM 容器                               │
│                                                         │
│  Qwen2.5-VL-32B-Instruct ──► 多模态推理生成             │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 2.2 服务依赖关系

```text
VideoRagService
  ├── VideoRetrievalService
  │     ├── VideoTranscribeService
  │     │     ├── VideoProcessService
  │     │     ├── WhisperService
  │     │     └── MineruService
  │     ├── DetectionService
  │     │     ├── AksBlipService
  │     │     │     └── BlipService
  │     │     └── MMDINOService
  │     ├── FaissService
  │     │     └── QwenEmbeddingService
  │     └── QwenRerankerService
  └── QwenVLService → 外部 vLLM 容器（OpenAI API）
```

## 3. 环境要求

### 3.1 硬件要求

- **NPU**：Atlas800I A2(Ascend 910B， 4 卡)
- **内存**：≥ 128 GB
- **磁盘**：≥ 400 GB（模型权重 + 缓存 + 数据集）

### 3.2 软件依赖

- 操作系统：Ubuntu 22.04
- Python：3.11
- CANN：9.0.0
- Docker

## 4. 部署指南

### 4.1 镜像准备

#### 4.1.1 镜像

vrag镜像 `swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.0-910b-ubuntu22.04-py3.11`
大模型镜像 `quay.io/vllm/vllm-ascend:v0.15.0rc1`

```bash
docker pull swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.0-910b-ubuntu22.04-py3.11
docker pull quay.io/vllm/vllm-ascend:v0.15.0rc1
```

### 4.2 准备模型权重

用户需预先下载以下 6 个模型，并放置到可挂载的目录中：

| 模型名称 | 配置字段 | 加载位置 | 说明 |
|----------|----------|----------|------|
| [Qwen2.5-VL-32B-Instruct](https://modelscope.cn/models/Qwen/Qwen2.5-VL-32B-Instruct) | `qwenvl_model_name` + vLLM `MODEL_PATH` | vLLM 容器 | 多模态大模型，通过 vLLM API 调用 |
| [whisper-large-v3-turbo](https://modelscope.cn/models/openai-mirror/whisper-large-v3-turbo) | `whisper_model_path` | VRAG 容器 | 语音识别 |
| [mm_grounding_dino_tiny_o365v1_goldg_v3det](https://hf-mirror.com/openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_v3det) | `mmdino_model_path` | VRAG 容器 | 目标检测 |
| [Qwen3-Embedding-0.6B](https://modelscope.cn/models/Qwen/Qwen3-Embedding-0.6B) | `embedding_model_path` | VRAG 容器 | 文本嵌入 |
| [Qwen3-Reranker-0.6B](https://modelscope.cn/models/Qwen/Qwen3-Reranker-0.6B) | `reranker_model_path` | VRAG 容器 | 检索重排序 |
| [blip2-itm-vit-g-coco](https://modelscope.cn/models/Salesforce/blip2-itm-vit-g-coco) | `blip_model_path` | VRAG 容器 | 图文相似度 |

建议目录结构：

```text
/models/
  ├── Qwen2.5-VL-32B-Instruct/
  ├── whisper-large-v3-turbo/
  ├── mm_grounding_dino_tiny_o365v1_goldg_v3det/
  ├── Qwen3-Embedding-0.6B/
  ├── Qwen3-Reranker-0.6B/
  └── blip2-itm-vit-g-coco/
```

### 4.3 启动 vLLM 推理容器

```bash
docker run -it \
  --name vllm-server \
  --privileged \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /models/:/models/ \
  -p 8000:8000 \
  quay.io/vllm/vllm-ascend:v0.15.0rc1 \
  /bin/bash
```

进入容器后，手动启动 vLLM 服务：

```bash
vllm serve /models/Qwen2.5-VL-32B-Instruct \
  --served-model-name=qwen2.5-vl-32b \
  --max-model-len 120000 \
  --gpu-memory-utilization 0.70 \
  --tensor-parallel-size 4 \
  --no-enable-prefix-caching \
  --host 0.0.0.0 \
  --port 8000
```

等待模型加载完成后，验证 API 可用：

```bash
curl http://<宿主机IP>:8000/v1/models
```

### 4.4 启动 VRAG 服务容器

```bash
docker run -it \
  --name vrag \
  --privileged \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /models/:/models/ \
  swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.0-910b-ubuntu22.04-py3.11 \
  /bin/bash
```

进入容器后，克隆代码并安装依赖：

```bash
# 克隆代码
git clone https://gitcode.com/Ascend/MultimodalSDK.git
cd MultimodalSDK/examples/AutoOptimizeAndSample

# 设置 PYTHONPATH
export PYTHONPATH=$(pwd):$PYTHONPATH

# 安装系统依赖
apt-get update && apt-get install -y libgl1 libglib2.0-0 libaio-dev ffmpeg libavcodec-extra libopenblas-dev swig

# 安装 Python 依赖
pip install -r requirements.txt
```

## 5. 配置说明

### 5.1 命令行参数

**VideoRag 服务**

```bash
python -m vrag.benchmark.video_rag \
  --config, -c    # TOML 配置文件路径
  --host, -H      # 服务监听地址，默认 0.0.0.0
  --port, -p      # 服务监听端口，默认 7860
```

**Baseline 服务**

```bash
python -m vrag.benchmark.baseline \
  --config, -c    # TOML 配置文件路径
  --host, -H      # 服务监听地址，默认 0.0.0.0
  --port, -p      # 服务监听端口，默认 7861
```

**评估脚本**

```bash
python -m vrag.benchmark.evaluation \
  --config, -c          # VideoMME 测试配置路径
  --datasets, -d        # VideoMME 数据集路径
  --host, -H            # VRAG/Baseline 服务地址，默认 http://localhost:7860
  --timeout, -t         # 请求超时时间（秒），默认 3600
  --output, -o          # 输出目录，默认 output
  --max_concurrent      # 最大并发数，默认 10
```

### 5.2 TOML 配置文件

系统支持通过 TOML 配置文件覆盖默认参数。配置文件通过 `--config` 参数传入，与默认值深度合并。

示例配置文件 `vrag_config.toml`，包含所有可配置参数及其默认值：

```toml
# ============================================================
# AutoOptimizeAndSample — Video RAG 配置文件
# 所有值均为默认值，按需修改
# ============================================================

# ---- 全局服务配置 ----
service_timeout = 3600.0                       # 所有服务的请求超时时间（秒）
service_max_concurrency = 40                   # 最大并发请求数
service_connection_request_multiplier = 2.0    # Runner 连接请求倍数

# ---- BLIP2 图文相似度 ----
blip_model_path = ""                           # BLIP2 模型本地路径（必填，如 /models/blip2-itm-vit-g-coco）
blip_device = "npu:1"                          # 推理设备
blip_batch_size = 68                           # 推理批大小
blip_cache_size = 4096                         # LRU 缓存容量

# ---- AKS 自适应关键帧选择 ----
default_target_frame_count = 24                # 目标关键帧数
default_max_recursion_depth = 5                # 最大递归深度
default_mean_diff_threshold = 0.05             # 均值差异阈值（低于此值不再分割）
default_std_dev_threshold = 0                  # 标准差阈值（低于此值不再分割）

# ---- Whisper 语音识别 ----
whisper_model_path = ""                        # Whisper 模型本地路径（必填，如 /models/whisper-large-v3-turbo）
whisper_device = "npu:0"                       # 推理设备
whisper_batch_size = 80                        # 推理批大小
whisper_cache_size = 4096                      # LRU 缓存容量

# ---- MinerU OCR ----
mineru_device = "npu:0"                        # 推理设备
mineru_batch_ratio = 12                        # 批处理比率
mineru_cache_size = 4096                       # LRU 缓存容量
default_formula_enable = false                 # 是否启用公式检测
default_table_enable = false                   # 是否启用表格检测
default_lang = "ch_lite"                       # OCR 识别语言
default_line_threshold_ratio = 0.6             # OCR 行聚类阈值比率

# ---- MMDINO 目标检测 ----
mmdino_model_path = ""                         # MMDINO 模型本地路径（必填，如 /models/mm_grounding_dino_tiny_o365v1_goldg_v3det）
mmdino_device = "npu:2"                        # 推理设备
mmdino_batch_size = 8                          # 推理批大小
mmdino_cache_size = 4096                       # LRU 缓存容量
default_mmdino_threshold = 0.43                # 目标检测置信度阈值

# ---- Qwen Embedding 文本嵌入 ----
embedding_model_path = ""                      # Qwen Embedding 模型本地路径（必填，如 /models/Qwen3-Embedding-0.6B）
embedding_device = "npu:1"                     # 推理设备
embedding_batch_size = 7                       # 推理批大小
embedding_cache_size = 4096                    # LRU 缓存容量
default_normalize = true                       # 是否 L2 归一化嵌入向量

# ---- Qwen Reranker 重排序 ----
reranker_model_path = ""                       # Qwen Reranker 模型本地路径（必填，如 /models/Qwen3-Reranker-0.6B）
reranker_device = "npu:3"                      # 推理设备
reranker_batch_size = 4                        # 推理批大小
default_max_length = 8192                      # 最大 token 长度
default_top_k = 5                              # 重排序后返回的 top-k 文档数

# ---- QwenVL vLLM API ----
qwenvl_api_base = "http://<宿主机IP>:8000/v1"   # vLLM API 地址（跨容器访问，填写宿主机 IP）
qwenvl_api_key = "EMPTY"                       # API Key
qwenvl_model_name = "qwen2.5-vl-32b"          # 模型名称标识
default_max_completion_tokens = 1            # 最大生成 token 数
default_temperature = 0.0                      # 采样温度
default_top_p = 1.0                            # Top-p 采样参数
default_seed = 42                              # 随机种子
default_timeout = 3600                         # 请求超时时间（秒）

# ---- 视频处理 ----
video_process_cache_dir = "cache_store/video_process"  # 视频提取结果缓存目录
video_process_cache_dir_lock_timeout = 300     # 缓存目录文件锁超时（秒）
decord_workers = 0                             # decord 工作线程数，0 为自动检测
default_extract_frames = true                  # 是否提取视频帧
default_extract_audio = true                   # 是否提取音频
default_max_frames_num = 84                   # 最大采样帧数
default_fps = 0.05                              # 采样帧率
default_force_sample = true                    # 是否强制均匀采样
default_resolution = 720                       # 目标分辨率（高度像素）
default_audio_chunk_length = 30                # 音频分块长度（秒）
default_min_chunk_threshold = 1.0              # 尾部音频块最小保留时长（秒）
default_audio_sample_rate = 16000              # 音频重采样率（Hz）

# ---- 视频转录 ----
video_transcribe_cache_size = 4096             # LRU 缓存容量
default_use_ocr = false                        # 是否启用 OCR 提取
default_use_asr = true                        # 是否启用 ASR 提取
default_ocr_dedup = true                       # OCR 前是否去重帧
default_ocr_dedup_threshold = 2                # OCR 帧去重 Hamming 距离阈值
default_ocr_dedup_block_size = 12              # OCR 帧去重感知哈希块大小

# ---- 检测服务 ----
detection_cache_size = 4096                    # LRU 缓存容量
default_use_det = true                         # 是否启用目标检测
default_det_dedup_frames = true                # 检测前是否去重帧
default_det_dedup_threshold = 2                # 检测帧去重 Hamming 距离阈值
default_det_dedup_block_size = 12              # 检测帧去重感知哈希块大小
default_det_location = true                    # 是否包含位置描述
default_det_relation = true                    # 是否包含空间关系描述
default_det_number = true                      # 是否包含物体计数描述
default_retrieve_frame_only = false             # 是否仅检索关键帧（不生成场景描述）

# ---- FAISS 向量检索 ----
default_faiss_threshold = 0.25                 # FAISS 余弦相似度阈值
default_faiss_enable_dedup = true              # 是否启用检索结果去重
default_faiss_dedup_threshold = 0.97           # 去重余弦相似度阈值
default_faiss_dedup_lap = 2                    # 去重迭代次数
default_faiss_sparse_search = false            # 是否使用稀疏搜索（按查询独立搜索）

# ---- 视频检索 ----
retrieval_cache_size = 4096                    # LRU 缓存容量
default_ocr_discard_min_length = 36             # OCR 文档最小文本长度
default_asr_discard_min_length = 5             # ASR 文档最小文本长度
default_retrieval_enable_fallback = true       # 检索无结果时是否回退均匀采样
default_retrieval_fallback_uniform_samples_k = 32  # 回退均匀采样帧数
default_retrieval_infer_always_use_frames = false  # 是否始终在推理中包含帧
default_retrieval_dedup_related_frames = true  # 是否对检索相关帧去重
default_retrieval_dedup_related_frames_threshold = 2   # 相关帧去重 Hamming 距离阈值
default_retrieval_dedup_related_frames_block_size = 12 # 相关帧去重感知哈希块大小
default_retrieval_ocr_top_k = 3                # OCR 文档检索 top-k
default_retrieval_ocr_retrieve_span_length = 0 # OCR 文档上下文扩展长度，0 不扩展
default_retrieval_ocr_related_frames = true    # 是否为 OCR 文档附加相关帧
default_retrieval_ocr_related_frames_top_k = 1 # 每条 OCR 文档附加相关帧数
default_retrieval_always_related_asr_docs = false  # 是否始终附加检测帧相关 ASR 文档
default_retrieval_asr_top_k = 84               # ASR 文档检索 top-k
default_retrieval_asr_retrieve_span_length = 0 # ASR 文档上下文扩展长度，0 不扩展
default_retrieval_asr_related_frames = true    # 是否为 ASR 文档附加相关帧
default_retrieval_asr_related_frames_top_k = 10  # 每条 ASR 文档附加相关帧数
default_retrieval_asr_max_related_frames = 1   # 每条 ASR 文档最大相关帧数，0 不限

# ---- Video RAG ----
default_det_retrieval_frames_only = false       # 是否仅使用检测检索帧（不含 OCR/ASR 指令）
default_rag_discard_empty_detection = true      # 是否丢弃空检测结果
default_return_retrieval_result = false         # 是否在响应中包含检索结果
```

### 5.3 设备分配参考

以下为 4 卡 910B 的推荐分配方案，VL模型和小模型会共同利用 4 卡：

**vLLM 容器（4 卡）**

| NPU | 用途 |
|-----|------|
| npu:0 | Qwen2.5-VL (TP=4) |
| npu:1 | Qwen2.5-VL (TP=4) |
| npu:2 | Qwen2.5-VL (TP=4) |
| npu:3 | Qwen2.5-VL (TP=4) |

**VRAG 容器（4 卡）**

| NPU | 用途 |
|-----|------|
| npu:0 | Whisper + MinerU OCR |
| npu:1 | BLIP2 + Qwen Embedding |
| npu:2 | MMDINO |
| npu:3 | Qwen Reranker |

## 6. 快速开始

### 6.1 启动 vLLM 服务

```bash
# 启动 vLLM 容器并进入
docker run -it \
  --name vllm-server \
  --privileged \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /models/:/models/ \
  -p 8000:8000 \
  quay.io/vllm/vllm-ascend:v0.15.0rc1 \
  /bin/bash

# 容器内启动 vLLM 服务
vllm serve /models/Qwen2.5-VL-32B-Instruct \
  --served-model-name=qwen2.5-vl-32b \
  --max-model-len 120000 \
  --gpu-memory-utilization 0.70 \
  --tensor-parallel-size 4 \
  --no-enable-prefix-caching \
  --host 0.0.0.0 \
  --port 8000

# 等待模型加载完成，验证 API
curl http://<宿主机IP>:8000/v1/models
```

### 6.2 启动 VRAG 服务

```bash
# 启动 VRAG 容器并进入
docker run -it \
  --name vrag \
  --privileged \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /models/:/models/ \
  swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.0-910b-ubuntu22.04-py3.11 \
  /bin/bash

# 容器内克隆代码并安装依赖
git clone https://gitcode.com/Ascend/MultimodalSDK.git
cd MultimodalSDK/examples/AutoOptimizeAndSample
export PYTHONPATH=$(pwd):$PYTHONPATH
apt-get update && apt-get install -y libgl1 libglib2.0-0 libaio-dev ffmpeg libavcodec-extra libopenblas-dev swig
pip install -r requirements.txt

# 如果配置文件中启用了 OCR（default_use_ocr = true），需预先下载 MinerU 模型权重
# 系统通过 MINERU_MODEL_SOURCE=local 限制仅使用本地权重，不会自动从网络下载
# 因此必须在启动服务前手动执行以下命令缓存模型：
mineru-models-download -s modelscope -m pipeline

# 启动 VRAG 服务
python -m vrag.benchmark.video_rag -c vrag_config.toml -H 0.0.0.0 -p 7860
```

### 6.3 调用 API

Python 调用示例：

```python
import bentoml

client = bentoml.SyncHTTPClient("http://localhost:7860")
result = client.ask(
    video_path="/data/test_video.mp4",
    question="What is happening in this video?"
)
print(result.answer)
```

### 6.4 使用配置文件启动

```bash
# 创建最小配置文件（仅覆盖必填的模型路径和 API 地址）
cat > vrag_config.toml << 'EOF'
blip_model_path = "/models/blip2-itm-vit-g-coco"
whisper_model_path = "/models/whisper-large-v3-turbo"
mmdino_model_path = "/models/mm_grounding_dino_tiny_o365v1_goldg_v3det"
embedding_model_path = "/models/Qwen3-Embedding-0.6B"
reranker_model_path = "/models/Qwen3-Reranker-0.6B"
qwenvl_api_base = "http://<宿主机IP>:8000/v1"
qwenvl_model_name = "Qwen2.5-VL-32B-Instruct"
EOF

# 使用配置文件启动
python -m vrag.benchmark.video_rag -c vrag_config.toml -H 0.0.0.0 -p 7860
```

## 7. 服务 API 参考

### 7.1 VideoRagService

**端点**：`POST /ask`

**请求参数**：

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| video_path | str | 是 | 视频文件路径 |
| question | str | 是 | 提问内容 |
| config | VideoRagConfig | 否 | 请求级配置覆盖 |

**响应结构**：

```json
{
  "question": "What is happening in this video?",
  "answer": "The video shows...",
  "digested_info": "Retrieved context used for generation...",
  "processing_time": 12.5
}
```

### 7.2 Baseline (SimpleQwenVLQAService)

**端点**：`POST /ask`

参数与响应格式同上，但不经过检索增强，直接将帧和问题送入 LLM。

### 7.3 其他服务

各子服务（WhisperService、MineruService、BlipService 等）作为 BentoML 内部依赖，由 VideoRagService 编排调用，不直接对外暴露。

## 8. Benchmark 评估

### 8.1 评估流程

1. 准备数据集 [VideoMME](https://modelscope.cn/datasets/lmms-lab/Video-MME)，包含视频文件和测试配置文件，在目录下解压 zip 文件，视频位于 `data` 目录下
2. 下载 [videomme_json_file.json](https://github.com/Leon1207/Video-RAG-master/blob/main/evals/videomme_json_file.json) 测试文件
3. 启动 vLLM 容器
4. 启动 VRAG 或 Baseline 容器
5. 在 VRAG 容器内运行评估脚本

### 8.2 Video RAG 模式评估

```bash
python -m vrag.benchmark.evaluation \
  -c /data/videomme/videomme_json_file.json \
  -d /data/videomme \
  -H http://localhost:7860 \
  -t 3600 \
  -o output/vrag_results \
  --max_concurrent 10
```

### 8.3 Baseline 模式评估

```bash
python -m vrag.benchmark.evaluation \
  -c /data/videomme/videomme_json_file.json \
  -d /data/videomme \
  -H http://localhost:7861 \
  -t 3600 \
  -o output/baseline_results \
  --max_concurrent 10
```

### 8.4 评估结果

评估结果输出为 `results.json`，包含：

| 字段 | 说明 |
|------|------|
| `elapsed_time` | 总评估耗时 |
| `total_samples` | 总样本数 |
| `processed_samples` | 已处理样本数 |
| `passed_samples` | 正确样本数 |
| `results` | 逐样本详细结果 |
| `timestamp` | 评估时间戳 |

## 9. 注意事项

- **模型权重必须本地加载**：所有 Transformers 模型均使用 `local_files_only=True`，不支持在线下载。服务启动时会校验模型路径是否存在，路径不存在将抛出异常。
- **MinerU下载模型**：MinerU OCR 限制本地加载模型，如需使用，请提前下载模型。
- **vLLM 容器需先于 VRAG 容器启动**：VRAG 服务的 QwenVLService 依赖 vLLM API，需确保 vLLM 服务就绪后再启动 VRAG。
- **qwenvl_api_base 必须正确配置**：VRAG 与 vLLM 运行在不同容器中，需填写宿主机 IP，如 `http://192.168.1.100:8000/v1`。
- **缓存目录**：视频处理结果默认缓存到 `cache_store/video_process`，可通过 `video_process_cache_dir` 参数修改。
