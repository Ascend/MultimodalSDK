# FlashVid 启动说明

FlashVid 对视频视觉 token 进行筛选和合并，以减少送入语言模型的 token 数量。本目录包含 Qwen2.5-VL、Qwen3-VL 和 Qwen3.5 的运行时补丁，通过 Multimodal SDK 的 vLLM 插件入口加载。

## 镜像与运行环境

FlashVid 需要以下运行环境：

| 项目 | 版本或要求 |
| --- | --- |
| 基础镜像 | `swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12` |
| vLLM | `v0.23.0` |
| vllm-ascend | `v0.23.0rc1` |
| Multimodal SDK | 包含本目录补丁、插件入口和已构建原生扩展的当前源码版本 |

基础镜像仅提供 CANN 等基础环境，还需要安装上述版本的 vLLM、vllm-ascend 和包含 FlashVid 补丁的 Multimodal SDK。

`tech_v26.2.0` 现有 Dockerfile 使用的 `vllm-ascend:v0.8.5rc1` 缺少本补丁所需的模型和剪枝接口，不能用于下面的命令。更换镜像后应重新安装当前源码中的 Python 包，确保 vLLM 能发现 `flashvid_patch` 插件入口。若设置了 `VLLM_PLUGINS` 插件白名单，需要将 `flashvid_patch` 包含在内。

FlashVid 使用 vLLM 自带的媒体加载和模型预处理流程，不需要开启 SDK 的 SCC、预处理或媒体解码补丁。

本集成入口依赖同目录的 `token_selection.py`、`method.py` 和 `patch_qwen2_5_vl.py`。安装前需确保这些基础工具、压缩算法及 Qwen2.5-VL 补丁已一并合入源码；仅安装集成入口不能启用 FlashVid。

## 启动前准备

以下命令均在上述环境的 Linux 容器内执行。完成 SDK 编译安装、确认其原生扩展可加载后，在当前源码目录注册 Python 包及插件入口：

```bash
export SDK_REPO=/workspace/MultimodalSDK
python -m pip install --no-deps -e "${SDK_REPO}/MultimodalSDK"
```

将 `MODEL_PATH` 设置为容器内的实际权重目录。以下路径任选一种：

```bash
export MODEL_PATH=/models/Qwen2.5-VL-7B-Instruct
# export MODEL_PATH=/models/Qwen3-VL-8B-Instruct
# export MODEL_PATH=/models/Qwen3.5-9B
```

下面两组命令使用相同的服务名、端口、上下文长度和显存比例。先停止优化前的服务，再启动优化后的服务。模型无法在单卡容纳时，应为两组命令同时增加相同的 `--tensor-parallel-size` 参数，并按实际硬件调整上下文长度和显存比例。

## 优化前：关闭 FlashVid

关闭 FlashVid 和编码前剪枝，同时不设置视频剪枝率，运行 vLLM 原生流程。

```bash
export ENABLE_FLASHVID=0
export ENABLE_ENCODE_PRUNE=0

vllm serve "${MODEL_PATH}" \
  --served-model-name flashvid-model \
  --max-model-len 16384 \
  --max-num-batched-tokens 16384 \
  --no-enable-prefix-caching \
  --async-scheduling \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --port 3050 \
  --gpu-memory-utilization 0.9 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --skip-mm-profiling \
  --mm-processor-cache-gb 0
```

## 优化后：开启 FlashVid

下面同时开启 FlashVid 和可选的编码前剪枝。若只比较 FlashVid 本身，可将 `ENABLE_ENCODE_PRUNE` 改为 `0`。

```bash
export ENABLE_FLASHVID=1
export ENABLE_ENCODE_PRUNE=1

vllm serve "${MODEL_PATH}" \
  --served-model-name flashvid-model \
  --max-model-len 16384 \
  --max-num-batched-tokens 16384 \
  --no-enable-prefix-caching \
  --async-scheduling \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --port 3050 \
  --gpu-memory-utilization 0.9 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --skip-mm-profiling \
  --mm-processor-cache-gb 0 \
  --video_pruning_rate 0.5
```

## 参数说明与验证

| 参数 | 含义 |
| --- | --- |
| `ENABLE_FLASHVID=1` | 启动时加载三个模型的 FlashVid 补丁；默认关闭 |
| `video_pruning_rate=0.5` | 视频视觉 token 的目标剪枝比例为 50%，对应保留比例为 50%；需与 FlashVid 开关一起设置 |
| `ENABLE_ENCODE_PRUNE=1` | 开启可选的视觉编码前剪枝；默认关闭 |
| `MODEL_PATH` | 容器内的模型权重目录 |

环境变量应在启动服务前设置。仅设置 `ENABLE_FLASHVID=1` 而未设置非零 `video_pruning_rate` 时，算法配置仍会关闭 FlashVid 压缩。

启动日志中的 `patch flash vid enabled` 表示插件入口已加载补丁；执行视频请求后，还应检查 `[FlashVID] compression enabled: yes` 及保留比例日志，以确认实际进入压缩路径。发送请求时使用的模型名为 `flashvid-model`。

比较优化前后效果时，应使用相同的视频、请求参数、硬件和并行配置，同时记录耗时、显存占用与任务精度。启用剪枝可能影响任务精度，实际收益需要在目标模型和数据集上测量。当前实现尚未在上述镜像和昇腾硬件上完成端到端验证。
