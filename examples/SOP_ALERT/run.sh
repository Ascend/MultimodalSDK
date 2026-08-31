#!/bin/bash
# -------------------------------------------------------------------------
#  This file is part of the MultimodalSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MultimodalSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------
# SOP 违规自动预警 —— 一键运行脚本
#
# 用法 （需要昇腾 NPU + CLIP 模型 + VLM 推理服务）:
#   bash run.sh --video data/cases/case1/video/sop.mp4 --sop data/cases/case1/sop.json
#
#   可通过环境变量覆盖默认配置:
#     CLIP_MODEL_PATH  CLIP 模型权重路径（默认 /models/chinese-clip-vit-large-patch14-336px）
#     DEVICE_ID        NPU 设备 ID（默认 0）
#     VLM_URL          VLM 推理服务地址（默认 http://127.0.0.1:8111/v1）
#     VLM_MODEL_NAME   VLM 模型名称（默认 Qwen2.5-VL-32B-Instruct）

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

CLIP_MODEL_PATH="${CLIP_MODEL_PATH:-/models/chinese-clip-vit-large-patch14-336px}"
DEVICE_ID="${DEVICE_ID:-0}"
VLM_URL="${VLM_URL:-http://127.0.0.1:8111/v1}"
VLM_MODEL_NAME="${VLM_MODEL_NAME:-Qwen2.5-VL-32B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/output}"

usage() {
    grep '^#' "$0" | sed -n '17,30p' | sed 's/^# \{0,1\}//'
    exit 1
}

VIDEO=""
SOP=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --video) VIDEO="$2"; shift 2 ;;
        --sop) SOP="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "未知参数: $1"; usage ;;
    esac
done

if [[ -z "${VIDEO}" || -z "${SOP}" ]]; then
    echo "错误: 完整流水线需要同时指定 --video 与 --sop"
    usage
fi

echo "=========================================="
echo "SOP 违规自动预警 —— 完整流水线"
echo "=========================================="
echo "  视频:       ${VIDEO}"
echo "  SOP:        ${SOP}"
echo "  CLIP 模型:  ${CLIP_MODEL_PATH}"
echo "  NPU 设备:   ${DEVICE_ID}"
echo "  VLM 服务:   ${VLM_URL} (${VLM_MODEL_NAME})"
echo "  输出目录:   ${OUTPUT_DIR}"
echo ""

python3 sop_alert_pipeline.py \
    --video "${VIDEO}" \
    --sop "${SOP}" \
    --clip-model-path "${CLIP_MODEL_PATH}" \
    --device-id "${DEVICE_ID}" \
    --vlm-url "${VLM_URL}" \
    --vlm-model-name "${VLM_MODEL_NAME}" \
    --output-dir "${OUTPUT_DIR}" \
    --print-markdown
