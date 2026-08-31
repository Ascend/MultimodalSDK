#!/bin/bash
# ============================================================================
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
# ============================================================================
# E2E test script for SOP_ALERT (SOP compliance alert pipeline).
# This script uses two containers:
#   1. vLLM container - vllm-ascend image (deleted and recreated on each run)
#   2. Test container - official MultimodalSDK image (reused, SDK preinstalled)
#
# Prerequisites:
#   - Atlas 800I A2 server with NPU driver installed (npu-smi info works)
#   - Docker is installed and running
#   - Qwen2.5-VL and chinese-clip models downloaded to ${HOST_MODEL_DIR}
#   - A real factory operation video (mp4, with audio track if possible)
#
# Verified on: 8x Ascend 910B4, driver 24.1.0.3, Docker 26.1.3 (aarch64)

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

HOST_MODEL_DIR="${HOST_MODEL_DIR:-/data1/sop_alert_models}"

# vLLM container (每次重建)
VLLM_CONTAINER_NAME="vllm-sop-alert"
VLLM_IMAGE_NAME="${VLLM_IMAGE_NAME:-quay.io/ascend/vllm-ascend:v0.21.0rc1}"
VLLM_MODEL_PATH="Qwen2.5-VL-7B-Instruct"
VLLM_MODEL_NAME="qwen2.5-vl-7b"
VLLM_TENSOR_PARALLEL=2
VLLM_PORT=18002
VISIBLE_DEVICES="${VISIBLE_DEVICES:-6,7}"

# 根据 VISIBLE_DEVICES 生成 vLLM 容器的 --device 参数（如 "6,7" -> "--device /dev/davinci6 --device /dev/davinci7"）
VLLM_NPU_DEVICES=""
IFS=',' read -ra _vllm_dev_ids <<< "${VISIBLE_DEVICES}"
for _dev_id in "${_vllm_dev_ids[@]}"; do
    VLLM_NPU_DEVICES+="--device /dev/davinci${_dev_id} "
done
VLLM_NPU_DEVICES="${VLLM_NPU_DEVICES% }"

# Test container (复用，SDK 已预装)
TEST_CONTAINER_NAME="test-sop-alert"
TEST_IMAGE_NAME="${TEST_IMAGE_NAME:-swr.cn-south-1.myhuaweicloud.com/ascendhub/multimodalsdk:26.1.0-cann9.1.0-torch_npu2.6.0.post5-910b-ubuntu22.04-py3.12-aarch64}"
SDK_SET_ENV="/usr/local/multimodal/script/set_env.sh"
DEVICE_ID=${DEVICE_ID:-4}
# 根据 DEVICE_ID 生成测试容器的 --device 参数（如 "4" -> "--device /dev/davinci4"）
TEST_NPU_DEVICES="--device /dev/davinci${DEVICE_ID}"
CLIP_MODEL_PATH="chinese-clip-vit-large-patch14-336px"

# Test configuration
# 通过命令行参数传入（必须提供）
# 格式: bash test_e2e.sh <case_directory>
# 例如: bash test_e2e.sh data/cases/case1
CASE_DIR="${1:-}"

# 从 case 目录读取视频和 SOP
if [[ -n "${CASE_DIR}" ]]; then
    VIDEO_PATH="${CASE_DIR}/video/sop.mp4"
    SOP_PATH="${CASE_DIR}/sop.json"
else
    VIDEO_PATH=""
    SOP_PATH=""
fi

SAMPLE_NUM=12

# 是否复用已存在的测试容器（默认 true，避免重复安装 ffmpeg）
REUSE_CONTAINER="${REUSE_CONTAINER:-true}"

# 输出目录基础路径
OUTPUT_BASE_DIR="output"

# ============================================================================
# Utility Functions
# ============================================================================
log_info() { echo -e "\033[34m[INFO]\033[0m $1"; }
log_success() { echo -e "\033[32m[SUCCESS]\033[0m $1"; }
log_warn() { echo -e "\033[33m[WARN]\033[0m $1"; }
log_error() { echo -e "\033[31m[ERROR]\033[0m $1"; }

container_exists() { docker ps -a --format '{{.Names}}' | grep -q "^$1$"; }
container_running() { docker ps --format '{{.Names}}' | grep -q "^$1$"; }

# 检查参数
check_args() {
    if [[ -z "${CASE_DIR}" ]]; then
        echo "Usage: $0 <case_directory>"
        echo ""
        echo "Example:"
        echo "  $0 data/cases/case1"
        echo ""
        echo "The case directory should contain:"
        echo "  - sop.json: SOP definition file"
        echo "  - video/sop.mp4: Operation video file"
        echo ""
        echo "Environment Variables:"
        echo "  ENABLE_AUDIO      Enable audio extraction (default: true)"
        echo "                    Set to 'false' to skip ffmpeg installation"
        echo ""
        echo "  REUSE_CONTAINER   Reuse existing test container (default: true)"
        echo "                    Set to 'false' to recreate container"
        echo ""
        echo "  Examples:"
        echo "    # Reuse container (fast, recommended)"
        echo "    bash $0 data/cases/case1"
        echo ""
        echo "    # Skip ffmpeg installation"
        echo "    ENABLE_AUDIO=false bash $0 data/cases/case1"
        echo ""
        echo "    # Recreate container (slow, will reinstall ffmpeg)"
        echo "    REUSE_CONTAINER=false bash $0 data/cases/case1"
        exit 1
    fi

    if [[ ! -d "${CASE_DIR}" ]]; then
        log_error "Case directory not found: ${CASE_DIR}"
        exit 1
    fi

    if [[ ! -f "${VIDEO_PATH}" ]]; then
        log_error "Video file not found: ${VIDEO_PATH}"
        log_info "Expected location: <case_dir>/video/sop.mp4"
        exit 1
    fi

    if [[ ! -f "${SOP_PATH}" ]]; then
        log_error "SOP file not found: ${SOP_PATH}"
        log_info "Expected location: <case_dir>/sop.json"
        exit 1
    fi
}

# ============================================================================
# vLLM Container
# ============================================================================
start_vllm_container() {
    if container_exists "${VLLM_CONTAINER_NAME}"; then
        log_info "Removing existing vLLM container..."
        docker rm -f "${VLLM_CONTAINER_NAME}" 2>/dev/null || true
    fi

    log_info "Creating vLLM container..."
    docker run -itd --name "${VLLM_CONTAINER_NAME}" \
        --network=host \
        ${VLLM_NPU_DEVICES} \
        --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
        -v /usr/local/dcmi:/usr/local/dcmi \
        -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
        -v "${HOST_MODEL_DIR}":/models \
        "${VLLM_IMAGE_NAME}" bash

    sleep 3
    container_running "${VLLM_CONTAINER_NAME}" || { log_error "vLLM container failed"; exit 1; }
    log_success "vLLM container started"
}

start_vllm_service() {
    log_info "Starting vLLM service..."
    curl -s "http://localhost:${VLLM_PORT}/v1/models" &>/dev/null && {
        log_warn "vLLM already running"; return 0
    }

    docker exec "${VLLM_CONTAINER_NAME}" bash -c "
        export ASCEND_RT_VISIBLE_DEVICES=${VISIBLE_DEVICES}
        nohup vllm serve /models/${VLLM_MODEL_PATH} \
            --served-model-name=${VLLM_MODEL_NAME} \
            --max-model-len=32768 \
            --tensor-parallel-size=${VLLM_TENSOR_PARALLEL} \
            --enforce-eager \
            --host 0.0.0.0 --port ${VLLM_PORT} >> /tmp/vllm.log 2>&1 &
    "

    log_info "Waiting for vLLM (up to 5 min)..."
    for i in {1..30}; do
        curl -s "http://localhost:${VLLM_PORT}/v1/models" &>/dev/null && {
            log_success "vLLM ready"; return 0
        }
        sleep 10
    done
    log_error "vLLM failed to start"
    docker exec "${VLLM_CONTAINER_NAME}" cat /tmp/vllm.log 2>/dev/null | tail -50
    exit 1
}

# ============================================================================
# Test Container (SDK preinstalled)
# ============================================================================
start_test_container() {
    local cases_dir
    cases_dir="$(cd "data/cases" && pwd)"

    # 检查是否需要复用容器
    if [[ "${REUSE_CONTAINER}" == "true" ]] && container_exists "${TEST_CONTAINER_NAME}"; then
        log_info "Reusing existing test container (REUSE_CONTAINER=true)"
        container_running "${TEST_CONTAINER_NAME}" || docker start "${TEST_CONTAINER_NAME}"
        return 0
    fi

    # 如果不复用容器，删除已存在的容器
    if container_exists "${TEST_CONTAINER_NAME}"; then
        log_info "Removing existing test container"
        docker rm -f "${TEST_CONTAINER_NAME}" >/dev/null 2>&1
    fi

    log_info "Creating test container..."

    docker run -itd --name "${TEST_CONTAINER_NAME}" \
        --network=host \
        ${TEST_NPU_DEVICES} \
        --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
        -v /usr/local/dcmi:/usr/local/dcmi \
        -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
        -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
        -v /etc/ascend_install.info:/etc/ascend_install.info \
        -v "${HOST_MODEL_DIR}":/models \
        -v "$(pwd):/workspace" \
        -v "${cases_dir}:/cases:ro" \
        "${TEST_IMAGE_NAME}" bash

    sleep 3
    container_running "${TEST_CONTAINER_NAME}" || { log_error "Test container failed"; exit 1; }
    log_success "Test container started"
}

setup_test_environment() {
    log_info "Setting up test environment..."

    # 检查是否需要安装 ffmpeg（包括 ffprobe）
    # ffprobe 用于获取视频信息，ffmpeg 用于音轨分离
    log_info "Installing ffmpeg (for video info and audio extraction)..."
    docker exec "${TEST_CONTAINER_NAME}" bash -c "
        if ! command -v ffprobe &> /dev/null; then
            echo 'Updating package lists...'
            apt-get update -qq
            echo 'Installing ffmpeg...'
            apt-get install -y ffmpeg
            echo 'Verifying installation...'
            which ffmpeg ffprobe || {
                echo 'ERROR: ffmpeg installation failed'
                exit 1
            }
            echo 'ffmpeg installed successfully'
        else
            echo 'ffmpeg already installed'
        fi
    " || {
        log_error "ffmpeg installation failed"
        exit 1
    }

    # 装 accelerate（vLLM 需要）
    log_info "Installing accelerate..."
    docker exec "${TEST_CONTAINER_NAME}" bash -c \
        "pip install --quiet accelerate -i https://mirrors.aliyun.com/pypi/simple/"

    log_success "Environment setup completed"
}

# ============================================================================
# Run Test
# ============================================================================
run_test() {
    log_info "Running SOP_ALERT E2E test..."

    # 从 CASE_DIR 提取 case 名称（例如：data/cases/case1 -> case1）
    local case_name=$(basename "${CASE_DIR}")
    local video_container="/cases/${case_name}/video/$(basename "${VIDEO_PATH}")"
    local sop_container="/cases/${case_name}/$(basename "${SOP_PATH}")"

    # 先使用临时输出目录
    local temp_output_dir="/workspace/output/temp_analysis"

    # 修复文件权限（SDK 要求）- 在宿主机上执行
    # 1. 修复 CLIP 模型目录权限
    find "${HOST_MODEL_DIR}/${CLIP_MODEL_PATH}" -type d -exec chmod 750 {} \; 2>/dev/null || true
    find "${HOST_MODEL_DIR}/${CLIP_MODEL_PATH}" -type f -exec chmod 640 {} \; 2>/dev/null || true

    # 2. 修复视频和 SOP 文件权限（在宿主机上，挂载前）
    chmod 640 "${VIDEO_PATH}" 2>/dev/null || true
    chmod 640 "${SOP_PATH}" 2>/dev/null || true

    # 运行流水线
    docker exec "${TEST_CONTAINER_NAME}" bash -c "
        source '${SDK_SET_ENV}'
        cd /workspace
        python3 sop_alert_pipeline.py \
            --video '${video_container}' \
            --sop '${sop_container}' \
            --clip-model-path /models/${CLIP_MODEL_PATH} \
            --device-id ${DEVICE_ID} \
            --clip-model-type cn_clip \
            --vlm-url http://127.0.0.1:${VLLM_PORT}/v1 \
            --vlm-model-name ${VLLM_MODEL_NAME} \
            --sample-num ${SAMPLE_NUM} \
            --output-dir ${temp_output_dir} \
            --print-markdown
    " || return 1

    # 读取报告中的分析结果，确定输出目录名称
    local report_json="/workspace/output/temp_analysis/sop_report.json"

    # 从 analysis 字段读取诊断结果
    local analysis_result=$(docker exec "${TEST_CONTAINER_NAME}" bash -c "
        source '${SDK_SET_ENV}'
        python3 -c \"
import json, sys
data = json.load(open('${report_json}'))
analysis = data.get('analysis', {})
compliant = analysis.get('compliant', False)
score = analysis.get('compliance_score', 0)
alerts = data.get('alerts', [])

# 判断诊断结果
if compliant:
    print('normal')
else:
    # 检查告警类型
    alert_types = [a.get('alert_type', '') for a in alerts]
    if any('STEP_MISSING' in t for t in alert_types):
        print('missing')
    elif any('ORDER' in t for t in alert_types):
        print('wrong_order')
    else:
        print('unknown')
\"
    ")

    # 根据分析结果确定输出目录后缀
    local output_suffix="${analysis_result:-unknown}"

    # 生成时间戳（格式：YYYYMMDD_HHMMSS）
    local timestamp=$(date +"%Y%m%d_%H%M%S")

    # 重命名输出目录：使用 case 名称 + 诊断结果 + 时间戳
    local final_output_dir="/workspace/output/${case_name}_${output_suffix}_${timestamp}"
    docker exec "${TEST_CONTAINER_NAME}" bash -c "
        rm -rf \"${final_output_dir}\"
        mv \"${temp_output_dir}\" \"${final_output_dir}\"
    "

    log_info "Report saved to: output/${case_name}_${output_suffix}_${timestamp}/"
    return 0
}

# ============================================================================
# Main
# ============================================================================
main() {
    cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

    # 检查参数
    check_args

    echo "=========================================="
    echo "SOP_ALERT E2E Test (Dual Container)"
    echo "=========================================="
    echo "vLLM: ${VLLM_CONTAINER_NAME} (${VLLM_IMAGE_NAME})"
    echo "SDK:  ${TEST_CONTAINER_NAME} (${TEST_IMAGE_NAME})"
    echo "Case: ${CASE_DIR}"
    echo "Video: ${VIDEO_PATH}"
    echo "SOP:  ${SOP_PATH}"
    echo ""

    start_vllm_container
    start_vllm_service
    start_test_container
    setup_test_environment

    echo ""
    echo "=========================================="
    echo "Running Test"
    echo "=========================================="

    if run_test; then
        log_success "E2E test completed successfully!"
        exit 0
    else
        log_error "E2E test failed!"
        exit 1
    fi
}

main "$@"
