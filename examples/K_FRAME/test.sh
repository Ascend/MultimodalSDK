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
# E2E test script for K_FRAME (Key Frame Selector).
# This script uses two containers:
#   1. vLLM container - vllm-ascend:v0.15.0rc1 (can be deleted and recreated)
#   2. Test container - multimodalsdk image (reused, do not delete)
#
# Prerequisites:
#   - Docker is installed and running
#   - Model files are downloaded to /models directory
#   - Test video is available

set -euo pipefail

# ============================================================================
# Configuration - Modify these variables to customize the test
# ============================================================================

# Model mount configuration (宿主机模型目录 -> 容器内固定目录/models)
HOST_MODEL_DIR="/models"

# vLLM configuration (vLLM容器每次调用删除重建)
VLLM_CONTAINER_NAME="vllm-k-frame"
VLLM_IMAGE_NAME="quay.io/ascend/vllm-ascend:v0.15.0rc1"
VLLM_MODEL_PATH="Qwen2.5-VL-7B-Instruct" # 相对于HOST_MODEL_DIR
VLLM_MODEL_NAME="qwen2.5-vl-7b"
VLLM_TENSOR_PARALLEL=2
VLLM_HOST="0.0.0.0"
VLLM_PORT="18001"

# Test container configuration (测试容器复用，需手动删除)
TEST_CONTAINER_NAME="test-k-frame"
TEST_IMAGE_NAME="swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.0-910b-ubuntu22.04-py3.11"

# Multimodal SDK installation package (手动提供本地路径)
MULTIMODAL_SDK_PACKAGE="./Ascend-mindxsdk-multimodal_26.1.0_linux-aarch64.run"

# CLIP model path (相对于/models)
CLIP_MODEL_PATH="chinese-clip-vit-large-patch14-336px"

# Device configuration (至少2张卡，用于vllm)
VISIBLE_DEVICES="0,1"

# Test configuration
VIDEO_PATH="/path/to/test_video.mp4"
QUERY="视频中出现了哪些物体？"
SAMPLE_NUM=4
DEVICE_LIST="0" # 设置clip模型运行设备

# Test mode: "simple" (离散关键帧) or "range" (区间关键帧)
TEST_MODE="simple"

# ============================================================================
# Utility Functions
# ============================================================================
log_info() {
    echo -e "\033[34m[INFO]\033[0m $1"
}

log_success() {
    echo -e "\033[32m[SUCCESS]\033[0m $1"
}

log_warn() {
    echo -e "\033[33m[WARN]\033[0m $1"
}

log_error() {
    echo -e "\033[31m[ERROR]\033[0m $1"
}

# ============================================================================
# Check Prerequisites
# ============================================================================
check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi

    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi

    # Check vLLM image
    if ! docker image inspect "${VLLM_IMAGE_NAME}" &> /dev/null; then
        log_warn "Docker image ${VLLM_IMAGE_NAME} not found locally"
        log_info "Pulling image... (this may take a while)"
        docker pull "${VLLM_IMAGE_NAME}"
    fi

    # Check test image
    if ! docker image inspect "${TEST_IMAGE_NAME}" &> /dev/null; then
        log_warn "Docker image ${TEST_IMAGE_NAME} not found locally"
        log_info "Pulling image... (this may take a while)"
        docker pull "${TEST_IMAGE_NAME}"
    fi

    # Check model directory
    if [[ ! -d "${HOST_MODEL_DIR}" ]]; then
        log_error "Model directory does not exist: ${HOST_MODEL_DIR}"
        exit 1
    fi

    # Check vLLM model
    if [[ ! -d "${HOST_MODEL_DIR}/${VLLM_MODEL_PATH}" ]]; then
        log_error "vLLM model not found: ${HOST_MODEL_DIR}/${VLLM_MODEL_PATH}"
        exit 1
    fi

    # Check CLIP model
    if [[ ! -d "${HOST_MODEL_DIR}/${CLIP_MODEL_PATH}" ]]; then
        log_error "CLIP model not found: ${HOST_MODEL_DIR}/${CLIP_MODEL_PATH}"
        exit 1
    fi

    # Check test video
    if [[ ! -f "${VIDEO_PATH}" ]]; then
        log_error "Test video not found: ${VIDEO_PATH}"
        exit 1
    fi

    log_success "Prerequisites check passed"
}

# ============================================================================
# Container Helpers
# ============================================================================
container_exists() {
    local name="$1"
    docker ps -a --format '{{.Names}}' | grep -q "^${name}$"
}

container_running() {
    local name="$1"
    docker ps --format '{{.Names}}' | grep -q "^${name}$"
}

# ============================================================================
# Start/Reset vLLM Container (always recreate)
# ============================================================================
start_vllm_container() {
    # Remove existing container if exists
    if container_exists "${VLLM_CONTAINER_NAME}"; then
        log_info "Removing existing vLLM container..."
        docker rm -f "${VLLM_CONTAINER_NAME}" 2>/dev/null || true
    fi

    log_info "Creating vLLM container..."

    docker run -itd --name "${VLLM_CONTAINER_NAME}" \
        --privileged \
        --network=host \
        --device /dev/davinci_manager \
        --device /dev/devmm_svm \
        --device /dev/hisi_hdc \
        -v /usr/local/dcmi:/usr/local/dcmi \
        -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
        -v ${HOST_MODEL_DIR}:/models \
        "${VLLM_IMAGE_NAME}" bash

    if [[ $? -ne 0 ]]; then
        log_error "Failed to create vLLM container"
        exit 1
    fi

    sleep 3

    if ! container_running "${VLLM_CONTAINER_NAME}"; then
        log_error "vLLM container failed to start"
        docker logs "${VLLM_CONTAINER_NAME}"
        exit 1
    fi

    log_success "vLLM container started: ${VLLM_CONTAINER_NAME}"
}

# ============================================================================
# Start vLLM Service
# ============================================================================
start_vllm_service() {
    log_info "Starting vLLM service..."

    # Check if service is already running
    if curl -s "http://localhost:${VLLM_PORT}/v1/models" &>/dev/null; then
        log_warn "vLLM service is already running at port ${VLLM_PORT}"
        return 0
    fi

    local start_cmd="export MODEL_PATH=/models/${VLLM_MODEL_PATH} && \
        export ASCEND_RT_VISIBLE_DEVICES=${VISIBLE_DEVICES} && \
        echo \"Starting vLLM at \$(date)...\" >> /tmp/vllm_start.log && \
        echo \"VISIBLE_DEVICES=\${ASCEND_RT_VISIBLE_DEVICES} TP=${VLLM_TENSOR_PARALLEL}\" >> /tmp/vllm_start.log && \
        nohup vllm serve \${MODEL_PATH} \
            --served-model-name=${VLLM_MODEL_NAME} \
            --max-model-len=100000 \
            --tensor-parallel-size=${VLLM_TENSOR_PARALLEL} \
            --mm_processor_cache_gb=0 \
            --host ${VLLM_HOST} \
            --port ${VLLM_PORT} >> /tmp/vllm_start.log 2>&1 &
        sleep 2 && ps aux | grep vllm | grep -v grep || echo \"vllm process not found\""

    log_info "Starting vLLM: /models/${VLLM_MODEL_PATH} -> ${VLLM_MODEL_NAME}, TP=${VLLM_TENSOR_PARALLEL}"
    docker exec "${VLLM_CONTAINER_NAME}" bash -c "${start_cmd}"

    # Wait for service to be ready
    log_info "Waiting for vLLM service to be ready (this may take a few minutes)..."
    local max_wait=300
    local waited=0
    while [[ $waited -lt $max_wait ]]; do
        if curl -s "http://localhost:${VLLM_PORT}/v1/models" &>/dev/null; then
            log_success "vLLM service is ready at http://localhost:${VLLM_PORT}/v1"
            return 0
        fi
        sleep 10
        waited=$((waited + 10))
        echo -n "."
    done

    echo ""
    log_error "vLLM service failed to start within ${max_wait} seconds"
    log_info "vLLM start log:"
    docker exec "${VLLM_CONTAINER_NAME}" cat /tmp/vllm_start.log 2>/dev/null || echo "No log found"
    exit 1
}

# ============================================================================
# Start Test Container (reuse if exists)
# ============================================================================
start_test_container() {
    if container_exists "${TEST_CONTAINER_NAME}"; then
        log_info "Test container '${TEST_CONTAINER_NAME}' already exists, reusing it"
        if container_running "${TEST_CONTAINER_NAME}"; then
            log_info "Test container is running"
        else
            log_info "Starting existing test container..."
            docker start "${TEST_CONTAINER_NAME}"
        fi
        return 0
    fi

    log_info "Creating test container..."

    # Get absolute video path
    local video_host_path
    local video_filename
    if [[ "${VIDEO_PATH}" == /* ]]; then
        video_host_path="${VIDEO_PATH}"
        video_filename="$(basename "${VIDEO_PATH}")"
    else
        if [[ "${VIDEO_PATH}" == */* ]]; then
            video_host_path="$(cd "$(dirname "${VIDEO_PATH}")" && pwd)/$(basename "${VIDEO_PATH}")"
        else
            video_host_path="$(pwd)/${VIDEO_PATH}"
        fi
        video_filename="$(basename "${VIDEO_PATH}")"
    fi
    local video_container_path="/video/${video_filename}"

    log_info "Mounting video: ${video_host_path} -> ${video_container_path}"

    docker run -itd --name "${TEST_CONTAINER_NAME}" \
        --privileged \
        --network=host \
        --device /dev/davinci_manager \
        --device /dev/devmm_svm \
        --device /dev/hisi_hdc \
        -v /usr/local/dcmi:/usr/local/dcmi \
        -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
        -v ${HOST_MODEL_DIR}:/models \
        -v "$(pwd):/workspace" \
        -v "${video_host_path}:${video_container_path}:ro" \
        -v "$(cd "$(dirname "${MULTIMODAL_SDK_PACKAGE}")" && pwd)/$(basename "${MULTIMODAL_SDK_PACKAGE}"):/tmp/multimodal_sdk.run:ro" \
        "${TEST_IMAGE_NAME}" bash

    if [[ $? -ne 0 ]]; then
        log_error "Failed to create test container"
        exit 1
    fi

    sleep 3

    if ! container_running "${TEST_CONTAINER_NAME}"; then
        log_error "Test container failed to start"
        docker logs "${TEST_CONTAINER_NAME}"
        exit 1
    fi

    log_success "Test container started: ${TEST_CONTAINER_NAME}"
}

# ============================================================================
# Setup Test Container Environment
# ============================================================================
setup_test_environment() {
    log_info "Setting up test container environment..."

    # Install system dependencies
    log_info "Installing system dependencies..."
    docker exec "${TEST_CONTAINER_NAME}" bash -c "apt-get update && apt-get install -y git curl wget libgl1-mesa-glx libglib2.0-0"

    # Install Python dependencies
    log_info "Installing Python dependencies..."
    docker exec "${TEST_CONTAINER_NAME}" bash -c "pip install transformers==4.51.3 'pillow>=11.2.1' numpy==1.26.4 opencv-python decord2 qwen_vl_utils openai einops accelerate decorator scipy attrs torch_npu -i https://mirrors.aliyun.com/pypi/simple/"

    # Install Multimodal SDK
    log_info "Installing Multimodal SDK..."
    docker exec "${TEST_CONTAINER_NAME}" bash -c "bash /usr/local/multimodal/script/uninstall.sh 2>/dev/null || true && bash /tmp/multimodal_sdk.run --install --install-path=/usr/local/"

    log_success "Test container environment setup completed"
}

# ============================================================================
# Run Test
# ============================================================================
run_test() {
    log_info "Running K_FRAME test..."
    log_info "Mode: ${TEST_MODE}"
    log_info "Video: ${VIDEO_PATH}"
    log_info "Query: ${QUERY}"

    # Get video container path
    local video_filename="$(basename "${VIDEO_PATH}")"
    local video_container_path="/video/${video_filename}"

    local vlm_url="http://127.0.0.1:${VLLM_PORT}/v1"

    # Build Python test code
    local python_code
    if [[ "${TEST_MODE}" == "range" ]]; then
        python_code="
import sys
sys.path.insert(0, '/workspace')
from k_range_frame_selector_example import RangeDetectionQaDemo

demo = RangeDetectionQaDemo(
    model_path='/models/${CLIP_MODEL_PATH}',
    device_list=[${DEVICE_LIST}],
    similar_threshold=0.03,
    image_similar_threshold=0.015,
    vlm_url='${vlm_url}',
    api_key='None',
    vlm_model_name='${VLLM_MODEL_NAME}',
    model_type='cn_clip'
)

answer = demo.qa('${QUERY}', '${video_container_path}', sample_num=${SAMPLE_NUM})
print(f'查询: ${QUERY}')
print(f'回答: {answer}')
"
    else
        python_code="
import sys
sys.path.insert(0, '/workspace')
from k_frame_selector_example import SimpleQaDemo

demo = SimpleQaDemo(
    model_path='/models/${CLIP_MODEL_PATH}',
    device_list=[${DEVICE_LIST}],
    vlm_url='${vlm_url}',
    api_key='None',
    vlm_model_name='${VLLM_MODEL_NAME}',
    model_type='cn_clip'
)

answer = demo.qa('${QUERY}', '${video_container_path}', sample_num=${SAMPLE_NUM})
print(f'查询: ${QUERY}')
print(f'回答: {answer}')
"
    fi

    # Run test in test container
    local start_time
    start_time=$(date +%s)

    log_info "Executing test in container..."
    local response_file="/tmp/kframe_test_output_$$.txt"
    docker exec "${TEST_CONTAINER_NAME}" bash -c "source /usr/local/multimodal/script/set_env.sh && cd /workspace && python3 -c \"${python_code}\"" > "${response_file}" 2>&1

    local exit_code=$?
    local end_time
    end_time=$(date +%s)
    local processing_time=$((end_time - start_time))

    cat "${response_file}"
    rm -f "${response_file}"

    if [[ ${exit_code} -eq 0 ]]; then
        log_success "Test completed successfully"
        echo ""
        echo "=========================================="
        echo "Test Result: PASS"
        echo "=========================================="
        echo "Processing time: ${processing_time}s"
        echo ""
        return 0
    else
        log_error "Test failed"
        echo ""
        echo "=========================================="
        echo "Test Result: FAIL"
        echo "=========================================="
        return 1
    fi
}

# ============================================================================
# Main
# ============================================================================
main() {
    echo "=========================================="
    echo "K_FRAME E2E Test Script"
    echo "=========================================="
    echo ""
    echo "Configuration:"
    echo "  vLLM Container: ${VLLM_CONTAINER_NAME}"
    echo "  Test Container: ${TEST_CONTAINER_NAME}"
    echo "  Host Model Dir: ${HOST_MODEL_DIR} -> /models"
    echo "  vLLM Model: ${VLLM_MODEL_PATH} -> ${VLLM_MODEL_NAME}"
    echo "  CLIP Model: ${CLIP_MODEL_PATH}"
    echo "  vLLM TP: ${VLLM_TENSOR_PARALLEL}"
    echo "  Visible Devices: ${VISIBLE_DEVICES}"
    echo "  Test Mode: ${TEST_MODE}"
    echo "  Video: ${VIDEO_PATH}"
    echo "  Query: ${QUERY}"
    echo ""

    check_prerequisites

    # Setup and start vLLM (always recreate)
    start_vllm_container
    start_vllm_service

    # Setup and start test container (reuse)
    start_test_container
    setup_test_environment

    echo ""
    echo "=========================================="
    echo "Running Test"
    echo "=========================================="

    if run_test; then
        echo ""
        log_success "E2E test completed successfully!"
        exit 0
    else
        echo ""
        log_error "E2E test failed!"
        exit 1
    fi
}

main "$@"
