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
# E2E test script for AutoOptimizeAndSample (Video RAG) service.
# This script:
#   1. Starts the vLLM container (if ENABLE_VLLM=true)
#   2. Starts the VRAG container
#   3. Installs dependencies in VRAG container
#   4. Starts the vLLM service
#   5. Starts the VRAG service
#   6. Runs the E2E test
#
# Usage: bash test.sh
#
# Prerequisites:
#   - Docker is installed and running
#   - Model files are downloaded to /models directory
#   - Test video file is available

set -euo pipefail

# ============================================================================
# Configuration - Modify these variables to customize the test
# ============================================================================

# Model mount configuration (宿主机模型目录 -> 容器内固定目录/models)
HOST_MODEL_DIR="/models"

# vLLM configuration
VLLM_CONTAINER_NAME="vllm-video-rag"
VLLM_IMAGE_NAME="quay.io/ascend/vllm-ascend:v0.15.0rc1"
VLLM_MODEL_PATH="Qwen2.5-VL-7B-Instruct"  # 模型路径，相对于/models
VLLM_MODEL_NAME="qwen2.5-vl-7b"           # 模型别名
VLLM_TENSOR_PARALLEL=4                      # TP数
VLLM_HOST="0.0.0.0"
VLLM_PORT="18001"

# VRAG configuration
VRAG_CONTAINER_NAME="vrag-video-rag"
VRAG_IMAGE_NAME="swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.0-910b-ubuntu22.04-py3.11"

# VRAG models (容器内访问路径为 /models/xxx)
MODEL_BLIP="blip2-itm-vit-g-coco"
MODEL_WHISPER="whisper-large-v3-turbo"
MODEL_MMDINO="mm_grounding_dino_tiny_o365v1_goldg_v3det"
MODEL_EMBEDDING="Qwen3-Embedding-0.6B"
MODEL_RERANKER="Qwen3-Reranker-0.6B"

# VRAG service configuration
VRAG_PORT="7860"
VRAG_HOST="0.0.0.0"

# SDK configuration
SDK_BRANCH="master"

# Device configuration
VISIBLE_DEVICES="0,1,2,3"

# Test configuration
VIDEO_PATH="/path/to/test_video.mp4"
QUESTION="What is happening in this video? Please describe the main content."
TIMEOUT="3600"

# vLLM开关 (设为false表示vLLM已在外部启动，只启动VRAG)
ENABLE_VLLM=true

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

    # Check if Docker daemon is running
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

    # Check VRAG image
    if ! docker image inspect "${VRAG_IMAGE_NAME}" &> /dev/null; then
        log_warn "Docker image ${VRAG_IMAGE_NAME} not found locally"
        log_info "Pulling image... (this may take a while)"
        docker pull "${VRAG_IMAGE_NAME}"
    fi

    # Check model directory
    if [[ ! -d "${HOST_MODEL_DIR}" ]]; then
        log_error "Model directory does not exist: ${HOST_MODEL_DIR}"
        exit 1
    fi

    # Check vLLM model
    if [[ "${ENABLE_VLLM}" == "true" ]]; then
        if [[ ! -d "${HOST_MODEL_DIR}/${VLLM_MODEL_PATH}" ]]; then
            log_error "vLLM model not found: ${HOST_MODEL_DIR}/${VLLM_MODEL_PATH}"
            exit 1
        fi
    fi

    # Check VRAG models
    local models=("${MODEL_BLIP}" "${MODEL_WHISPER}" "${MODEL_MMDINO}" "${MODEL_EMBEDDING}" "${MODEL_RERANKER}")
    for model in "${models[@]}"; do
        if [[ ! -d "${HOST_MODEL_DIR}/${model}" ]]; then
            log_error "VRAG model not found: ${HOST_MODEL_DIR}/${model}"
            exit 1
        fi
    done

    log_success "Prerequisites check passed"
}

# ============================================================================
# Check if Container Exists
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
# Start vLLM Container
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
        -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
        -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
        -v ${HOST_MODEL_DIR}:/models \
        -p ${VLLM_PORT}:${VLLM_PORT} \
        "${VLLM_IMAGE_NAME}" bash

    if [[ $? -ne 0 ]]; then
        log_error "Failed to start vLLM container"
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
# Start VRAG Container
# ============================================================================
start_vrag_container() {
    if container_exists "${VRAG_CONTAINER_NAME}"; then
        log_warn "VRAG container '${VRAG_CONTAINER_NAME}' already exists, reusing it"
        if container_running "${VRAG_CONTAINER_NAME}"; then
            log_info "VRAG container is running"
        else
            log_info "Starting existing VRAG container..."
            docker start "${VRAG_CONTAINER_NAME}"
        fi
        return 0
    fi

    log_info "Creating VRAG container..."

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

    docker run -itd --name "${VRAG_CONTAINER_NAME}" \
        --privileged \
        --device /dev/davinci_manager \
        --device /dev/devmm_svm \
        --device /dev/hisi_hdc \
        -v /usr/local/dcmi:/usr/local/dcmi \
        -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
        -v ${HOST_MODEL_DIR}:/models \
        -v "${video_host_path}:${video_container_path}:ro" \
        -p ${VRAG_PORT}:${VRAG_PORT} \
        "${VRAG_IMAGE_NAME}" bash

    if [[ $? -ne 0 ]]; then
        log_error "Failed to create VRAG container"
        exit 1
    fi

    sleep 3

    if ! docker ps --format '{{.Names}}' | grep -q "^${VRAG_CONTAINER_NAME}$"; then
        log_error "VRAG container failed to start"
        docker logs "${VRAG_CONTAINER_NAME}"
        exit 1
    fi

    log_success "VRAG container started: ${VRAG_CONTAINER_NAME}"
}

# ============================================================================
# Setup VRAG Environment
# ============================================================================
setup_vrag_environment() {
    log_info "Setting up VRAG environment..."

    # Install system dependencies
    log_info "Installing system dependencies..."
    docker exec "${VRAG_CONTAINER_NAME}" bash -c "
        apt-get update && apt-get install -y \
            libgl1 libglib2.0-0 libaio-dev ffmpeg libavcodec-extra libopenblas-dev swig \
            git curl
    "

    # Clone MultimodalSDK
    log_info "Cloning MultimodalSDK (branch: ${SDK_BRANCH})..."
    docker exec "${VRAG_CONTAINER_NAME}" bash -c "mkdir -p /workspace && cd /workspace && rm -rf MultimodalSDK && git clone -b ${SDK_BRANCH} https://gitcode.com/Ascend/MultimodalSDK.git"

    # Install Python dependencies (use China mirror, exclude en-core-web-sm)
    log_info "Installing Python dependencies..."
    docker exec "${VRAG_CONTAINER_NAME}" bash -c ' \
        cd /workspace/MultimodalSDK/examples/AutoOptimizeAndSample && \
        grep -v "en-core-web-sm" requirements.txt > requirements_filtered.txt && \
        pip install -r requirements_filtered.txt -i https://mirrors.aliyun.com/pypi/simple/ && \
        pip show en-core-web-sm >/dev/null 2>&1 || python -m spacy download en_core_web_sm \
    '

    log_success "VRAG environment setup completed"
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
            --max-model-len 120000 \
            --gpu-memory-utilization 0.70 \
            --tensor-parallel-size ${VLLM_TENSOR_PARALLEL} \
            --no-enable-prefix-caching \
            --host ${VLLM_HOST} \
            --port ${VLLM_PORT} >> /tmp/vllm_start.log 2>&1 &
        sleep 2 && ps aux | grep vllm | grep -v grep || echo \"vllm process not found\""

    log_info "Starting vLLM: /models/${VLLM_MODEL_PATH} -> ${VLLM_MODEL_NAME}, TP=${VLLM_TENSOR_PARALLEL}"
    docker exec "${VLLM_CONTAINER_NAME}" bash -c "${start_cmd}"

    # Wait for service to be ready
    log_info "Waiting for vLLM service to be ready..."
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
    log_info "Container logs:"
    docker logs --tail 50 "${VLLM_CONTAINER_NAME}" 2>&1
    exit 1
}

# ============================================================================
# Create VRAG Config
# ============================================================================
create_vrag_config() {
    log_info "Creating VRAG config file..."

    docker exec "${VRAG_CONTAINER_NAME}" bash -c "cat > /workspace/vrag_config.toml << EOF
blip_model_path = \"/models/${MODEL_BLIP}\"
whisper_model_path = \"/models/${MODEL_WHISPER}\"
mmdino_model_path = \"/models/${MODEL_MMDINO}\"
embedding_model_path = \"/models/${MODEL_EMBEDDING}\"
reranker_model_path = \"/models/${MODEL_RERANKER}\"
qwenvl_api_base = \"http://host.docker.internal:${VLLM_PORT}/v1\"
qwenvl_api_key = \"EMPTY\"
qwenvl_model_name = \"${VLLM_MODEL_NAME}\"
default_target_frame_count = 5
default_max_frames_num = 5
default_retrieval_ocr_top_k = 5
default_retrieval_ocr_related_frames_top_k = 5
default_retrieval_asr_top_k = 5
default_retrieval_asr_related_frames_top_k = 5
default_retrieval_asr_max_related_frames = 5
default_retrieval_fallback_uniform_samples_k = 5
EOF
"

    # Add host.docker.internal to /etc/hosts if not exists
    docker exec "${VRAG_CONTAINER_NAME}" bash -c "
        if ! grep -q 'host.docker.internal' /etc/hosts 2>/dev/null; then
            echo '172.17.0.1 host.docker.internal' >> /etc/hosts
        fi
    "

    log_success "VRAG config created"
}

# ============================================================================
# Start VRAG Service
# ============================================================================
start_vrag_service() {
    log_info "Starting VRAG service..."

    # Check if service is already running
    if curl -s "http://localhost:${VRAG_PORT}/status" &>/dev/null; then
        log_warn "VRAG service is already running at port ${VRAG_PORT}"
        return 0
    fi

    local start_cmd="cd /workspace/MultimodalSDK/examples/AutoOptimizeAndSample && \
        export PYTHONPATH=\$(pwd):\$PYTHONPATH && \
        export ASCEND_RT_VISIBLE_DEVICES=${VISIBLE_DEVICES} && \
        echo \"Starting VRAG service at \$(date)...\" >> /tmp/vrag_start.log && \
        python -m vrag.benchmark.video_rag \
            -c /workspace/vrag_config.toml \
            -H ${VRAG_HOST} \
            -p ${VRAG_PORT} >> /tmp/vrag_start.log 2>&1 &"

    log_info "Starting VRAG service..."
    docker exec -d "${VRAG_CONTAINER_NAME}" bash -c "${start_cmd}"

    # Wait for service to be ready
    log_info "Waiting for VRAG service to be ready..."
    local max_wait=600
    local waited=0
    while [[ $waited -lt $max_wait ]]; do
        if curl -s "http://localhost:${VRAG_PORT}/status" &>/dev/null; then
            log_success "VRAG service is ready at http://localhost:${VRAG_PORT}"
            return 0
        fi
        sleep 5
        waited=$((waited + 5))
        echo -n "."
    done

    echo ""
    log_error "VRAG service failed to start within ${max_wait} seconds"
    log_info "VRAG start log:"
    docker exec "${VRAG_CONTAINER_NAME}" cat /tmp/vrag_start.log 2>/dev/null || echo "No log found"
    log_info "Container logs:"
    docker logs --tail 50 "${VRAG_CONTAINER_NAME}" 2>&1
    exit 1
}

# ============================================================================
# Run Test
# ============================================================================
run_test_request() {
    log_info "Running E2E test request..."

    # Get video container path
    local video_filename="$(basename "${VIDEO_PATH}")"
    local video_container_path="/video/${video_filename}"

    # Build request payload
    local payload_file="/tmp/vrag_request_$$.json"
    cat > "${payload_file}" << EOF
{
    "video_path": "${video_container_path}",
    "question": "${QUESTION}"
}
EOF

    # Send request
    local response_file="/tmp/vrag_response_$$.txt"
    local start_time
    start_time=$(date +%s)

    log_info "Sending request to http://localhost:${VRAG_PORT}/ask..."
    log_info "Video: ${video_container_path} (host: ${VIDEO_PATH})"
    log_info "Question: ${QUESTION}"

    local http_code
    http_code=$(curl -s -w "%{http_code}" -X POST "http://localhost:${VRAG_PORT}/ask" \
        -H "Content-Type: application/json" \
        -d @"${payload_file}" \
        --max-time "${TIMEOUT}" \
        -o "${response_file}" 2>&1) || {
            log_error "Request failed"
            cat "${response_file}" 2>/dev/null
            rm -f "${payload_file}" "${response_file}"
            exit 1
        }

    rm -f "${payload_file}"

    local end_time
    end_time=$(date +%s)
    local processing_time=$((end_time - start_time))

    # Parse response
    if [[ "${http_code}" == "200" ]]; then
        log_success "Request completed successfully"
        echo ""
        echo "=========================================="
        echo "Test Result: PASS"
        echo "=========================================="
        echo "Processing time: ${processing_time}s"

        # Extract and display answer
        local answer
        answer=$(python3 -c "
import sys, json
with open('${response_file}', 'r') as f:
    data = json.load(f)
print(data.get('answer', ''))
" 2>/dev/null || cat "${response_file}")

        echo ""
        echo "Answer:"
        echo "${answer}"
        echo ""

        rm -f "${response_file}"
        return 0
    else
        log_error "Request failed with HTTP ${http_code}"
        echo ""
        echo "=========================================="
        echo "Test Result: FAIL"
        echo "=========================================="
        echo "Response body:"
        cat "${response_file}"
        echo ""

        rm -f "${response_file}"
        return 1
    fi
}

# ============================================================================
# Main
# ============================================================================
main() {
    echo "=========================================="
    echo "AutoOptimizeAndSample E2E Test Script"
    echo "=========================================="
    echo ""
    echo "Configuration:"
    echo "  vLLM Container: ${VLLM_CONTAINER_NAME}"
    echo "  VRAG Container: ${VRAG_CONTAINER_NAME}"
    echo "  Host Model Dir: ${HOST_MODEL_DIR} -> /models"
    echo "  vLLM Model: ${VLLM_MODEL_PATH} -> ${VLLM_MODEL_NAME}"
    echo "  vLLM TP: ${VLLM_TENSOR_PARALLEL}"
    echo "  vLLM Port: ${VLLM_PORT}"
    echo "  Enable vLLM: ${ENABLE_VLLM}"
    echo "  VRAG Port: ${VRAG_PORT}"
    echo "  SDK Branch: ${SDK_BRANCH}"
    echo "  Visible Devices: ${VISIBLE_DEVICES}"
    echo ""

    check_prerequisites

    if [[ "${ENABLE_VLLM}" == "true" ]]; then
        start_vllm_container
        start_vllm_service
    else
        log_warn "Skipping vLLM setup (ENABLE_VLLM=false)"
    fi

    start_vrag_container
    setup_vrag_environment
    create_vrag_config
    start_vrag_service

    echo ""
    echo "=========================================="
    echo "Running Test"
    echo "=========================================="

    if run_test_request; then
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
