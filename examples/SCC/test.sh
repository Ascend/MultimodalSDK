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
# End-to-end test script for SCC (Semantic Connected Components) visual token compression.
# This script automates the full testing workflow including:
#   1. Starting the vLLM container
#   2. Applying SCC patches
#   3. Starting the vLLM service
#   4. Running the image request test
#
# Usage: bash test.sh
#
# Prerequisites:
#   - Docker is installed and running
#   - Model files are downloaded to /models directory
#   - Test image is available

set -euo pipefail

# ============================================================================
# Configuration - Modify these variables to customize the test
# ============================================================================

# Container configuration
CONTAINER_NAME="vllm-ascend-scc"
IMAGE_NAME="quay.io/ascend/vllm-ascend:v0.18.0"

# Model mount configuration (宿主机模型目录 -> 容器内固定目录/models)
HOST_MODEL_DIR="/models"

# Model configuration
MODEL_PATH="Qwen2.5-VL-7B-Instruct"

# Test image configuration
IMAGE_PATH="/path/to/test.jpg"

# vLLM service configuration
VLLM_HOST="localhost"
VLLM_PORT="18000"
HOST="http://${VLLM_HOST}:${VLLM_PORT}"
MAX_TOKENS="256"
TIMEOUT="300"

# SCC configuration
SCC_RATIO="0.5"
SCC_TAU="0.98"

# MultimodalSDK configuration
SDK_BRANCH="master"

# Device configuration (固定2张卡)
VISIBLE_DEVICES="0,1"

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

    # Check if container image exists
    if ! docker image inspect "${IMAGE_NAME}" &> /dev/null; then
        log_warn "Docker image ${IMAGE_NAME} not found locally"
        log_info "Pulling image... (this may take a while)"
        docker pull "${IMAGE_NAME}"
    fi

    # Check model directory
    if [[ ! -d "${HOST_MODEL_DIR}" ]]; then
        log_error "Model directory does not exist: ${HOST_MODEL_DIR}"
        exit 1
    fi

    # Check model path
    if [[ ! -d "${HOST_MODEL_DIR}/${MODEL_PATH}" ]]; then
        log_error "Model path does not exist: ${HOST_MODEL_DIR}/${MODEL_PATH}"
        log_info "Please download the model first:"
        log_info "  ModelScope:"
        log_info "    pip install modelscope"
        log_info "    from modelscope.hub.snapshot_download import snapshot_download"
        log_info "    snapshot_download('Qwen/Qwen2.5-VL-7B-Instruct', local_dir='${HOST_MODEL_DIR}/${MODEL_PATH}')"
        exit 1
    fi

    # Check model files
    if [[ ! -f "${HOST_MODEL_DIR}/${MODEL_PATH}/config.json" ]]; then
        log_error "Model config.json not found in: ${HOST_MODEL_DIR}/${MODEL_PATH}"
        exit 1
    fi

    # Check test image
    if [[ ! -f "${IMAGE_PATH}" ]]; then
        log_error "Test image not found: ${IMAGE_PATH}"
        exit 1
    fi

    log_success "Prerequisites check passed"
}

# ============================================================================
# Stop and Remove Existing Container
# ============================================================================
cleanup_container() {
    log_info "Cleaning up existing container..."

    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
            log_info "Stopping container ${CONTAINER_NAME}..."
            docker stop "${CONTAINER_NAME}" 2>/dev/null || true
        fi
        log_info "Removing container ${CONTAINER_NAME}..."
        docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true
    fi

    log_success "Cleanup completed"
}

# ============================================================================
# Start vLLM Container
# ============================================================================
start_container() {
    log_info "Starting vLLM container..."

    docker run -itd --name "${CONTAINER_NAME}" \
        --net=host \
        --privileged \
        --device /dev/davinci_manager \
        --device /dev/devmm_svm \
        --device /dev/hisi_hdc \
        -v /usr/local/dcmi:/usr/local/dcmi \
        -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
        -v ${HOST_MODEL_DIR}:/models \
        "${IMAGE_NAME}" bash

    if [[ $? -ne 0 ]]; then
        log_error "Failed to start container"
        exit 1
    fi

    # Wait for container to be ready
    log_info "Waiting for container to start..."
    sleep 3

    # Check if container is running
    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        log_error "Container failed to start"
        docker logs "${CONTAINER_NAME}"
        exit 1
    fi

    log_success "Container started: ${CONTAINER_NAME}"
}

# ============================================================================
# Apply SCC Patches
# ============================================================================
apply_patches() {
    log_info "Applying SCC patches..."

    # Clone or update MultimodalSDK into container
    log_info "Checking for MultimodalSDK in container..."
    if ! docker exec "${CONTAINER_NAME}" test -d /workspace/MultimodalSDK 2>/dev/null; then
        log_info "Cloning MultimodalSDK (branch: ${SDK_BRANCH}) into container..."
        docker exec "${CONTAINER_NAME}" bash -c "mkdir -p /workspace && cd /workspace && git clone -b ${SDK_BRANCH} https://gitcode.com/Ascend/MultimodalSDK.git"
    else
        # Check current branch
        local current_branch
        current_branch=$(docker exec "${CONTAINER_NAME}" bash -c "cd /workspace/MultimodalSDK && git branch --show-current" 2>/dev/null || echo "")
        if [[ "${current_branch}" != "${SDK_BRANCH}" ]]; then
            log_info "Switching MultimodalSDK from branch '${current_branch}' to '${SDK_BRANCH}'..."
            docker exec "${CONTAINER_NAME}" bash -c "cd /workspace/MultimodalSDK && git fetch origin && git checkout -b ${SDK_BRANCH} origin/${SDK_BRANCH} 2>/dev/null || git checkout ${SDK_BRANCH}"
        else
            log_info "Updating MultimodalSDK on branch: ${SDK_BRANCH}..."
            docker exec "${CONTAINER_NAME}" bash -c "cd /workspace/MultimodalSDK && git pull origin ${SDK_BRANCH}"
        fi
    fi

    # Patches are in the cloned MultimodalSDK repository
    local patch_dir="/workspace/MultimodalSDK/examples/SCC/patches"
    if ! docker exec "${CONTAINER_NAME}" test -d "${patch_dir}" 2>/dev/null; then
        log_warn "Patches directory not found: ${patch_dir}"
        log_warn "Skipping patch application"
        return
    fi

    # Copy patches to a temp location
    log_info "Copying patches to container..."
    docker exec "${CONTAINER_NAME}" bash -c "mkdir -p /workspace/patches && cp ${patch_dir}/* /workspace/patches/"

    # Apply patches
    log_info "Applying patches..."
    docker exec "${CONTAINER_NAME}" bash -c "cd /vllm-workspace/vllm-ascend && \
        for patch in /workspace/patches/*.patch; do \
            if [[ -f \"\$patch\" ]]; then \
                echo \"Applying: \$patch\" && \
                git apply \"\$patch\" && \
                echo \"Success: \$patch\" || echo \"Failed: \$patch\"; \
            fi \
        done"

    log_success "Patches applied"
}


# ============================================================================
# Start vLLM Service
# ============================================================================
start_vllm_service() {
    log_info "Starting vLLM service with SCC enabled..."

    # Check if service is already running
    if curl -s "${HOST}/v1/models" &>/dev/null; then
        log_warn "vLLM service is already running at ${HOST}"
        log_info "Stopping existing service..."
        docker exec "${CONTAINER_NAME}" pkill -f "vllm serve" || true
        sleep 2
    fi

    # Prepare environment and start command (使用nohup记录日志)
    local start_cmd="cd /vllm-workspace/vllm-ascend && \
        export MODEL_PATH=/models/${MODEL_PATH} && \
        export VLLM_ASCEND_ENABLE_SCC=1 && \
        export VLLM_ASCEND_SCC_RATIO=${SCC_RATIO} && \
        export VLLM_ASCEND_SCC_TAU=${SCC_TAU} && \
        export VLLM_HOST=${VLLM_HOST} && \
        export PORT=${VLLM_PORT} && \
        ${VISIBLE_DEVICES:+export ASCEND_RT_VISIBLE_DEVICES=${VISIBLE_DEVICES}} && \
        echo \"Starting vLLM at \$(date)...\" >> /tmp/vllm_start.log && \
        echo \"PORT=\${PORT} VLLM_HOST=\${VLLM_HOST}\" >> /tmp/vllm_start.log && \
        nohup bash start_vllm_server.sh >> /tmp/vllm_start.log 2>&1 &
        sleep 2 && ps aux | grep vllm | grep -v grep || echo \"vllm process not found\""

    log_info "Executing: vllm serve with SCC (ratio=${SCC_RATIO}, tau=${SCC_TAU}, devices=${VISIBLE_DEVICES:-all})"
    docker exec "${CONTAINER_NAME}" bash -c "${start_cmd}"

    # Wait for service to be ready
    log_info "Waiting for vLLM service to be ready (this may take a few minutes)..."
    local max_wait=300
    local waited=0
    while [[ $waited -lt $max_wait ]]; do
        if curl -s --max-time 300 "${HOST}/v1/models" &>/dev/null; then
            log_success "vLLM service is ready"
            return 0
        fi
        sleep 10
        waited=$((waited + 10))
        echo -n "."
    done

    echo ""
    log_error "vLLM service failed to start within ${max_wait} seconds"
    log_info "vLLM start log:"
    docker exec "${CONTAINER_NAME}" cat /tmp/vllm_start.log 2>/dev/null || echo "No log found"
    log_info "Container logs:"
    docker logs --tail 50 "${CONTAINER_NAME}" 2>&1
    exit 1
}

# ============================================================================
# Run Test Request
# ============================================================================
run_test_request() {
    log_info "Running E2E test request..."

    local model_name
    model_name="/models/${MODEL_PATH}"

    # Encode image to base64
    local image_b64
    image_b64=$(base64 -w 0 "${IMAGE_PATH}")

    # Build request payload to temp file
    local payload_file="/tmp/scc_request_$$.json"
    cat > "${payload_file}" << EOF
{
    "model": "${model_name}",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,${image_b64}"}
                },
                {"type": "text", "text": "Please describe this image in detail."}
            ]
        }
    ],
    "max_tokens": ${MAX_TOKENS},
    "temperature": 0.0
}
EOF

    # Send request using file
    local start_time
    start_time=$(date +%s)
    local response_file="/tmp/scc_response_$$.txt"

    log_info "Sending request to ${HOST}/v1/chat/completions..."
    curl -s -w "\n%{http_code}" -X POST "${HOST}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d @"${payload_file}" \
        --max-time "${TIMEOUT}" > "${response_file}" 2>&1 || {
            log_error "Request failed"
            cat "${response_file}" 2>/dev/null
            rm -f "${payload_file}" "${response_file}"
            exit 1
        }

    rm -f "${payload_file}"

    local end_time
    end_time=$(date +%s)
    local processing_time=$((end_time - start_time))

    # Parse response from file
    local http_code
    http_code=$(tail -n1 "${response_file}")
    local body
    body=$(sed '$d' "${response_file}")

    if [[ "${http_code}" == "200" ]]; then
        log_success "Request completed successfully"
        echo ""
        echo "=========================================="
        echo "Test Result: PASS"
        echo "=========================================="
        echo "Processing time: ${processing_time}s"

        # Extract and display answer
        local answer
        answer=$(echo "${body}" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data['choices'][0]['message']['content'])" 2>/dev/null || echo "${body}")

        echo ""
        echo "Answer:"
        # Use printf to safely handle special characters in LLM response
        printf '%s' "${answer}" | head -c 500
        if [[ ${#answer} -gt 500 ]]; then
            echo "..."
        fi
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
        # Use printf to safely handle special characters in error response
        printf '%s' "${body}"
        echo ""

        rm -f "${response_file}"
        return 1
    fi
}

# ============================================================================
# Cleanup
# ============================================================================
cleanup() {
    log_info "Cleaning up..."
    # Note: Container is kept running for debugging
    log_success "Cleanup completed"
}

# ============================================================================
# Main
# ============================================================================
main() {
    echo "=========================================="
    echo "SCC E2E Test Script"
    echo "=========================================="
    echo ""
    echo "Configuration:"
    echo "  Container Name: ${CONTAINER_NAME}"
    echo "  Image Name: ${IMAGE_NAME}"
    echo "  Host Model Dir: ${HOST_MODEL_DIR} -> /models"
    echo "  Model Path: ${MODEL_PATH} -> /models/${MODEL_PATH}"
    echo "  Image Path: ${IMAGE_PATH}"
    echo "  SCC Ratio: ${SCC_RATIO}"
    echo "  SCC Tau: ${SCC_TAU}"
    echo "  SDK Branch: ${SDK_BRANCH}"
    echo "  Visible Devices: ${VISIBLE_DEVICES:-all}"
    echo "  vLLM Host: ${HOST}"
    echo ""

    check_prerequisites
    cleanup_container
    start_container
    apply_patches
    start_vllm_service

    echo ""
    echo "=========================================="
    echo "Running Test"
    echo "=========================================="

    if run_test_request; then
        echo ""
        log_success "E2E test completed successfully!"
        cleanup
        exit 0
    else
        echo ""
        log_error "E2E test failed!"
        cleanup
        exit 1
    fi
}

main
