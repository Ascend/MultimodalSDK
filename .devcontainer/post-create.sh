#!/bin/bash
# Post-create hook for MultimodalSDK Dev Container.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_PREFIX="${PYTHON_PREFIX:-/opt/buildtools/python-3.11.4}"

verify_sha256() {
    local file="$1" expected="$2"
    local actual
    actual="$(sha256sum "$file" | awk '{print $1}')"
    if [ "$actual" != "$expected" ]; then
        echo "[ERROR] SHA256 mismatch for $(basename "$file"): expected ${expected}, got ${actual}" >&2
        return 1
    fi
}

mkdir -p /root/.pip /root/.config/pip
printf '%s\n' \
    '[global]' \
    'index-url=https://mirrors.aliyun.com/pypi/simple/' \
    'trusted-host=mirrors.aliyun.com' \
    'timeout=200' \
    > /root/.pip/pip.conf
cp /root/.pip/pip.conf /root/.config/pip/pip.conf

echo "[INFO] Installing Python development dependencies..."
"${PYTHON_PREFIX}/bin/pip3" install -r "${REPO_ROOT}/.devcontainer/requirements-dev.txt" \
    -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
rm -rf /root/.cache/pip

# Git LFS is required by this repo's post-checkout hook used during pre-commit stash/restore.
if ! command -v git-lfs >/dev/null 2>&1; then
    echo "[INFO] Installing git-lfs..."
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq && apt-get install -y --no-install-recommends git-lfs \
        && apt-get clean && rm -rf /var/lib/apt/lists/*
fi

echo "[INFO] Installing pre-commit hooks..."
pre-commit install
pre-commit install-hooks || echo "[WARN] pre-commit install-hooks failed; run it manually later." >&2

GITLEAKS_BIN="${REPO_ROOT}/gitleaks"
if [ ! -x "${GITLEAKS_BIN}" ]; then
    echo "[INFO] Downloading gitleaks binary for pre-commit..."
    GITLEAKS_VERSION="8.24.2"
    ARCH="$(uname -m)"
    case "${ARCH}" in
        aarch64|arm64)
            GITLEAKS_ARCH="arm64"
            # SHA256 from gitleaks v8.24.2 release checksums (gitleaks_8.24.2_checksums.txt).
            GITLEAKS_SHA256="574a6d52573c61173add7ddb5e3cc68c0e82cb0735818a1eeb9a0a2de1643fbc"
            ;;
        x86_64|amd64)
            GITLEAKS_ARCH="x64"
            GITLEAKS_SHA256="fa0500f6b7e41d28791ebc680f5dd9899cd42b58629218a5f041efa899151a8e"
            ;;
        *)
            echo "[WARN] Unsupported architecture for gitleaks download: ${ARCH}" >&2
            GITLEAKS_ARCH=""
            ;;
    esac
    if [ -n "${GITLEAKS_ARCH}" ]; then
        GITLEAKS_URL="https://github.com/gitleaks/gitleaks/releases/download/v${GITLEAKS_VERSION}/gitleaks_${GITLEAKS_VERSION}_linux_${GITLEAKS_ARCH}.tar.gz"
        TMP_DIR="$(mktemp -d)"
        if curl -fsSL "${GITLEAKS_URL}" -o "${TMP_DIR}/gitleaks.tar.gz"; then
            if verify_sha256 "${TMP_DIR}/gitleaks.tar.gz" "${GITLEAKS_SHA256}"; then
                tar -xzf "${TMP_DIR}/gitleaks.tar.gz" -C "${TMP_DIR}"
                install -m 0755 "${TMP_DIR}/gitleaks" "${GITLEAKS_BIN}"
                echo "[INFO] gitleaks installed to ${GITLEAKS_BIN}"
            else
                echo "[ERROR] gitleaks download failed integrity check; refusing to install." >&2
            fi
        else
            echo "[WARN] Failed to download gitleaks; pre-commit gitleaks hook may fail." >&2
        fi
        rm -rf "${TMP_DIR}"
    fi
fi

echo "[INFO] Fetching build dependencies (opensource, makeself, etc.)..."
chmod +x "${REPO_ROOT}/build_script/fetch_deps.sh"
bash "${REPO_ROOT}/build_script/fetch_deps.sh"
