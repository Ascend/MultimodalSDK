#!/usr/bin/env bash
set -e

echo "======================================"
echo "[INFO] Pre-smoke test start"
echo "======================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRESMOKE_DIR="${SCRIPT_DIR}/test/presmoke"

if [ ! -d "$PRESMOKE_DIR" ]; then
    echo "[ERROR] presmoke directory not found: $PRESMOKE_DIR"
    exit 1
fi

# ------------------------------------------------------------------------------
# Execute install.sh
# ------------------------------------------------------------------------------
INSTALL_SCRIPT="$PRESMOKE_DIR/install.sh"

if [ ! -f "$INSTALL_SCRIPT" ]; then
    echo "[ERROR] Install script not found: $INSTALL_SCRIPT"
    exit 1
fi

echo "[INFO] Running install script: $(basename "$INSTALL_SCRIPT")"
echo "--------------------------------------"

chmod u+x "$INSTALL_SCRIPT"
bash "$INSTALL_SCRIPT"

echo "[INFO] Install script completed successfully"

# ------------------------------------------------------------------------------
# performing tests
# ------------------------------------------------------------------------------
echo "--------------------------------------"
echo "[INFO] Running mm_acc test cases with pytest"
echo "--------------------------------------"

INSTALL_PATH="${SCRIPT_DIR}/presmoke_install"

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH
source "${INSTALL_PATH}/multimodal/script/set_env.sh"
export PYTHONPATH="${PYTHONPATH}:$PRESMOKE_DIR"

pip3 install pillow==12.0.0
pip3 install torchvision==0.24.1
pip3 install librosa

# Run pytest on all test cases in mm_acc directory
python3 -m pytest "$PRESMOKE_DIR/mm_acc/" -v

if [ $? -eq 0 ]; then
    echo "[INFO] All mm_acc test cases passed"
else
    echo "[ERROR] Some mm_acc test cases failed"
    exit 1
fi

echo

echo "======================================"
echo "[SUCCESS] All presmoke cases PASSED"
echo "======================================"
