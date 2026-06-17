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
# Determine which tests to run based on changed files
# ------------------------------------------------------------------------------
RUN_ALL_TESTS=true
TEST_TARGETS=()
CHANGE_FILE="${SCRIPT_DIR}/../change.txt"
CONFIG_FILE="${SCRIPT_DIR}/presmoke_config.yaml"

echo "[INFO] Checking changed files from $CHANGE_FILE..."

# Use Python to parse config and determine test strategy
if [ -f "$CHANGE_FILE" ] && [ -f "$CONFIG_FILE" ]; then
    CHANGED_FILES=$(cat "$CHANGE_FILE" | tr -d '\r')

    if [ -n "$CHANGED_FILES" ]; then
        echo "[INFO] Changed files:"
        echo "$CHANGED_FILES" | sed 's/^/  - /'

        # Pass changed files to Python to determine test strategy
        PYTHON_SCRIPT="
import yaml
import sys
import os
import traceback

try:
    SCRIPT_DIR = sys.argv[1]
    CONFIG_FILE = os.path.join(SCRIPT_DIR, 'presmoke_config.yaml')
    CHANGE_FILE = os.path.join(SCRIPT_DIR, '../change.txt')

    with open(CONFIG_FILE, 'r') as f:
        config = yaml.safe_load(f)

    with open(CHANGE_FILE, 'r') as f:
        changed_files = [line.rstrip('\\r').strip() for line in f if line.strip()]

    # Convert all path separators to / (Linux style)
    changed_files = [f.replace('\\\\', '/') for f in changed_files]

    all_test_patterns = config.get('all_test', [])
    path_mapping = config.get('path_mapping', {})
    single_file_test_dirs = config.get('single_file_test_dirs', [])

    run_all_test = False
    test_dirs = []
    test_files = []

    # Check if any file matches all_test patterns
    import fnmatch
    for f in changed_files:
        for pattern in all_test_patterns:
            matched = False
            if '*' in pattern:
                # Any pattern with *, use fnmatch to handle all cases
                if fnmatch.fnmatch(f, pattern):
                    matched = True
            else:
                # No *, use prefix or exact match
                if f.startswith(pattern) or f == pattern:
                    matched = True

            if matched:
                run_all_test = True
                break
        if run_all_test:
            break

    if run_all_test:
        print('all_test')
        sys.exit(0)

    # Check single_file_test_dirs
    for f in changed_files:
        for dir_prefix in single_file_test_dirs:
            if f.startswith(dir_prefix):
                # Check if it's a .py file
                if f.endswith('.py'):
                    full_test_file = os.path.join(SCRIPT_DIR, f)
                    if full_test_file not in test_files:
                        test_files.append(full_test_file)

    # Check path_mapping
    for f in changed_files:
        for src_path, test_dir in path_mapping.items():
            if f == src_path or f.startswith(src_path):
                full_test_dir = os.path.join(SCRIPT_DIR, test_dir)
                if full_test_dir not in test_dirs:
                    test_dirs.append(full_test_dir)

    # Determine what to run: collect all targets
    all_test_targets = []
    for f in test_files:
        all_test_targets.append(f)
    for d in test_dirs:
        all_test_targets.append(d)

    if all_test_targets:
        print('test_targets')
        for t in all_test_targets:
            print(t)
    else:
        print('skip_test')
except Exception as e:
    print('error')
    print(traceback.format_exc())
"

        RESULT=$(python3 -c "$PYTHON_SCRIPT" "$SCRIPT_DIR" 2>&1)
        FIRST_LINE=$(echo "$RESULT" | head -1)

        if [ "$FIRST_LINE" = "all_test" ]; then
            echo "[INFO] All test trigger matched - will run all tests"
            RUN_ALL_TESTS=true
        elif [ "$FIRST_LINE" = "test_targets" ]; then
            echo "[INFO] Test targets found:"
            while IFS= read -r line; do
                if [ "$line" != "test_targets" ] && [ -n "$line" ]; then
                    TEST_TARGETS+=("$line")
                    echo "  - $line"
                fi
            done <<< "$RESULT"
            RUN_ALL_TESTS=false
        elif [ "$FIRST_LINE" = "skip_test" ]; then
            echo "[INFO] No matching paths found - will skip all presmoke tests"
            echo "======================================"
            echo "[SUCCESS] Presmoke test skipped"
            echo "======================================"
            exit 0
        else
            echo "[WARNING] Failed to determine test strategy - will run all tests"
            echo "[DEBUG] Python output:"
            echo "$RESULT"
            RUN_ALL_TESTS=true
        fi
    else
        echo "[INFO] No changed files in $CHANGE_FILE - will run all tests"
    fi
else
    if [ ! -f "$CHANGE_FILE" ]; then
        echo "[INFO] $CHANGE_FILE not found - will run all tests"
    fi
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "[INFO] $CONFIG_FILE not found - will run all tests"
    fi
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

# torchvision is used only as a reference implementation for presmoke accuracy checks.
# Keep it aligned with torch/torch-npu 2.9.1; it is not a fixed Multimodal SDK runtime dependency.
pip3 install "pillow>=11.2.1"
pip3 install torchvision==0.24.1
pip3 install librosa
pip3 install torch-npu==2.9.1
pip3 install transformers==4.51.3
pip3 install einops

# Run pytest
TEST_PASSED=true

if [ "$RUN_ALL_TESTS" = true ]; then
    echo "[INFO] Running all test cases"
    python3 -m pytest "$PRESMOKE_DIR/mm_acc/" -v
    if [ $? -ne 0 ]; then
        TEST_PASSED=false
    fi
elif [ ${#TEST_TARGETS[@]} -gt 0 ]; then
    echo "[INFO] Running specific test targets:"
    python3 -m pytest "${TEST_TARGETS[@]}" -v
    if [ $? -ne 0 ]; then
        TEST_PASSED=false
    fi
else
    echo "[WARNING] No test targets found, but RUN_ALL_TESTS is false - this should not happen"
    echo "[INFO] Will run all tests as fallback"
    python3 -m pytest "$PRESMOKE_DIR/mm_acc/" -v
    if [ $? -ne 0 ]; then
        TEST_PASSED=false
    fi
fi

if [ "$TEST_PASSED" = true ]; then
    echo "[INFO] All test cases passed"
else
    echo "[ERROR] Some test cases failed"
    exit 1
fi

echo

echo "======================================"
echo "[SUCCESS] All presmoke cases PASSED"
echo "======================================"
