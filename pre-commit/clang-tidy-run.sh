#!/usr/bin/env bash
# Run clang-tidy with include paths required by AccSDK and acc_data.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ACC_SDK="${ROOT}/AccSDK"
ACC_DATA_CPP="${ACC_SDK}/acc_data/src/cpp"

CLANG_TIDY="${CLANG_TIDY:-clang-tidy}"
if ! command -v "${CLANG_TIDY}" >/dev/null 2>&1 && [ -x /opt/llvm/bin/clang-tidy ]; then
    CLANG_TIDY=/opt/llvm/bin/clang-tidy
fi

if ! command -v "${CLANG_TIDY}" >/dev/null 2>&1; then
    echo "clang-tidy not found; install clang-tidy or set CLANG_TIDY" >&2
    exit 1
fi

is_3rdparty_path() {
    case "$1" in
        */3rdparty/* | */third_party/*) return 0 ;;
        *) return 1 ;;
    esac
}

# Drop diagnostics (and their trailing notes) from vendor trees under 3rdparty/ or third_party/.
filter_3rdparty_diagnostics() {
    awk '
        /^Error while processing / {
            if ($0 ~ /\/(3rdparty|third_party)\//) {
                skip_block = 1
                pending_error = ""
                next
            }
            pending_error = $0
            next
        }
        /^error: too many errors emitted/ {
            next
        }
        /^[^ ]+:[0-9]+:[0-9]+:/ {
            if ($0 ~ /: note:/) {
                if (skip_block) {
                    next
                }
                print
                next
            }
            if ($1 ~ /\/(3rdparty|third_party)\//) {
                skip_block = 1
                next
            }
            if (pending_error != "") {
                print pending_error
                pending_error = ""
            }
            skip_block = 0
            print
            next
        }
        skip_block { next }
        { print }
    '
}

EXTRA_ARGS=(
    --config-file="${ROOT}/.clang-tidy"
    --quiet
    # LLVM Regex is POSIX ERE (no PCRE lookaround); rely on AWK to drop 3rdparty diagnostics.
    --header-filter='.*'
    --extra-arg=-std=c++17
)

add_include() {
    if [ -d "$1" ]; then
        EXTRA_ARGS+=(--extra-arg=-I"$1")
    fi
}

add_system_include() {
    if [ -d "$1" ]; then
        EXTRA_ARGS+=(--extra-arg=-isystem"$1")
    fi
}

# AccSDK public headers
add_include "${ACC_SDK}/include"
add_include "${ACC_SDK}/source/inc"
add_include "${ACC_SDK}/source/py/include"

# acc_data internal headers (resolves common/*.h, operator/*.h, etc.)
add_include "${ACC_DATA_CPP}"
add_include "${ACC_DATA_CPP}/interface"

# Third-party / toolkit headers (use -isystem to avoid tidy noise in vendor code)
add_system_include "${ACC_SDK}/opensource/libjpeg-turbo/include"
add_system_include "${ACC_SDK}/opensource/FFmpeg/include"
add_system_include "${ACC_SDK}/opensource/soxr/include"
add_system_include "${ACC_SDK}/acc_data/3rdparty/pybind/pybind11/include"

ASCEND_HOME="${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}"
add_system_include "${ASCEND_HOME}/acllib/include"
add_system_include /usr/local/Ascend/driver

PYTHON_INCLUDE="${PYTHON_INCLUDE:-}"
if [ -z "${PYTHON_INCLUDE}" ]; then
    PYTHON_INCLUDE="$(python3 -c 'import sysconfig; print(sysconfig.get_path("include"))' 2>/dev/null || true)"
fi
if [ -n "${PYTHON_INCLUDE}" ] && [ -d "${PYTHON_INCLUDE}" ]; then
    add_system_include "${PYTHON_INCLUDE}"
fi

if [ -f "${ROOT}/compile_commands.json" ]; then
    EXTRA_ARGS+=(-p "${ROOT}")
fi

TARGETS=()
for file in "$@"; do
    if is_3rdparty_path "${file}"; then
        continue
    fi
    TARGETS+=("${file}")
done

if [ "${#TARGETS[@]}" -eq 0 ]; then
    exit 0
fi

TIDY_STATUS=0
OUTPUT="$("${CLANG_TIDY}" "${EXTRA_ARGS[@]}" -- "${TARGETS[@]}" 2>&1)" || TIDY_STATUS=$?
FILTERED="$(printf '%s\n' "${OUTPUT}" | filter_3rdparty_diagnostics)"

if [ -n "${FILTERED}" ]; then
    printf '%s\n' "${FILTERED}"
fi

# Fail on errors only. WarningsAsErrors elevates gated checks; naming stays advisory.
if printf '%s\n' "${FILTERED}" | grep -qE ':[0-9]+:[0-9]+: error:'; then
    exit 1
fi

# Non-zero without diagnostic lines => tool/config failure (do not silently pass).
if [ "${TIDY_STATUS}" -ne 0 ]; then
    if [ -z "${FILTERED}" ] && [ -n "${OUTPUT}" ]; then
        printf '%s\n' "${OUTPUT}" >&2
    fi
    echo "clang-tidy failed with exit code ${TIDY_STATUS}" >&2
    exit 1
fi
