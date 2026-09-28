#!/bin/bash
set -e
serviceName=$1


# 读取变更文件列表
MODIFY_FILE="change.txt"
FILE_ARRAY=()
exec 3< "${MODIFY_FILE}"
while IFS= read -r raw_line <&3 || [[ -n "${raw_line}" ]]; do
    fp=$(echo "${raw_line}" | sed -e 's/^"//' -e 's/"$//' -e 's/^[ \t]*//' -e 's/[ \t]*$//')
    if [[ -n "${fp}" ]]; then
        FILE_ARRAY+=("${fp}")
    fi
done
exec 3<&-

# 过滤只保留源码文件传入检测
CHECK_ARRAY=()
for file in "${FILE_ARRAY[@]}"; do
    if [[ "${file}" =~ [.](cpp|cc|cxx|c)$ ]]; then
        CHECK_ARRAY+=("${file}")
    fi
done

if [[ ${#CHECK_ARRAY[@]} -eq 0 ]]; then
    echo "INFO: 本次变更没有 .cpp / .cc / .cxx / .c 文件，无需代码检查，脚本退出"
    exit 0
fi

echo "===== 开始安装clang与pre-commit并执行静态检测 ====="
ARCH=$(uname -m)
echo "检测到架构: ${ARCH}"

# 根据架构选择包名
if [[ "${ARCH}" == "x86_64" ]]; then
    PKG_NAME="LLVM-22.1.8-Linux-X64.tar.xz"
elif [[ "${ARCH}" == "aarch64" ]]; then
    PKG_NAME="LLVM-22.1.8-Linux-ARM64.tar.xz"
else
    echo "ERROR: 不支持的架构 ${ARCH}"
    exit 1
fi

# 下载地址
RAW_URL="https://mindcluster.obs.cn-north-4.myhuaweicloud.com/blueImageDependency/clang-tidy/22.1.8/${PKG_NAME}"
DOWNLOAD_URL="${RAW_URL}"

echo "准备下载包: ${PKG_NAME}"
wget -q --no-host-directories -c --no-check-certificate "${DOWNLOAD_URL}"

mkdir -p clang-tools
# 解压
tar -Jxf "${PKG_NAME}" -C ./clang-tools --strip-components=1

# PATH
export PATH=$PATH:${ATOMGIT_WORKSPACE}/clang-tools/bin

# 验证
clang-tidy --version


python3 -m pip install --upgrade pre-commit
echo "===== precommit ====="
python3 -m pre_commit --version
cd ./${serviceName}
echo "待检测C/C++源文件总数：${#CHECK_ARRAY[@]}"
# cat ./pre-commit/clang-tidy-run.sh
dos2unix ./pre-commit/*.sh
echo "实际传参：${CHECK_ARRAY[@]}"
bash ./pre-commit/clang-tidy-run.sh ${CHECK_ARRAY[@]}
