#!/bin/bash
set -e
serviceName=$1

WORKSPACE=$(echo "$(pwd)" | awk -F'/output' '{print $1}')
echo "WORKSPACE=${WORKSPACE}"
cd "${WORKSPACE}"
branch=$(git rev-parse --abbrev-ref HEAD)
commit_id=$(git rev-parse HEAD)
echo "branch=${branch}  commit_id=${commit_id}"
cd -

build_time="$(date -u -d '+8 hours' +%Y%m%d%H%M%S)"

# 遍历当前目录下所有文件（排除已经生成的 .json 和 .sha256sum，避免自循环）
for file in ./*; do
    # 跳过目录、跳过自身产物
    [ -f "$file" ] || continue
    case "$file" in
        *.json|*.sha256sum) continue ;;
    esac

    filename=$(basename "$file")
    echo "===== processing: ${filename} ====="

    # 计算 sha256
    sha256_val=$(sha256sum "$filename" | awk '{print $1}')

    # 生成 .json 元数据
    cat > "${filename}.json" <<EOF
{
    "sha256Sum": "${sha256_val}",
    "repoInfo": [{
        "repoUrl": "http://gitcode.com/Ascend/${serviceName}.git",
        "repoBranch": "${branch}",
        "commitId": "${commit_id}"
    }],
    "buildTime": "${build_time}"
}
EOF

    # 生成 .sha256sum
    echo "${sha256_val}  ${filename}" > "${filename}.sha256sum"
done

ls -al ./
echo "----- generated json & sha256sum -----"
for f in ./*.json ./*.sha256sum; do
    [ -f "$f" ] || continue
    echo ">>> $f"
    cat "$f"
    echo
done
