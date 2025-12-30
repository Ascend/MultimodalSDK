#!/bin/bash

# -------------------------------------------------------------------------
#  This file is part of the MultimodalSDK project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
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
# -------------------------------------------------------------------------
# Description: Build script of ACC SDK opensource.
# Author: ACC SDK
# Create: 2025
# History: NA
set -e

OPEN_SOURCE_DIR="../opensource"
SCRIPT_DIR_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OPEN_SOURCE_ABS_ROOT=$(realpath "${SCRIPT_DIR_ROOT}/${OPEN_SOURCE_DIR}")

# libjpeg-turbo
compile_libjpeg_turbo() {
    local libjpeg_root="${OPEN_SOURCE_ABS_ROOT}/libjpeg-turbo"
    local src_dir="${libjpeg_root}"          # 解压到这里
    local build_dir="${src_dir}/build"
    local install_dir="${libjpeg_root}"      # 安装到这里

    echo "start compile libjpeg-turbo..."

    # 创建 build 目录
    mkdir -p "${build_dir}" || return 1

    export CFLAGS="-O2 -fPIC -fstack-protector-strong -D_FORTIFY_SOURCE=2 -ftrapv"
    export CXXFLAGS="$CFLAGS"
    export LDFLAGS="-Wl,-z,relro,-z,now -Wl,-z,noexecstack -s"

    # 构建
    cmake -S "${src_dir}" -B "${build_dir}" -DCMAKE_INSTALL_PREFIX="${install_dir}" -DCMAKE_C_FLAGS="${CFLAGS}" -DCMAKE_CXX_FLAGS="${CXXFLAGS}" -DCMAKE_EXE_LINKER_FLAGS="${LDFLAGS}" -DCMAKE_SHARED_LINKER_FLAGS="${LDFLAGS}" -DCMAKE_SKIP_RPATH=TRUE -DCMAKE_SKIP_BUILD_RPATH=TRUE -DCMAKE_SKIP_INSTALL_RPATH=TRUE || return 1
    cmake --build "${build_dir}" -j64 || return 1
    cmake --install "${build_dir}" || return 1

    echo "libjpeg-turbo compile success"
    export LD_LIBRARY_PATH="${install_dir}/lib:${LD_LIBRARY_PATH}"
    echo "libjpeg-turbo ld_library_path set success"

}


# ffmpeg
compile_ffmpeg() {

    local script_dir
    script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
    local open_source_abs
    open_source_abs=$(realpath "${script_dir}/${OPEN_SOURCE_DIR}")

    local ffmpeg_root="${open_source_abs}/FFmpeg"
    local install_dir="${open_source_abs}/FFmpeg"

    export CFLAGS="-O2 -fPIC -fstack-protector-strong -D_FORTIFY_SOURCE=2 -ftrapv"
    export CXXFLAGS="${CFLAGS}"
    export LDFLAGS="-Wl,-z,relro,-z,now -Wl,-z,noexecstack"

    echo "start compile FFmpeg..."

    if [ ! -d "${ffmpeg_root}" ]; then
        echo "FFmpeg source directory does not exist: ${ffmpeg_root}"
        return 1
    fi

    # 判断是否已经 configure 过
    if [ ! -f "${ffmpeg_root}/config.log" ]; then
        echo "run FFmpeg configure..."
        (cd "${ffmpeg_root}" && ./configure --prefix="${install_dir}" --disable-demuxer=sap --disable-muxer=sap --disable-network --disable-x86asm --enable-shared --disable-static --extra-cflags="${CFLAGS}" --extra-ldflags="${LDFLAGS}") || { echo "FFmpeg configure failed"; return 1; }
    else
        echo "FFmpeg already configured, skipping configure step"
    fi

    echo "start make FFmpeg..."
    (cd "${ffmpeg_root}" && make -j64) || { echo "FFmpeg make failed"; return 1; }

    echo "start make install FFmpeg..."
    (cd "${ffmpeg_root}" && make install) || { echo "FFmpeg make install failed"; return 1; }

    echo "FFmpeg compile success"
    export LD_LIBRARY_PATH="${install_dir}/lib:${LD_LIBRARY_PATH}"
    echo "FFmpeg ld_library_path set success"
}



main() {
    linux_type=$(uname -m)
    compile_libjpeg_turbo &
    pid1=$!
    compile_ffmpeg &
    pid2=$!

    wait $pid1
    status1=$?
    wait $pid2
    status2=$?

    if [ $status1 -ne 0 ]; then
        echo "libjpeg-turbo compile failed"
        exit 1
    fi
    if [ $status2 -ne 0 ]; then
        echo "ffmpeg compile failed"
        exit 1
    fi

    echo "opensource compilation task success"
}

main
