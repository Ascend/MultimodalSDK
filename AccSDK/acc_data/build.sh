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
# Used to build project files

set -e
CURRENT_PATH=$(cd "$(dirname "$0")"; pwd)

function print_help_info()
{
    Magenta="\e[32m"
    ENDCOLOR="\e[0m"
    echo "$0 options"
    echo "options"
    echo -e "${Magenta}    -t <release|debug|clean|ut>${ENDCOLOR}  Build The default value is 'release'"
    echo -e "${Magenta}         release${ENDCOLOR}              Build in release mode"
    echo -e "${Magenta}         debug${ENDCOLOR}                Build in debug mode"
    echo -e "${Magenta}         clean${ENDCOLOR}                Deleting Temporary Files Generated During Build"
    echo -e "${Magenta}         ut${ENDCOLOR}                   Build in ut mode"
    echo -e "${Magenta}         fuzz${ENDCOLOR}                 Build in dt-fuzz mode"
    echo -e "${Magenta}    -c <on|off>${ENDCOLOR}               Instrument code coverage, default is no instrumentation when not specified, [default: 'off']"
    echo -e "${Magenta}    -s <on|off>${ENDCOLOR}               Address sanitizer, default is on when not specified, [default: 'off']"
    echo -e "${Magenta}    --enable_submodule_update${ENDCOLOR} git submodule update --init --recursive --remote"
    echo -e "${Magenta}    --disable_src${ENDCOLOR}             Do not perform source code compilation."
    echo -e "${Magenta}    -h|--help${ENDCOLOR}                 Show this message"
    echo "Example:"
    echo "  $0"
    echo "  $0 -t release"
    echo "  $0 -t debug"
    echo "  $0 -t debug -c ON -S ON"
    echo "  $0 -t clean"
}

function build_all()
{
    cd "${CURRENT_PATH}"
    if [ "$DISABLE_SRC_BUILD" = "True" ]; then
        return
    fi
    if [ ! -d "build" ]; then
            echo "Creating 'build' directory"
            mkdir build
    fi
    echo "USING_COVERAGE is $USING_COVERAGE"
    if ! bash buildscript/build_src.sh "$CMAKE_BUILD_TYPE" "$USING_COVERAGE" "$USING_XSCAN"; then
        echo "Failed to build_src" 1>&2
        return 1
    fi
    mkdir -p "${CURRENT_PATH}/output"
}
function read_input_args()
{
    BUILD_TYPE="RELEASE"
    if [ "$1" = "" ] || [ "$1" = "-t" ]; then
        BUILD_TYPE="RELEASE"
    fi
    ENABLE_SUBMODULE_UPDATE="False"
    USING_COVERAGE="False"
    DISABLE_COMPILE_OP="False"
    DISABLE_SRC_BUILD="False"
    echo "read_input_args"
    set -- $(getopt -o h::t:c:s: --long xsan:,isa:,os:,compiler:,instrument:,entry-point:,help,enable_submodule_update,disable_src -n "$0" -- "$@")
    while true
    do
        case "$1" in
            -t)
                BUILD_TYPE=$(echo "$2"|tr a-z A-Z|tr -d "'")
                shift
                ;;
            -c)
                USING_COVERAGE=$(echo "$2"|tr a-z A-Z|tr -d "'")
                shift
                ;;
            -h|--help)
                BUILD_TYPE="-H"
                shift
                break
                ;;
            --enable_submodule_update)
                ENABLE_SUBMODULE_UPDATE="True"
                shift
                break
                ;;
            --disable_src)
                DISABLE_SRC_BUILD="True"
                shift
                break
                ;;
            -s)
                USING_XSCAN=$(echo "$2"|tr a-z A-Z|tr -d "'")
                shift
                ;;
            --)
                if [ "$BUILD_TYPE" = "" ]; then
                    BUILD_TYPE="UNKNOWN"
                fi
                shift
                break
                ;;
            *)
                if [ "$BUILD_TYPE" = "" ]; then
                    BUILD_TYPE="UNKNOWN"
                fi
                break
                ;;
        esac
        shift
    done
}
function clean_build_files()
{
    source "${CURRENT_PATH}/buildscript/build_env.sh"
    rm -rf "${ACCDATA_ROOT_PATH}/output"
    mkdir -p "${ACCDATA_ROOT_PATH}/output"
    touch "${ACCDATA_ROOT_PATH}/output/.gitkeep"
    rm -rf "${ACCDATA_ROOT_PATH}/install"
    mkdir -p "${ACCDATA_ROOT_PATH}/install"
    touch "${ACCDATA_ROOT_PATH}/install/.gitkeep"
    rm -rf "${ACCDATA_ROOT_PATH}/build"
    mkdir -p "${ACCDATA_ROOT_PATH}/build"
    touch "${ACCDATA_ROOT_PATH}/build/.gitkeep"
    echo "Clean succeed!"
}
function package_output()
{
    echo "Packaging..."
    VERSION=$(grep -E '^project' CMakeLists.txt | sed -n 's/.*VERSION[[:space:]]\+\([0-9.]*\).*/\1/p')
    if [ -n "$VERSION" ]; then
        echo "Found version: $VERSION"
    else
        echo "Version not found in CMakeLists.txt, use default version 0.0.1"
        VERSION="0.0.1"
    fi
    ARCH="aarch64"
    PACKAGE_NAME="BeiMing-AccData-${VERSION}-1.Linux.${ARCH}.tar.gz"
    cp -r "${CURRENT_PATH}/dist/" "${CURRENT_PATH}/output/AccData"
    tar -czvf "${CURRENT_PATH}/output/${PACKAGE_NAME}" -C "${CURRENT_PATH}/output" AccData
    echo "Packaging completed: $PACKAGE_NAME"
}
function main()
{
    read_input_args $@

    echo "BUILD_TYPE=${BUILD_TYPE}"
    if [ "${BUILD_TYPE}" = "RELEASE" ] || [ "${BUILD_TYPE}" = "" ]; then
        CMAKE_BUILD_TYPE="Release"
    elif [ "${BUILD_TYPE}" = "UT" ]; then
        CMAKE_BUILD_TYPE="Ut"
    elif [ "${BUILD_TYPE}" = "DEBUG" ]; then
        CMAKE_BUILD_TYPE="Debug"
    elif [ "${BUILD_TYPE}" = "FUZZ" ]; then
        CMAKE_BUILD_TYPE="Fuzz"
    elif [ "${BUILD_TYPE}" = "-H" ] || [ "${BUILD_TYPE}" = "--HELP" ]; then
        print_help_info
        exit
    elif [ "${BUILD_TYPE}" = "CLEAN" ]; then
        clean_build_files
        exit
    else
        print_help_info
        exit 1
    fi
    if [ "${ENABLE_SUBMODULE_UPDATE}" = "True" ]; then
        git submodule update --init --recursive --remote
    fi
    if ! build_all; then
        exit 1
    fi
    if [ "${CMAKE_BUILD_TYPE}" = "Release" ]; then
        package_output
    fi
}

main "$@"
exit $?