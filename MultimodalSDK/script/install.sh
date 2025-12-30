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

# 自定义变量
install_path="${USER_PWD}"
PACKAGE_LOG_NAME=MultiModalSDk
LOG_SIZE_THRESHOLD=$((10*1024*1024))
declare -A param_dict=()               # 参数个数统计
version_number=""
mxsdk_manufacture_name=""
mxsdk_new_ame=""
arch_name="aarch64"
invalid_param_flag=n
self_dir=""

info_record_path="${HOME}/log/multimodal"
info_record_file="deployment.log"

#标识符
install_flag=n
print_version_flag=n
install_path_flag=n
upgrade_flag=n
quiet_flag=n

ms_deployment_log_rotate() {
  if [ -L "${info_record_path}" ]; then
    echo "The directory path of deployment.log cannot be a symlink." >&2
    exit 1
  fi
  if [[ ! -d "${info_record_path}" ]];then
    mkdir -p "${info_record_path}"
    chmod 750 "${info_record_path}"
  fi
  record_file_path="${info_record_path}"/"${info_record_file}"
  if [ -L "${record_file_path}" ]; then
    echo "The deployment.log cannot be a symlink." >&2
    exit 1
  fi
  if [[ ! -f "${record_file_path}" ]];then
    touch "${record_file_path}" 2>/dev/null
  fi
  record_file_path_bk="${info_record_path}"/"${info_record_file}".bk
  if [ -L "${record_file_path_bk}" ]; then
    echo "The deployment.log.bk cannot be a symlink." >&2
    exit 1
  fi
  log_size=$(find "${record_file_path}" -exec ls -l {} \; | awk '{ print $5 }')
  if [[ "${log_size}" -ge "${LOG_SIZE_THRESHOLD}" ]];then
    mv -f "${record_file_path}" "${record_file_path_bk}"
    touch "${record_file_path}" 2>/dev/null
    chmod 400 "${record_file_path_bk}"
  fi
  chmod 600 "${record_file_path}"
}

ms_log()
{
  ms_deployment_log_rotate
  record_file_path="${info_record_path}/${info_record_file}"
  chmod 640 "${record_file_path}"
  user_ip=$(who am i | awk '{print $NF}' | sed 's/[()]//g')
  [[ -z "${user_ip}" ]] && user_ip="localhost"
  user_name=$(whoami)
  host_name=$(hostname)
  timestamp=$(date "+%Y-%m-%d %H:%M:%S")

  log_line="[${timestamp}][${user_ip}][${user_name}][${host_name}]: $1"
  echo "${log_line}" >> "${record_file_path}"
  chmod 440 "${record_file_path}"
  echo "$1"
}



function print() {
  # 将关键信息打印到屏幕上
  echo "[${PACKAGE_LOG_NAME}] [$(date +%Y%m%d-%H:%M:%S)] [user: ${user_n}] [${ip_n}] [$1] $2"
}


readonly user_n=$(whoami)
readonly WHO_PATH=$(which who)
readonly CUT_PATH=$(which cut)
ip_n=$(${WHO_PATH} -m | ${CUT_PATH} -d '(' -f 2 | ${CUT_PATH} -d ')' -f 1)
if [ "${ip_n}" = "" ]; then
  ip_n="localhost"
fi
readonly ip_n

###  公用函数
function print_usage() {
  ms_log "Please input this command for more help: --help"
}

### 脚本入参的相关处理函数
function check_script_args() {
  ######################  check params confilct ###################
  if [ "${invalid_param_flag}" = y ]; then
    ms_log "ERROR: check script para failed"
    print_usage
    exit 1
  fi

  if [ $# -lt 3 ]; then
    print_usage
  fi
  # 重复参数检查
  for key in "${!param_dict[@]}";do
    if [ "${param_dict[${key}]}" -gt 1 ]; then
      ms_log "ERROR: parameter error! ${key} is repeat."
      exit 1
    fi
  done

  if [ "${print_version_flag}" = y ]; then
    if [ "${install_flag}" = y ] || [ "${upgrade_flag}" = "y" ]; then
      ms_log "ERROR: --version param cannot config with install or upgrade param."
      exit 1
    fi
  fi

  if [ "${install_path_flag}" = y ]; then
    if [[ ! "${install_path}" =~ ^/.* ]]; then
      ms_log "ERROR: parameter error ${install_path}, must absolute path."
      exit 1
    fi
  fi

  if [ "${upgrade_flag}" = y ]; then
    if [ "${install_flag}" = y ] || [ "${print_version_flag}" = "y" ]; then
      ms_log "ERROR: --install and --upgrade para cannot be configed together."
      exit 1
    fi
  fi
  if [ "${install_flag}" = y ]; then
    if [ "${upgrade_flag}" = "y" ] || [ "${print_version_flag}" = "y" ]; then
      ms_log "ERROR: Unsupported parameters,  install operation must not use with upgrade or version."
      exit 1
    fi
  fi
  if [ "${install_path_flag}" = y ]; then
    if [ "${install_flag}" = "n" ] && [ "${upgrade_flag}" = "n" ]; then
      ms_log "ERROR: Unsupported separate 'install-path' used independently."
      exit 1
    fi
  fi
}

check_target_dir()
{
  if [[ "${install_path}" =~ [^a-zA-Z0-9_./-] ]]; then
    ms_log "Multimodal SDK dir contains invalid char, please check path."
    exit 1
  fi
}

function check_sha256sum()
{
  if [ ! -e "/usr/bin/sha256sum" ] && [ ! -e "/usr/bin/shasum" ]; then
    ms_log "ERROR:Sha256 check Failed."
    exit 1
  fi
}

# 解析脚本自身的参数
function parse_script_args() {
  ms_log "INFO: start to run"
  local all_para_len="$*"
  if [[ ${#all_para_len} -gt 1024 ]]; then
    ms_log "The total length of the parameter is too long"
    exit 1
  fi
  local num=0
  while true; do
    if [[ "$1" == "" ]]; then
      break
    fi
    if [[ "${1: 0: 2}" == "--" ]]; then
      num=$((num + 1))
    fi
    if [[ ${num} -gt 2 ]]; then
      break
    fi
    shift 1
  done
  while true; do
    case "$1" in
    --check)
      check_sha256sum
      exit 0
      ;;
    --version)
      print_version_flag=y
      ((param_dict["version"]++)) || true
      shift
      ;;
    --install)
      check_platform
      install_flag=y
      ((param_dict["install"]++)) || true
      shift
      ;;
    --install-path=*)
      check_platform
      # 去除指定安装目录后所有的 "/"
      install_path=$(echo "$1" | cut -d"=" -f2 | sed "s/\/*$//g")
      check_target_dir
      if [[ "${install_path}" != /* ]]; then
        install_path="${USER_PWD}/${install_path}"
      fi
      existing_dir="${install_path}"
      while [[ ! -d "${existing_dir}" && "${existing_dir}" != "/" ]]; do
        existing_dir=$(dirname "${existing_dir}")
      done
      abs_existing_dir=$(readlink -f "${existing_dir}")
      nonexistent_suffix="${install_path#$existing_dir}"
      install_path="${abs_existing_dir}${nonexistent_suffix}"
      install_path_flag=y
      ((param_dict["install-path"]++)) || true
      shift
      ;;
    --upgrade)
      check_platform
      upgrade_flag=y
      ((param_dict["upgrade"]++)) || true
      shift
      ;;
    --quiet)
      quiet_flag=y
      ((param_dict["quiet"]++)) || true
      shift
      ;;
    -*)
      ms_log "WARNING: Unsupported parameters: $1"
      print_usage
      invalid_param_flag=y
      shift
      ;;
    *)
      if [ "x$1" != "x" ]; then
        ms_log "WARNING: Unsupported parameters: $1"
        print_usage
        invalid_param_flag=y
      fi
      break
      ;;
    esac
  done
}

ms_save_upgrade_info()
{
  path="$1"
  user_ip=$(who am i | awk '{print $NF}' | sed 's/(//g' | sed 's/)//g')
  if [[ -z "${user_ip}" ]]; then
    user_ip=localhost
  fi
  user_name=$(whoami)
  host_name=$(hostname)
  append_text="[$(date "+%Y-%m-%d %H:%M:%S")][${user_ip}][${user_name}][${host_name}]:"
  echo "${append_text}" >> "${path}"
  append_text="${old_version_info}"
  append_text+="    ->    "
  append_text+=${new_version_info}
  echo "${append_text:+$append_text }Upgrade Multimodal SDK successfully." >> "${path}"
}

ms_save_install_info()
{
  path="$1"
  user_ip=$(who am i | awk '{print $NF}' | sed 's/(//g' | sed 's/)//g')
  if [[ -z "${user_ip}" ]]; then
    user_ip=localhost
  fi
  user_name=$(whoami)
  host_name=$(hostname)
  append_text="[$(date "+%Y-%m-%d %H:%M:%S")][${user_ip}][${user_name}][${host_name}]:"
  echo "$append_text${new_version_info:+ $new_version_info} Install Multimodal SDK successfully." >> "${path}"
}


ms_record_operator_info()
{
  ms_deployment_log_rotate
  find "${record_file_path}" -type f -exec chmod 750 {} +
  if test x"${install_flag}" = xy; then
    ms_save_install_info "${record_file_path}"
    echo "Install Multimodal SDK successfully." >&2
  fi

  if test x"${upgrade_flag}" = xy; then
    ms_save_upgrade_info "${record_file_path}"
    echo "Upgrade Multimodal SDK successfully." >&2
  fi
  abs_path=$(readlink -f "${install_path}/script/set_env.sh")
  echo "Please execute '. ${abs_path}' to activate environment variables."

  find "${record_file_path}" -type f -exec chmod 440 {} +
}

ms_set_env()
{
  sdkhome_tmp="${install_path}"
  sed -i "s#export MULTIMODAL_SDK_HOME=.*#export MULTIMODAL_SDK_HOME=\"${sdkhome_tmp}\"#g" "${sdkhome_tmp}"/script/set_env.sh
}

function check_platform()
{
  plat="$(uname -m)"
  result="$(echo ${arch_name} | grep ${plat})"
  if test x"${result}" = x""; then
    ms_print_warning "Warning: Platform(${plat}) mismatch for ${arch_name}, please check it."
    ms_log "Warning: Platform(${plat}) mismatch for ${arch_name}, please check it."
  fi
}

function install_whl()
{
  cd "${self_dir}/multimodal"

  whl_file_name=$(find ./ -maxdepth 1 -type f -name 'mm*.whl')
  if test x"${quiet_flag}" = xn; then
    ms_log "INFO: Begin to install wheel package(${whl_file_name##*/})."
  fi

  if [[ -f "${whl_file_name}" ]];then
    if test x"${quiet_flag}" = xy; then
      python3 -m pip install --no-index --upgrade --force-reinstall --no-dependencies "${whl_file_name##*/}" --user > /dev/null 2>&1
    else
      python3 -m pip install --no-index --upgrade --force-reinstall --no-dependencies "${whl_file_name##*/}" --user
    fi
    if test $? -ne 0; then
      ms_log "ERROR: Install wheel package failed."
      return 1
    else
      if test x"${quiet_flag}" = xn; then
        ms_log "INFO: Install wheel package successfully."
      fi
    fi
    rm -rf "${whl_file_name##*/}"
    return 0
  else
    ms_log "ERROR: There is no wheel package to install."
    return 1
  fi
  cd - > /dev/null
}


function untar_file() {
  if [ "${print_version_flag}" = y ]; then
    self_dir="$(cd "$(dirname "$0")"; pwd)"
    tar -xzf "${self_dir}/multimodal-sdk-_linux-aarch64.tar.gz" -C "${self_dir}"
    cat "${self_dir}/version.info"
  elif [ "${install_flag}" = y ] || [ "${upgrade_flag}" = "y" ]; then

    eula_action="install"

    if test x"${upgrade_flag}" = xy; then
      eula_action="upgrade"

      if [[ -d "${install_path}/multimodal/lib" ]] && \
        [[ -d "${install_path}/multimodal/script" ]] && \
        [[ -d "${install_path}/multimodal/opensource" ]] && \
        [[ -f "${install_path}/multimodal/version.info" ]]; then
        unset doupgrade

        if test x"${quiet_flag}" = xn; then
          echo "Check install path (\"${install_path}\")."
          echo "Found an existing installation."
          read -t 60 -n1 -re -p "Do you want to upgrade by removing the old installation? [Y/N] " answer
          case "${answer}" in
            Y|y)
              doupgrade=y
              ;;
            *)
              doupgrade=n
              ;;
          esac
        else
          doupgrade=y
        fi

        if [[ x"${doupgrade}" == "xn" ]]; then
          ms_log "Warning: user rejected to upgrade, nothing changed"
          echo "Upgradation cancelled. Nothing changed."
          exit 1
        else
          ms_log "Info: user choose to upgrade"
          echo "Removing old installation at ${install_path}/multimodal ..."
          cd "${install_path}/multimodal/script"
          bash ./uninstall.sh
          if [ $? -ne 0 ]; then
            ms_log "ERROR: Failed to uninstall."
            exit 1
          fi
          cd - > /dev/null
          ms_log "Remove old installation success!"
          if test $? -ne 0; then
            ms_log "Error: failed to remove old installation at ${install_path}/multimodal"
            echo "Error: failed to remove old installation."
            exit 1
          fi
          echo "Old installation removed. Proceeding with new installation..."
        fi

      else
        ms_log "Error:There is no Multimodal SDK installed in current install path, please check it."
        exit 1
      fi

    fi
    self_dir="$(cd "$(dirname "$0")"; pwd)"
    mxsdk_manufacture_name="multimodal"

    if [ -e "${install_path}/${mxsdk_manufacture_name}" ] && [ ! -L "${install_path}/${mxsdk_manufacture_name}" ]; then
      ms_log "Error: ./${mxsdk_manufacture_name} exists and is not a symlink, cannot overwrite. Please check the installation path."
      exit 1
    fi

    if [ -d "${install_path}/${mxsdk_manufacture_name}" ]; then
      if test x"${install_flag}" = xy; then
        ms_log "ERROR: the same name directory already exists, install failed."
        exit 1
      fi
      ms_log "Info: Target directory ${install_path}/${mxsdk_manufacture_name} exists. Merging contents!"
    fi

    mkdir -p "${self_dir}/${mxsdk_manufacture_name}"
    if [ $? -ne 0 ]; then
      ms_log "ERROR: Failed to create tmp directory."
      exit 1
    fi
    tar -xzf "${self_dir}/multimodal-sdk-_linux-aarch64.tar.gz" -C "${self_dir}/${mxsdk_manufacture_name}" --no-same-owner
    version_number=$(head -n 1 "${self_dir}/${mxsdk_manufacture_name}/version.info" | cut -d ':' -f2 | tr -d '[:space:]')

    mxsdk_new_ame="${mxsdk_manufacture_name}-${version_number}"
    version_dir_exist=n
    if [ -d "${install_path}/${mxsdk_new_ame}" ]; then
      if test x"${install_flag}" = xy; then
        ms_log "ERROR: the same name directory already exists, install failed."
        exit 1
      fi
      ms_log "Info: Target directory ${install_path}/${mxsdk_new_ame} exists. Merging contents!"
      version_dir_exist=y
    else
      mkdir -p "${install_path}/${mxsdk_new_ame}"
      if [ $? -ne 0 ]; then
        ms_log "ERROR: Failed to create install directory"
        exit 1
      fi
      chmod 750 "${install_path}/${mxsdk_new_ame}"
    fi

    install_whl
    if [ $? -ne 0 ] && [ "${upgrade_flag}" = "n" ]; then
      ms_log "ERROR: Failed to install wheel package, delete incomplete installation files."
      rm -rf "${install_path}/${mxsdk_new_ame}"
      rm -rf "${install_path}/${mxsdk_manufacture_name}"
      exit 1
    fi

    mv "${self_dir}/${mxsdk_manufacture_name}/lib" "${install_path}/${mxsdk_new_ame}/lib"
    mv "${self_dir}/${mxsdk_manufacture_name}/version.info" "${install_path}/${mxsdk_new_ame}/version.info"
    mv "${self_dir}/${mxsdk_manufacture_name}/opensource" "${install_path}/${mxsdk_new_ame}/opensource"
    mv "${self_dir}/${mxsdk_manufacture_name}/script" "${install_path}/${mxsdk_new_ame}/script"
    if [ $? -ne 0 ]; then
      ms_log "ERROR: Failed to rename ${mxsdk_manufacture_name} to ${mxsdk_new_ame}"
      exit 1
    fi
    cd ${install_path}
    ln -snf "./${mxsdk_new_ame}" "${install_path}/${mxsdk_manufacture_name}"

    install_path=${install_path}/${mxsdk_new_ame}
    ms_set_env
    find "${install_path}" -type d -name "script" -exec chmod 550 {} +
    find "${install_path}" -type d -name "lib" -exec chmod 550 {} +
    find "${install_path}" -type d -name "opensource" -exec chmod -R 550 {} +
    find "${install_path}" -type f -path "*/script/set_env.sh" -exec chmod 500 {} +
    find "${install_path}" -type f -path "*/script/uninstall.sh" -exec chmod 500 {} +
    find "${install_path}" -type f -name "*.so*" -exec chmod 440 {} +

    cd - > /dev/null
    ms_record_operator_info
  else
    ms_log "Info: Do not proceed with installation or upgrade and exit."
  fi
}


# 程序开始
function main() {
  parse_script_args "$@"
  check_script_args "$@"
  untar_file
}

main "$@"