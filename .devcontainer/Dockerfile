# Dev Container image for MultimodalSDK source development.
# Build: docker build --network host -t multimodalsdk-dev:ubuntu22.04 .devcontainer/

FROM quay.io/ascend/vllm-ascend:v0.8.5rc1

WORKDIR /tmp

RUN cp -a /etc/apt/sources.list /etc/apt/sources.list.bak && \
    sed -i "s@http://.*archive.ubuntu.com@http://mirrors.huaweicloud.com@g" /etc/apt/sources.list && \
    sed -i "s@http://.*security.ubuntu.com@http://mirrors.huaweicloud.com@g" /etc/apt/sources.list

RUN export DEBIAN_FRONTEND=noninteractive && \
    export TZ=Asia/Shanghai && \
    apt update && \
    apt install --no-install-recommends ca-certificates -y && \
    apt install --no-install-recommends wget vim dos2unix net-tools ssh lsof sshpass -y && \
    apt install --no-install-recommends curl gcc cmake g++ make pkg-config unzip autoconf git patch bison -y && \
    apt install --no-install-recommends zlib1g-dev xz-utils libssl-dev libffi-dev libsqlite3-dev \
        libbz2-dev libreadline-dev libncurses5-dev libncursesw5-dev libpcre2-dev -y && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /root/.pip /root/.config/pip && \
    rm -f /root/.config/pip/pip.conf && \
    printf '%s\n' \
        '[global]' \
        'index-url=https://mirrors.aliyun.com/pypi/simple/' \
        'trusted-host=mirrors.aliyun.com' \
        'timeout=200' \
        > /root/.pip/pip.conf && \
    cp /root/.pip/pip.conf /root/.config/pip/pip.conf

# AccSDK/MultimodalSDK build scripts expect Python under /opt/buildtools/python-${PYTHON_VERSION}.
ARG PYTHON_VERSION=3.11.4
ARG PYTHON_PREFIX=/opt/buildtools/python-${PYTHON_VERSION}
ENV PYTHON_VERSION=${PYTHON_VERSION} \
    PYTHON_PREFIX=${PYTHON_PREFIX} \
    Python3_EXECUTABLE=${PYTHON_PREFIX}/bin/python3

RUN curl -fsSL "https://mirrors.aliyun.com/python-release/source/Python-${PYTHON_VERSION}.tgz" -o Python.tgz && \
    tar -xzf Python.tgz && \
    cd "Python-${PYTHON_VERSION}" && \
    ./configure --prefix="${PYTHON_PREFIX}" --enable-shared --with-ensurepip=install && \
    make -j"$(nproc)" && make install && \
    cd .. && rm -rf Python.tgz "Python-${PYTHON_VERSION}" && \
    ldconfig "${PYTHON_PREFIX}/lib"

RUN "${PYTHON_PREFIX}/bin/pip3" install -U pip setuptools wheel \
        -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com && \
    "${PYTHON_PREFIX}/bin/pip3" install \
        torch==2.5.1 torch-npu==2.5.1.post1 torchvision==0.20.1 transformers==4.51.3 \
        einops==0.8.2 pillow==11.2.1 numpy==1.26.4 \
        -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com && \
    rm -rf /root/.cache/pip

ARG SWIG_VERSION=4.2.1
# SHA256 from upstream swig-4.2.1.tar.gz (FreeBSD ports distinfo).
ARG SWIG_SHA256=fa045354e2d048b2cddc69579e4256245d4676894858fcf0bab2290ecf59b7d8
RUN curl -fsSL "https://downloads.sourceforge.net/project/swig/swig/swig-${SWIG_VERSION}/swig-${SWIG_VERSION}.tar.gz" -o swig.tgz && \
    echo "${SWIG_SHA256}  swig.tgz" | sha256sum -c - && \
    tar -xzf swig.tgz && \
    cd "swig-${SWIG_VERSION}" && \
    ./configure --prefix=/usr/local && make -j"$(nproc)" && make install && \
    cd .. && rm -rf swig.tgz "swig-${SWIG_VERSION}"

RUN PY_SITE="$("${PYTHON_PREFIX}/bin/python3" -c 'import site; print(site.getsitepackages()[0])')" && \
    mkdir -p "/opt/python${PYTHON_VERSION}/lib" && \
    ln -sf "${PY_SITE}" "/opt/python${PYTHON_VERSION}/lib/python3.11"

ARG ASCEND_BASE=/usr/local/Ascend
ARG TOOLKIT_PATH=$ASCEND_BASE/ascend-toolkit/latest

USER root

ENV LD_LIBRARY_PATH=${PYTHON_PREFIX}/lib:$LD_LIBRARY_PATH
ENV PATH=/usr/local/bin:${PYTHON_PREFIX}/bin:$PATH
ENV TBE_IMPL_PATH=$TOOLKIT_PATH/opp/op_impl/built-in/ai_core/tbe
ENV PATH=$TOOLKIT_PATH/atc/ccec_compiler/bin:$TOOLKIT_PATH/atc/bin:$PATH
ENV ASCEND_OPP_PATH=$TOOLKIT_PATH/opp
ENV ASCEND_AICPU_PATH=$TOOLKIT_PATH
ENV PYTHONPATH=$TBE_IMPL_PATH:$PYTHONPATH
ENV ASCEND_HOME=${ASCEND_BASE}

RUN sed -i '$a\source /usr/local/Ascend/ascend-toolkit/set_env.sh' ~/.bashrc && \
    sed -i '$a\export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH' ~/.bashrc

WORKDIR /workspace/MultimodalSDK
