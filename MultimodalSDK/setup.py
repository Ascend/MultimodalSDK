#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
from setuptools import setup, find_packages

setup(
    name='mm',
    version='1.0.0',
    packages=find_packages(where='source'),
    package_dir={'': 'source'},
    package_data={
        'mm.acc._impl': [
            '_acc.so',
            'lib/*.so*',
            'opensource/FFmpeg/lib/*.so*',
            'opensource/libjpeg-turbo/lib/*.so*',
            'opensource/soxr/lib/*.so*',
        ],
    },
    extras_require={
        'quality': [
            'transformers>=4.51.3',
            'pyiqa>=0.1.15',
            'ultralytics>=8.4.0',
            'ptlflow>=0.4.2',
            'opencv-python>=4.8.0',
            'scipy>=1.10.0',
            'scikit-image>=0.21.0',
            'scikit-video>=1.1.11',
            'timm>=0.9.0',
            'einops>=0.7.0',
            'pyyaml>=6.0',
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
