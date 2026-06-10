#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the MultimodalSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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
# Video rag service.

import argparse

import bentoml

from vrag.bentos.video_rag import VideoRagService

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", "-c", type=str, help="video rag toml config path.")
    parser.add_argument("--host", "-H", type=str, default="0.0.0.0", help="video rag host.")
    parser.add_argument("--port", "-p", type=int, default=7860, help="video rag port.")

    args = parser.parse_args()

    bentoml.serve(
        VideoRagService, blocking=True, host=args.host, port=args.port, args=({"config_file_path": args.config})
    )
