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
# Baseline service for VL ask.

import argparse
import time
from typing import Optional

import bentoml

from vrag.bentos.qwenvl import QwenVLService
from vrag.bentos.video_process import VideoProcessService
from vrag.bentos.video_rag import VideoRagArgs, VideoRagConfig, VideoRagInferenceResult
from vrag.shared import retry_async_request, vrag_service
from vrag.tools.base64 import encode_frames_async
from vrag.tools.render import generate_final_prompt

args = bentoml.use_arguments(VideoRagArgs).override()


@vrag_service(args)
class SimpleQwenVLQAService:
    qwen_vl: QwenVLService = bentoml.depends(QwenVLService)
    process: VideoProcessService = bentoml.depends(VideoProcessService)

    @bentoml.api
    async def ask(
        self, video_path: str, question: str, config: Optional[VideoRagConfig] = None
    ) -> VideoRagInferenceResult:
        start = time.time()

        merged_config = VideoRagConfig.merge_config(config)

        video_process_config = merged_config.retrieval.transcribe.video_process
        video_process_config.extract_audio = False

        frame_extraction, _ = await retry_async_request(
            lambda: self.process.extract(video_path=video_path, config=video_process_config), "baseline_extract_video"
        )

        rendered = generate_final_prompt(question=question, frame_extraction=frame_extraction)

        frames_b64 = await encode_frames_async(frame_extraction.frames)

        ans = await retry_async_request(
            lambda: self.qwen_vl.generate(query=rendered, frames_b64=frames_b64, config=merged_config.qwenvl),
            "baseline_llm_infer",
        )

        return VideoRagInferenceResult(
            question=question, answer=ans, digested_info=rendered, processing_time=time.time() - start
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", "-c", type=str, help="baseline toml config path.")
    parser.add_argument("--host", "-H", type=str, default="0.0.0.0", help="baseline host.")
    parser.add_argument("--port", "-p", type=int, default=7861, help="baseline port.")

    args = parser.parse_args()

    bentoml.serve(
        SimpleQwenVLQAService, blocking=True, host=args.host, port=args.port, args=({"config_file_path": args.config})
    )
