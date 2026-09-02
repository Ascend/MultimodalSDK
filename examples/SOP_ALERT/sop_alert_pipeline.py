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
"""
完整流水线（video 模式，需要昇腾 NPU + VLM 推理服务）::
    python sop_alert_pipeline.py \
        --video /path/to/operation.mp4 \
        --sop data/cases/case1/sop.json \
        --clip-model-path /models/chinese-clip-vit-large-patch14-336px \
        --device-id 0 \
        --vlm-url http://127.0.0.1:8111/v1 \
        --vlm-model-name Qwen2.5-VL-32B-Instruct \
        --output-dir ./output
"""

import argparse
import os
import sys

from report_generator import build_report, render_markdown, save_reports
from sop_comparator import compare_with_sop
from sop_loader import load_sop
from step_extractor import load_observations


def parse_args():
    parser = argparse.ArgumentParser(description="SOP compliance alert pipeline based on MultimodalSDK")
    parser.add_argument("--sop", required=True, help="标准 SOP JSON 文件路径")
    parser.add_argument("--output-dir", default="./output", help="报告输出目录，默认 ./output")
    parser.add_argument("--report-name", default="sop_report", help="报告文件名（不含扩展名）")
    parser.add_argument("--print-markdown", action="store_true", help="在终端打印 Markdown 报告")

    # 完整流水线（video 模式）
    parser.add_argument("--video", help="操作视频路径（mp4）。指定后走完整流水线，需要 NPU 与 VLM 服务")
    parser.add_argument("--clip-model-path", help="CLIP 模型权重路径（关键帧选择器使用）")
    parser.add_argument("--device-id", type=int, default=0, help="NPU 设备 ID，默认 0")
    parser.add_argument(
        "--clip-model-type", default="cn_clip", choices=["clip", "cn_clip"], help="CLIP 模型类型，默认 cn_clip"
    )
    parser.add_argument("--vlm-url", help="VLM 推理服务 URL（OpenAI 兼容）")
    parser.add_argument(
        "--vlm-api-key",
        default=None,
        help="VLM API Key，默认读取环境变量 VLM_API_KEY。推荐使用环境变量方式，禁止将密钥写入命令行",
    )
    parser.add_argument("--vlm-model-name", help="VLM 推理服务模型名称")
    parser.add_argument("--sample-fps", type=float, default=1.0, help="视频采样帧率，默认 1 帧/秒")
    parser.add_argument("--sample-num", type=int, default=12, help="每个步骤最大关键帧数量，默认 12")
    parser.add_argument("--disable-audio", action="store_true", help="关闭音频活动检测")

    # 离线验证（observations 模式）
    parser.add_argument("--observations", help="步骤观测序列 JSON 路径。指定后跳过媒体处理层，无需 NPU")
    return parser.parse_args()


def run_video_pipeline(args):
    """完整流水线：媒体处理层（MultimodalSDK）→ 步骤理解层 → 观测序列。"""
    required = {
        "--clip-model-path": args.clip_model_path,
        "--vlm-url": args.vlm_url,
        "--vlm-model-name": args.vlm_model_name,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError(f"video 模式缺少必要参数: {', '.join(missing)}")

    from openai import OpenAI

    from media_processor import StepRangeLocator, decode_video, load_audio_activity
    from step_extractor import StepExtractor

    sop = load_sop(args.sop)

    print(f"[1/4] mm.video_decode 解码视频: {args.video} (采样率 {args.sample_fps} fps)")
    video = decode_video(args.video, sample_fps=args.sample_fps)
    print(f"      解码得到 {len(video.frames)} 帧，视频时长 {video.duration_s:.1f}s")

    audio = None
    if not args.disable_audio:
        print("[2/4] mm.load_audio 加载音轨并进行声音活动检测")
        audio = load_audio_activity(args.video)
        print(f"      {'音轨加载成功' if audio else '视频无音轨，跳过音频检测'}")
    else:
        print("[2/4] 音频检测已关闭")

    print(f"[3/4] mm.KRangFrameSelector 定位步骤区间 + VLM 逐步骤确认（共 {len(sop.steps)} 个 SOP 步骤）")
    locator = StepRangeLocator(args.clip_model_path, args.device_id, args.clip_model_type)
    # API Key 优先命令行参数，其次环境变量 VLM_API_KEY，避免凭证通过 ps/proc 泄露
    api_key = args.vlm_api_key or os.environ.get("VLM_API_KEY")
    extractor = StepExtractor(
        locator=locator,
        vlm_client=OpenAI(base_url=args.vlm_url, api_key=api_key),
        vlm_model_name=args.vlm_model_name,
        sample_num=args.sample_num,
    )
    extraction = extractor.extract(sop, video, audio)
    print(f"      提取到 {len(extraction.observations)} 个实际操作步骤")
    return sop, extraction


def run_offline(args):
    """离线验证：直接加载预生成的步骤观测序列。"""
    sop = load_sop(args.sop)
    print(f"[离线模式] 加载步骤观测序列: {args.observations}")
    extraction = load_observations(args.observations)
    print(f"      共 {len(extraction.observations)} 个观测步骤")
    return sop, extraction


def main():
    args = parse_args()
    if bool(args.video) == bool(args.observations):
        print("错误: --video 与 --observations 必须且只能指定其中一个", file=sys.stderr)
        return 1

    try:
        if args.video:
            sop, extraction = run_video_pipeline(args)
            source = args.video
        else:
            sop, extraction = run_offline(args)
            source = args.observations

        print("[4/4] SOP 比对分析与报告生成")
        result = compare_with_sop(sop, extraction)
        report = build_report(sop, extraction, result, source, args.sop)
        json_path, md_path = save_reports(report, args.output_dir, args.report_name)
    except Exception as e:
        print(f"Pipeline 执行失败: {e}", file=sys.stderr)
        return 1

    print()
    print(
        f"合规评分: {result.compliance_score}/100  "
        f"结论: {'合规' if result.is_compliant else '存在违规'}  "
        f"告警数: {len(result.alerts)}"
    )
    for alert in result.alerts:
        print(f"  [{alert.level}] {alert.alert_type}: {alert.message}")
    print()
    print(f"JSON 报告:     {json_path}")
    print(f"Markdown 报告: {md_path}")

    if args.print_markdown:
        print()
        print(render_markdown(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
