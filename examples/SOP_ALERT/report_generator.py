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
"""报告层：生成 JSON 与 Markdown 两种格式的分析报告。"""

import json
import os
from datetime import datetime, timezone
from typing import Tuple

from common import format_ts
from sop_comparator import ComplianceResult, LEVEL_ERROR, LEVEL_INFO, LEVEL_WARNING
from sop_loader import Sop
from step_extractor import ExtractionResult

_LEVEL_BADGE = {LEVEL_ERROR: "🔴 ERROR", LEVEL_WARNING: "🟡 WARNING", LEVEL_INFO: "🔵 INFO"}


def _escape_md_cell(text: str) -> str:
    """转义 Markdown 表格单元格中的 | 和换行符，防止破坏表格结构。"""
    if not text:
        return text
    return str(text).replace("|", "\\|").replace("\n", " ")


def build_report(
    sop: Sop, extraction: ExtractionResult, result: ComplianceResult, video_path: str, sop_path: str
) -> dict:
    """汇总各层输出，构建结构化报告数据。"""
    return {
        "report_type": "sop_compliance_alert",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "input": {
            "video_path": video_path,
            "sop_path": sop_path,
            "video_duration_s": round(extraction.video_duration_s, 3),
            "audio_enabled": extraction.audio_enabled,
        },
        "sop": {
            "name": sop.name,
            "description": sop.description,
            "total_steps": len(sop.steps),
            "steps": [{"id": s.step_id, "name": s.name, "critical": s.critical} for s in sop.steps],
        },
        "observed_steps": [obs.to_dict() for obs in extraction.observations],
        "analysis": {
            "matched_step_count": len(result.matched_step_ids),
            "matched_step_ids": result.matched_step_ids,
            "compliance_score": result.compliance_score,
            "compliant": result.is_compliant,
            "alert_count": {
                "error": sum(1 for a in result.alerts if a.level == LEVEL_ERROR),
                "warning": sum(1 for a in result.alerts if a.level == LEVEL_WARNING),
                "info": sum(1 for a in result.alerts if a.level == LEVEL_INFO),
            },
        },
        "alerts": [alert.to_dict() for alert in result.alerts],
    }


def render_markdown(report: dict) -> str:
    """将结构化报告渲染为 Markdown 文本。"""
    analysis = report["analysis"]
    counts = analysis["alert_count"]
    conclusion = "✅ 合规" if analysis["compliant"] else "❌ 存在违规，需人工介入"

    lines = [
        "# 工厂操作视频 SOP 合规分析报告",
        "",
        f"- **生成时间**：{report['generated_at']}",
        f"- **操作视频**：`{report['input']['video_path']}`",
        f"- **标准 SOP**：`{report['input']['sop_path']}`（{report['sop']['name']}）",
        f"- **视频时长**：{format_ts(report['input']['video_duration_s'])}",
        f"- **音频检测**：{'启用' if report['input']['audio_enabled'] else '未启用'}",
        "",
        "## 1 总体结论",
        "",
        "| 合规评分 | 结论 | ERROR | WARNING | INFO |",
        "|--------|------|-------|---------|------|",
        f"| **{analysis['compliance_score']} / 100** | {conclusion} "
        f"| {counts['error']} | {counts['warning']} | {counts['info']} |",
        "",
        "## 2 SOP 步骤执行情况",
        "",
        "| SOP 步骤 | 名称 | 关键步骤 | 是否检测到 | 实际时间段 | 置信度 |",
        "|---------|------|---------|-----------|-----------|--------|",
    ]

    observed = {o["step_id"]: o for o in report["observed_steps"] if o.get("step_id")}
    for step in report["sop"]["steps"]:
        obs = observed.get(step["id"])
        if obs:
            time_range = f"{format_ts(obs['start_s'])}-{format_ts(obs['end_s'])}"
            lines.append(
                f"| {_escape_md_cell(step['id'])} | {_escape_md_cell(step['name'])} | {'是' if step['critical'] else '否'} "
                f"| ✅ | {time_range} | {obs['confidence']:.2f} |"
            )
        else:
            lines.append(
                f"| {_escape_md_cell(step['id'])} | {_escape_md_cell(step['name'])} | {'是' if step['critical'] else '否'} | ❌ 未检测到 | - | - |"
            )

    lines += ["", "## 3 实际观测步骤序列", ""]
    if report["observed_steps"]:
        lines += [
            "| # | 操作 | 时间段 | 耗时 | 置信度 | 证据 |",
            "|---|------|--------|------|--------|------|",
        ]
        for i, obs in enumerate(report["observed_steps"], start=1):
            time_range = f"{format_ts(obs['start_s'])}-{format_ts(obs['end_s'])}"
            evidence = obs.get("evidence") or "-"
            lines.append(
                f"| {i} | {_escape_md_cell(obs['name'])} | {time_range} | {obs['duration_s']:.1f}s "
                f"| {obs['confidence']:.2f} | {_escape_md_cell(evidence)} |"
            )
    else:
        lines.append("未从视频中观测到任何有效操作步骤。")

    lines += ["", "## 4 违规告警明细", ""]
    if report["alerts"]:
        lines += [
            "| 级别 | 告警类型 | 步骤 | 时间段 | 说明 |",
            "|------|---------|------|--------|------|",
        ]
        for alert in report["alerts"]:
            badge = _LEVEL_BADGE.get(alert["level"], alert["level"])
            step = f"{alert['step_name']}" + (f"({alert['step_id']})" if alert["step_id"] else "")
            lines.append(
                f"| {badge} | {_escape_md_cell(alert['alert_type'])} | {_escape_md_cell(step)} "
                f"| {alert['time_range'] or '-'} | {_escape_md_cell(alert['message'])} |"
            )
    else:
        lines.append("未发现违规项，本次操作完全符合 SOP。🎉")

    lines += [
        "",
        "---",
        "",
        "> 本报告由 MultimodalSDK SOP 违规自动预警 Pipeline 自动生成："
        "`mm.video_decode` 视频解码 → `mm.load_audio` 音频活动检测 → "
        "`mm.KRangFrameSelector` 步骤关键帧定位 → VLM 步骤确认 → SOP 比对分析。",
        "",
    ]
    return "\n".join(lines)


def save_reports(report: dict, output_dir: str, basename: str = "sop_report") -> Tuple[str, str]:
    """将报告写入 JSON 与 Markdown 文件。

    :return: (json 路径, markdown 路径)
    """
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f"{basename}.json")
    md_path = os.path.join(output_dir, f"{basename}.md")

    # 报告含操作视频路径与画面证据等个人信息，写入时收紧权限为 0o600，与输入视频 ≤0640 的隐私基线一致
    fd = os.open(json_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    fd = os.open(md_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(render_markdown(report))
    return json_path, md_path
