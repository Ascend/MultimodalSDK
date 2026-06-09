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
# Prompt template rendering utilities.


from datetime import timedelta
from typing import Optional, Tuple, List

from vrag.constants import (
    RETRIEVE_PROMPT_TEMPLATE,
    FINAL_PROMPT_TEMPLATE,
    DET_RETRIEVE_PROMPT_TEMPLATE,
    NOTHING,
    ASR_RETRIEVE_PROMPT_TEMPLATE,
    OCR_RETRIEVE_PROMPT_TEMPLATE,
)
from vrag.logger import logger
from vrag.tools.decord import FrameExtraction
from vrag.tools.lingua import lang_msg_align_to
from vrag.types import Timestamp


def generate_final_prompt(
    question: str,
    frame_extraction: Optional[FrameExtraction],
    det_instruction: Optional[str] = None,
    asr_instruction: Optional[str] = None,
    ocr_instruction: Optional[str] = None,
    additional_instruction: Optional[str] = None,
) -> str:
    return FINAL_PROMPT_TEMPLATE.render(
        frame_timestamps=[_formatted_stamp(stmp) for stmp in frame_extraction.frame_timestamps]
        if frame_extraction
        else None,
        det_instruction=det_instruction,
        asr_instruction=asr_instruction,
        ocr_instruction=ocr_instruction,
        video_duration=_formatted_stamp(frame_extraction.video_duration) if frame_extraction else "<Unknown>",
        question=question,
        frame_height=frame_extraction.frame_height if frame_extraction else -1,
        frame_width=frame_extraction.frame_width if frame_extraction else -1,
        lang_msg=lang_msg_align_to(question),
        additional_instruction=additional_instruction,
    ).strip()


def generate_detection_instruction(
    det_docs: List[Tuple[str, float]], targets: List[str], discard_empty: bool = True, sort_by_time: bool = True
) -> str:
    def _discard_docs(docs: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        ret = list(filter(lambda doc: doc[0], docs))
        msg = f"Discard {len(docs) - len(ret)} empty DET docs."
        logger.debug(msg)
        return ret

    if sort_by_time:
        det_docs = sorted(det_docs, key=lambda doc: doc[1])

    final_det_docs = _discard_docs(det_docs) if discard_empty else [(d or NOTHING, t) for d, t in det_docs]

    return DET_RETRIEVE_PROMPT_TEMPLATE.render(
        det_docs=[(d[0], _formatted_stamp(d[1])) for d in final_det_docs],
        targets=targets,
    ).strip()


def generate_asr_instruction(
    asr_docs: List[Tuple[str, Tuple[float, float]]], discard_empty: bool = True, sort_by_time: bool = True
) -> str:
    def _discard_docs(docs: List[Tuple[str, Tuple[float, float]]]) -> List[Tuple[str, Tuple[float, float]]]:
        ret = list(filter(lambda doc: doc[0], docs))
        msg = f"Discard {len(docs) - len(ret)} empty ASR docs."
        logger.debug(msg)
        return ret

    if sort_by_time:
        asr_docs = sorted(asr_docs, key=lambda doc: doc[1][0])

    final_asr_docs = _discard_docs(asr_docs) if discard_empty else (asr_docs if asr_docs else [("", (0, 0))])

    return ASR_RETRIEVE_PROMPT_TEMPLATE.render(
        asr_docs=[(a[0], (_formatted_stamp(a[1][0]), _formatted_stamp(a[1][1]))) for a in final_asr_docs]
    ).strip()


def generate_ocr_instruction(
    ocr_docs: List[Tuple[str, float]],
    discard_empty: bool = True,
    sort_by_time: bool = True,
) -> str:
    def _discard_docs(docs: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        ret = list(filter(lambda doc: doc[0], docs))
        msg = f"Discard {len(docs) - len(ret)} empty OCR docs."
        logger.debug(msg)
        return ret

    if sort_by_time:
        ocr_docs = sorted(ocr_docs, key=lambda doc: doc[1])

    final_ocr_docs = _discard_docs(ocr_docs) if discard_empty else (ocr_docs if ocr_docs else [("", 0)])

    return OCR_RETRIEVE_PROMPT_TEMPLATE.render(
        ocr_docs=[(o[0], _formatted_stamp(o[1])) for o in final_ocr_docs]
    ).strip()


def generate_retrieval_prompt(question: str, schema: str) -> str:
    return RETRIEVE_PROMPT_TEMPLATE.render(question=question, schema=schema)


def _formatted_stamp(t: Timestamp) -> str:
    td = timedelta(seconds=int(t))
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
