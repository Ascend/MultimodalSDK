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
# Language detection utilities using lingua library.


from lingua import Language, LanguageDetector, LanguageDetectorBuilder

from vrag.shared import once


def lang_msg_align_to(question: str) -> str:
    """
    Generate a language instruction string aligned with the question's language.

    Args:
        question: The user's question

    Returns:
        Instruction string in the detected language.

    Raises:
        NotImplementedError: If unsupported language is detected.
    """
    lang = _detect_language_multi_vote(question)

    match lang:
        case Language.CHINESE:
            return "使用与问题相同的语言进行回答！当然，术语或专有名词可保留语言以避免歧义。"
        case Language.ENGLISH:
            return (
                "Answer in the same language as the question! However, "
                "technical terms or proper nouns may remain in their original language to prevent ambiguity."
            )
        case _:
            raise NotImplementedError(f"Unsupported language: {lang}")


@once
def _get_lang_detector() -> LanguageDetector:
    """Get a cached language detector for Chinese and English"""

    return LanguageDetectorBuilder.from_languages(Language.CHINESE, Language.ENGLISH).build()


def _detect_language_multi_vote(text: str) -> Language:
    """
    Detect language using multi-vote sampling.

    Divide text into chunks and votes on the most common language detected.
    """
    detector = _get_lang_detector()

    if len(text) <= 5:
        return detector.detect_language_of(text) or Language.ENGLISH

    # tail of text will be ignored
    chunk_size = max(1, len(text) // 5)
    chunks = [text[i * chunk_size : (i + 1) * chunk_size] for i in range(5)]

    votes: dict[Language, int] = {}

    for chunk in chunks:
        if not chunk.strip():
            continue
        lang = detector.detect_language_of(chunk)
        if lang:
            votes[lang] = votes.get(lang, 0) + 1

    if len(votes) == 0:
        return Language.ENGLISH
    else:
        sorted_votes = sorted(votes.items(), key=lambda x: x[1], reverse=True)
        return sorted_votes[0][0]
