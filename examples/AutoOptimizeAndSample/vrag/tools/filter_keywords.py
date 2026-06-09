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
# Keyword filtering utilities based on part-of-speech tags.


from functools import cache
from typing import List

import spacy
from spacy import Language
from spacy.tokens import Doc

from vrag.constants import DEFAULT_LANG
from vrag.logger import logger


def filter_keywords(keywords: List[str], lang: str = DEFAULT_LANG) -> List[str]:
    """
    Filter a list of keywords phrases based on part-of-speech patterns.

    Keeps:
        - Single tokens that are NOUN, ADJ, VERB or PROPN.
        - Two-token phrases matching:
            * ADJ + NOUN/PROPN
            * NOUN/PROPN + NOUN/PROPN
            * VERB + NOUN/PROPN
        - Three-token phrases matching:
            * ADJ + NOUN/PROPN + NOUN/PROPN
    Excludes some literal phrases in all cases.
    Removes duplicates automatically. Order is not preserved.
    """
    nlp_lang = _get_nlp(lang)

    keywords = list({k.lower() for k in keywords})

    passed = [phrase for phrase in keywords if _is_valid_phrase(nlp_lang(phrase))]

    msg = f"Filtering phrase\nRaw: {keywords}\nPassed: {passed}"
    logger.debug(msg)
    return passed


@cache
def _get_nlp(lang: str = DEFAULT_LANG) -> Language:
    return spacy.load(f"{lang}_core_web_sm")


def _is_valid_single_token(doc: Doc) -> bool:
    """Check if a single-token doc is a valid keyword."""
    return doc[0].pos_ in {"NOUN", "ADJ", "VERB", "PROPN"}


def _is_valid_two_token(doc: Doc) -> bool:
    """Check if a two-token doc matches allowed POS pattern."""
    pos0 = doc[0].pos_
    pos1 = doc[1].pos_
    return (
        (pos0 == "ADJ" and pos1 in {"NOUN", "PROPN"})
        or (pos0 in {"NOUN", "PROPN"} and pos1 in {"NOUN", "PROPN"})
        or (pos0 == "VERB" and pos1 in {"NOUN", "PROPN"})
    )


def _is_valid_three_token(doc: Doc) -> bool:
    """Check if a three-token doc matches allowed POS pattern."""
    pos0 = doc[0].pos_
    pos1 = doc[1].pos_
    pos2 = doc[2].pos_
    return pos0 == "ADJ" and pos1 in {"NOUN", "PROPN"} and pos2 in {"NOUN", "PROPN"}


def _should_exclude(phrase: str) -> bool:
    """Determine if a phrase should be excluded regardless of POS."""
    return phrase in {"video", "the video", "me", "available options", "a", "b", "c", "d", "e", "f", "code"}


def _is_valid_phrase(doc: Doc) -> bool:
    """Determine if the phrase (as DOC) meets filtering criteria."""
    length = len(doc)
    if _should_exclude(doc.text.lower()):
        return False
    match length:
        case 1:
            return _is_valid_single_token(doc)
        case 2:
            return _is_valid_two_token(doc)
        case 3:
            return _is_valid_three_token(doc)
    return False
