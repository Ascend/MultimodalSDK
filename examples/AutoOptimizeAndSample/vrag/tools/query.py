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
# Query utilities for video rag search.


import json
from functools import cached_property
from typing import ClassVar, List, Literal, Optional

from json_repair import repair_json
from pydantic.json_schema import GenerateJsonSchema, JsonSchemaValue

from vrag.shared import ConfigBase
from vrag.tools.filter_keywords import filter_keywords


class DetectionConfig(ConfigBase):
    scene_desc: List[str]
    """
    One or more English scene descriptions used to roughly retrieve video frames.
    Each description MUST be a single sentence that is:
    - Highly specific and visual: include explicit attributes (colors, materials, clothing, posture, facial expression, lighting, etc.)
    - Grounded in concrete actions or states: use clear, observable verbs (e.g., 'pouring tea', 'reading a book', 'looking out of a window')
    - Spatially clear: describe where objects and subjects are in relation to each other and the environment
    - Resembling a COCO-style caption: only describe what can be seen, avoid abstract concepts, metaphors, or emotions
    - Using common, unambiguous vocabulary that a vision-language model (like BLIP) would easily align with natural images

    If you have multiple sub-questions, provide multiple such descriptions to cover them all.
    """

    retrieve_scene_subtitles: bool = False
    """Set True ONLY if user want to know the exact talking content when these scenes show up."""

    scene_occurrence_count: Optional[int] = None
    """How many time the scene of `scene_desc` appears in the video.
    If user specified the count of occurrence of the scene, writes it."""

    class_names: List[str]
    """List of class names of tangible physical objects that a visual detector can locate with bounding boxes.
    No abstract concepts, no actions, and no verbs allowed. Class names SHALL always be in lowercase English!
    Acceptable phrases includes single tokens that are NOUN or ADJ;
    two-token phrases matching (ADJ + NOUN/PROPN) or (NOUN/PROPN + NOUN/PROPN);
    and three-token phrases matching (ADJ + NOUN/PROPN + NOUN/PROPN).
    If the question asks for action detection (e.g., "find the person running"),
    you must extract the visible object noun (e.g., "person") instead of the verb phrase.
    For relationship detection, specify the anchor object (e.g., if detecting "dog behind chair",
    only list "dog" as class_name, use rel=True for position logic).
    Visual grounding models detect objects present in frames, not dynamic events.
    """

    loc: bool = False
    """Return bounding boxes."""

    num: bool = False
    """Return object counts."""

    rel: bool = False
    """Return spatial relationships between objects."""


class SubtitlesConfig(ConfigBase):
    relevant_subtitles: List[str] | Literal["all"]
    """
    Intent-based guesses of the exact spoken words or short phrases likely
    to appear in the video when the relevant content is being discussed.
    These are used to search ASR transcripts and jump to precise timestamps.

    HARD RULES:
    1. NEVER copy, rephrase, or synonymize the question itself.
       BAD: question "What are the primary themes?" \u2192 "primary themes"
       GOOD: concrete words actually spoken, e.g. "sunken ruins", "Roman navy"

    2. ALWAYS guess highly concrete nouns, proper names, numbers, specific
       verbs, or very short spoken phrases that people would actually SAY.
       Think: "Roman harbor", "sonar scan", "expansion of the empire",
       "trade routes", not "key topics" or "main idea".

    3. If answer options are provided, you MUST mine them for likely spoken
       vocabulary \u2014 keep the concrete objects, places, people and actions,
       never the abstract category labels.
       Example options:
         A: Secrets about Rome found underwater.
         B: How Rome grew into an empire.
         C: The Roman Maritime's prosperity.
         D: The high technologies used to detect sank ships.
       ACCEPTABLE guesses: ["sunken city", "Roman navy", "sonar technology",
                            "Mediterranean trade", "ancient shipwreck",
                            "expansion of the empire"]
       UNACCEPTABLE: ["primary themes", "main topics", "overview"]

    4. Generate 2-4 guesses. If your guesses look like exam question words
       (e.g. "theme", "summary", "key idea"), discard them and try again.

    5. STRICTLY avoid "all" for thematic/main-idea questions. Only use "all"
       if the user explicitly asks for full transcription, or the answer
       irreducibly requires the entire video timeline (extremely rare).
       If you cannot generate good concrete guesses for a thematic question,
       it usually means the answer depends on scenes rather than speech;
       in that case, leave relevant_subtitles empty and rely on visual
       retrieval mechanisms (DetectionConfig) instead.
    """

    retrieve_related_frames: bool = False
    """
    When True, the system will attach the nearby video frames from the
    timestamp of each retrieved subtitle segment. These frames provide
    visual context that may help answer the question, but they are NOT
    used for detection / bounding-box localization.

    DECISION GUIDE:
    - Set to True if the retrieved subtitle segments likely benefit from
      showing the corresponding imagery (e.g., a speaker's expression,
      an object being discussed while mentioned, a visual event that
      the speech refers to).
    - In practice, most "what/who/where/when" questions with concrete
      subjects should set this to True, because seeing the frame is
      almost always helpful.
    - Set to False ONLY when the answer can be derived from pure text
      or audio without any visual reinforcement, OR when the question
      has no subtitle retrieval at all (empty relevant_subtitles) and
      visual information is provided exclusively via DetectionConfig.
    - This flag is independent of DetectionConfig. It is perfectly valid
      to have retrieve_related_frames=True without any detection, or
      to have detection without this flag. The two sets of frames will
      be merged if both are requested.
    """

    @property
    def retrieve_all_subtitles(self) -> bool:
        if isinstance(self.relevant_subtitles, str):
            return self.relevant_subtitles == "all"
        if self.relevant_subtitles:
            return self.relevant_subtitles[0] == "all"
        return False


class GenerateJsonSchemaUnsort(GenerateJsonSchema):
    def sort(self, value: JsonSchemaValue, parent_key: str | None = None) -> JsonSchemaValue:
        """Not sort the value"""
        return value


class Query(ConfigBase):
    """
    To search the video for more intel.

    CHOOSING det AND sub - YOU MUST FOLLOW THESE RULES:
    - If the question asks about what a person/character said, mentioned, explained,
      or any dialogue/speech, you MUST include `sub`.
    - If the question also describes a visual anchor to find that person or scene
      (e.g., "the man in a blue shirt", "when the car appears"), you MUST include `det`
      alongside `sub`. Both are required in such mixed cases.
    - Use ONLY `det` when the question is purely visual (e.g., objects, actions, colours,
      spatial relations) and does NOT ask for spoken content.
    - Use ONLY `sub` when the question is purely about the spoken/written message
      (e.g., "What is the main topic of the lecture?") with no visual anchor needed
      to identify the speaker.
    """

    filter_keywords: ClassVar[bool] = True

    det: Optional[DetectionConfig] = None
    """Set this when the query relies on visual facts (e.g., objects, people, scenes) to
    locate the relevant moment. Can be combined with `sub`."""

    sub: Optional[SubtitlesConfig] = None
    """Set this whenever the answer depends on spoken/subtitle content. Can be combined with `det`."""

    @property
    def retrieve_all_subtitles(self) -> bool:
        return self.sub.retrieve_all_subtitles if self.sub else False

    @property
    def retrieve_related_frames(self) -> bool:
        return self.sub.retrieve_related_frames if self.sub else False

    @property
    def retrieve_related_docs_for_det_frames(self) -> bool:
        return self.det.retrieve_scene_subtitles if self.det else False

    @classmethod
    def schema_string(cls) -> str:
        return json.dumps(cls.model_json_schema(schema_generator=GenerateJsonSchemaUnsort), indent=2)

    @classmethod
    def from_raw_json(cls, j: str) -> "Query":
        return cls.model_validate_json(repair_json(j))

    @cached_property
    def access_filtered_targets(self) -> List[str]:
        if self.filter_keywords:
            return filter_keywords(self.det.class_names) if self.det else []
        return self.det.class_names if self.det else []

    @cached_property
    def access_scene_desc(self) -> List[str]:
        return self.det.scene_desc if self.det else []

    @cached_property
    def access_related_subtitles(self) -> List[str]:
        return self.sub.relevant_subtitles if self.sub and isinstance(self.sub.relevant_subtitles, list) else []
