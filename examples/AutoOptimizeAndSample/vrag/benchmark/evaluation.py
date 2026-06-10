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
# Evaluate video rag service or baseline.

import argparse
import asyncio
import json
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Callable, Optional

import bentoml
from pydantic import BaseModel, Field

from vrag.bentos.video_rag import VideoRagInferenceResult
from vrag.logger import logger


class EvalVideoRagResult(BaseModel):
    video_path: Path
    question: str
    processing_time: float
    digested_info: str
    answer: str
    reference_answer: str
    choice_right: bool


class Summary(BaseModel):
    elapsed_time: float
    total_samples: int
    processed_samples: int
    passed_samples: int
    results: List[EvalVideoRagResult]
    timestamp: str = Field(default_factory=lambda: datetime.now(tz=timezone.utc).strftime("%Y%m%d-%H%M%S"))


class MMEQuestionAnswer(BaseModel):
    question_id: str = Field(..., description="Question identifier")
    task_type: str = Field(..., description="Type of task")
    question: str = Field(..., description="Question text")
    options: List[str] = Field(..., default_factory=list, description="Available answers")
    answer: str = Field(..., description="Correct answer")

    @property
    def make_question(self) -> str:
        selection = "\n".join(self.options)
        return f"{self.question}\n{selection}"


class VideoMME(BaseModel):
    video_id: str = Field(..., description="Video identifier")
    duration: str = Field(..., description="Duration category (e.g., 'short')")
    domain: str = Field(..., description="Main domain category")
    sub_category: str = Field(..., description="Sub domain category")
    url: str = Field(..., description="YouTube url and also video name")

    question_answer: Optional[MMEQuestionAnswer] = Field(None, description="Question-Answer")

    @staticmethod
    def parse_raw_json(file: Path) -> List["VideoMME"]:
        data = json.loads(file.read_text(encoding="utf-8"))

        if not isinstance(data, list):
            data = [data]

        result_items: List[VideoMME] = []

        for video_data in data:
            video_item = VideoMME.model_validate(video_data)

            for question_answer_data in video_data["questions"]:
                qa = MMEQuestionAnswer.model_validate(question_answer_data)

                video_item_copy = deepcopy(video_item)
                video_item_copy.question_answer = qa
                result_items.append(video_item_copy)

        return result_items

    def normalize_video_path(self, datasets_path: Path) -> Path:
        return datasets_path / "data" / f"{self.url}.mp4"

    def key(self) -> str:
        return self.domain


async def run_inference_with_video_mme(
    client: bentoml.AsyncHTTPClient, question: VideoMME, datasets_path: Path, retries: int = 5
) -> Optional[EvalVideoRagResult]:
    for retry in range(retries):
        try:
            result = await client.ask(
                question.normalize_video_path(datasets_path).as_posix(), question.question_answer.make_question
            )
            result = VideoRagInferenceResult.model_validate(result)

            choice_right = False
            if result.answer and question.question_answer.answer:
                choice_right = result.answer[0] == question.question_answer.answer[0]

            return EvalVideoRagResult(
                video_path=question.url,
                question=question.question_answer.make_question,
                processing_time=result.processing_time,
                digested_info=result.digested_info,
                answer=result.answer,
                reference_answer=question.question_answer.answer,
                choice_right=choice_right,
            )
        except Exception as e:
            msg = (
                f"Failed to run inference on {question.video_id}/{question.question_answer.question_id} "
                f"with {e} at {retry + 1}/{retries} attempt"
            )
            logger.exception(msg)

    return None


async def async_map_video_question(
    client: bentoml.AsyncHTTPClient,
    datasets_path: Path,
    questions: List[VideoMME],
    skip_when: Optional[Callable[[VideoMME], bool]] = None,
    max_concurrent: Optional[int] = None,
) -> Summary:
    valid_questions: List[VideoMME] = []

    for question in questions:
        if skip_when and skip_when(question):
            msg = f"Skipping eval for {question.video_id} by hooks."
            logger.debug(msg)
            continue

        if not question.normalize_video_path(datasets_path).exists():
            msg = f"Skipping eval for {question.video_id} by video not exists."
            logger.debug(msg)
            continue

        valid_questions.append(question)

    msg = f"Processing {len(valid_questions)} Q/A."
    logger.info(msg)

    acc_cal = _AccurateCalculator()

    async def process_question(idx: int, question: VideoMME, sem: asyncio.Semaphore) -> Optional[EvalVideoRagResult]:
        async with sem:
            msg = f"Processing [{idx}] [{question.video_id}] [{question.question_answer.question_id}] [{question.url}]"
            logger.info(msg)

            result = await run_inference_with_video_mme(client, question, datasets_path)

            if result is None:
                msg = f"Processing [{idx}] [{question.video_id}] [{question.question_answer.question_id}] return none."
                logger.warning(msg)
                return None

            acc_cal.add(result.choice_right)

            msg = (
                f"To [{idx}] [{question.video_id}] [{question.question_answer.question_id}]\n"
                f"Q:\n{question.question_answer.make_question}\n"
                f"Ref:\n{result.reference_answer}\n"
                f"Pred:\n{result.answer}\n"
                f"ACC:{acc_cal}|Time:{result.processing_time:.2f}s"
            )
            logger.info(msg)

            return result

    semaphore = asyncio.Semaphore(max_concurrent or len(valid_questions))

    start = time.time()
    tasks = [process_question(idx, question, semaphore) for idx, question in enumerate(valid_questions)]

    results = [result for result in await asyncio.gather(*tasks) if result is not None]
    end = time.time()

    return Summary(
        elapsed_time=end - start,
        total_samples=len(valid_questions),
        processed_samples=acc_cal.processed,
        passed_samples=acc_cal.passed,
        results=results,
    )


async def run_evaluation_loop(
    host: str, timeout: int, config_path: Path, datasets_path: Path, output_path: Path, max_concurrent: int = 10
):
    video_qas = VideoMME.parse_raw_json(config_path)

    def skip_when(question: VideoMME) -> bool:
        return question.duration in ["short", "medium"]

    async with bentoml.AsyncHTTPClient(host, timeout=timeout) as client:
        summary = await async_map_video_question(
            client, datasets_path, video_qas, skip_when=skip_when, max_concurrent=max_concurrent
        )

    accuracy = summary.passed_samples / summary.total_samples if summary.total_samples > 0 else 0.0
    avg_time = summary.elapsed_time / summary.total_samples if summary.total_samples > 0 else 0.0

    summary_msg = (
        f"Total elapse time: {summary.elapsed_time:.2f}s\n"
        f"Total samples: {summary.total_samples}\n"
        f"Processed samples: {summary.processed_samples}\n"
        f"Passed samples: {summary.passed_samples}\n"
        f"Accuracy: {accuracy:.2%}\n"
        f"Average Time: {avg_time:.2f}s"
    )
    logger.info(summary_msg)

    if not output_path.exists():
        output_path.mkdir(exist_ok=True, parents=True)

    result_path = output_path / f"results_{summary.timestamp}.json"
    msg = f"Save results to {result_path.as_posix()}"
    logger.info(msg)
    result_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")


class _AccurateCalculator:
    def __init__(self):
        self.processed: int = 0
        self.passed: int = 0

    @property
    def accuracy(self) -> float:
        if self.processed == 0:
            return 0.0

        return self.passed / self.processed

    def add(self, passed: bool) -> "_AccurateCalculator":
        self.passed += 1 if passed else 0
        self.processed += 1
        return self

    def __str__(self):
        return f"[{self.passed}/{self.processed}]=>[{self.accuracy:.2%}]"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", "-c", type=str, help="Video-MME test config path.")
    parser.add_argument("--datasets", "-d", type=str, help="Video-MME datasets path.")
    parser.add_argument("--host", "-H", type=str, default="http://localhost:7860", help="vrag or baseline host.")
    parser.add_argument("--timeout", "-t", type=int, default=3600, help="timeout(s) to visit bentoml service.")
    parser.add_argument("--output", "-o", type=str, default="output", help="output directory.")
    parser.add_argument("--max_concurrent", type=int, default=10, help="max concurrent to visit bentoml service.")

    args = parser.parse_args()

    asyncio.run(
        run_evaluation_loop(
            config_path=Path(args.config),
            datasets_path=Path(args.datasets),
            output_path=Path(args.output),
            host=args.host,
            timeout=args.timeout,
            max_concurrent=args.max_concurrent,
        )
    )
