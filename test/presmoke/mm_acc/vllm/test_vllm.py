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
# pylint: disable=redefined-outer-name
"""
End-to-end pre-smoke test for the vLLM + MultimodalSDK integration.

The test boots a vLLM OpenAI-compatible server as a subprocess with the
``MM_SCC_RATE`` environment variable exported, waits for the server to be
ready, then issues one chat completion for an image and one for a video.

The server fixture is parametrized over ``MODEL_PATHS`` so each configured
model gets its own server (on a unique port) and a fresh round of image /
video inference calls.
"""

import logging
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from urllib import request as urlrequest
from urllib.error import URLError

import pytest
from mm_test.common import (
    IMAGE_PATH,
    MM_SCC_RATE,
    MODEL_PATHS,
    USER_PROMPT,
    VIDEO_PATH,
    VLLM_EXTRA_ARGS,
    VLLM_HOST,
    VLLM_PORT,
    VLLM_SERVER_READY_INTERVAL,
    VLLM_SERVER_READY_TIMEOUT,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _require_path(path: str, kind: str) -> str:
    """Fail fast if a constant has not been filled in or does not exist."""
    if not path:
        pytest.skip(f"{kind} is not configured in mm_test/common.py")
    if not os.path.exists(path):
        pytest.skip(f"{kind} does not exist: {path}")
    return path


def _to_file_url(path: str) -> str:
    """Convert a local path into a ``file:///`` URL the vLLM server can load."""
    abs_path = os.path.abspath(path)
    return f"file://{abs_path}"


@dataclass
class VllmServerCtx:
    """Per-model state shared with the test functions."""

    model_path: str
    served_model_name: str
    base_url: str
    proc: subprocess.Popen


def _build_vllm_command(model_path: str, served_model_name: str, port: int) -> list:
    """Construct the ``vllm serve`` command line for one model."""
    cmd = [
        "vllm",
        "serve",
        model_path,
        "--host",
        VLLM_HOST,
        "--port",
        str(port),
        "--served-model-name",
        served_model_name,
        "--max-model-len",
        "20480",
        "--allowed-local-media-path",
        "/workspace",
    ]
    cmd.extend(VLLM_EXTRA_ARGS)
    return cmd


def _wait_for_server(base_url: str, timeout: int, interval: int) -> None:
    """Poll ``/v1/models`` until the server responds successfully."""
    deadline = time.monotonic() + timeout
    url = f"{base_url}/models"
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urlrequest.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    logger.info("vLLM server is ready at %s", base_url)
                    return
        except (URLError, ConnectionError, OSError) as exc:
            last_error = exc
            logger.info("Waiting for vLLM server at %s: %s", base_url, exc)
        time.sleep(interval)
    raise RuntimeError(
        f"vLLM server did not become ready within {timeout}s "
        f"(last error: {last_error!r}); see [vllm/out] / [vllm/err] log lines above"
    )


def _resolve_served_model_name(model_path: str) -> str:
    """Derive the served-model-name from the model directory name."""
    return os.path.basename(os.path.normpath(model_path)) or model_path


def _stream_subprocess_output(stream, prefix: str, level: int) -> None:
    """Forward every line of ``stream`` to the module logger until EOF."""
    for raw in stream:
        line = raw.rstrip()
        if line:
            logger.log(level, "%s%s", prefix, line)
    stream.close()


def _drain_subprocess_output(proc: subprocess.Popen) -> tuple[threading.Thread, threading.Thread]:
    """Spawn reader threads so the child's pipe never prints and never blocks."""
    out_thread = threading.Thread(
        target=_stream_subprocess_output,
        args=(proc.stdout, "[vllm/out] ", logging.INFO),
        daemon=True,
        name="vllm-stdout-drain",
    )
    err_thread = threading.Thread(
        target=_stream_subprocess_output,
        args=(proc.stderr, "[vllm/err] ", logging.WARNING),
        daemon=True,
        name="vllm-stderr-drain",
    )
    out_thread.start()
    err_thread.start()
    return out_thread, err_thread


@pytest.fixture(scope="module", params=MODEL_PATHS, ids=os.path.basename)
def vllm_server(request):
    """Launch a vLLM server for each entry in ``MODEL_PATHS`` and tear it down."""
    model_path = request.param
    _require_path(model_path, "MODEL_PATHS entry")

    # Pick a unique port per model so parametrized runs do not collide.
    port = VLLM_PORT + MODEL_PATHS.index(model_path)
    served_model_name = _resolve_served_model_name(model_path)
    base_url = f"http://{VLLM_HOST}:{port}/v1"

    env = os.environ.copy()
    env["MM_SCC_RATE"] = MM_SCC_RATE
    env["MM_PREPROCESSOR"] = "True"

    cmd = _build_vllm_command(model_path, served_model_name, port)
    logger.info("Starting vLLM subprocess for %s (MM_SCC_RATE=%s)", model_path, MM_SCC_RATE)
    logger.info("Command: %s", cmd)
    # ``with subprocess.Popen`` only closes the pipes and then ``wait()``s for
    # the child to exit; it never sends a signal. Terminate explicitly inside
    # the ``with`` block so ``__exit__``'s final ``wait()`` is a no-op.
    # stdout/stderr are piped separately so two reader threads can drain them
    # into the module logger in real time -- otherwise a startup crash would
    # only surface as "vLLM did not become ready" with no error context.
    with subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    ) as proc:
        out_thread, err_thread = _drain_subprocess_output(proc)
        try:
            _wait_for_server(
                base_url=base_url,
                timeout=VLLM_SERVER_READY_TIMEOUT,
                interval=VLLM_SERVER_READY_INTERVAL,
            )
            yield VllmServerCtx(
                model_path=model_path,
                served_model_name=served_model_name,
                base_url=base_url,
                proc=proc,
            )
        finally:
            if proc.poll() is None:
                logger.info("Terminating vLLM subprocess pid=%s", proc.pid)
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    logger.warning("vLLM did not exit, killing pid=%s", proc.pid)
                    proc.kill()
                    proc.wait(timeout=30)
            # Give the drainer threads a moment to flush any final lines.
            for t in (out_thread, err_thread):
                t.join(timeout=5)


def _chat_completion(ctx: VllmServerCtx, content: list) -> dict:
    """POST a multimodal chat completion to the local server."""
    import json  # local import keeps the module import-cheap when skipped

    payload = {
        "model": ctx.served_model_name,
        "messages": [
            {
                "role": "user",
                "content": content,
            }
        ],
        "max_tokens": 256,
    }
    body = json.dumps(payload).encode("utf-8")
    req = urlrequest.Request(
        f"{ctx.base_url}/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlrequest.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _extract_text(response: dict) -> str:
    choices = response.get("choices") or []
    assert choices, f"no choices in response: {response}"
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return content or ""


def _log_response(label: str, ctx: VllmServerCtx, response: dict, text: str) -> None:
    """Print the full inference reply (header + extracted text + raw payload)."""
    sep = "=" * 72
    header = f"{sep}\n[{ctx.model_path}] {label}\n{sep}"
    print(header)
    print(f"served-model-name: {ctx.served_model_name}")
    print(f"endpoint         : {ctx.base_url}/chat/completions")
    print(f"reply text       :\n{text}")
    print(sep)
    logger.info("%s\nreply text: %s", header, text)
    logger.info("raw response payload: %s", response)


def test_image_inference(vllm_server):
    """Run one chat completion against an image and verify a non-empty reply."""
    image_path = _require_path(IMAGE_PATH, "IMAGE_PATH")
    image_url = _to_file_url(image_path)

    response = _chat_completion(
        vllm_server,
        [
            {"type": "text", "text": USER_PROMPT},
            {"type": "image_url", "image_url": {"url": image_url}},
        ],
    )
    text = _extract_text(response)
    _log_response("image inference reply", vllm_server, response, text)
    assert text.strip(), "vLLM returned an empty reply for the image"


def test_video_inference(vllm_server):
    """Run one chat completion against a video and verify a non-empty reply."""
    video_path = _require_path(VIDEO_PATH, "VIDEO_PATH")
    video_url = _to_file_url(video_path)

    response = _chat_completion(
        vllm_server,
        [
            {"type": "text", "text": USER_PROMPT},
            {"type": "video_url", "video_url": {"url": video_url}},
        ],
    )
    text = _extract_text(response)
    _log_response("video inference reply", vllm_server, response, text)
    assert text.strip(), "vLLM returned an empty reply for the video"
