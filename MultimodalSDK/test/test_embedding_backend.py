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

# pylint: skip-file

import base64
import sys
from unittest.mock import MagicMock, patch

# NOTE: importing ``mm`` on a dev machine without the built ``_acc``
# extension requires the stub injection provided by ``test/conftest.py``
# (added by the sibling KTS-modules PR; merge that PR first).

# The tests mock the OpenAI client entirely; on CI/dev machines without the
# optional ``openai`` package installed, inject a minimal placeholder package
# so ``patch("openai.OpenAI")`` and the lazy type imports inside
# ``OpenAIEmbeddingBackend._init_client`` still resolve.
try:
    import openai  # noqa: F401
except ImportError:
    import types

    _openai_stub = types.ModuleType("openai")
    _openai_stub.__path__ = []
    _openai_stub.OpenAI = object  # replaced by mock.patch per test

    _types_stub = types.ModuleType("openai.types")
    _types_stub.__path__ = []
    _openai_stub.types = _types_stub

    _create_embedding_stub = types.ModuleType("openai.types.create_embedding_response")
    _create_embedding_stub.CreateEmbeddingResponse = object
    _types_stub.create_embedding_response = _create_embedding_stub
    _types_stub.CreateEmbeddingResponse = object  # version fallback path

    sys.modules.setdefault("openai", _openai_stub)
    sys.modules.setdefault("openai.types", _types_stub)
    sys.modules.setdefault("openai.types.create_embedding_response", _create_embedding_stub)

import numpy as np
import pytest

from mm.core.segmenter.embedding_backend import (
    EmbeddingBackend,
    OpenAIEmbeddingBackend,
    _l2_normalize,
)

VALID_BASE_URL = "http://127.0.0.1:8000/v1"
VALID_MODEL = "Qwen3-VL-Embedding-2B"
SYSTEM_INSTRUCTION = "Represent the user's input."


def _make_embedding_response(values):
    """Builds a mock CreateEmbeddingResponse-like object."""
    item = MagicMock()
    item.embedding = list(values)
    resp = MagicMock()
    resp.data = [item]
    return resp


@pytest.fixture
def backend_client():
    """Creates an 'openai' protocol backend with a mocked OpenAI client.

    Yields (backend, client, openai_cls_mock).
    """
    with patch("openai.OpenAI") as openai_cls:
        client = MagicMock()
        openai_cls.return_value = client
        backend = OpenAIEmbeddingBackend(base_url=VALID_BASE_URL)
        yield backend, client, openai_cls
        backend.close()


def _write_frames(tmp_path, contents):
    paths = []
    for i, data in enumerate(contents):
        p = tmp_path / f"frame_{i:03d}.jpg"
        p.write_bytes(data)
        paths.append(str(p))
    return paths


# ------------------------------------------------------------------ #
# construction contract (§6.6)
# ------------------------------------------------------------------ #
class TestInitValidation:
    @pytest.mark.parametrize(
        "bad_url",
        [
            "ftp://127.0.0.1:8000/v1",  # not http(s)
            "http://127.0.0.1:8000/api",  # not ending with /v1
            "http://127.0.0.1:8000/v1\x00",  # null byte
        ],
    )
    def test_base_url_value_errors(self, bad_url):
        with pytest.raises(ValueError):
            OpenAIEmbeddingBackend(base_url=bad_url)

    def test_base_url_type_error(self):
        with pytest.raises(TypeError):
            OpenAIEmbeddingBackend(base_url=123)

    @pytest.mark.parametrize("bad_model", ["", None, 42])
    def test_model_name_invalid(self, bad_model):
        with pytest.raises(ValueError):
            OpenAIEmbeddingBackend(base_url=VALID_BASE_URL, model_name=bad_model)

    @pytest.mark.parametrize("bad_timeout", [0, -1, "60", None, True])
    def test_timeout_invalid(self, bad_timeout):
        with pytest.raises(ValueError):
            OpenAIEmbeddingBackend(base_url=VALID_BASE_URL, timeout=bad_timeout)

    @pytest.mark.parametrize("bad_concurrency", [0, -1, True, "32", 2.5])
    def test_max_concurrent_invalid(self, bad_concurrency):
        with pytest.raises(ValueError):
            OpenAIEmbeddingBackend(base_url=VALID_BASE_URL, max_concurrent=bad_concurrency)

    def test_protocol_invalid(self):
        with pytest.raises(TypeError):
            OpenAIEmbeddingBackend(base_url=VALID_BASE_URL, protocol="grpc")

    def test_timeout_passed_to_client(self):
        with patch("openai.OpenAI") as openai_cls:
            openai_cls.return_value = MagicMock()
            OpenAIEmbeddingBackend(base_url=VALID_BASE_URL, timeout=12.5)
            kwargs = openai_cls.call_args.kwargs
            assert kwargs["timeout"] == 12.5
            assert kwargs["base_url"] == VALID_BASE_URL
            assert kwargs["api_key"] == "EMPTY"

    @pytest.mark.parametrize("bad_api_key", ["", 123, True])
    def test_api_key_invalid(self, bad_api_key):
        with pytest.raises(ValueError):
            OpenAIEmbeddingBackend(base_url=VALID_BASE_URL, api_key=bad_api_key)

    def test_api_key_passthrough(self):
        with patch("openai.OpenAI") as openai_cls:
            openai_cls.return_value = MagicMock()
            OpenAIEmbeddingBackend(base_url=VALID_BASE_URL, api_key="sk-custom-key")
            assert openai_cls.call_args.kwargs["api_key"] == "sk-custom-key"

    def test_openai_missing_raises_import_error(self):
        with patch.dict(sys.modules, {"openai": None}):
            with pytest.raises(ImportError, match="openai"):
                OpenAIEmbeddingBackend(base_url=VALID_BASE_URL)

    def test_backend_id(self):
        with patch("openai.OpenAI"):
            backend = OpenAIEmbeddingBackend(base_url=VALID_BASE_URL, model_name="mymodel")
            assert backend.backend_id == "openai:mymodel"
            backend.close()


# ------------------------------------------------------------------ #
# image protocol: vLLM Chat Embeddings wire format (§8.2, field-by-field)
# ------------------------------------------------------------------ #
class TestEncodeImages:
    def test_wire_format_field_by_field(self, backend_client, tmp_path):
        backend, client, _ = backend_client
        content = b"\xff\xd8fakejpegbytes"
        (path,) = _write_frames(tmp_path, [content])
        client.post.return_value = _make_embedding_response([3.0, 4.0])

        backend.encode_images([path])

        assert client.post.call_count == 1
        args, kwargs = client.post.call_args
        assert args[0] == "/embeddings"
        body = kwargs["body"]
        messages = body["messages"]
        # three-segment structure
        assert len(messages) == 3
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == [{"type": "text", "text": SYSTEM_INSTRUCTION}]
        assert messages[1]["role"] == "user"
        user_content = messages[1]["content"]
        assert len(user_content) == 2
        expected_b64 = base64.b64encode(content).decode("ascii")
        assert user_content[0] == {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{expected_b64}"},
        }
        assert user_content[1] == {"type": "text", "text": ""}
        assert messages[2]["role"] == "assistant"
        assert messages[2]["content"] == [{"type": "text", "text": ""}]
        # scalar fields
        assert body["model"] == VALID_MODEL
        assert body["encoding_format"] == "float"
        assert body["continue_final_message"] is True
        assert body["add_special_tokens"] is True
        # response type cast
        assert kwargs["cast_to"] is backend._resp_type

    def test_result_shape_and_l2_normalization(self, backend_client, tmp_path):
        backend, client, _ = backend_client
        paths = _write_frames(tmp_path, [b"a", b"b", b"c"])
        client.post.side_effect = [
            _make_embedding_response([3.0, 4.0]),
            _make_embedding_response([0.0, 5.0]),
            _make_embedding_response([1.0, 0.0]),
        ]
        mat = backend.encode_images(paths)
        assert mat.shape == (3, 2)
        assert mat.dtype == np.float32
        np.testing.assert_allclose(np.linalg.norm(mat, axis=1), 1.0, atol=1e-6)

    def test_order_preserved_under_concurrency(self, backend_client, tmp_path):
        backend, client, _ = backend_client
        contents = [bytes([i]) * 8 for i in range(8)]
        paths = _write_frames(tmp_path, contents)
        vectors = [[float(i), 1.0] for i in range(8)]

        def fake_post(url, body=None, cast_to=None):
            url_field = body["messages"][1]["content"][0]["image_url"]["url"]
            b64 = url_field.split(",", 1)[1]
            idx = contents.index(base64.b64decode(b64))
            return _make_embedding_response(vectors[idx])

        client.post.side_effect = fake_post
        mat = backend.encode_images(paths)
        # each row must correspond to its own input path
        for i in range(8):
            np.testing.assert_allclose(mat[i], _l2_normalize(np.array(vectors[i])), atol=1e-7)

    def test_missing_file_raises(self, backend_client):
        backend, _, _ = backend_client
        with pytest.raises(FileNotFoundError):
            backend.encode_images(["/nonexistent/frame.jpg"])

    def test_max_concurrent_configurable(self, tmp_path):
        captured = {}

        class FakeExecutor:
            def __init__(self, max_workers=None):
                captured["max_workers"] = max_workers

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def map(self, fn, *iterables):
                return [fn(p) for p in iterables[0]]

        with (
            patch("openai.OpenAI") as openai_cls,
            patch("mm.core.segmenter.embedding_backend.ThreadPoolExecutor", FakeExecutor),
        ):
            client = MagicMock()
            openai_cls.return_value = client
            backend = OpenAIEmbeddingBackend(base_url=VALID_BASE_URL, max_concurrent=8)
            try:
                (path,) = _write_frames(tmp_path, [b"frame"])
                client.post.return_value = _make_embedding_response([3.0, 4.0])
                backend.encode_images([path])
            finally:
                backend.close()
            assert captured["max_workers"] == 8

    @pytest.mark.parametrize("bad_input", [[], "not-a-list", [123], None])
    def test_invalid_input(self, backend_client, bad_input):
        backend, _, _ = backend_client
        with pytest.raises((ValueError, TypeError)):
            backend.encode_images(bad_input)

    def test_server_error_wrapped_as_runtime_error(self, backend_client, tmp_path):
        backend, client, _ = backend_client
        (path,) = _write_frames(tmp_path, [b"frame"])
        client.post.side_effect = RuntimeError("connection refused by server")
        with pytest.raises(RuntimeError, match="connection refused by server"):
            backend.encode_images([path])

    def test_invalid_response_shape(self, backend_client, tmp_path):
        backend, client, _ = backend_client
        (path,) = _write_frames(tmp_path, [b"frame"])
        empty_resp = MagicMock()
        empty_resp.data = []
        client.post.return_value = empty_resp
        with pytest.raises(RuntimeError, match="invalid embedding response"):
            backend.encode_images([path])


# ------------------------------------------------------------------ #
# text protocol: standard /embeddings primary path (§8.2-3)
# ------------------------------------------------------------------ #
class TestEncodeText:
    def test_standard_path_request(self, backend_client):
        backend, client, _ = backend_client
        client.embeddings.create.return_value = _make_embedding_response([3.0, 4.0])
        vec = backend.encode_text("what traffic signs appear")

        assert client.embeddings.create.call_count == 1
        kwargs = client.embeddings.create.call_args.kwargs
        assert kwargs["model"] == VALID_MODEL
        assert kwargs["input"] == ["what traffic signs appear"]
        # client.post must not be used on the standard path
        client.post.assert_not_called()
        assert vec.shape == (2,)
        np.testing.assert_allclose(np.linalg.norm(vec), 1.0, atol=1e-6)

    def test_chat_path_wire_format_field_by_field(self, backend_client):
        backend, client, _ = backend_client
        client.post.return_value = _make_embedding_response([1.0, 0.0])
        backend.encode_text_chat("query text")

        args, kwargs = client.post.call_args
        assert args[0] == "/embeddings"
        body = kwargs["body"]
        messages = body["messages"]
        assert len(messages) == 3
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == [{"type": "text", "text": SYSTEM_INSTRUCTION}]
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == [{"type": "text", "text": "query text"}]
        assert messages[2]["role"] == "assistant"
        assert messages[2]["content"] == [{"type": "text", "text": ""}]
        assert body["model"] == VALID_MODEL
        assert body["encoding_format"] == "float"
        assert body["continue_final_message"] is True
        assert body["add_special_tokens"] is True

    def test_standard_path_error_wrapped(self, backend_client):
        backend, client, _ = backend_client
        client.embeddings.create.side_effect = TimeoutError("request timed out")
        with pytest.raises(RuntimeError, match="request timed out"):
            backend.encode_text("query")

    def test_chat_path_error_wrapped(self, backend_client):
        backend, client, _ = backend_client
        client.post.side_effect = ValueError("bad payload")
        with pytest.raises(RuntimeError, match="bad payload"):
            backend.encode_text_chat("query")

    @pytest.mark.parametrize("bad_text", ["", None, 123])
    def test_invalid_text(self, backend_client, bad_text):
        backend, _, _ = backend_client
        with pytest.raises((ValueError, TypeError)):
            backend.encode_text(bad_text)

    def test_zero_vector_returned_as_is(self, backend_client):
        backend, client, _ = backend_client
        client.embeddings.create.return_value = _make_embedding_response([0.0, 0.0])
        vec = backend.encode_text("query")
        np.testing.assert_array_equal(vec, np.array([0.0, 0.0], dtype=np.float32))


# ------------------------------------------------------------------ #
# lifecycle
# ------------------------------------------------------------------ #
class TestLifecycle:
    def test_close_idempotent(self):
        with patch("openai.OpenAI"):
            backend = OpenAIEmbeddingBackend(base_url=VALID_BASE_URL)
            backend.close()
            backend.close()  # must not raise

    def test_closed_rejects_encoding(self, backend_client):
        backend, _, _ = backend_client
        backend.close()
        with pytest.raises(RuntimeError, match="closed"):
            backend.encode_text("query")
        with pytest.raises(RuntimeError, match="closed"):
            backend.encode_images(["/any/frame.jpg"])

    def test_close_failure_does_not_propagate(self):
        with patch("openai.OpenAI") as openai_cls:
            client = MagicMock()
            client.close.side_effect = OSError("pool already closed")
            openai_cls.return_value = client
            backend = OpenAIEmbeddingBackend(base_url=VALID_BASE_URL)
            backend.close()  # must not raise
            client.close.assert_called_once()

    def test_is_embedding_backend_subclass(self, backend_client):
        backend, _, _ = backend_client
        assert isinstance(backend, EmbeddingBackend)
