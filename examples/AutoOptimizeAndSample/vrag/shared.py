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
# Shared utilities and helper functions for the VRAG.
# This module provides common utilities including:
# - Validation helpers
# - Decorators (once, execute_time)
# - Configuration base class (ArgsBase)

import asyncio
import copy
from functools import wraps
from itertools import chain, filterfalse
from pathlib import Path
from time import time
from typing import Any, Callable, Iterable, List, Optional, ParamSpec, Tuple, Type, TypeVar, Awaitable

import numpy as np
import tomli
from pydantic import BaseModel, ConfigDict, Field

import bentoml

from vrag.constants import DEFAULT_RETRY_EXCEPTIONS
from vrag.logger import logger

_S = TypeVar("_S", bound="ArgsBase")
_T = TypeVar("_T")
_R = TypeVar("_R")
_P = ParamSpec("_P")
U8_MAX = 255


def into_u8_frames(n_frames: np.ndarray) -> np.ndarray:
    if np.issubdtype(n_frames.dtype, np.floating):
        return (n_frames * U8_MAX).astype(np.uint8)
    elif n_frames.dtype != np.uint8:
        return n_frames.astype(np.uint8)
    return n_frames


def once(func: Callable[_P, _R]) -> Callable[_P, _R]:
    """
    Decorator that ensures a function is executed only once and caches the result.

    Support both synchronous functions.

    Args:
        func: The function to wrap.

    Returns:
        A wrapped function that returns the cached result on subsequent calls.
    """
    __ret_value: Optional[_R] = None
    _executed: bool = False

    @wraps(func)
    def _wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        nonlocal __ret_value, _executed
        if _executed:
            return __ret_value
        __ret_value = func(*args, **kwargs)
        _executed = True
        return __ret_value

    return _wrapper


def execute_time(func: Callable[_P, _R]) -> Callable[_P, _R]:
    """
    Decorators that logs the execution time of a function.

    Args:
        func: The synchronous function to be timing.

    Returns:
        The wrapped function with logging capabilities.
    """

    @wraps(func)
    def _wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        start = time()
        msg = f"Executing {func.__name__}"
        logger.debug(msg)
        res = func(*args, **kwargs)
        msg = f"Executing {func.__name__} in {time() - start:.4f}s"
        logger.debug(msg)
        return res

    return _wrapper


def first_available(*val: Optional[_T]) -> _T:
    """
    Return the first non-None value from the provided arguments.

    Args:
        val: Variable number of optional values to check.

    Returns:
        The first non-None value among the provided arguments.

    Raises:
        RuntimeError: If all provided values are None.
    """

    def ok(obj: Optional[_T], msg: str = "Value is None") -> _T:
        if obj is None:
            raise RuntimeError(msg)
        return obj

    return ok(next(v for v in val if v is not None), "No non-None values available")


def downsample(data: List[int], target_length: int) -> List[int]:
    """
    Downsample a list of integers to a specified length (only supports target_length <= len(data)).

    Args:
        data: Optional list of integers.
        target_length: Target length (must be <= original length and >= 1).

    Returns:
        Downsample list of integers to target_length.
    """

    n = len(data)
    if target_length > n:
        raise ValueError("Target length should not be larger than original length (downsample)")
    if target_length <= 0:
        raise ValueError("Target length should be positive (downsample)")
    if target_length == n:
        return data.copy()
    if target_length == 1:
        return [data[n // 2]]
    indices: List[int] = np.linspace(0, n - 1, target_length, dtype=int).tolist()
    return [data[idx] for idx in indices]


class ConfigBase(BaseModel):
    model_config = ConfigDict(use_attribute_docstrings=True)


class ArgsBase(BaseModel):
    model_config = ConfigDict(use_attribute_docstrings=True)
    config_file_path: Optional[Path] = None
    """Path to the TOML configuration file for overriding default arguments."""
    service_timeout: float = 3600.0
    """The request session timeout for all services."""
    service_max_concurrency: int = 20
    """The maximum number of concurrent service requests allowed."""
    service_connection_request_multiplier: float = Field(2.0, gt=0)
    """The multiplier for the number of concurrent service requests allowed per runner connection."""

    @classmethod
    @once
    def default(cls: Type[_S]) -> "_S":
        return cls()

    def override_by(self: _S, path: Optional[Path]) -> _S:
        if path is None or not path.exists():
            return self

        override_data = tomli.loads(path.read_text(encoding="utf-8"))
        current_data = self.model_dump(mode="json")

        merged = self._merge(current_data, override_data)
        return self.model_validate(merged)

    def override(self: _S) -> _S:
        return self.override_by(self.config_file_path)

    @classmethod
    def _merge(cls, base: dict, override: dict) -> dict:
        _merged = copy.deepcopy(base)
        for k, v in override.items():
            if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                _merged[k] = cls._merge(base[k], v)
            else:
                _merged[k] = v
        return _merged


def flatten(list_of_list: Iterable[Iterable[_T]]) -> Iterable[_T]:
    return chain.from_iterable(list_of_list)


def vrag_service(args: ArgsBase):
    return bentoml.service(
        traffic={"timeout": args.service_timeout, "max_concurrency": args.service_max_concurrency},
        runner_connection={
            "max_age": args.service_timeout,
            "max_request": args.service_max_concurrency * args.service_connection_request_multiplier,
        },
        max_runner_connections=args.service_max_concurrency,
    )


def unique_everseen(iterable: Iterable[_T], key: Optional[Callable[[_T], _R]] = None) -> Iterable[_T]:
    seen = set()
    seen_add = seen.add
    if key is None:
        for element in filterfalse(seen.__contains__, iterable):
            seen_add(element)
            yield element
    else:
        for element in iterable:
            k = key(element)
            if k not in seen:
                seen_add(k)
                yield element


async def retry_async_request(
    func: Callable[[], Awaitable[Any]],
    label: str,
    *,
    retries: int = 3,
    delay: float = 0.2,
    max_delay: float = 2.0,
    backoff: float = 2.0,
    retry_exceptions: Tuple[Type[Exception], ...] = DEFAULT_RETRY_EXCEPTIONS,
) -> Any:
    attempt = 0
    current_delay = delay

    while True:
        try:
            return await func()
        except retry_exceptions as e:
            if attempt >= retries:
                msg = f"[retry] function: {label} failed after {retries} tries"
                logger.error(msg)
                raise

            sleep_time = min(current_delay, max_delay)

            msg = (
                f"[retry] function: {label} attempt={attempt + 1}/{retries} "
                f"error={type(e).__name__} sleep={sleep_time:.2f}s"
            )
            logger.warning(msg)

            await asyncio.sleep(sleep_time)

            attempt += 1
            current_delay *= backoff
