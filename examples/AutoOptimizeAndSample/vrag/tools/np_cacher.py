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
# Numpy caching decorators and utilities.

import hashlib
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Concatenate, List, Optional, ParamSpec, TypeAlias, TypeVar, Union, overload, Coroutine

import numpy as np

from vrag.logger import logger

P = ParamSpec("P")
T = TypeVar("T")
S = TypeVar("S")
ArrayInput: TypeAlias = Union[np.ndarray, List[np.ndarray]]


class CacherBase(metaclass=ABCMeta):
    @staticmethod
    def _gather_missed(
        all_keys: List[str], is_dense_arr: bool, miss_indices: List[int], work_list: List[np.ndarray]
    ) -> tuple[ArrayInput, List[str]]:
        msg = f"Cache miss count (sync): {len(miss_indices)} / {len(work_list)}"
        logger.debug(msg)

        missed_arrays = [work_list[idx] for idx in miss_indices]
        missed_keys = [all_keys[idx] for idx in miss_indices]

        batch_input = np.stack(missed_arrays, axis=0) if is_dense_arr else missed_arrays
        return batch_input, missed_keys

    @staticmethod
    def _validate_results(result_map: dict[int, T], expected_length: int) -> List[T]:
        final_result: List[T] = []
        for i in range(expected_length):
            if i not in result_map:
                raise RuntimeError(f"Logic error: missing result for index {i}")
            final_result.append(result_map[i])
        return final_result

    @staticmethod
    def _normalized_arr(arr_input: ArrayInput) -> tuple[bool, List[np.ndarray]]:
        if isinstance(arr_input, np.ndarray):
            work_list: List[np.ndarray] = list(arr_input)
            is_dense_arr = True
        else:
            work_list = arr_input
            is_dense_arr = False
        return is_dense_arr, work_list

    @staticmethod
    def _get_array_hash(arr: np.ndarray) -> str:
        """Generate hash including shape, dtype, and content to prevent collisions."""
        arr_contiguous = np.ascontiguousarray(arr)
        shape_bytes = str(arr.shape).encode("utf-8")
        dtype_bytes = str(arr.dtype).encode("utf-8")
        data_bytes = arr_contiguous.tobytes()

        combined = shape_bytes + b"|" + dtype_bytes + b"|" + data_bytes
        return hashlib.sha256(combined).hexdigest()

    @abstractmethod
    def get(self, key: str) -> Any:
        raise NotImplementedError(f"{self.__class__} not implements get")

    @abstractmethod
    def put(self, key: str, data: Any) -> None:
        raise NotImplementedError(f"{self.__class__} not implements put")

    def cached_sync(
        self, func: Callable[Concatenate[ArrayInput, P], List[T]]
    ) -> Callable[Concatenate[ArrayInput, P], List[T]]:
        """Sync version of cache."""

        @wraps(func)
        def _wrapper(arr_input: ArrayInput, *args: P.args, **kwargs: P.kwargs) -> List[T]:
            return self._cache_logic_sync(arr_input, args, kwargs, func, self._get_array_hash)

        return _wrapper

    def cached_sync_with(
        self, make_key_suffix: Callable[P, str]
    ) -> Callable[[Callable[Concatenate[ArrayInput, P], List[T]]], Callable[Concatenate[ArrayInput, P], List[T]]]:
        def decorator(
            func: Callable[Concatenate[ArrayInput, P], List[T]],
        ) -> Callable[Concatenate[ArrayInput, P], List[T]]:
            @wraps(func)
            def _wrapper(arr_input: ArrayInput, *args: P.args, **kwargs: P.kwargs) -> List[T]:
                suffix = make_key_suffix(*args, **kwargs)
                return self._cache_logic_sync(
                    arr_input, args, kwargs, func, lambda arr: self._get_combined_key(arr, suffix)
                )

            return _wrapper

        return decorator

    def cached(
        self, func: Callable[Concatenate[ArrayInput, P], Coroutine[Any, Any, List[T]]]
    ) -> Callable[Concatenate[ArrayInput, P], Coroutine[Any, Any, List[T]]]:
        @wraps(func)
        async def _wrapper(arr_input: ArrayInput, *args: P.args, **kwargs: P.kwargs) -> List[T]:
            return await self._cache_logic(arr_input, args, kwargs, func, self._get_array_hash)

        return _wrapper

    def cached_with(
        self, make_key_suffix: Callable[P, str]
    ) -> Callable[
        [Callable[Concatenate[ArrayInput, P], Coroutine[Any, Any, List[T]]]],
        Callable[Concatenate[ArrayInput, P], Coroutine[Any, Any, List[T]]],
    ]:
        def decorator(
            func: Callable[Concatenate[ArrayInput, P], Coroutine[Any, Any, List[T]]],
        ) -> Callable[Concatenate[ArrayInput, P], Coroutine[Any, Any, List[T]]]:
            @wraps(func)
            async def _wrapper(arr_input: ArrayInput, *args: P.args, **kwargs: P.kwargs) -> List[T]:
                suffix = make_key_suffix(*args, **kwargs)
                return await self._cache_logic(
                    arr_input, args, kwargs, func, lambda arr: self._get_combined_key(arr, suffix)
                )

            return _wrapper

        return decorator

    def cached_method(
        self, func: Callable[Concatenate[S, ArrayInput, P], Coroutine[Any, Any, List[T]]]
    ) -> Callable[Concatenate[S, ArrayInput, P], Coroutine[Any, Any, List[T]]]:
        @wraps(func)
        async def _wrapper(slf: S, arr_input: ArrayInput, *args: P.args, **kwargs: P.kwargs) -> List[T]:
            return await self._cache_logic(arr_input, args, kwargs, func, self._get_array_hash, slf=slf)

        return _wrapper

    def cached_method_with(
        self, make_key_suffix: Callable[P, str]
    ) -> Callable[
        [Callable[Concatenate[S, ArrayInput, P], Coroutine[Any, Any, List[T]]]],
        Callable[Concatenate[S, ArrayInput, P], Coroutine[Any, Any, List[T]]],
    ]:
        def decorator(
            func: Callable[Concatenate[S, ArrayInput, P], Coroutine[Any, Any, List[T]]],
        ) -> Callable[Concatenate[S, ArrayInput, P], Coroutine[Any, Any, List[T]]]:
            @wraps(func)
            async def _wrapper(slf: S, arr_input: ArrayInput, *args: P.args, **kwargs: P.kwargs) -> List[T]:
                suffix = make_key_suffix(*args, **kwargs)
                return await self._cache_logic(
                    arr_input, args, kwargs, func, lambda arr: self._get_combined_key(arr, suffix), slf=slf
                )

            return _wrapper

        return decorator

    def _merge_result(
        self, batch_result: List[T], miss_indices: List[int], miss_keys: List[str], result_map: dict[int, T]
    ):
        if len(batch_result) != len(miss_indices):
            raise ValueError(f"Merge failed with {len(batch_result)} items, expected {len(miss_indices)}")

        for i, idx in enumerate(miss_indices):
            res_item = batch_result[i]
            result_map[idx] = res_item
            self.put(miss_keys[i], res_item)

    def _gather_hits_misses(
        self, make_key: Callable[[np.ndarray], str], work_list: List[np.ndarray]
    ) -> tuple[List[str], List[int], dict[int, T]]:
        result_map: dict[int, T] = {}
        miss_indices: List[int] = []

        all_keys = [make_key(arr) for arr in work_list]

        for idx, key in enumerate(all_keys):
            cached_val = self.get(key)
            if cached_val is not None:
                result_map[idx] = cached_val
            else:
                miss_indices.append(idx)

        return all_keys, miss_indices, result_map

    def _get_combined_key(self, arr: np.ndarray, extra_key: str) -> str:
        return f"{self._get_array_hash(arr)}_{extra_key}"

    @overload
    def _cache_logic_sync(
        self,
        arr_input: ArrayInput,
        args: P.args,
        kwargs: P.kwargs,
        func: Callable[Concatenate[ArrayInput, P], List[T]],
        make_key: Callable[[np.ndarray], str],
    ) -> List[T]:
        """Cache for function."""

    @overload
    def _cache_logic_sync(
        self,
        arr_input: ArrayInput,
        args: P.args,
        kwargs: P.kwargs,
        func: Callable[Concatenate[S, ArrayInput, P], List[T]],
        make_key: Callable[[np.ndarray], str],
        /,
        slf: S = None,
    ) -> List[T]:
        """Cache for method."""

    def _cache_logic_sync(
        self,
        arr_input: ArrayInput,
        args: P.args,
        kwargs: P.kwargs,
        func: Callable[Concatenate[ArrayInput, P], List[T]] | Callable[Concatenate[S, ArrayInput, P], List[T]],
        make_key: Callable[[np.ndarray], str],
        /,
        slf: S = None,
    ) -> List[T]:
        """Sync version of cache."""
        is_dense_arr, work_list = self._normalized_arr(arr_input)
        all_keys, miss_indices, result_map = self._gather_hits_misses(make_key, work_list)
        if miss_indices:
            batch_input, miss_keys = self._gather_missed(all_keys, is_dense_arr, miss_indices, work_list)
            batch_results = (func(slf, batch_input, *args, **kwargs)) if slf else func(batch_input, *args, **kwargs)
            self._merge_result(batch_results, miss_indices, miss_keys, result_map)
        else:
            logger.debug("Cache all hit (sync)!")
        return self._validate_results(result_map, len(work_list))

    @overload
    async def _cache_logic(
        self,
        arr_input: ArrayInput,
        args: P.args,
        kwargs: P.kwargs,
        func: Callable[Concatenate[ArrayInput, P], Coroutine[Any, Any, List[T]]],
        make_key: Callable[[np.ndarray], str],
    ) -> List[T]:
        """Cache for function"""

    @overload
    async def _cache_logic(
        self,
        arr_input: ArrayInput,
        args: P.args,
        kwargs: P.kwargs,
        func: Callable[Concatenate[S, ArrayInput, P], Coroutine[Any, Any, List[T]]],
        make_key: Callable[[np.ndarray], str],
        /,
        slf: S = None,
    ) -> List[T]:
        """Cache for method"""

    async def _cache_logic(
        self,
        arr_input: ArrayInput,
        args: P.args,
        kwargs: P.kwargs,
        func: Callable[Concatenate[ArrayInput, P], Coroutine[Any, Any, List[T]]]
        | Callable[Concatenate[S, ArrayInput, P], Coroutine[Any, Any, List[T]]],
        make_key: Callable[[np.ndarray], str],
        /,
        slf: S = None,
    ) -> List[T]:
        """Async version of cache."""
        is_dense_arr, work_list = self._normalized_arr(arr_input)
        all_keys, miss_indices, result_map = self._gather_hits_misses(make_key, work_list)
        if miss_indices:
            batch_input, miss_keys = self._gather_missed(all_keys, is_dense_arr, miss_indices, work_list)
            if slf:
                batch_results = await func(slf, batch_input, *args, **kwargs)
            else:
                batch_results = await func(batch_input, *args, **kwargs)
            self._merge_result(batch_results, miss_indices, miss_keys, result_map)
        else:
            logger.debug("Cache all hit (async)!")
        return self._validate_results(result_map, len(work_list))


def get_cacher(cap: int) -> CacherBase:
    return _NumpyLRUCache(capacity=cap)


@dataclass
class _NumpyLRUCache(CacherBase):
    capacity: int
    cache: OrderedDict = field(init=False, default_factory=OrderedDict)

    def get(self, key: str) -> Optional[Any]:
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: str, data: Any) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.capacity:
            self.cache.popitem(last=False)
        self.cache[key] = data
