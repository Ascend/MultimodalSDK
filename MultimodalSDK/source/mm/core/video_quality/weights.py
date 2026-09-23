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

"""Weight resolver — unified weight path resolution with explicit-param priority.

Resolution priority for every weight role:
    1. Explicit ``weights`` value passed by the caller (highest priority).
    2. Environment variable declared in ``defaults.WEIGHT_ROLES``.
    3. ``ValueError`` with actionable message.

For single-weight scorers, ``weights`` is a ``str | None``.
For dual-weight scorers, ``weights`` is a ``dict[str, str] | None`` keyed by
role name, and each role is resolved independently.
"""

from __future__ import annotations

import os
from typing import Any

from .defaults import WeightRole, get_weight_roles


class WeightResolver:
    """Resolve weight paths for a scorer using explicit-param > env-var > error."""

    def __init__(self, scorer_name: str):
        self._scorer_name = scorer_name
        self._roles: list[WeightRole] = get_weight_roles(scorer_name)

    @property
    def scorer_name(self) -> str:
        return self._scorer_name

    @property
    def roles(self) -> list[WeightRole]:
        return self._roles

    def resolve_single(self, weights: str | None, role: str | None = None) -> str:
        """Resolve a single-weight scorer's path.

        Only valid for scorers declaring exactly one weight role.  Multi-role
        scorers must go through :meth:`resolve_multi` (or pass *role*
        explicitly), otherwise their remaining roles would be silently
        dropped.  If *role* is given it must match a declared role name.
        """
        if not self._roles:
            raise ValueError(f"Scorer '{self._scorer_name}' does not declare any weight roles.")
        if role is None and len(self._roles) != 1:
            raise ValueError(
                f"Scorer '{self._scorer_name}' declares {len(self._roles)} weight roles "
                f"{[r.role for r in self._roles]}; pass a dict via resolve_multi() "
                f"or specify role=..."
            )

        if role is None:
            target_role = self._roles[0]
        else:
            target_role = self._find_role(role)

        if weights is not None:
            return weights
        return self._from_env(target_role)

    def resolve_multi(self, weights: dict[str, str] | None) -> dict[str, str]:
        """Resolve a multi-weight scorer's paths.

        Returns a dict ``{role_name: resolved_path}`` for every declared role.
        Keys in *weights* that do not match any declared role are ignored.
        """
        if not self._roles:
            raise ValueError(f"Scorer '{self._scorer_name}' does not declare any weight roles.")

        weights = weights or {}
        resolved: dict[str, str] = {}
        for role in self._roles:
            value = weights.get(role.role)
            if value is not None:
                resolved[role.role] = value
            else:
                resolved[role.role] = self._from_env(role)
        return resolved

    def resolve(self, weights: Any) -> Any:
        """Auto-dispatch: single str → resolve_single, dict → resolve_multi."""
        if isinstance(weights, dict):
            return self.resolve_multi(weights)
        return self.resolve_single(weights)

    def _find_role(self, role: str) -> WeightRole:
        for r in self._roles:
            if r.role == role:
                return r
        raise ValueError(
            f"Unknown weight role '{role}' for scorer '{self._scorer_name}'. "
            f"Declared roles: {[r.role for r in self._roles]}"
        )

    @staticmethod
    def _from_env(role: WeightRole) -> str:
        value = os.environ.get(role.env_var, "")
        if value:
            return value
        raise ValueError(
            f"Weight role '{role.role}' for scorer is not provided. "
            f"Either pass it explicitly via the `weights` parameter, "
            f"or set the environment variable {role.env_var}. "
            f"Description: {role.description or role.role}"
        )
