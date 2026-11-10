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
"""CPU regression tests for compression configuration, clustering and merging."""

from dataclasses import fields
import importlib.util
import logging
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import torch


SOURCE = Path(__file__).resolve().parents[1] / 'source/mm/patcher/vllm'


def load_method():
    """Load the pure algorithm without native SDK libraries or a vLLM install."""
    package = types.ModuleType('flashvid_algorithm_test')
    package.__path__ = [str(SOURCE / 'flashvid')]
    vllm_log = types.ModuleType('vllm.logger')
    vllm_log.init_logger = logging.getLogger
    sdk_log = types.ModuleType('mm.comm.log')
    sdk_log._Logger = logging.getLogger('flashvid_test')
    with patch.dict(sys.modules, {'flashvid_algorithm_test': package, 'vllm.logger': vllm_log, 'mm.comm.log': sdk_log}):
        spec = importlib.util.spec_from_file_location('mm.patcher.vllm.constants', SOURCE / 'constants.py')
        constants = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(constants)
        with patch.dict(sys.modules, {'mm.patcher.vllm.constants': constants}):
            spec = importlib.util.spec_from_file_location(
                'flashvid_algorithm_test.method', SOURCE / 'flashvid/method.py'
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
    return module


method = load_method()


class TestFlashVidCompression(unittest.TestCase):
    def test_config_can_load_without_token_selection(self):
        self.assertTrue(callable(method.get_prune_config))
        self.assertNotIn('flashvid_algorithm_test.token_selection', sys.modules)

    def test_pruning_rate_is_validated_before_inference(self):
        for rate in (-0.1, 1.0, 2.0, float('nan'), float('inf'), '0.5'):
            with self.subTest(rate=rate), self.assertRaisesRegex(ValueError, 'video_pruning_rate'):
                method.get_prune_config(rate)

    def test_no_pruning_returns_disabled_config(self):
        for rate in (None, 0.0):
            config = method.get_prune_config(rate)
            self.assertFalse(config.enable_flashvid)
            self.assertEqual(config.retention_ratio, 1.0)

    def test_environment_and_dataclass_defaults_agree(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(method, 'ENABLE_FLASHVID', True):
            config = method.get_prune_config(0.5)
        defaults = method.FlashVidConfig()
        for name in ('do_segment', 'complementary_segment', 'enable_encode_prune'):
            self.assertEqual(getattr(config, name), getattr(defaults, name))
        self.assertIn('visual_token_length', {field.name for field in fields(config)})

    def test_boolean_parsing_accepts_shared_spellings(self):
        for raw, expected in ((' on ', True), ('OFF', False), ('yes', True), ('0', False)):
            with self.subTest(raw=raw), patch.dict(os.environ, {'FLAG': raw}):
                self.assertEqual(method._env_flag('FLAG', False), expected)

    def test_budget_is_exact(self):
        budget = method.compute_flashvid_token_budget(7, 3, 0.5, 0.7)
        self.assertEqual(budget.retained_per_frame, 4)
        self.assertEqual(budget.total_tokens, 12)
        self.assertEqual(budget.adts_per_frame + budget.sttm_per_frame, 4)
        self.assertEqual(sum(method.allocate_flashvid_sttm_tokens([1, 4, 3], 5)), 5)

    def test_dpc_clamps_neighbor_count_to_short_sequences(self):
        features = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        for neighbors in (7, [7, 7]):
            with self.subTest(neighbors=neighbors):
                labels, centers = method.dpc_knn(features, [1, 2], neighbors)
                self.assertEqual([len(row) for row in centers], [1, 2])
                self.assertTrue(all(row.numel() == 3 for row in labels))

    def test_dpc_uses_only_valid_candidates(self):
        features = torch.randn(2, 4, 3, generator=torch.Generator().manual_seed(1))
        valid = torch.tensor([[True, False, False, False], [True, True, True, False]])
        _, centers = method.dpc_knn(features, [1, 2], [7, 7], valid)
        for row, center in zip(valid, centers):
            self.assertTrue(row[center].all())

    def test_temporal_merge_propagates_accumulated_counts(self):
        features = torch.tensor([[[1.0, 0.0], [0.0, 1.0]], [[2.0, 0.0], [0.0, 1.0]], [[3.0, 0.0], [0.0, 1.0]]])
        original = features.clone()
        candidates = torch.tensor([[True, False]] * 3)
        tokens, indices = method.spatiotemporal_compression(features, 0.8, candidates, 0)
        self.assertTrue(torch.equal(tokens[0], torch.tensor([[2.0, 0.0]])))
        self.assertEqual([row.numel() for row in indices], [1, 0, 0])
        self.assertTrue(torch.equal(features, original))

    def test_zero_vectors_do_not_mutate_or_create_nan(self):
        features = torch.zeros(2, 3, 4)
        tokens, _ = method.spatiotemporal_compression(features, 0.8, torch.ones(2, 3, dtype=torch.bool), 1)
        self.assertTrue(all(torch.isfinite(row).all() for row in tokens))
        self.assertTrue(torch.equal(features, torch.zeros_like(features)))

    def test_compression_preserves_inputs_and_fixed_budget(self):
        features = torch.randn(3, 4, 2, generator=torch.Generator().manual_seed(3))
        original = features.clone()
        config = method.FlashVidConfig(enable_flashvid=True, retention_ratio=0.5, alpha=0.0, temporal_threshold=1.0)
        tokens, indices = method.flashvid_compression(features, torch.ones(3, 4), config)
        self.assertEqual(tokens.shape, (6, 2))
        self.assertEqual(indices.numel(), 6)
        self.assertEqual(indices.unique().numel(), 6)
        self.assertEqual(config.visual_token_length, 6)
        self.assertTrue(torch.equal(features, original))


if __name__ == '__main__':
    unittest.main()
