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
"""CPU regression tests for FlashVid token selection and restoration."""

import importlib.util
from pathlib import Path
import unittest

import torch


SOURCE = Path(__file__).resolve().parents[1] / 'source/mm/patcher/vllm/flashvid/token_selection.py'
SPEC = importlib.util.spec_from_file_location('flashvid_token_selection', SOURCE)
selection = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(selection)


class TestFlashVidTokenSelection(unittest.TestCase):
    def test_v2_is_independent_of_other_batch_items(self):
        generator = torch.Generator().manual_seed(7)
        features = torch.randn(3, 12, 4, generator=generator)
        attention = torch.rand(3, 12, generator=generator)
        _, batched = selection.attn_div_v2_based_token_selection(features, attention, 5)
        for batch in range(3):
            _, individual = selection.attn_div_v2_based_token_selection(
                features[batch : batch + 1], attention[batch : batch + 1], 5
            )
            self.assertTrue(torch.equal(batched[batch], individual[0]))

    def test_v2_does_not_reselect_identical_tokens(self):
        features = torch.ones(2, 8, 4)
        attention = torch.ones(2, 8)
        selected, indices = selection.attn_div_v2_based_token_selection(features, attention, 8)
        self.assertEqual(selected.shape, (2, 8, 4))
        for row in indices:
            self.assertTrue(torch.equal(row, torch.arange(8)))

    def test_zero_vectors_have_finite_cosine_distances(self):
        features = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]], dtype=torch.float16)
        distances = selection.pairwise_cosine_distances(features)
        self.assertTrue(torch.isfinite(distances).all())
        selected, indices = selection.attn_div_v3_based_token_selection(features, torch.ones(1, 3), 2)
        self.assertTrue(torch.isfinite(selected).all())
        self.assertEqual(indices.unique().numel(), 2)

    def test_pruning_accepts_both_single_sequence_layouts(self):
        features = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        for value in (features, features.unsqueeze(0), features.unsqueeze(1)):
            with self.subTest(shape=value.shape):
                keep, remove = selection.prune_embeds(value)
                self.assertTrue(torch.equal(keep, torch.tensor([0, 2])))
                self.assertTrue(torch.equal(remove, torch.tensor([1, 3])))

    def test_pruning_rejects_multiple_sequences(self):
        with self.assertRaises(ValueError):
            selection.prune_embeds(torch.ones(2, 3, 4))

    def test_empty_and_singleton_indices_preserve_device(self):
        for length in (0, 1):
            for device in ('cpu', 'meta'):
                with self.subTest(length=length, device=device):
                    keep, remove = selection.prune_embeds(torch.empty(length, 4, device=device))
                    self.assertEqual(keep.device.type, device)
                    self.assertEqual(remove.device.type, device)
                    self.assertEqual(keep.numel(), length)
                    self.assertEqual(remove.numel(), 0)

    def test_restore_preserves_input_rank_dtype_and_values(self):
        features = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float16)
        attention = torch.tensor([0.25, 0.75])
        keep, remove = torch.tensor([0, 2]), torch.tensor([1, 3])
        for value in (features, features.unsqueeze(1)):
            with self.subTest(shape=value.shape):
                restored, restored_attention = selection.restore_hidden_states(value, 4, keep, remove, attention)
                self.assertEqual(restored.shape, (4, *value.shape[1:]))
                self.assertEqual(restored.dtype, value.dtype)
                self.assertEqual(restored.device, value.device)
                self.assertTrue(torch.equal(restored[keep], value))
                self.assertTrue(torch.equal(restored[remove], value))
                self.assertTrue(torch.equal(restored_attention, torch.tensor([0.25, 0.25, 0.75, 0.75])))

    def test_restore_consecutive_and_leading_removed_tokens(self):
        features = torch.tensor([[10.0], [20.0]])
        restored, attention = selection.restore_hidden_states(
            features, 6, torch.tensor([1, 4]), torch.tensor([0, 2, 3, 5]), torch.tensor([0.1, 0.2])
        )
        self.assertTrue(torch.equal(restored[:, 0], torch.tensor([10.0, 10.0, 10.0, 10.0, 20.0, 20.0])))
        self.assertTrue(torch.equal(attention, torch.tensor([0.1, 0.1, 0.1, 0.1, 0.2, 0.2])))

    def test_restore_handles_unsorted_keep_indices(self):
        restored, attention = selection.restore_hidden_states(
            torch.tensor([[20.0], [10.0]]), 4, torch.tensor([2, 0]), torch.tensor([1, 3])
        )
        self.assertTrue(torch.equal(restored[:, 0], torch.tensor([10.0, 10.0, 20.0, 20.0])))
        self.assertIsNone(attention)

    def test_prune_restore_round_trip_keeps_retained_values(self):
        features = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        keep, remove = selection.prune_embeds(features)
        restored, _ = selection.restore_hidden_states(features[keep], len(features), keep, remove)
        self.assertTrue(torch.equal(restored, features))

    def test_restore_rejects_invalid_index_partitions(self):
        for keep, remove in (([0, 0], [1, 2]), ([0, 2], [1]), ([0, 2], [1, 4])):
            with self.subTest(keep=keep, remove=remove), self.assertRaises(ValueError):
                selection.restore_hidden_states(torch.ones(2, 3), 4, torch.tensor(keep), torch.tensor(remove))

    def test_restore_without_pruning_and_for_empty_sequences(self):
        for length in (0, 1, 4):
            with self.subTest(length=length):
                features = torch.arange(length * 2, dtype=torch.float32).reshape(length, 2)
                restored, attention = selection.restore_hidden_states(
                    features, length, torch.arange(length), torch.empty(0, dtype=torch.long)
                )
                self.assertTrue(torch.equal(restored, features))
                self.assertIsNone(attention)


if __name__ == '__main__':
    unittest.main()
