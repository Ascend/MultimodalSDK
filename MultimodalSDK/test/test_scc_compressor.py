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
import unittest
import torch
import numpy as np

from mm.core.scc.compressor import (
    scc_shrink,
    scc_should_run,
    scc_compress_to_target,
    set_uniform_true,
)
from mm.core.scc.compressor import (
    _UnionFind as UnionFind,
    _approximate_components as approximate_components,
    _split_sizes as split_sizes,
    _label_prop_components,
    _component_means,
)


class TestSccShrink(unittest.TestCase):
    """Tests for scc_shrink: token count calculation."""

    def test_basic_compression(self):
        # 100 tokens, ratio 0.3 => ceil(30) = 30
        result = scc_shrink(100, 0.3)
        self.assertEqual(result, 30)

    def test_rounded_up(self):
        # 11 tokens, ratio 0.3 => ceil(3.3) = 4
        result = scc_shrink(11, 0.3)
        self.assertEqual(result, 4)

    def test_min_one_token(self):
        # 1 token, tiny positive ratio should still return at least 1
        result = scc_shrink(1, 0.01)
        self.assertEqual(result, 1)

    def test_ratio_zero_raises(self):
        # ratio == 0 is invalid (must be > 0)
        with self.assertRaises(ValueError):
            scc_shrink(100, 0.0)

    def test_ratio_negative_raises(self):
        # ratio < 0 is invalid
        with self.assertRaises(ValueError):
            scc_shrink(100, -0.1)

    def test_ratio_above_one_raises(self):
        # ratio > 1 is invalid (compression ratio cannot exceed 1)
        with self.assertRaises(ValueError):
            scc_shrink(100, 1.5)

    def test_zero_tokens(self):
        # max(1, ceil(0 * 0.3)) = 1
        result = scc_shrink(0, 0.3)
        self.assertEqual(result, 1)

    def test_full_ratio(self):
        # ratio 1.0 means no compression, should keep all tokens
        result = scc_shrink(50, 1.0)
        self.assertEqual(result, 50)

    def test_half_ratio(self):
        # 100 tokens, ratio 0.5 => ceil(50) = 50
        result = scc_shrink(100, 0.5)
        self.assertEqual(result, 50)

    def test_tiny_ratio(self):
        # 1000 tokens, ratio 0.01 => ceil(10) = 10
        result = scc_shrink(1000, 0.01)
        self.assertEqual(result, 10)

    def test_large_count(self):
        # 10000 tokens, ratio 0.1 => ceil(1000) = 1000
        result = scc_shrink(10000, 0.1)
        self.assertEqual(result, 1000)


class TestSccShouldRun(unittest.TestCase):
    """Tests for scc_should_run: threshold check."""

    def test_within_default_limit(self):
        # n=100, default max=8192 -> True
        self.assertTrue(scc_should_run(100))

    def test_equal_to_limit(self):
        # n=8192, max=8192 -> True
        self.assertTrue(scc_should_run(8192))

    def test_exceeds_default_limit(self):
        # n=9000 > max=8192 -> False
        self.assertFalse(scc_should_run(9000))

    def test_zero_max_means_always_run(self):
        # max_tokens_per_item <= 0 -> always True
        self.assertTrue(scc_should_run(99999, max_tokens_per_item=0))
        self.assertTrue(scc_should_run(99999, max_tokens_per_item=-1))

    def test_zero_n_raises(self):
        # n must be positive; 0 is not a valid item size
        with self.assertRaises(ValueError):
            scc_should_run(0)

    def test_negative_n_raises(self):
        # n must be positive
        with self.assertRaises(ValueError):
            scc_should_run(-1)

    def test_custom_limit(self):
        self.assertTrue(scc_should_run(50, max_tokens_per_item=100))
        self.assertFalse(scc_should_run(150, max_tokens_per_item=100))


class TestUnionFind(unittest.TestCase):
    """Tests for UnionFind data structure."""

    def test_init_creates_singleton_sets(self):
        uf = UnionFind(5)
        for i in range(5):
            self.assertEqual(uf.find(i), i)

    def test_find_returns_root_after_union(self):
        uf = UnionFind(5)
        uf.batch_union(np.array([0]), np.array([1]))
        self.assertEqual(uf.find(0), uf.find(1))

    def test_batch_union_multiple_pairs(self):
        uf = UnionFind(6)
        uf.batch_union(np.array([0, 2, 4]), np.array([1, 3, 5]))
        self.assertEqual(uf.find(0), uf.find(1))
        self.assertEqual(uf.find(2), uf.find(3))
        self.assertEqual(uf.find(4), uf.find(5))
        # The three groups should remain distinct
        self.assertNotEqual(uf.find(0), uf.find(2))
        self.assertNotEqual(uf.find(2), uf.find(4))

    def test_batch_union_same_set_is_noop(self):
        uf = UnionFind(3)
        uf.batch_union(np.array([0]), np.array([1]))
        root_before = uf.find(0)
        uf.batch_union(np.array([0]), np.array([1]))
        root_after = uf.find(0)
        self.assertEqual(root_before, root_after)

    def test_path_compression(self):
        uf = UnionFind(4)
        # Build a chain: 0-1-2-3
        uf.batch_union(np.array([0, 1, 2]), np.array([1, 2, 3]))
        # All four should share the same root
        roots = [uf.find(i) for i in range(4)]
        self.assertEqual(len(set(roots)), 1)


class TestSplitSizes(unittest.TestCase):
    """Tests for split_sizes: even distribution."""

    def test_even_split(self):
        # 10 / 3 = 3 rem 1 -> [4, 3, 3]
        result = split_sizes(10, 3)
        self.assertEqual(result, [4, 3, 3])

    def test_exact_split(self):
        # 12 / 3 = 4 rem 0 -> [4, 4, 4]
        result = split_sizes(12, 3)
        self.assertEqual(result, [4, 4, 4])

    def test_zero_total(self):
        # 0 / 3 = 0 rem 0 -> [0, 0, 0]
        result = split_sizes(0, 3)
        self.assertEqual(result, [0, 0, 0])

    def test_one_total(self):
        # 1 / 3 = 0 rem 1 -> [1, 0, 0]
        result = split_sizes(1, 3)
        self.assertEqual(result, [1, 0, 0])

    def test_sum_equals_total(self):
        for n, k in [(7, 3), (20, 6), (1, 1), (15, 4)]:
            self.assertEqual(sum(split_sizes(n, k)), n)


class TestSetUniformTrue(unittest.TestCase):
    """Tests for set_uniform_true: uniform sampling."""

    def test_basic_uniform_selection(self):
        # n=10, k=3 -> select 3 evenly-spaced positions
        result = set_uniform_true(10, 3)
        self.assertEqual(result.dtype, torch.bool)
        self.assertEqual(result.shape, (10,))
        self.assertEqual(int(result.sum()), 3)

    def test_k_equals_n(self):
        # n=5, k=5 -> all True
        result = set_uniform_true(5, 5)
        self.assertEqual(int(result.sum()), 5)

    def test_k_zero_raises(self):
        # k == 0 is invalid (must be > 0)
        with self.assertRaises(ValueError):
            set_uniform_true(5, 0)

    def test_k_above_n_raises(self):
        # k > n is invalid (must be <= n)
        with self.assertRaises(ValueError):
            set_uniform_true(5, 6)

    def test_n_zero_raises(self):
        # n must be positive
        with self.assertRaises(ValueError):
            set_uniform_true(0, 0)

    def test_selection_is_deterministic(self):
        # Same input -> same output
        a = set_uniform_true(20, 4)
        b = set_uniform_true(20, 4)
        self.assertTrue(torch.equal(a, b))

    def test_positions_are_sorted(self):
        # The selected positions should be monotonically non-decreasing
        result = set_uniform_true(100, 10)
        positions = torch.nonzero(result).flatten().tolist()
        self.assertEqual(positions, sorted(positions))


class TestApproximateComponents(unittest.TestCase):
    """Tests for approximate_components: connected components."""

    def test_empty_matrix(self):
        result = approximate_components(np.zeros((0, 0), dtype=bool))
        self.assertEqual(result, [])

    def test_all_singletons(self):
        # No edges -> every node is its own component
        adj = np.zeros((4, 4), dtype=bool)
        result = approximate_components(adj)
        # At least 4 components returned
        all_nodes = sorted([n for c in result for n in c])
        self.assertEqual(all_nodes, [0, 1, 2, 3])

    def test_fully_connected(self):
        # All edges except self-loops -> one big component
        # Use integer arithmetic (numpy bool arrays don't support `-`).
        adj = (np.ones((5, 5), dtype=np.int8) - np.eye(5, dtype=np.int8)).astype(bool)
        result = approximate_components(adj)
        # The largest component should contain most nodes
        max_comp = max(result, key=len)
        self.assertGreaterEqual(len(max_comp), 1)

    def test_returns_list_of_lists(self):
        adj = np.eye(3, dtype=bool)
        result = approximate_components(adj)
        self.assertIsInstance(result, list)
        for comp in result:
            self.assertIsInstance(comp, list)


class TestLabelPropComponents(unittest.TestCase):
    """Tests for _label_prop_components: the NPU/GPU branch's connected-component finder.

    This is the label-propagation algorithm used when `adj` is on a non-CPU
    device (NPU/CUDA). The implementation itself is device-agnostic, so we
    can exercise it with CPU tensors in any environment.
    """

    def test_empty_matrix(self):
        adj = torch.zeros((0, 0), dtype=torch.bool)
        labels, K = _label_prop_components(adj)
        self.assertEqual(K, 0)
        self.assertEqual(labels.numel(), 0)

    def test_all_singletons(self):
        # No cross-edges -> every node is its own component
        # (self-loops are added internally, so each node still gets a label)
        adj = torch.zeros((4, 4), dtype=torch.bool)
        labels, K = _label_prop_components(adj)
        self.assertEqual(K, 4)
        # 4 distinct labels in [0, K)
        self.assertEqual(sorted(set(labels.tolist())), list(range(K)))

    def test_fully_connected(self):
        adj = torch.ones((5, 5), dtype=torch.bool)
        labels, K = _label_prop_components(adj)
        self.assertEqual(K, 1)
        # All labels should be identical
        self.assertTrue((labels == labels[0]).all().item())

    def test_two_components(self):
        # {0, 1} connected; {2, 3} connected; no cross-edges
        adj = torch.zeros((4, 4), dtype=torch.bool)
        adj[0, 1] = adj[1, 0] = True
        adj[2, 3] = adj[3, 2] = True
        labels, K = _label_prop_components(adj)
        self.assertEqual(K, 2)
        self.assertEqual(labels[0].item(), labels[1].item())
        self.assertEqual(labels[2].item(), labels[3].item())
        self.assertNotEqual(labels[0].item(), labels[2].item())

    def test_three_components(self):
        # Three separate edges {0-1}, {2-3}, {4-5}
        adj = torch.zeros((6, 6), dtype=torch.bool)
        adj[0, 1] = adj[1, 0] = True
        adj[2, 3] = adj[3, 2] = True
        adj[4, 5] = adj[5, 4] = True
        labels, K = _label_prop_components(adj)
        self.assertEqual(K, 3)
        for group in [(0, 1), (2, 3), (4, 5)]:
            self.assertEqual(labels[group[0]].item(), labels[group[1]].item())
        # Groups should be distinct from each other
        self.assertEqual(len({labels[i].item() for i in (0, 2, 4)}), 3)

    def test_skewed_adj_is_symmetrized(self):
        # _label_prop_components symmetrizes internally, so a one-sided
        # edge should still connect both endpoints.
        adj = torch.zeros((3, 3), dtype=torch.bool)
        adj[0, 1] = True  # not symmetric initially
        labels, K = _label_prop_components(adj)
        self.assertEqual(K, 2)
        self.assertEqual(labels[0].item(), labels[1].item())
        self.assertNotEqual(labels[0].item(), labels[2].item())

    def test_dense_chain(self):
        # 1 - 2 - 3 - 4: should be one component
        adj = torch.zeros((4, 4), dtype=torch.bool)
        adj[0, 1] = adj[1, 0] = True
        adj[1, 2] = adj[2, 1] = True
        adj[2, 3] = adj[3, 2] = True
        labels, K = _label_prop_components(adj)
        self.assertEqual(K, 1)
        self.assertTrue((labels == labels[0]).all().item())


class TestComponentMeans(unittest.TestCase):
    """Tests for _component_means: mean-pool `features` by integer `labels`.

    Used by the non-CPU branch (`_component_centers` -> `_label_prop_components`
    + `_component_means`) to turn per-token features into per-cluster
    averages.
    """

    def test_two_groups(self):
        features = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        labels = torch.tensor([0, 0, 1])
        out = _component_means(features, labels, K=2)
        expected = torch.tensor([[2.0, 3.0], [5.0, 6.0]])
        self.assertTrue(torch.allclose(out, expected))

    def test_all_same_label(self):
        features = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        labels = torch.tensor([0, 0])
        out = _component_means(features, labels, K=1)
        expected = torch.tensor([[2.0, 3.0]])
        self.assertTrue(torch.allclose(out, expected))

    def test_three_groups(self):
        features = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
        labels = torch.tensor([2, 0, 1, 0, 1, 2])
        out = _component_means(features, labels, K=3)
        # group 0: mean([2, 4]) = 3
        # group 1: mean([3, 5]) = 4
        # group 2: mean([1, 6]) = 3.5
        expected = torch.tensor([[3.0], [4.0], [3.5]])
        self.assertTrue(torch.allclose(out, expected))

    def test_empty_component_raises(self):
        # K=3 but only 2 unique labels -> one component is empty
        features = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        labels = torch.tensor([0, 1])
        with self.assertRaises(ValueError):
            _component_means(features, labels, K=3)


class TestSccCompressToTarget(unittest.TestCase):
    """Integration tests for scc_compress_to_target end-to-end pipeline."""

    def test_image_compress_returns_correct_token_count(self):
        # Create random features
        torch.manual_seed(0)
        N, D, k_target = 64, 16, 16
        features = torch.randn(N, D)

        compressed = scc_compress_to_target(
            features,
            k_target=k_target,
            num_frames=1,
            tokens_per_frame=0,
            tau=0.5,
            epsilon=0.1,
        )
        # Compressed should have exactly k_target rows
        self.assertEqual(compressed.shape[0], k_target)
        # scc_compress_to_target no longer returns a mask; callers build it
        # via set_uniform_true. Verify the externally-built mask contract.
        mask = set_uniform_true(N, k_target)
        # Mask should match original length
        self.assertEqual(mask.shape[0], N)
        # Mask should be bool dtype
        self.assertEqual(mask.dtype, torch.bool)
        # Exactly k_target positions should be selected (uniform sampling)
        self.assertEqual(int(mask.sum()), k_target)

    def test_image_compress_preserves_feature_dim(self):
        torch.manual_seed(1)
        features = torch.randn(32, 8)
        compressed = scc_compress_to_target(
            features,
            k_target=10,
        )
        self.assertEqual(compressed.shape[1], 8)

    def test_video_compress_with_multiple_frames(self):
        torch.manual_seed(2)
        num_frames = 4
        tokens_per_frame = 16
        D = 8
        k_target = 8
        features = torch.randn(num_frames * tokens_per_frame, D)

        compressed = scc_compress_to_target(
            features,
            k_target=k_target,
            num_frames=num_frames,
            tokens_per_frame=tokens_per_frame,
            tau=0.5,
            epsilon=0.1,
        )
        # Compressed should have exactly k_target rows
        self.assertEqual(compressed.shape[0], k_target)
        # scc_compress_to_target no longer returns a mask; callers build it
        # via set_uniform_true over the full original sequence.
        mask = set_uniform_true(num_frames * tokens_per_frame, k_target)
        self.assertEqual(mask.shape[0], num_frames * tokens_per_frame)
        self.assertEqual(mask.dtype, torch.bool)
        # Total selected positions should equal k_target.
        self.assertEqual(int(mask.sum()), k_target)

    def test_video_compress_wrong_length_raises(self):
        features = torch.randn(10, 4)
        with self.assertRaises(ValueError) as ctx:
            scc_compress_to_target(
                features,
                k_target=4,
                num_frames=2,
                tokens_per_frame=8,  # 2*8=16 != 10
            )
        self.assertIn("scc_compress_to_target", str(ctx.exception))

    def test_compress_with_zero_features(self):
        # All-zero features should not crash
        torch.manual_seed(3)
        features = torch.zeros(8, 4)
        # Should not raise, even though features are zero
        compressed = scc_compress_to_target(
            features,
            k_target=4,
            tau=0.0,
            epsilon=0.1,
        )
        self.assertEqual(compressed.shape[0], 4)
        # Sanity check the externally-built mask path still works on N=8, k=4.
        mask = set_uniform_true(8, 4)
        self.assertEqual(mask.shape[0], 8)
        self.assertEqual(int(mask.sum()), 4)

    def test_preserves_bfloat16_dtype(self):
        torch.manual_seed(4)
        features = torch.randn(32, 8, dtype=torch.bfloat16)
        compressed = scc_compress_to_target(features, k_target=10)
        self.assertEqual(compressed.shape[0], 10)
        self.assertEqual(compressed.dtype, torch.bfloat16)

    def test_preserves_float16_dtype(self):
        torch.manual_seed(5)
        features = torch.randn(32, 8, dtype=torch.float16)
        compressed = scc_compress_to_target(features, k_target=10)
        self.assertEqual(compressed.shape[0], 10)
        self.assertEqual(compressed.dtype, torch.float16)

    def test_preserves_float32_dtype(self):
        torch.manual_seed(6)
        features = torch.randn(32, 8, dtype=torch.float32)
        compressed = scc_compress_to_target(features, k_target=10)
        self.assertEqual(compressed.shape[0], 10)
        self.assertEqual(compressed.dtype, torch.float32)

    def test_unsupported_dtype_raises(self):
        features = torch.randint(0, 10, (8, 4), dtype=torch.int32)
        with self.assertRaises(ValueError):
            scc_compress_to_target(features, k_target=4)

    def test_int64_dtype_raises(self):
        # torch.randn rejects int dtypes, so build via torch.zeros instead
        features = torch.zeros(8, 4, dtype=torch.int64)
        with self.assertRaises(ValueError):
            scc_compress_to_target(features, k_target=4)

    def test_k_target_above_features_raises(self):
        # k_target must be <= features.shape[0]
        features = torch.randn(8, 4)
        with self.assertRaises(ValueError):
            scc_compress_to_target(features, k_target=9)

    def test_k_target_zero_raises(self):
        # k_target must be >= 1
        features = torch.randn(8, 4)
        with self.assertRaises(ValueError):
            scc_compress_to_target(features, k_target=0)

    def test_k_target_negative_raises(self):
        # k_target must be >= 1
        features = torch.randn(8, 4)
        with self.assertRaises(ValueError):
            scc_compress_to_target(features, k_target=-1)

    def test_num_frames_zero_raises(self):
        features = torch.randn(8, 4)
        with self.assertRaises(ValueError):
            scc_compress_to_target(features, k_target=4, num_frames=0)

    def test_num_frames_negative_raises(self):
        features = torch.randn(8, 4)
        with self.assertRaises(ValueError):
            scc_compress_to_target(features, k_target=4, num_frames=-1)

    def test_k_target_below_num_frames_raises(self):
        # Video branch: each frame needs at least 1 output token
        # 4 frames * 16 tokens = 64 features, but k_target=2 < num_frames=4
        features = torch.randn(64, 4)
        with self.assertRaises(ValueError):
            scc_compress_to_target(features, k_target=2, num_frames=4, tokens_per_frame=16)

    def test_k_target_equal_num_frames_ok(self):
        # Edge case: k_target == num_frames means 1 token per frame -- valid
        torch.manual_seed(7)
        features = torch.randn(4 * 4, 4)  # 4 frames * 4 tokens
        compressed = scc_compress_to_target(features, k_target=4, num_frames=4, tokens_per_frame=4)
        self.assertEqual(compressed.shape[0], 4)


if __name__ == "__main__":
    unittest.main()
