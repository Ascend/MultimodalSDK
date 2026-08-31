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
#
# SCC (Semantic Connected Components) vision token compression.
# Algorithm adapted from LLaVA-CR. Per-frame SCC, then per-frame token
# merge; video runs each frame independently and concatenates (no
# cross-frame feature fusion).
import math

import numpy as np
import torch

DEFAULT_TAU = 0.98
DEFAULT_EPSILON = 0.05
NORM_EPS = 1e-12
MIN_LOG_GRAPH_SIZE = 2
DETERMINISTIC_SEED_MASK = 0x7FFFFFFF
CPU_DEVICE_TYPE = "cpu"
DEFAULT_LABEL_PROP_MAX_ITERS = 64
LABEL_PROP_FLATTEN_ITERS = 8
CENTER_SELF_COUNT = 1


class _UnionFind:
    def __init__(self, size: int):
        self.parent = np.arange(size, dtype=np.int64)
        self.rank = np.zeros(size, dtype=np.int32)

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def batch_union(self, x_arr: np.ndarray, y_arr: np.ndarray) -> None:
        for x, y in zip(x_arr, y_arr):
            x_root = self.find(int(x))
            y_root = self.find(int(y))
            if x_root == y_root:
                continue
            if self.rank[x_root] < self.rank[y_root]:
                self.parent[x_root] = y_root
            else:
                self.parent[y_root] = x_root
                if self.rank[x_root] == self.rank[y_root]:
                    self.rank[x_root] += 1


def _approximate_components(adj_matrix: np.ndarray, epsilon: float = DEFAULT_EPSILON):
    """Approximate connected components on a dense boolean adjacency matrix.

    Samples O(log n / eps^2) anchors, unions them with their neighbours, and
    reports the resulting clusters. Unsampled nodes that have no edge to any
    sampled anchor become singleton clusters. Output is a list of int lists.
    """
    n = adj_matrix.shape[0]
    if n == 0:
        return []

    unassigned_nodes = np.ones(n, dtype=bool)

    sample_size = min(
        n,
        int(np.ceil(np.log(max(n, MIN_LOG_GRAPH_SIZE)) / epsilon**2)),
    )
    seed = int(adj_matrix.sum()) & DETERMINISTIC_SEED_MASK
    rng = np.random.default_rng(seed)
    sampled_nodes = rng.choice(n, size=sample_size, replace=False)
    unassigned_nodes[sampled_nodes] = False

    neighbor_dict: dict[int, np.ndarray] = {}
    for i in sampled_nodes:
        neighbors = np.nonzero(adj_matrix[i])[0]
        neighbor_dict[i] = neighbors
        unassigned_nodes[neighbors] = False

    remain_nodes = np.nonzero(unassigned_nodes)[0]
    remain_nodes = [[int(e)] for e in remain_nodes]

    uf = _UnionFind(n)
    all_x, all_y = [], []
    for i in sampled_nodes:
        for j in neighbor_dict[i]:
            all_x.append(int(i))
            all_y.append(int(j))
    if all_x:
        uf.batch_union(np.array(all_x), np.array(all_y))

    sampled_roots = np.array([uf.find(int(i)) for i in sampled_nodes])
    unique_roots = np.unique(sampled_roots)

    roots_all = np.array([uf.find(int(i)) for i in range(n)])
    components = []
    for root in unique_roots:
        cluster = np.where(roots_all == root)[0].tolist()
        if len(cluster) > 0:
            components.append([int(x) for x in cluster])
    components.extend(remain_nodes)

    degrees = np.count_nonzero(adj_matrix, axis=1)

    def get_sort_key(cluster):
        max_degree = -1
        min_node = float("inf")
        for node in cluster:
            current_degree = degrees[node]
            if (current_degree > max_degree) or (current_degree == max_degree and node < min_node):
                max_degree = current_degree
                min_node = node
        return min_node

    components.sort(key=get_sort_key)
    return components


def _l2_normalize(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    norm = torch.norm(x, p=2, dim=dim, keepdim=True).clamp_min(NORM_EPS)
    return x / norm


def _cosine_adjacency(x: torch.Tensor, tau: float) -> torch.Tensor:
    """Return adjacency where entry (i, j) means cos(x_i, x_j) > tau."""
    x_n = _l2_normalize(x)
    if x.dim() == 2:
        sim = torch.matmul(x_n, x_n.transpose(0, 1))
    elif x.dim() == 3:
        sim = torch.matmul(x_n, x_n.transpose(1, 2))
    else:
        raise ValueError(f"Unsupported feature dim: {x.dim()}")
    return sim > tau


def _merge_by_components(features: torch.Tensor, components) -> torch.Tensor:
    """Mean-pool tokens in each component. features: [N, D] -> [K, D]."""
    out = []
    for comp in components:
        selected = features[comp]
        out.append(selected.mean(dim=0))
    return torch.stack(out, dim=0)


def _token_merge_back(centers: torch.Tensor, originals: torch.Tensor) -> torch.Tensor:
    """Fold each original token into its nearest center; the center's own vector is counted once (CENTER_SELF_COUNT) so empty components stay at their original value."""
    c_n = _l2_normalize(centers, dim=1)
    o_n = _l2_normalize(originals, dim=1)
    sim = torch.matmul(o_n, c_n.t())  # [N, K]
    closest = torch.argmax(sim, dim=1)  # [N]

    K = centers.shape[0]
    merged = torch.zeros_like(centers)
    merged.scatter_add_(
        dim=0,
        index=closest.view(-1, 1).expand(-1, centers.shape[1]),
        src=originals,
    )
    counts = torch.bincount(closest, minlength=K).to(centers.dtype)
    counts = counts + CENTER_SELF_COUNT
    out = centers + merged
    out = out / counts.unsqueeze(1)

    return out


# NPU path uses exact label propagation; `epsilon` only affects the CPU fallback sampler.
def _label_prop_components(
    adj: torch.Tensor,
    max_iters: int = DEFAULT_LABEL_PROP_MAX_ITERS,
):
    """Connected-component labels via label propagation. adj: [N, N] bool (symmetric; self-loops added internally). Returns (labels[N] in [0, K), K)."""
    N = adj.shape[0]
    device = adj.device
    if N == 0:
        return torch.zeros(0, dtype=torch.long, device=device), 0

    adj = adj | adj.transpose(0, 1)
    eye = torch.eye(N, dtype=torch.bool, device=device)
    adj = adj | eye

    sentinel = N
    labels = torch.arange(N, dtype=torch.long, device=device)

    for _ in range(max_iters):
        neighbor_labels = torch.where(
            adj,
            labels.unsqueeze(0).expand(N, N),
            torch.full((1,), sentinel, dtype=torch.long, device=device),
        )
        new_labels = neighbor_labels.min(dim=1).values
        new_labels = torch.minimum(new_labels, labels[new_labels.clamp_max(N - 1)])
        if torch.equal(new_labels, labels):
            labels = new_labels
            break
        labels = new_labels

    for _ in range(LABEL_PROP_FLATTEN_ITERS):
        jumped = labels[labels]
        if torch.equal(jumped, labels):
            break
        labels = jumped

    unique_roots, inverse = torch.unique(labels, sorted=True, return_inverse=True)
    K = int(unique_roots.numel())
    return inverse, K


def _component_means(features: torch.Tensor, labels: torch.Tensor, K: int) -> torch.Tensor:
    """Mean-pool `features` ([N, D]) by `labels` ([N] in [0, K)) -> [K, D]."""
    D = features.shape[1]
    sums = torch.zeros((K, D), dtype=features.dtype, device=features.device)
    sums.scatter_add_(0, labels.unsqueeze(-1).expand(-1, D), features)
    counts = torch.bincount(labels, minlength=K)
    if counts.numel() != K or torch.any(counts == 0):
        raise ValueError(
            "_component_means received non-contiguous or empty component "
            f"labels: K={K}, counts={counts.detach().cpu().tolist()}"
        )
    counts = counts.to(features.dtype)
    return sums / counts.unsqueeze(-1)


def _component_centers(
    features: torch.Tensor,
    adj: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """Return SCC component centers using the backend implied by `adj.device`."""
    if adj.device.type == CPU_DEVICE_TYPE:
        components = _approximate_components(adj.cpu().numpy(), epsilon=epsilon)
        return _merge_by_components(features, components)

    labels, num_components = _label_prop_components(adj)
    return _component_means(features, labels, num_components)


def _scc_frame(
    features: torch.Tensor,
    tau: float,
    epsilon: float,
) -> torch.Tensor:
    adj = _cosine_adjacency(features, tau)
    centers = _component_centers(features, adj, epsilon)
    return _token_merge_back(centers, features)


def _select_topk_by_degree(degrees: torch.Tensor, k: int) -> torch.Tensor:
    indices = torch.arange(
        degrees.numel(),
        device=degrees.device,
        dtype=torch.int64,
    )
    scores = degrees.to(torch.int64) * (degrees.numel() + 1) - indices
    return torch.topk(scores, k=k, largest=True).indices.sort().values


def _align_image_tokens(compressed: torch.Tensor, k_target: int, tau: float):
    if compressed.shape[0] > k_target:
        c_n = _l2_normalize(compressed.float(), dim=1)
        sim_full = torch.matmul(c_n, c_n.t())
        degrees = (sim_full > tau).sum(dim=1)
        keep_idx = _select_topk_by_degree(degrees, k_target)
        survivors = compressed[keep_idx]
        compressed = _token_merge_back(survivors, compressed)

    elif compressed.shape[0] == k_target:
        pass

    else:
        pad_count = k_target - compressed.shape[0]
        pad = compressed.mean(dim=0, keepdim=True).expand((pad_count, -1)).clone()
        compressed = torch.cat([compressed, pad], dim=0)

    return compressed


def _split_sizes(n, k):
    """Split `n` into `k` parts (first `n % k` are one larger). Sum equals `n`. E.g. split_sizes(7, 3) == [3, 2, 2]."""
    base = n // k
    rem = n % k
    return [base + 1] * rem + [base] * (k - rem)


def scc_shrink(tokens_per_frame: int, ratio: float) -> int:
    """Return the fixed placeholder/feature target length used by SCC."""
    if not (0.0 < ratio <= 1.0):
        raise ValueError(f"ratio must be in (0, 1], got {ratio}")
    return max(1, math.ceil(tokens_per_frame * ratio))


def scc_should_run(n: int, max_tokens_per_item: int = 8192) -> bool:
    """Whether to run SCC on a single image/video item.

    SCC's per-item compute grows with the input token count, and above a
    certain size the SCC overhead exceeds the downstream savings from
    having fewer LLM tokens -- for very large items the LLM is better off
    seeing the raw tokens. The default threshold (8192 tokens) is a
    heuristic empirical cutoff: below it compression usually pays for
    itself, above it the time spent compressing outweighs the benefit.

    Returns True when `n <= max_tokens_per_item` (item is small enough to
    be worth compressing); False otherwise. `max_tokens_per_item <= 0`
    disables the threshold and always returns True.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    return max_tokens_per_item <= 0 or n <= max_tokens_per_item


def set_uniform_true(n, k):
    """Length-`n` bool tensor with `k` uniformly spaced `True`s (via linspace+round). When multiple samples round to the same index, that slot is marked once."""
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if k <= 0 or k > n:
        raise ValueError(f"k must be in (0, n], got k={k}, n={n}")
    arr = torch.zeros(n, dtype=torch.bool)
    indices = torch.linspace(0, n - 1, k).round().long()
    arr[indices] = True
    return arr


def scc_compress_to_target(
    features: torch.Tensor,
    k_target: int,
    num_frames: int = 1,
    tokens_per_frame: int = 0,
    tau: float = DEFAULT_TAU,
    epsilon: float = DEFAULT_EPSILON,
) -> torch.Tensor:
    """Compress `features` to exactly `k_target` tokens. Video branch runs per-frame SCC independently and concatenates; the result is then adjusted to `k_target` (trim least-connected centers / pad with mean). Does NOT return a `sparse_mask` -- callers that need one should use `set_uniform_true(features.shape[0], k_target)`."""
    if features.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError(f"features must be one of bfloat16 / float16 / float32, got {features.dtype}")
    if num_frames < 1:
        raise ValueError(f"num_frames must be >= 1, got {num_frames}")
    if k_target < 1:
        raise ValueError(f"k_target must be >= 1, got {k_target}")
    if k_target > features.shape[0]:
        raise ValueError(f"k_target ({k_target}) must be <= features.shape[0] ({features.shape[0]})")
    if k_target < num_frames:
        raise ValueError(
            f"k_target ({k_target}) must be >= num_frames ({num_frames}); each frame needs at least one output token"
        )

    # Promote to float32 for SCC math, restore the caller's dtype on return.
    orig_dtype = features.dtype
    features = features.float()

    if num_frames > 1 and tokens_per_frame > 0:
        if tokens_per_frame * num_frames != features.shape[0]:
            raise ValueError(
                f"scc_compress_to_target: N={features.shape[0]} must equal "
                f"num_frames*tokens_per_frame={num_frames}*{tokens_per_frame}="
                f"{num_frames * tokens_per_frame}"
            )

        targets = _split_sizes(k_target, num_frames)

        all_compressed = []
        for i in range(num_frames):
            frame_feature = features[i * tokens_per_frame : (i + 1) * tokens_per_frame]
            compressed = _scc_frame(frame_feature, tau=tau, epsilon=epsilon)
            compressed = _align_image_tokens(compressed, k_target=targets[i], tau=tau)
            all_compressed.append(compressed)

        return torch.cat(all_compressed, dim=0).to(orig_dtype)

    compressed = _scc_frame(features, tau=tau, epsilon=epsilon)
    return _align_image_tokens(compressed, k_target=k_target, tau=tau).to(orig_dtype)
