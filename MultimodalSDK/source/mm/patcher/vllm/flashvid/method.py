import math
import os
from dataclasses import dataclass, field
from importlib import import_module
from typing import Tuple, Optional, Union, List

import torch
import torch.nn.functional as F
from torch import Tensor

from vllm.logger import init_logger
from mm.patcher.vllm.constants import ENABLE_FLASHVID, _parse_bool

logger = init_logger("vllm.flashvid")


@dataclass
class FlashVidConfig:
    # enable flash vid compress
    enable_flashvid: bool = field(default=False)
    # Average retention ratio.
    retention_ratio: float = field(default=0.5)

    # 1) Token Selection Method. Defaults to ADTS.
    alpha: float = field(default=0.7)  # Ratio of ADTS tokens.
    token_selection_method: str = field(default="attn_div")

    # 2) Tree-based Spatio-Temporal Token Merging.
    temporal_threshold: float = field(default=0.8)

    # Dynamic Video Segmentation (DySeg).
    do_segment: bool = field(default=False)
    segment_threshold: float = field(default=0.9)
    min_segment_num: int = field(default=4)
    complementary_segment: bool = field(default=False)

    # compress encode before vit
    enable_encode_prune: bool = field(default=False)
    encode_prune_threshold: float = field(default=0.7)
    visual_token_length: int = field(default=0, init=False)


@dataclass(frozen=True)
class FlashVidTokenBudget:
    """Fixed token budget shared by the processor and compressor."""

    retained_per_frame: int
    total_tokens: int
    adts_per_frame: int
    sttm_per_frame: int


ALL_TOKEN_SELECTION_METHOD = {
    "attn": "attn_based_token_selection",
    "attn_div_v2": "attn_div_v2_based_token_selection",
    "attn_div": "attn_div_v3_based_token_selection",
    "div": "div_based_token_selection",
}


def _env_flag(name: str, default: bool) -> bool:
    return _parse_bool(name, os.getenv(name, ""), default)


def get_prune_config(video_pruning_rate: float | None) -> FlashVidConfig:
    if video_pruning_rate is not None and (
        not isinstance(video_pruning_rate, (int, float))
        or not math.isfinite(video_pruning_rate)
        or not 0.0 <= video_pruning_rate < 1.0
    ):
        raise ValueError("video_pruning_rate must be finite and in [0, 1)")
    if video_pruning_rate is None or video_pruning_rate == 0.0:
        return FlashVidConfig(enable_flashvid=False, retention_ratio=1.0)

    enable_flashvid = ENABLE_FLASHVID
    enable_encode_prune = _env_flag("ENABLE_ENCODE_PRUNE", False)
    retention_ratio = 1 - video_pruning_rate
    if not enable_flashvid:
        return FlashVidConfig(enable_flashvid=False, retention_ratio=1.0, enable_encode_prune=enable_encode_prune)

    segment_threshold = float(os.getenv("FLASHVID_SEGMENT_THRESHOLD", "0.9"))
    min_segment_num = int(os.getenv("FLASHVID_MIN_SEGMENT_NUM", "4"))
    method = os.getenv("FLASHVID_TOKEN_SELECTION_METHOD", "attn_div").lower()
    alpha = float(os.getenv("FLASHVID_ALPHA", "0.70"))
    temporal_threshold = float(os.getenv("FLASHVID_TEMPORAL_THRESHOLD", "0.8"))
    encode_prune_threshold = float(os.getenv("FLASHVID_ENCODE_PRUNE_THRESHOLD", "0.7"))

    if not 0.0 <= segment_threshold <= 1.0:
        raise ValueError("FLASHVID_SEGMENT_THRESHOLD must be in [0, 1]")
    if min_segment_num < 1:
        raise ValueError("FLASHVID_MIN_SEGMENT_NUM must be at least 1")
    if method not in {"attn", "div", "attn_div", "attn_div_v2"}:
        raise ValueError(f"Unsupported FlashVID token selection method: {method!r}")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("FLASHVID_ALPHA must be in [0, 1]")
    if not 0.0 <= temporal_threshold <= 1.0:
        raise ValueError("FLASHVID_TEMPORAL_THRESHOLD must be in [0, 1]")
    if not 0.0 <= encode_prune_threshold <= 1.0:
        raise ValueError("FLASHVID_ENCODE_PRUNE_THRESHOLD must be in [0, 1]")

    config = FlashVidConfig(
        enable_flashvid=True,
        retention_ratio=retention_ratio,
        do_segment=_env_flag("FLASHVID_DO_SEGMENT", False),
        segment_threshold=segment_threshold,
        min_segment_num=min_segment_num,
        complementary_segment=_env_flag("FLASHVID_COMPLEMENTARY_SEGMENT", False),
        token_selection_method=method,
        alpha=alpha,
        temporal_threshold=temporal_threshold,
        enable_encode_prune=enable_encode_prune,
        encode_prune_threshold=encode_prune_threshold,
    )
    logger.info(f"FlashVID config: {config}")
    return config


def compute_flashvid_token_budget(
    tokens_per_frame: int,
    num_frames: int,
    retention_ratio: float,
    alpha: float,
) -> FlashVidTokenBudget:
    """Compute the exact FlashVID output budget.

    Args:
        tokens_per_frame: Number of merged vision tokens in each frame.
        num_frames: Number of frames in the video.
        retention_ratio: Configured FlashVID retention ratio.
        alpha: Fraction of retained tokens assigned to ADTS.

    Returns:
        The per-frame and total token budgets.
    """
    if tokens_per_frame <= 0 or num_frames <= 0:
        return FlashVidTokenBudget(0, 0, 0, 0)
    if retention_ratio <= 0.0:
        raise ValueError("retention_ratio must be greater than zero")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")

    retained_per_frame = math.ceil(tokens_per_frame * retention_ratio)
    retained_per_frame = min(tokens_per_frame, max(1, retained_per_frame))
    adts_per_frame = min(retained_per_frame, math.ceil(retained_per_frame * alpha))
    sttm_per_frame = retained_per_frame - adts_per_frame

    return FlashVidTokenBudget(
        retained_per_frame=retained_per_frame,
        total_tokens=num_frames * retained_per_frame,
        adts_per_frame=adts_per_frame,
        sttm_per_frame=sttm_per_frame,
    )


def allocate_flashvid_sttm_tokens(capacities: list[int], target: int) -> list[int]:
    """Allocate an exact STTM target proportionally across frames.

    Uses the largest-remainder method with integer arithmetic. Frame index is
    the deterministic tie breaker.

    Args:
        capacities: Maximum number of candidate tokens in each frame.
        target: Total number of STTM tokens to allocate.

    Returns:
        Per-frame cluster counts whose sum is exactly ``target``.

    Raises:
        ValueError: If the target or capacities are invalid.
    """
    if target < 0:
        raise ValueError("target must be non-negative")
    if any(capacity < 0 for capacity in capacities):
        raise ValueError("capacities must be non-negative")

    total_capacity = sum(capacities)
    if target > total_capacity:
        raise ValueError(f"FlashVID STTM target exceeds candidate capacity: target={target}, capacity={total_capacity}")
    if target == 0:
        return [0] * len(capacities)

    allocation = [target * capacity // total_capacity for capacity in capacities]
    remainders = [target * capacity % total_capacity for capacity in capacities]
    remaining = target - sum(allocation)
    order = sorted(range(len(capacities)), key=lambda index: (-remainders[index], index))
    for index in order:
        if remaining == 0:
            break
        if allocation[index] < capacities[index]:
            allocation[index] += 1
            remaining -= 1

    if remaining != 0 or sum(allocation) != target:
        raise RuntimeError("Failed to allocate the exact FlashVID STTM token budget")
    return allocation


def flashvid_compression(
    video_features: torch.Tensor,
    cls_attention: torch.Tensor,
    prune_config: FlashVidConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_frames, num_visual_tokens, feat_dim = video_features.shape

    # 1. Partition the video frames into segments based on transition similarities.
    if prune_config.do_segment:
        segment_lengths = segment(
            video_features=video_features.mean(1),
            segment_threshold=prune_config.segment_threshold,
            min_segment_num=prune_config.min_segment_num,
            complementary_segment=prune_config.complementary_segment,
        )
    else:
        # Treat the whole video as a single segment.
        segment_lengths = torch.tensor([num_frames], dtype=torch.long, device=video_features.device)

    num_segments = segment_lengths.shape[0]
    global_indices = torch.arange(num_frames * num_visual_tokens, dtype=torch.long, device=video_features.device)

    # 2. Apply Attention and Diversity-based Token Selection (ADTS).
    budget = compute_flashvid_token_budget(
        tokens_per_frame=num_visual_tokens,
        num_frames=num_frames,
        retention_ratio=prune_config.retention_ratio,
        alpha=prune_config.alpha,
    )

    all_segment_features = []
    all_segment_indices = []
    offset = 0
    for seg_idx in range(num_segments):
        seg_len = int(segment_lengths[seg_idx])
        segment_features = video_features[offset : offset + seg_len]
        segment_cls_attention = cls_attention[offset : offset + seg_len]
        segment_global_indices = global_indices.view(num_frames, num_visual_tokens)[offset : offset + seg_len]
        segment_features, segment_global_indices = segment_compression(
            segment_features=segment_features,
            segment_global_indices=segment_global_indices,
            cls_attention=segment_cls_attention,
            prune_config=prune_config,
            num_attn_div_tokens=budget.adts_per_frame,
            num_sttm_tokens=budget.sttm_per_frame,
        )
        all_segment_features.append(segment_features)
        all_segment_indices.append(segment_global_indices)
        offset += seg_len
    final_tokens = torch.cat(all_segment_features, dim=0)  # (num_final_tokens, feat_dim)
    final_global_indices = torch.cat(all_segment_indices, dim=0)  # (num_final_tokens,)

    sorted_indices = final_global_indices.argsort()
    sorted_tokens = final_tokens[sorted_indices]  # Sort by global indices.
    if len(sorted_tokens) != budget.total_tokens:
        raise RuntimeError(
            "FlashVID output does not match its fixed token budget: "
            f"actual={len(sorted_tokens)}, expected={budget.total_tokens}"
        )
    if len(final_global_indices) != budget.total_tokens:
        raise RuntimeError(
            "FlashVID representative indices do not match its token budget: "
            f"actual={len(final_global_indices)}, "
            f"expected={budget.total_tokens}"
        )
    # Store the final token length in the `prune_config`.
    prune_config.visual_token_length = sorted_tokens.shape[0]
    # print(f"#Visual Tokens After Vision-Side Compression : {prune_config.visual_token_length}")
    return sorted_tokens, final_global_indices[sorted_indices]


def segment_compression(
    segment_features: torch.Tensor,
    segment_global_indices: torch.Tensor,
    cls_attention: torch.Tensor,
    prune_config: FlashVidConfig,
    num_attn_div_tokens: int,
    num_sttm_tokens: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compress the segment features by applying Temporal Average Merging (TAM) and Spatial Merging.

    Args:
        segment_features (torch.Tensor): The features of the video segment, of shape (num_frames, num_visual_tokens, feat_dim).
        segment_global_indices (torch.Tensor): The global indices of the video segment, of shape (num_frames, num_visual_tokens).
        cls_attention (torch.Tensor): [CLS] attentions used for per-frame token selection, of shape (num_frames, num_visual_tokens).
        prune_config (FlashVidConfig): The configuration for FlashVid.
        num_attn_div_tokens: Number of ADTS tokens retained per frame.
        num_sttm_tokens: Number of STTM tokens retained per frame.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The final tokens and their global indices after compression.
    """
    num_frames, num_visual_tokens, feat_dim = segment_features.shape
    # 1. Apply Attention and Diversity-based Token Selection (ADTS).
    if num_attn_div_tokens > 0:
        additional_kwargs = {"cls_attention": cls_attention} if "attn" in prune_config.token_selection_method else {}
        selectors = import_module('.token_selection', __package__)
        selector = getattr(selectors, ALL_TOKEN_SELECTION_METHOD[prune_config.token_selection_method])
        selected_features, selected_indices = selector(
            features=segment_features,
            num_retained_tokens=num_attn_div_tokens,
            **additional_kwargs,
        )
        selected_global_indices = segment_global_indices.gather(1, index=selected_indices).view(-1)
    else:
        selected_features = segment_features.new_empty((num_frames, 0, feat_dim))
        selected_indices = torch.empty(
            (num_frames, 0),
            dtype=torch.long,
            device=segment_features.device,
        )
        selected_global_indices = segment_global_indices.new_empty((0,))

    mask = torch.ones(num_frames, num_visual_tokens, dtype=torch.bool, device=segment_features.device)
    mask.scatter_(1, selected_indices, False)
    num_other_tokens = num_sttm_tokens * num_frames
    # 1. Apply Temporal Average Merging (TAM) to the segment features.
    if num_other_tokens > 0:
        can_merge_frames = num_frames > 1
        if can_merge_frames and prune_config.temporal_threshold < 1.0:
            (
                temp_merged_token_list,
                temp_merged_indices_list,
            ) = spatiotemporal_compression(
                video_features=segment_features,
                temporal_threshold=prune_config.temporal_threshold,
                token_mask=mask,
                retained_tokens_per_frame=(num_attn_div_tokens + num_sttm_tokens),
            )
            temp_merged_global_indices_list = [
                segment_global_indices[i][temp_merged_indices]
                for i, temp_merged_indices in enumerate(temp_merged_indices_list)
            ]
        else:
            # With no temporal merging, use only tokens not selected by ADTS.
            temp_merged_indices_list = [torch.where(mask[frame_index])[0] for frame_index in range(num_frames)]
            temp_merged_token_list = [
                segment_features[frame_index][frame_indices]
                for frame_index, frame_indices in enumerate(temp_merged_indices_list)
            ]
            temp_merged_global_indices_list = [
                segment_global_indices[frame_index][frame_indices]
                for frame_index, frame_indices in enumerate(temp_merged_indices_list)
            ]
    else:
        # No spatial-temporal merging needed.
        temp_merged_token_list = []
        temp_merged_global_indices_list = []

    all_tokens = [selected_features.view(-1, feat_dim)]
    all_global_indices = [selected_global_indices]
    # 2. Apply Spatial Merging to the tokens after temporal merging.
    if num_other_tokens > 0:
        capacities = [len(tokens) for tokens in temp_merged_token_list]
        num_current_retained_tokens = sum(capacities)
        if num_current_retained_tokens < num_other_tokens:
            raise RuntimeError(
                "Insufficient FlashVID STTM candidates: "
                f"actual={num_current_retained_tokens}, "
                f"required={num_other_tokens}"
            )

        num_clusters_list = allocate_flashvid_sttm_tokens(capacities, num_other_tokens)
        if num_current_retained_tokens > num_other_tokens:
            num_frames_in_segment = len(temp_merged_token_list)
            max_num_tokens = max(len(tokens) for tokens in temp_merged_token_list)
            batched_tokens = torch.zeros(
                (num_frames_in_segment, max_num_tokens, feat_dim),
                dtype=segment_features.dtype,
                device=segment_features.device,
            )
            valid_token_mask = torch.zeros(
                (num_frames_in_segment, max_num_tokens),
                dtype=torch.bool,
                device=segment_features.device,
            )
            k_list = []
            for i, temp_merged_tokens in enumerate(temp_merged_token_list):
                num_tokens = len(temp_merged_tokens)
                batched_tokens[i, :num_tokens] = temp_merged_tokens
                valid_token_mask[i, :num_tokens] = True
                k_list.append(max(1, min(num_clusters_list[i], 7)))
            cluster_indices_list, cluster_center_indices_list = dpc_knn(
                features=batched_tokens,
                num_clusters=num_clusters_list,
                k=k_list,
                valid_token_mask=valid_token_mask,
            )
            for i, (
                temp_merged_tokens,
                temp_merged_global_indices,
            ) in enumerate(
                zip(
                    temp_merged_token_list,
                    temp_merged_global_indices_list,
                )
            ):
                num_clusters = num_clusters_list[i]
                if num_clusters == 0:
                    continue

                cluster_indices = cluster_indices_list[i][: len(temp_merged_tokens)]
                cluster_center_indices = cluster_center_indices_list[i]
                aggregated_tokens = torch.zeros(
                    (num_clusters, feat_dim),
                    dtype=segment_features.dtype,
                    device=segment_features.device,
                )
                aggregated_tokens.scatter_add_(
                    0,
                    cluster_indices.unsqueeze(-1).expand(-1, feat_dim),
                    temp_merged_tokens,
                )
                cluster_counts = (
                    torch.bincount(cluster_indices, minlength=num_clusters).unsqueeze(-1).to(segment_features.dtype)
                )
                aggregated_tokens = aggregated_tokens / cluster_counts
                global_token_indices = temp_merged_global_indices[cluster_center_indices]

                all_tokens.append(aggregated_tokens)
                all_global_indices.append(global_token_indices)
        else:
            for temp_merged_tokens, temp_merged_global_indices in zip(
                temp_merged_token_list,
                temp_merged_global_indices_list,
            ):
                all_tokens.append(temp_merged_tokens)
                all_global_indices.append(temp_merged_global_indices)

    segment_final_tokens = torch.cat(all_tokens, dim=0)  # (num_final_tokens, feat_dim)
    segment_final_global_indices = torch.cat(all_global_indices, dim=0)  # (num_final_tokens,)
    expected_tokens = (num_attn_div_tokens + num_sttm_tokens) * num_frames
    if len(segment_final_tokens) != expected_tokens:
        raise RuntimeError(
            "FlashVID segment output does not match its token budget: "
            f"actual={len(segment_final_tokens)}, expected={expected_tokens}"
        )
    if len(segment_final_global_indices) != expected_tokens:
        raise RuntimeError(
            "FlashVID segment indices do not match its token budget: "
            f"actual={len(segment_final_global_indices)}, "
            f"expected={expected_tokens}"
        )
    return segment_final_tokens, segment_final_global_indices


def segment(
    video_features: torch.Tensor,
    segment_threshold: float,
    min_segment_num: int,
    complementary_segment: bool = True,
) -> torch.Tensor:
    """Segments the video features into distinct segments based on similarity.

    Args:
        video_features (torch.Tensor): The video features to segment.
        segment_threshold (float): The threshold for segmenting.
        min_segment_num (int): The minimum number of segments required.
        complementary_segment (int): Use complementary segmentation to ensure `min_segment_num` constraint.

    Returns:
        torch.Tensor: The lengths of the segments.
    """
    num_frames, feat_dim = video_features.shape

    # 0. Calculate transition similarities
    normed_video_features = video_features / video_features.norm(p=2, dim=-1, keepdim=True)
    transition_similarities = torch.sum(normed_video_features[:-1] * normed_video_features[1:], dim=-1)

    # 1. Find cut indices based on the segment threshold
    cut_indices = torch.where(transition_similarities < segment_threshold)[0]

    # 2. Ensure at least `min_segment_num` segments (Top-K or Uniform complementary segment)
    segment_lengths = additional_segment(
        cut_indices=cut_indices,
        num_frames=num_frames,
        min_segment_num=min_segment_num,
        transition_similarities=transition_similarities,
        segment_threshold=segment_threshold,
        complementary_segment=complementary_segment,
    )
    return segment_lengths


def additional_segment(
    cut_indices: torch.Tensor,
    num_frames: int,
    min_segment_num: int,
    transition_similarities: torch.Tensor,
    segment_threshold: float,
    complementary_segment: bool = True,
):
    num_segments = cut_indices.numel() + 1
    if num_segments < min_segment_num and complementary_segment:
        num_remaining_cut_indices = min_segment_num - num_segments
        transition_similarities[transition_similarities < segment_threshold] = 1.0
        complementary_cut_indices = torch.topk(
            transition_similarities, k=min(num_remaining_cut_indices, transition_similarities.shape[0]), largest=False
        ).indices
        cut_indices = torch.cat([cut_indices, complementary_cut_indices]).sort().values

    padded_cut_indices = F.pad(cut_indices, (1, 1), value=0)
    padded_cut_indices[0] = -1
    padded_cut_indices[-1] = num_frames - 1
    segment_lengths = torch.diff(padded_cut_indices, n=1, dim=0)
    # print(f"segment lengths: {segment_lengths}")
    return segment_lengths


@torch.no_grad()
def dpc_knn(
    features: torch.Tensor,
    num_clusters: Union[int, List[int]],
    k: Union[int, List[int]] = 7,
    valid_token_mask: Optional[torch.Tensor] = None,
) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], Union[torch.Tensor, List[torch.Tensor]]]:
    """Apply DPC-kNN clustering algorithm to the pooled image features, generating preliminary clustering result.

    Args:
        features (torch.Tensor): Pooled image features (temporal features), of shape (batch_size, seq_len, feat_dim).
        num_clusters (int or List[int]): The number of clusters. If a list, specifies the number of clusters for each batch element.
        k (int or List[int]): The number of nearest neighbors to consider for local density. Default is 7.
        valid_token_mask (Optional[torch.Tensor]): Boolean Mask indicating valid tokens, of shape (batch_size, seq_len). Default is None.

    Returns:
        Tuple[Union[torch.Tensor, List[torch.Tensor]], Union[torch.Tensor, List[torch.Tensor]]]:
            Cluster indices and cluster center indices. If num_clusters is a list, returns lists of tensors.
    """
    invalid_token_mask = ~valid_token_mask if valid_token_mask is not None else None
    bsz, seq_len, feat_dim = features.shape
    if seq_len == 0:
        raise ValueError("DPC-kNN requires at least one candidate token")

    # Calculate euclidean distance and local density
    # dists = torch.cdist(features.float(), features.float()) / math.sqrt(feat_dim)

    features_fp32 = features.float()
    squared_norms = (features_fp32 * features_fp32).sum(dim=-1, keepdim=True)
    squared_dists = (
        squared_norms + squared_norms.transpose(1, 2) - 2 * torch.bmm(features_fp32, features_fp32.transpose(1, 2))
    ).clamp_min_(0.0)
    dists = torch.sqrt(squared_dists) / math.sqrt(feat_dim)

    # Mask out invalid tokens
    if valid_token_mask is not None:
        fill_val = dists.max() + 1
        dists.masked_fill_(invalid_token_mask.unsqueeze(1), fill_val)

    neighbors = k if isinstance(k, list) else [k] * bsz
    if len(neighbors) != bsz or any(value < 1 for value in neighbors):
        raise ValueError("k must specify a positive neighbor count for each batch item")
    capacities = valid_token_mask.sum(-1).tolist() if valid_token_mask is not None else [seq_len] * bsz
    neighbors = [min(value, max(1, capacity)) for value, capacity in zip(neighbors, capacities)]
    max_k = min(max(neighbors), seq_len)
    nearest_dist = torch.topk(dists, k=max_k, dim=-1, largest=False).values

    density = torch.stack(
        [torch.mean(-(nearest_dist[i, :, :count] ** 2), dim=-1).exp() for i, count in enumerate(neighbors)]
    )

    # ! [DISABLED] Add little random noise to ensure no tokens have the same density.
    # density = density + torch.rand_like(density, device=density.device, dtype=density.dtype) * 1e-6

    # Ensure the density of the empty token be 0
    if valid_token_mask is not None:
        density.masked_fill_(invalid_token_mask, 0.0)

    # Obtain the minimum distance to the point with higher density.
    mask = density[:, None, :] > density[:, :, None]
    max_dist = dists.view(bsz, -1).max(dim=-1)[0].view(-1, 1, 1)
    modified_dists = torch.where(mask, dists, max_dist)
    dist, _ = torch.min(modified_dists, dim=-1)

    # Calculate clustering score (clustering centers have the highest score)
    score = dist * density
    if invalid_token_mask is not None:
        score.masked_fill_(invalid_token_mask, float("-inf"))
    if isinstance(num_clusters, int):
        cluster_center_indices = torch.topk(score, k=num_clusters, dim=-1).indices
        # Obtain the distance matrix w.r.t cluster centers (batch_size, seq_len, num_clusters)
        dists = torch.gather(dists, dim=-1, index=cluster_center_indices.unsqueeze(1).expand(-1, seq_len, -1))
        cluster_indices = torch.argmin(dists, dim=-1)
        # Ensure each cluster center to merge with itself
        cluster_indices.scatter_(
            dim=-1,
            index=cluster_center_indices,
            src=torch.arange(num_clusters).to(cluster_indices).unsqueeze(0).expand(bsz, -1),
        )
        return cluster_indices, cluster_center_indices
    else:
        cluster_indices_list = []
        cluster_center_indices_list = []
        for i in range(bsz):
            k_i = num_clusters[i]
            if k_i == 0:
                cluster_center_indices_list.append(torch.tensor([], dtype=torch.long, device=features.device))
                cluster_indices_list.append(torch.zeros(seq_len, dtype=torch.long, device=features.device))
                continue
            cc_idx = torch.topk(score[i], k=k_i, dim=-1).indices
            cluster_center_indices_list.append(cc_idx)
            dists_i = torch.gather(dists[i], dim=-1, index=cc_idx.unsqueeze(0).expand(seq_len, -1))
            c_idx = torch.argmin(dists_i, dim=-1)
            c_idx.scatter_(
                dim=-1,
                index=cc_idx,
                src=torch.arange(k_i).to(c_idx),
            )
            cluster_indices_list.append(c_idx)
        return cluster_indices_list, cluster_center_indices_list


def spatiotemporal_compression(
    video_features: torch.Tensor,
    temporal_threshold: float,
    token_mask: torch.Tensor,
    retained_tokens_per_frame: int,
) -> tuple[list[Tensor], list[Tensor]]:
    """Merge temporal candidates without modifying the caller's features."""
    video_features = video_features.clone()
    num_frames, num_visual_tokens, feat_dim = video_features.shape
    # Selected ADTS tokens are masked from merging, so the lower bound includes
    # both ADTS and STTM tokens.
    lower_bound = retained_tokens_per_frame * num_frames
    normed_video_features = F.normalize(video_features.float(), p=2, dim=-1, eps=1e-12)
    cosine_similarities = torch.bmm(normed_video_features[1:], normed_video_features[:-1].transpose(1, 2))

    # Mask out the selected tokens.
    mask_next = ~token_mask[1:]
    mask_prev = ~token_mask[:-1]
    cosine_similarities.masked_fill_(mask_next.unsqueeze(-1), -1.0)
    cosine_similarities.masked_fill_(mask_prev.unsqueeze(1), -1.0)

    max_sims, max_sim_indices = torch.max(cosine_similarities, dim=-1)
    padded_max_sims = F.pad(max_sims, (0, 0, 1, 0), value=-1)
    padded_max_sim_indices = F.pad(max_sim_indices, (0, 0, 1, 0), value=-1)

    token_counts = torch.ones(num_frames, num_visual_tokens).to(video_features)
    mask = padded_max_sims > temporal_threshold
    retaining_token_mask = ~mask
    # Ensure the number of retained tokens after TAM does not fall below the
    # fixed output budget.
    if retaining_token_mask.int().sum() < lower_bound:
        total_tokens = num_frames * num_visual_tokens
        num_tokens_to_merge = total_tokens - lower_bound
        if num_tokens_to_merge == 0:
            mask = torch.zeros_like(mask)
            retaining_token_mask = ~mask
        else:
            soft_threshold = padded_max_sims.view(-1).topk(k=num_tokens_to_merge).values[-1]
            soft_threshold = max(soft_threshold, -1.0 + 1e-6)
            mask = padded_max_sims > soft_threshold
            retaining_token_mask = ~mask

    for frame_idx in range(num_frames - 1, -1, -1):
        frame_features = video_features[frame_idx]
        frame_token_counts = token_counts[frame_idx]
        frame_mask = mask[frame_idx]

        # Apply spatiotemporal average merging.
        retained_mask = ~frame_mask
        tokens_to_merge = frame_features[retained_mask]
        if tokens_to_merge.numel() > 0:
            to_merge_token_counts = frame_token_counts[retained_mask]
            aggregated_tokens = tokens_to_merge / to_merge_token_counts.unsqueeze(-1).to(tokens_to_merge.dtype)
            video_features[frame_idx][retained_mask] = aggregated_tokens
            token_counts[frame_idx][retained_mask] = 1

        # other tokens are connected to the previous frame's tokens
        other_tokens = frame_features[frame_mask]
        if other_tokens.numel() > 0:
            anchor_token_indices = padded_max_sim_indices[frame_idx][frame_mask]
            video_features[frame_idx - 1].index_add_(0, anchor_token_indices, other_tokens)
            token_counts[frame_idx - 1].index_add_(0, anchor_token_indices, frame_token_counts[frame_mask])
            token_counts[frame_idx][frame_mask] = 0
    # Filter final tokens
    combined_mask = retaining_token_mask & token_mask  # (num_frames, num_visual_tokens)
    final_tokens = []
    retained_token_indices = []
    for i in range(num_frames):
        frame_retained_indices = torch.where(combined_mask[i])[0]  # (frame_retained_tokens_num,)
        frame_retained_tokens = video_features[i][frame_retained_indices]  # (frame_retained_tokens_num, feat_dim)
        final_tokens.append(frame_retained_tokens)
        retained_token_indices.append(frame_retained_indices)

    return final_tokens, retained_token_indices
