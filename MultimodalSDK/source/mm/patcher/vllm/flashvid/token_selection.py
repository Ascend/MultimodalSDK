from typing import Tuple

from enum import Enum
import torch
import torch.nn.functional as F


class TokenSelectionMethod(str, Enum):
    ATTN = "attn"
    DIV = "div"
    ADTS = "attn_div"
    ADTS_v2 = "attn_div_v2"


def pairwise_cosine_distances(image_features: torch.Tensor) -> torch.Tensor:
    """Calculate pairwise cosine distances for a batch of feature vectors.

    Args:
        image_features (torch.Tensor): Visual features, of shape (bsz, num_visual_tokens, feat_dim)

    Returns:
        torch.Tensor: Pairwise cosine distances, of shape (bsz, num_visual_tokens, num_visual_tokens)
    """
    normed_features = F.normalize(image_features.float(), p=2, dim=-1, eps=1e-12)
    similarities = torch.bmm(normed_features, normed_features.transpose(-1, -2))
    return (1.0 - similarities).clamp_min(0.0)


def attn_div_based_token_selection1(
    features: torch.Tensor,
    cls_attention: torch.Tensor,
    num_retained_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select visual tokens based on attention and diversity.

    Args:
        features (torch.Tensor): Visual features, of shape (bsz, num_visual_tokens, feat_dim)
        cls_attention (torch.Tensor): [CLS] attention, of shape (bsz, num_visual_tokens)
        num_retained_tokens (int): Number of tokens to retain

    Returns:
        torch.Tensor: Pruned features, of shape (bsz, num_retained_tokens, feat_dim)
    """
    original_features = features
    features = features.float()
    cls_attention = cls_attention.float() * 1e6
    bsz, num_visual_tokens, feat_dim = features.shape
    dist_matrix = pairwise_cosine_distances(features)

    # Calibrate by [CLS] attention.
    dist_matrix = dist_matrix * cls_attention.unsqueeze(1)

    # Initialize keeping indices.
    keep_indices = torch.zeros(bsz, num_retained_tokens, dtype=torch.long, device=features.device)

    # select the first token.
    min_dist = torch.topk(dist_matrix, k=2, dim=1, largest=False).values[:, 1, :]
    keep_indices[:, 0] = torch.argmax(min_dist, dim=-1)

    batch_idx = torch.arange(bsz, device=features.device)

    # Maintain the running minimum distance from each token to the selected set.
    min_dist = dist_matrix[batch_idx, keep_indices[:, 0]]
    # Exclude the first selected token from reselection.
    min_dist.scatter_(1, keep_indices[:, :1], -1)

    # Select the rest of the tokens.
    for i in range(1, num_retained_tokens):
        keep_indices[:, i] = torch.argmax(min_dist, dim=-1)
        min_dist = torch.minimum(min_dist, dist_matrix[batch_idx, keep_indices[:, i]])
        min_dist.scatter_(1, keep_indices[:, i : i + 1], -1)

    keep_indices = keep_indices.sort().values
    selected_features = original_features[batch_idx.unsqueeze(1), keep_indices]

    return selected_features, keep_indices


# Original FlashVID implementation from flashvid/token_selection.py.
def attn_div_based_token_selection(
    features: torch.Tensor,
    cls_attention: torch.Tensor,
    num_retained_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select visual tokens based on attention and diversity.

    Args:
        features (torch.Tensor): Visual features, of shape (bsz, num_visual_tokens, feat_dim)
        cls_attention (torch.Tensor): [CLS] attention, of shape (bsz, num_visual_tokens)
        num_retained_tokens (int): Number of tokens to retain

    Returns:
        torch.Tensor: Pruned features, of shape (bsz, num_retained_tokens, feat_dim)
    """
    # Convert features and cls_attention to torch.float32
    original_features = features
    features = features.float()
    cls_attention = cls_attention.float() * 1e6  # Scale attention to avoid numerical issues
    bsz, num_visual_tokens, feat_dim = features.shape
    dist_matrix = pairwise_cosine_distances(features)

    # Calibrate by [CLS] attention.
    dist_matrix = dist_matrix * cls_attention.unsqueeze(1)  # (bsz, num_visual_tokens, num_visual_tokens)

    # Initialize keeping indices.
    keep_indices = torch.zeros(
        bsz, num_retained_tokens, dtype=torch.long, device=features.device
    )  # (bsz, num_retained_tokens)

    # select the first token.
    min_dist = torch.topk(dist_matrix, k=2, dim=1, largest=False).values[:, 1, :]  # (bsz, num_visual_tokens)
    keep_indices[:, 0] = torch.argmax(min_dist, dim=-1)  # (bsz,)

    # Select the rest of the tokens.
    for i in range(1, num_retained_tokens):
        # Get the distances to the already selected tokens.
        dist_sub_matrix = torch.gather(
            dist_matrix,
            dim=1,
            index=keep_indices[:, :i].unsqueeze(-1).expand(-1, -1, num_visual_tokens),
        )
        min_dist = torch.min(dist_sub_matrix, dim=1).values
        # Prevent select the same token again.
        min_dist.scatter_(1, keep_indices[:, :i], -1)  # (bsz, num_visual_tokens)
        keep_indices[:, i] = torch.argmax(min_dist, dim=-1)

    keep_indices = keep_indices.sort().values
    selected_features = torch.gather(
        original_features, dim=1, index=keep_indices.unsqueeze(-1).expand(-1, -1, feat_dim)
    )  # (bsz, num_retained_tokens, feat_dim)

    return selected_features, keep_indices


def attn_div_v3_based_token_selection(features, cls_attention, num_retained_tokens):
    bsz, n, dim = features.shape
    # 1. L2归一化求余弦距离
    dist = pairwise_cosine_distances(features)

    # 2. 和原版保持一致：注意力放大1e6，仅加权距离一次
    attn_scaled = cls_attention.float() * 1e6
    dist_weighted = dist * attn_scaled.unsqueeze(1)

    # 3. 每个token最近邻加权距离
    min_neighbor_dist = torch.topk(dist_weighted, k=2, dim=-1, largest=False).values[:, :, 1]

    # 4. 仅用加权离散度作为打分，不再二次乘attn，避免平方
    score = min_neighbor_dist

    # 一步筛选
    _, keep_indices = torch.topk(score, k=num_retained_tokens, dim=-1, largest=True)
    keep_indices = keep_indices.sort().values
    selected_features = torch.gather(features, 1, keep_indices.unsqueeze(-1).expand(-1, -1, dim))
    return selected_features, keep_indices


def attn_div_v2_based_token_selection(
    features: torch.Tensor,
    cls_attention: torch.Tensor,
    num_retained_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select visual tokens based on attention and diversity.

    Args:
        features (torch.Tensor): Visual features, of shape (bsz, num_visual_tokens, feat_dim)
        cls_attention (torch.Tensor): [CLS] attention, of shape (bsz, num_visual_tokens)
        num_retained_tokens (int): Number of tokens to retain

    Returns:
        torch.Tensor: Pruned features, of shape (bsz, num_retained_tokens, feat_dim)
    """
    original_features = features
    features = features.float()
    pooled_features = features.mean(1)
    global_cls_attention = cls_attention.float() * 1e6
    bsz, num_visual_tokens, feat_dim = features.shape
    dist_matrix = pairwise_cosine_distances(features)

    calibration_term1 = global_cls_attention.unsqueeze(1)
    local_cls_attention = torch.einsum("b n d, b d -> b n", features, pooled_features)
    calibration_term2 = local_cls_attention.unsqueeze(1)
    dist_matrix = dist_matrix * calibration_term1 * calibration_term2

    keep_indices = torch.zeros(bsz, num_retained_tokens, dtype=torch.long, device=features.device)

    min_dist = torch.topk(dist_matrix, k=2, dim=1, largest=False).values[:, 1, :]
    keep_indices[:, 0] = torch.argmax(min_dist, dim=-1)

    batch_idx = torch.arange(bsz, device=features.device)

    for i in range(1, num_retained_tokens):
        selected_idx = keep_indices[:, :i]
        dist_sub_matrix = dist_matrix[batch_idx.unsqueeze(1), selected_idx]
        min_dist = torch.min(dist_sub_matrix, dim=1).values
        min_dist.scatter_(1, selected_idx, float("-inf"))
        keep_indices[:, i] = torch.argmax(min_dist, dim=-1)

    keep_indices = keep_indices.sort().values
    selected_features = original_features[batch_idx.unsqueeze(1), keep_indices]

    return selected_features, keep_indices


def attn_based_token_selection(
    features: torch.Tensor,
    cls_attention: torch.Tensor,
    num_retained_tokens: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select visual tokens based on attention.

    Args:
        features (torch.Tensor): Visual features, of shape (bsz, num_visual_tokens, feat_dim)
        cls_attention (torch.Tensor): [CLS] attention, of shape (bsz, num_visual_tokens)
        num_retained_tokens (int): Number of tokens to retain

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Pruned features and their indices.
    """
    bsz, num_visual_tokens, feat_dim = features.shape
    topk_indices = torch.topk(cls_attention, k=num_retained_tokens, dim=-1).indices.sort().values
    batch_idx = torch.arange(bsz, device=features.device)
    selected_features = features[batch_idx.unsqueeze(1), topk_indices]
    return selected_features, topk_indices


def div_based_token_selection(
    features: torch.Tensor,
    num_retained_tokens: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select visual tokens based on diversity.

    Args:
        features (torch.Tensor): Visual features, of shape (bsz, num_visual_tokens, feat_dim)
        num_retained_tokens (int): Number of tokens to retain

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Pruned features and their indices.
    """
    original_features = features
    features = features.float()
    bsz, num_visual_tokens, feat_dim = features.shape
    dist_matrix = pairwise_cosine_distances(features)
    min_dist = torch.topk(dist_matrix, k=2, dim=1, largest=False).values[:, 1, :]

    keep_indices = torch.zeros(bsz, num_retained_tokens, dtype=torch.long, device=features.device)
    keep_indices[:, 0] = torch.argmax(min_dist, dim=-1)

    batch_idx = torch.arange(bsz, device=features.device)

    for i in range(1, num_retained_tokens):
        selected_idx = keep_indices[:, :i]
        dist_sub_matrix = dist_matrix[batch_idx.unsqueeze(1), selected_idx]
        min_dist = torch.min(dist_sub_matrix, dim=1).values
        min_dist.scatter_(1, selected_idx, -1)
        keep_indices[:, i] = torch.argmax(min_dist, dim=-1)

    keep_indices = keep_indices.sort().values
    selected_features = original_features[batch_idx.unsqueeze(1), keep_indices]

    return selected_features, keep_indices


def prune_embeds(image_embeds: torch.Tensor, threshold_sim: float = 0.7) -> tuple[torch.Tensor, torch.Tensor]:
    """Prune one sequence in [N, D], [N, 1, D], or [1, N, D] layout."""
    if image_embeds.dim() == 2:
        embeds = image_embeds
    elif image_embeds.dim() == 3 and image_embeds.shape[1] == 1:
        embeds = image_embeds.squeeze(1)
    elif image_embeds.dim() == 3 and image_embeds.shape[0] == 1:
        embeds = image_embeds.squeeze(0)
    else:
        raise ValueError("Expected a single sequence in [N, D], [N, 1, D], or [1, N, D] layout")
    seq_len = embeds.shape[0]

    # 不足2个token，全部保留
    if seq_len <= 1:
        keep_indices = torch.arange(seq_len, device=embeds.device)
        remove_indices = torch.empty(0, dtype=torch.long, device=embeds.device)
        return keep_indices, remove_indices

    # L2归一化，防止除0
    normed_embeds = F.normalize(embeds.float(), p=2, dim=-1, eps=1e-12)
    # 相邻两两余弦相似度 shape: [seq_len-1]
    transition_similarities = torch.sum(normed_embeds[:-1] * normed_embeds[1:], dim=-1)

    # 初始化掩码：偶数下标全部保留
    retention_mask = torch.zeros(seq_len, dtype=torch.bool, device=image_embeds.device)
    retention_mask[::2] = True

    # 取出所有奇数下标：1,3,5...
    odd_indices = torch.arange(start=1, end=seq_len, step=2, device=embeds.device)
    # 每个奇数位置对应其左侧相邻相似度索引 i-1
    sim_indices = odd_indices - 1
    odd_sims = transition_similarities[sim_indices]
    # 相似度 < 阈值则保留；>= 阈值判定冗余丢弃
    retention_mask[odd_indices] = odd_sims < threshold_sim

    # 获取保留和删除的索引
    keep_indices = retention_mask.nonzero().flatten()
    remove_indices = (~retention_mask).nonzero().flatten()

    return keep_indices, remove_indices


def restore_hidden_states(
    pruned_hidden: torch.Tensor,
    original_seq_len: int,
    keep_indices: torch.Tensor,
    rm_indices: torch.Tensor,
    pruned_attn: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    根据剪枝后特征 + 保留/删除索引，恢复原始长度时序hidden_states（可选同时恢复attn_weights）
    :param pruned_hidden: 剪枝后保留的有效特征 [M, D] 或 [M, 1, D]
    :param original_seq_len: 原始序列总长度
    :param keep_indices: 保留的索引数组
    :param rm_indices: 删除的索引数组
    :param pruned_attn: 可选，剪枝后保留的注意力权重 [M,], M <= original_seq_len
    :return (full_hidden, full_attn): full_hidden [original_seq_len, 1, D] 或 [original_seq_len, D], full_attn [original_seq_len,]

    保持输入特征的维度；删除位置使用最近的前置保留位置回填，序列开头的删除位置使用第一个保留位置。
    """
    if pruned_hidden.dim() not in (2, 3) or (pruned_hidden.dim() == 3 and pruned_hidden.shape[1] != 1):
        raise ValueError("Expected pruned_hidden in [M, D] or [M, 1, D] layout")
    if original_seq_len < 0:
        raise ValueError("original_seq_len must be non-negative")
    device = pruned_hidden.device
    for indices in (keep_indices, rm_indices):
        if indices.dim() != 1 or indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("Indices must be one-dimensional integer tensors")
    keep_indices = keep_indices.to(device=device, dtype=torch.long)
    rm_indices = rm_indices.to(device=device, dtype=torch.long)
    if keep_indices.numel() != pruned_hidden.shape[0]:
        raise ValueError("The number of retained indices must match pruned_hidden")
    partition = torch.cat((keep_indices, rm_indices)).sort().values
    if not torch.equal(partition, torch.arange(original_seq_len, device=device)):
        raise ValueError("Retained and removed indices must partition the original sequence")
    if original_seq_len and not keep_indices.numel():
        raise ValueError("Cannot restore a non-empty sequence without retained tokens")
    if pruned_attn is not None and (
        pruned_attn.dim() != 1 or pruned_attn.numel() != keep_indices.numel() or pruned_attn.device != device
    ):
        raise ValueError("Attention must have shape [M] and share the hidden-state device")

    full_hidden = pruned_hidden.new_zeros((original_seq_len, *pruned_hidden.shape[1:]))
    full_hidden[keep_indices] = pruned_hidden
    full_attn = None
    if pruned_attn is not None:
        full_attn = pruned_attn.new_zeros(original_seq_len)
        full_attn[keep_indices] = pruned_attn

    if rm_indices.numel():
        # Copy the closest preceding retained token. Leading gaps use the first
        # retained token, and consecutive gaps never read an unfilled position.
        sorted_keep = keep_indices.sort().values
        positions = torch.searchsorted(sorted_keep, rm_indices, right=True).sub(1).clamp_min(0)
        fill_source = sorted_keep[positions]
        full_hidden[rm_indices] = full_hidden[fill_source]
        if full_attn is not None:
            full_attn[rm_indices] = full_attn[fill_source]

    return full_hidden, full_attn
