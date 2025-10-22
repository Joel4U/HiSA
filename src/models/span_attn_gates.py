import torch
from typing import Optional, Tuple, Dict, List

# ==================== 距离核函数 ====================
def build_distance_kernel(
    dist: torch.Tensor, 
    mode: str = 'soft', 
    gamma: float = 1.0, 
    d0: int = 2
) -> torch.Tensor:
    """
    构建 token 级距离核 G[i,j] = g(dist[i,j])
    
    Args:
        dist: (B, L, L) token 对之间的距离矩阵
        mode: 'soft' 使用指数衰减，'hard' 使用阈值
        gamma: soft 模式下的衰减系数
        d0: 距离偏置/阈值
    
    Returns:
        G: (B, L, L) 距离核矩阵
        
    公式:
        - hard: G[i,j] = 1 if dist[i,j] <= d0 else 0
        - soft: G[i,j] = exp(-gamma * max(0, dist[i,j] - d0))
    """
    dist = dist.float()
    if mode == 'hard':
        return (dist <= d0).float()
    else:
        effective_dist = torch.clamp(dist - d0, min=0.0)
        return torch.exp(-gamma * effective_dist)

@torch.no_grad()
def pack_span_maps(span_maps: List[Tuple], Smax: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将 span_maps 列表打包为张量，便于后续完全向量化计算。
    Args:
        span_maps: [(span2id, id2lr), ...]，其中 id2lr: (Sb, 2)
        Smax: 最大 span 数量（邻居 builder 已确定）
    Returns:
        id2lr_pad: (B, Smax, 2) long 每个样本的 span (l, r)
        S_row_mask: (B, Smax) bool 有效 span 掩码
    """
    B = len(span_maps)
    id2lr_pad = torch.zeros(B, Smax, 2, dtype=torch.long, device=device)
    S_row_mask = torch.zeros(B, Smax, dtype=torch.bool, device=device)
    for b in range(B):
        _, id2lr_b = span_maps[b]
        Sb = 0 if id2lr_b is None else id2lr_b.size(0)
        if Sb > 0:
            id2lr_pad[b, :Sb] = id2lr_b
            S_row_mask[b, :Sb] = True
    return id2lr_pad, S_row_mask

@torch.no_grad()
def compute_softhead_overlap_gate(
    head_indices: torch.Tensor,   # (B, L, L, Ktok)
    head_weights: torch.Tensor,   # (B, L, L, Ktok)
    id2lr_pad: torch.Tensor,      # (B, S, 2)
    S_row_mask: torch.Tensor,     # (B, S)
    N_idx: torch.Tensor,          # (B, S, K)
    N_mask: torch.Tensor,         # (B, S, K)
    Ktok_limit: int = None,
    epsilon: float = 1e-8
) -> torch.Tensor:
    """
    基于 soft head 重叠度的门控（不使用距离）
    gate[s, k] = Jaccard 相似度 = 交集权重 / 并集权重
    或简化为：gate[s, k] = Σ_i min(w_s[i], w_n[i]) 对于共同的 token
    """
    device = head_indices.device
    B, L, _, Ktok_total = head_indices.shape
    S, K = id2lr_pad.size(1), N_idx.size(2)
    Ktok = Ktok_total if Ktok_limit is None else min(Ktok_limit, Ktok_total)
    
    # 获取每个 span 的 soft heads
    ls = id2lr_pad[..., 0].clamp(0, L - 1)
    rs = id2lr_pad[..., 1].clamp(0, L - 1)
    b_ids = torch.arange(B, device=device).view(B, 1).expand(B, S)
    
    idx_s = head_indices[b_ids, ls, rs, :Ktok].clamp(0, L - 1)  # (B, S, Ktok)
    w_s = head_weights[b_ids, ls, rs, :Ktok]                    # (B, S, Ktok)
    
    # 归一化权重
    valid_s = (idx_s >= 0) & (idx_s < L)
    w_s = w_s * valid_s.float()
    w_s = w_s / (w_s.sum(dim=-1, keepdim=True) + epsilon)
    
    # 邻居 span 的 soft heads
    N_idx_c = N_idx.clamp(0, S - 1)
    idx_n = torch.take_along_dim(
        idx_s.unsqueeze(2), 
        N_idx_c.unsqueeze(-1).expand(B, S, K, Ktok), 
        dim=1
    )  # (B, S, K, Ktok)
    w_n = torch.take_along_dim(
        w_s.unsqueeze(2), 
        N_idx_c.unsqueeze(-1).expand(B, S, K, Ktok), 
        dim=1
    )  # (B, S, K, Ktok)
    
    # 计算重叠度：找共同的 head token
    idx_s_exp = idx_s.unsqueeze(2).unsqueeze(-1)  # (B, S, 1, Ktok, 1)
    idx_n_exp = idx_n.unsqueeze(3)                # (B, S, K, 1, Ktok)
    same = (idx_s_exp == idx_n_exp).float()       # (B, S, K, Ktok, Ktok)
    
    # 交集权重：Σ_ij same[i,j] * min(w_s[i], w_n[j])
    w_s_exp = w_s.unsqueeze(2).unsqueeze(-1)      # (B, S, 1, Ktok, 1)
    w_n_exp = w_n.unsqueeze(3)                    # (B, S, K, 1, Ktok)
    min_w = torch.min(w_s_exp, w_n_exp)
    
    gate = (same * min_w).sum(dim=(-1, -2))       # (B, S, K) 交集权重
    
    # 可选：计算并集并归一化（完整 Jaccard）
    # union = w_s.sum(-1, keepdim=True) + w_n.sum(-1) - gate
    # gate = gate / (union + epsilon)
    
    # 掩码
    gate = gate * N_mask.float() * S_row_mask.unsqueeze(-1).float()
    return gate.clamp_min(epsilon)

@torch.no_grad()
def compute_span_distance_gate(
    id2lr_pad: torch.Tensor,   # (B, S, 2)
    N_idx: torch.Tensor,       # (B, S, K)
    N_mask: torch.Tensor,      # (B, S, K)
    gamma: float = 0.3,
    d0: int = 0,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Chebyshev 距离乘法门控：gate = exp(-γ·max(0, dist-d0))"""
    B, S, K = N_idx.shape
    ls, rs = id2lr_pad[..., 0], id2lr_pad[..., 1]
    N_idx_safe = N_idx.clamp(0, S - 1)
    
    n_ls = torch.take_along_dim(ls.unsqueeze(2), N_idx_safe, dim=1)
    n_rs = torch.take_along_dim(rs.unsqueeze(2), N_idx_safe, dim=1)
    
    dist = torch.maximum(
        (ls.unsqueeze(2) - n_ls).abs(),
        (rs.unsqueeze(2) - n_rs).abs()
    ).float()
    
    effective = (dist - d0).clamp_min(0.0)
    gate = torch.exp(-gamma * effective)
    return (gate * N_mask.float()).clamp_min(epsilon)

import torch
import torch.nn as nn

class SpanDistanceBias(nn.Module):
    """
    桶化距离的加法偏置（T5/ALiBi 风格）
    仅对指定头生效，其余头 bias=0
    """
    def __init__(
        self,
        num_heads: int,
        head_idx: int = 1,
        bucket_boundaries: tuple = (0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128),
        lambda_bias: float = 0.2,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_idx = head_idx
        self.lambda_bias = lambda_bias
        
        self.register_buffer(
            'boundaries', 
            torch.tensor(bucket_boundaries, dtype=torch.long),
            persistent=False
        )
        num_buckets = len(bucket_boundaries)
        
        # 每个桶一个可学习 bias
        self.bias_table = nn.Parameter(torch.zeros(num_buckets))
        nn.init.normal_(self.bias_table, mean=0.0, std=0.02)
    
    def forward(
        self,
        id2lr_pad: torch.Tensor,  # (B, S, 2)
        N_idx: torch.Tensor,      # (B, S, K)
        N_mask: torch.Tensor,     # (B, S, K)
    ) -> torch.Tensor:
        """返回 head_bias: (B, S, h, K)，仅 head_idx 处非 0"""
        device = id2lr_pad.device
        B, S, K = N_idx.shape
        
        # 计算 Chebyshev 距离（复用逻辑）
        ls, rs = id2lr_pad[..., 0], id2lr_pad[..., 1]
        N_idx_safe = N_idx.clamp(0, S - 1)
        n_ls = torch.take_along_dim(ls.unsqueeze(2), N_idx_safe, dim=1)
        n_rs = torch.take_along_dim(rs.unsqueeze(2), N_idx_safe, dim=1)
        dist = torch.maximum(
            (ls.unsqueeze(2) - n_ls).abs(),
            (rs.unsqueeze(2) - n_rs).abs()
        ).long()  # (B, S, K)
        
        # 桶化
        bucket_idx = torch.bucketize(dist, self.boundaries, right=False)
        bucket_idx = bucket_idx.clamp(0, len(self.boundaries) - 1)
        
        # 查表 + 缩放
        bias_vals = self.bias_table[bucket_idx] * self.lambda_bias
        bias_vals = bias_vals * N_mask.float()  # (B, S, K)
        
        # 拼成按头 bias
        head_bias = torch.zeros(B, S, self.num_heads, K, device=device, dtype=bias_vals.dtype)
        head_bias[:, :, self.head_idx, :] = bias_vals
        return head_bias

def build_multihead_gates(
    id2lr_pad: torch.Tensor,       # (B, S, 2)
    N_idx: torch.Tensor,           # (B, S, K)
    N_mask: torch.Tensor,          # (B, S, K)
    S_row_mask: torch.Tensor,      # (B, S)
    head_indices: torch.Tensor,    # 软 head 的索引
    head_weights: torch.Tensor,    # 软 head 的权重
    num_heads: int = 4, use_syntax_prior: bool = True, use_distance_prior: bool = True,
    gamma: float = 0.3, d0: int = 1, Ktok_limit: int = 3) -> torch.Tensor:
    """
    构建多头门控：
    - head0: 句法先验（softhead_overlap）
    - head1: 距离先验（Chebyshev 衰减）
    - head2-3: 自由注意力（gate=1）
    
    Returns:
        head_gate: (B, S, h, K)
    """
    device = id2lr_pad.device
    B, S, K = N_idx.shape
    
    # 初始化：所有头 gate=1
    head_gate = torch.ones(B, S, num_heads, K, device=device)
    
    # head0: 句法先验
    if use_syntax_prior:
        soft_gate = compute_softhead_overlap_gate(
            head_indices.detach(), head_weights.detach(), id2lr_pad, S_row_mask, N_idx, N_mask, Ktok_limit=Ktok_limit)  # (B, S, K)
        head_gate[:, :, 0, :] = soft_gate
    
    # head1: 距离先验
    if use_distance_prior:
        dist_gate = compute_span_distance_gate(id2lr_pad, N_idx, N_mask, gamma=gamma, d0=d0)  # (B, S, K)
        head_gate[:, :, 1, :] = dist_gate
    
    # head2-3: 保持 1（自由）
    return head_gate