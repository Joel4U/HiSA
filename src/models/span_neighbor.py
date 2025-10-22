import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional, Dict
# import matplotlib.pyplot as plt

# ============================================================
# 1. SoftHeadComputer (简化版，用于测试)
# ============================================================

class SoftHeadComputer(nn.Module):
    """计算 Span 的软 Head"""
    
    def __init__(
        self,
        w_coverage=2.0,
        w_degree=1.5,
        w_pos=0.8,
        w_deprel=0.5,
        w_medoid=0.5,
        w_direction=0.3,
        temperature=0.7,
        k=3
    ):
        super().__init__()
        self.w_coverage = w_coverage
        self.w_degree = w_degree
        self.w_pos = w_pos
        self.w_deprel = w_deprel
        self.w_medoid = w_medoid
        self.w_direction = w_direction
        self.temperature = temperature
        self.k = k
    
    def compute_soft_heads(
        self,
        dep_parents: torch.Tensor,      # (bsz, L)
        token_embeds: torch.Tensor,     # (bsz, dim, L)
        pos_batch: List[List[str]],
        deprel_batch: List[List[str]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        返回:
            soft_heads: (bsz, L, L, dim)
            head_weights: (bsz, L, L, k)
            head_indices: (bsz, L, L, k)
        """
        bsz, dim, L = token_embeds.size()
        device = token_embeds.device
        
        # 转置 token_embeds 到 (bsz, L, dim)
        token_embeds = token_embeds.transpose(1, 2)
        
        soft_heads = torch.zeros(bsz, L, L, dim, device=device)
        head_weights = torch.zeros(bsz, L, L, self.k, device=device)
        head_indices = torch.full((bsz, L, L, self.k), -1, dtype=torch.long, device=device)
        
        for b in range(bsz):
            for l in range(L):
                for r in range(l, L):
                    # 计算 span [l, r] 的头分数
                    scores = self._compute_head_scores(
                        l, r, dep_parents[b], pos_batch[b], deprel_batch[b], L
                    )
                    
                    # Softmax 归一化
                    probs = torch.softmax(scores / self.temperature, dim=0)
                    
                    # Top-K
                    topk_vals, topk_idx = torch.topk(probs, min(self.k, len(probs)))
                    
                    # 填充
                    k_actual = topk_idx.size(0)
                    head_indices[b, l, r, :k_actual] = topk_idx
                    head_weights[b, l, r, :k_actual] = topk_vals
                    
                    # 软 head 向量
                    soft_heads[b, l, r] = (
                        probs.unsqueeze(1) * token_embeds[b]
                    ).sum(dim=0)
        
        return soft_heads, head_weights, head_indices
    
    def _compute_head_scores(
        self, l: int, r: int, parents: torch.Tensor,
        pos: List[str], deprel: List[str], L: int
    ) -> torch.Tensor:
        """计算 span [l,r] 中每个 token 作为 head 的分数"""
        span_size = r - l + 1
        scores = torch.zeros(L)
        
        for t in range(l, r + 1):
            score = 0.0
            
            # 1. Coverage: 是否在 span 内
            if l <= t <= r:
                score += self.w_coverage
            
            # 2. Degree: 依存度数
            children = (parents == t).sum().item()
            score += self.w_degree * children
            
            # 3. POS 权重
            pos_weight = {
                'VERB': 2.0, 'NOUN': 1.5, 'ADJ': 1.0, 
                'ADP': 0.8, 'DET': 0.3
            }.get(pos[t], 0.5)
            score += self.w_pos * pos_weight
            
            # 4. Deprel 权重
            deprel_weight = {
                'root': 2.0, 'nsubj': 1.5, 'obj': 1.5,
                'obl': 1.2, 'amod': 1.0
            }.get(deprel[t], 0.5)
            score += self.w_deprel * deprel_weight
            
            # 5. Medoid: 到 span 中心的距离
            center = (l + r) / 2.0
            dist_to_center = abs(t - center)
            score += self.w_medoid * (1.0 / (1.0 + dist_to_center))
            
            scores[t] = score
        
        return scores


# ============================================================
# 2. 依存树距离计算
# ============================================================

def compute_dep_distance(parents: np.ndarray) -> torch.Tensor:
    """从父节点数组计算依存树距离矩阵"""
    L = len(parents)
    dist = np.full((L, L), L, dtype=np.int32)
    
    # Floyd-Warshall 算法
    # 初始化直接边
    for i in range(L):
        dist[i, i] = 0
        if parents[i] >= 0:
            p = parents[i]
            dist[i, p] = 1
            dist[p, i] = 1
    
    # 迭代更新
    for k in range(L):
        for i in range(L):
            for j in range(L):
                dist[i, j] = min(dist[i, j], dist[i, k] + dist[k, j])
    
    return torch.from_numpy(dist)


# ============================================================
# 3. SpanNeighborBuilder (修改版)
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional


# src/models/span_neighbor.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

class SpanNeighborBuilder(nn.Module):
    """
    向量化的 Span 邻居构建器（批量维度一次性处理），无 Python 级循环。
    - 候选集合 M：用 token 邻接和 top-1 head 的 token→span 映射做布尔矩阵乘法得到。
    - 门控打分 Gate：用 soft head 权重在 token 维的密集表示做 P @ G @ P^T 得到。
    - 几何填充：把 8 邻域转为 S×S 布尔矩阵，直接 OR 到候选集合。
    """
    def __init__(
        self,
        K: int = 10,                    # 每个 span 的邻居数
        Ktok: int = 3,                  # 每个 span 的 soft head 数量（取前 Ktok）
        d: int = 2,                     # 依存距离阈值（token 维）
        gamma: float = 1.0,             # 门控衰减系数
        max_width: Optional[int] = None,# 最大 span 宽度
        cap_spans_per_token: int = 128, # 每个 token 关联的最大 span 数（按枚举顺序保留前 cap 个）
        use_self_loop: bool = True,     # 使用自环补齐
        use_geom_fill: bool = True,     # 使用几何 8 邻域填充（作为候选集合的 OR）
    ):
        super().__init__()
        self.K = K
        self.Ktok = Ktok
        self.d = d
        self.gamma = gamma
        self.max_width = max_width
        self.cap_spans_per_token = cap_spans_per_token
        self.use_self_loop = use_self_loop
        self.use_geom_fill = use_geom_fill

        # 缓存 (L, max_width) → (span2id, id2lr, l_idx, r_idx)
        self._span_cache = {}

    def _recover_dist_from_prefix(self, dist_sum_prefix, dist_cnt_prefix, invalid_val=999.0):
        first_col_f = dist_sum_prefix[..., :1]
        first_col_c = dist_cnt_prefix[..., :1]
        sum_diff = torch.diff(dist_sum_prefix, dim=-1, prepend=torch.zeros_like(first_col_f))
        cnt_diff = torch.diff(dist_cnt_prefix, dim=-1, prepend=torch.zeros_like(first_col_c))
        dist = torch.where(
            cnt_diff > 0,
            sum_diff,  # 这就是 dist * 1
            torch.full_like(sum_diff, float(invalid_val)))
        return dist
    
    def _enumerate_spans(self, L: int, device: torch.device):
        """
        枚举满足宽度约束的所有 span，构造:
        - span2id: (L, L) 映射，非法为 -1
        - id2lr: (S, 2) 每个 span 的 (l,r)
        - l_idx, r_idx: (S,) 索引用于从 head_indices 中批量取值
        """
        key = (L, self.max_width)
        if key in self._span_cache:
            return self._span_cache[key]

        ar = torch.arange(L, device=device)
        l_grid = ar.view(L, 1).expand(L, L)     # (L,L)
        r_grid = ar.view(1, L).expand(L, L)     # (L,L)
        mask = (r_grid >= l_grid)
        if self.max_width is not None:
            mask &= (r_grid - l_grid + 1) <= self.max_width

        l_idx = l_grid[mask]  # (S,)
        r_idx = r_grid[mask]  # (S,)
        S = l_idx.numel()
        id2lr = torch.stack([l_idx, r_idx], dim=-1)  # (S,2)

        span2id = torch.full((L, L), -1, dtype=torch.long, device=device)
        span2id[l_idx, r_idx] = torch.arange(S, dtype=torch.long, device=device)

        self._span_cache[key] = (span2id, id2lr, l_idx, r_idx)
        return self._span_cache[key]

    def _build_geom_matrix(self, L: int, id2lr: torch.Tensor, span2id: torch.Tensor) -> torch.Tensor:
        """
        几何 8 邻域的 S×S 布尔矩阵 Geom：
        Geom[s, s2]=True 表示 s2 在 (l±1, r±1) 的邻域内且合法。
        """
        device = id2lr.device
        l_idx = id2lr[:, 0]  # (S,)
        r_idx = id2lr[:, 1]  # (S,)

        # 8 邻域偏移
        offsets = torch.tensor(
            [[-1, -1], [-1, 0], [-1, 1],
             [0, -1],           [0, 1],
             [1, -1],  [1, 0],  [1, 1]],
            dtype=torch.long, device=device
        )  # (8,2)

        # 生成邻域坐标 (S,8)
        nl = l_idx.unsqueeze(1) + offsets[:, 0].unsqueeze(0)  # (S,8)
        nr = r_idx.unsqueeze(1) + offsets[:, 1].unsqueeze(0)  # (S,8)
        valid = (nl >= 0) & (nr >= 0) & (nl <= nr) & (nr < L)
        if self.max_width is not None:
            valid &= (nr - nl + 1) <= self.max_width

        # 将 (nl,nr) 映射到 s2 id
        nl_safe = nl.clamp(0, L - 1)
        nr_safe = nr.clamp(0, L - 1)
        s2 = span2id[nl_safe, nr_safe]  # (S,8), 无效处为 -1

        # 构造 S×S 的几何邻接布尔矩阵
        S = id2lr.size(0)
        Geom = torch.zeros(S, S, dtype=torch.bool, device=device)

        rows = torch.arange(S, device=device).unsqueeze(1).expand(S, s2.size(1))  # (S,8)
        mask = valid & (s2 >= 0)
        rows_flat = rows[mask]        # (Nvalid,)
        cols_flat = s2[mask]          # (Nvalid,)
        Geom[rows_flat, cols_flat] = True
        # 自身不作为几何邻居
        diag = torch.arange(S, device=device)
        Geom[diag, diag] = False
        return Geom

    @torch.no_grad()
    def forward(
        self,
        head_indices: torch.Tensor,  # (B, L, L, Ktok_total)
        head_weights: torch.Tensor,  # (B, L, L, Ktok_total)
        dist: torch.Tensor,          # (B, L, L)
    ) -> Tuple[torch.Tensor, torch.Tensor, List]:
        """
        返回:
            N_idx:  (B, S, K) 邻居 span 的 ID
            N_mask: (B, S, K) 有效掩码
            span_maps: List[(span2id, id2lr)] 长度为 B（同一 L 下内容相同）
        说明:
            - 要求同一个 batch 内 L 一致（常见 pad 同长），否则需分桶处理。
            - 完全向量化，无 Python 级 b/sid 循环。
        """
        device = dist.device
        B, L, _L2, Ktok_total = head_indices.shape
        assert L == _L2, "dist/head_indices 的第二维应一致"
        Ktok = min(self.Ktok, Ktok_total)

        # 1) 枚举 span（按 max_width 约束）
        span2id, id2lr, l_idx, r_idx = self._enumerate_spans(L, device)
        S = id2lr.size(0)

        # Edge case: 若 S==0（极端设置），直接返回空
        if S == 0:
            empty_idx = torch.empty(B, 0, self.K, dtype=torch.long, device=device)
            empty_mask = torch.empty(B, 0, self.K, dtype=torch.bool, device=device)
            span_maps = [(span2id, id2lr) for _ in range(B)]
            return empty_idx, empty_mask, span_maps

        # 2) 取每个 span 的前 Ktok soft-head（批量）
        #    idx_all, w_all: (B, S, Ktok)
        idx_all = head_indices[:, l_idx, r_idx, :Ktok].contiguous()
        w_all = head_weights[:, l_idx, r_idx, :Ktok].contiguous()
        valid1 = idx_all.ge(0)
        idx_safe = idx_all.clamp(min=0)
        w_all = w_all * valid1

        # 3) 构造 token 维的权重矩阵 P_w: (B, S, L) 与二值矩阵 P_bin: (B, S, L)
        P_w = torch.zeros(B, S, L, dtype=w_all.dtype, device=device)
        P_w.scatter_add_(dim=2, index=idx_safe, src=w_all)
        P_bin = torch.zeros(B, S, L, dtype=torch.float32, device=device)
        P_bin.scatter_add_(dim=2, index=idx_safe, src=valid1.float())

        # 4) token 邻接 A: (B, L, L)，dist<=d；带自环
        A = (dist <= self.d).to(torch.float32)  # (B,L,L)
        eye = torch.eye(L, dtype=torch.float32, device=device).unsqueeze(0).expand(B, -1, -1)
        A = torch.maximum(A, eye)  # 加自环

        # 5) 候选 token 集 U_mask: (B, S, L)；候选 span 集 M: (B, S, S) 布尔
        U_hit = torch.bmm(P_bin, A)  # (B,S,L)
        U_mask = U_hit > 0.0

        # token→span 映射 E（只用 top-1 head）：(B, L, S)
        top1 = idx_all[:, :, 0].clamp(0, L - 1)               # (B,S)
        top1_valid = idx_all[:, :, 0].ge(0).float()           # (B,S)
        E = F.one_hot(top1, num_classes=L).float()            # (B,S,L)
        E = (E * top1_valid.unsqueeze(-1)).transpose(1, 2)    # (B,L,S)

        # 限制每个 token 关联的 span 数量（保留枚举顺序的前 cap 个）
        if self.cap_spans_per_token and self.cap_spans_per_token > 0:
            csum = torch.cumsum(E, dim=2)
            E = E * (csum <= float(self.cap_spans_per_token)).float()

        M = torch.bmm(U_mask.float(), E) > 0.0                # (B,S,S)
        # 去掉自身
        diag = torch.arange(S, device=device)
        M[:, diag, diag] = False

        # 几何 8 邻域 OR 融入候选集合（可选）
        if self.use_geom_fill:
            Geom = self._build_geom_matrix(L, id2lr, span2id)    # (S,S) bool
            M = M | Geom.unsqueeze(0)                            # 广播到 (B,S,S)

        # 6) 门控打分：Gate = P_w @ G @ P_w^T
        G = torch.exp(-self.gamma * dist.to(torch.float32))   # (B,L,L)
        A_pw = torch.bmm(P_w, G)                              # (B,S,L)
        Gate = torch.bmm(A_pw, P_w.transpose(1, 2))           # (B,S,S)

        # 7) 仅保留候选，其他置 -inf 后做 TopK（动态 k，避免 k>S 抛错）
        neg_inf = torch.tensor(float('-inf'), device=device, dtype=Gate.dtype)
        Gate_masked = torch.where(M, Gate, neg_inf)           # (B,S,S)

        k_eff = int(min(self.K, S))                           # 动态安全 k
        # 分配 (B,S,K) 输出，必要时填充
        N_idx = torch.full((B, S, self.K), -1, dtype=torch.long, device=device)
        N_mask = torch.zeros((B, S, self.K), dtype=torch.bool, device=device)

        if k_eff > 0:
            topk_vals, topk_idx = torch.topk(Gate_masked, k=k_eff, dim=2)  # (B,S,k_eff)
            N_idx[:, :, :k_eff] = topk_idx
            N_mask[:, :, :k_eff] = torch.isfinite(topk_vals)
        # 若 k_eff==0（极端情况 S==0 时已提前返回；此处仅防御）

        # 8) 自环补齐（不足 K 的位置填自身）
        if self.use_self_loop:
            rows = torch.arange(S, device=device).view(1, S, 1).expand(B, S, self.K)  # (B,S,K)
            fill_self = ~N_mask
            if fill_self.any():
                N_idx[fill_self] = rows[fill_self]
                N_mask[fill_self] = True

        # 9) span_maps（与原接口一致；同一 L 下内容相同，复用同一张量引用）
        span_maps = [(span2id, id2lr) for _ in range(B)]
        return N_idx.contiguous(), N_mask.contiguous(), span_maps

# ============================================================
# 4. 可视化函数
# ============================================================

def visualize_dependency_tree(tokens, parents, ax):
    """可视化依存树"""
    import networkx as nx
    
    G = nx.DiGraph()
    edges = []
    for i, p in enumerate(parents):
        if p >= 0:
            edges.append((p, i))
    
    G.add_edges_from(edges)
    
    # 层次布局
    root = np.where(parents == -1)[0][0]
    pos = hierarchy_pos(G, root)
    
    # 归一化位置
    if pos:
        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        
        for node in pos:
            x, y = pos[node]
            pos[node] = (
                (x - x_min) / (x_max - x_min + 1e-9),
                (y - y_min) / (y_max - y_min + 1e-9)
            )
    
    # 绘制
    nx.draw_networkx_nodes(G, pos, node_color='lightblue',
                           node_size=800, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color='gray',
                           arrows=True, arrowsize=15, ax=ax)
    
    labels = {i: f"{tokens[i]}\n({i})" for i in range(len(tokens))}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)
    
    ax.set_title("Dependency Tree", fontweight='bold')
    ax.axis('off')


def hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    """计算层次布局位置"""
    pos = _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)
    return pos


def _hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5,
                   pos=None, parent=None, parsed=None):
    if pos is None:
        pos = {root: (xcenter, vert_loc)}
    else:
        pos[root] = (xcenter, vert_loc)
    
    if parsed is None:
        parsed = [root]
    else:
        parsed.append(root)
    
    children = list(G.neighbors(root))
    if len(children) != 0:
        dx = width / len(children)
        nextx = xcenter - width/2 - dx/2
        for child in children:
            nextx += dx
            pos = _hierarchy_pos(G, child, width=dx, vert_gap=vert_gap,
                                vert_loc=vert_loc-vert_gap, xcenter=nextx,
                                pos=pos, parent=root, parsed=parsed)
    return pos


def visualize_span_neighbors(tokens, target_span, neighbors, id2lr, ax):
    """可视化 span 邻居关系"""
    import networkx as nx
    
    l, r = target_span
    target_text = " ".join(tokens[l:r+1])
    
    G = nx.Graph()
    target_sid = None
    
    # 找到目标 span id
    for sid, (sl, sr) in enumerate(id2lr):
        if sl == l and sr == r:
            target_sid = sid
            break
    
    if target_sid is None:
        ax.text(0.5, 0.5, "Target span not found", ha='center', va='center')
        ax.axis('off')
        return
    
    # 添加节点和边
    for sid in neighbors:
        G.add_node(sid)
        if sid != target_sid:
            G.add_edge(target_sid, sid)
    
    # 布局
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # 节点颜色
    node_colors = ['#FF6B6B' if sid == target_sid else '#4ECDC4' 
                   for sid in neighbors]
    
    # 绘制
    nx.draw_networkx_nodes(G, pos, nodelist=neighbors,
                           node_color=node_colors, node_size=1200,
                           alpha=0.9, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color='#95A5A6',
                           width=2, alpha=0.6, ax=ax)
    
    # 标签
    labels = {}
    for sid in neighbors:
        sl, sr = id2lr[sid]
        text = " ".join(tokens[sl:sr+1])
        if len(text) > 12:
            text = text[:12] + "..."
        labels[sid] = f"[{sl},{sr}]\n{text}"
    
    nx.draw_networkx_labels(G, pos, labels, font_size=7, ax=ax)
    
    ax.set_title(f"Neighbors of [{l},{r}] \"{target_text}\"",
                fontweight='bold', fontsize=10)
    ax.axis('off')


def print_neighbor_details(tokens, target_span, neighbors, id2lr, 
                          gate_scores=None):
    """打印邻居详情"""
    l, r = target_span
    target_text = " ".join(tokens[l:r+1])
    
    print("\n" + "="*80)
    print(f"Target Span: [{l},{r}] \"{target_text}\"")
    print("="*80)
    
    print(f"\n{'Rank':<6}{'Span ID':<10}{'Range':<12}{'Text':<25}{'Gate Score':<12}")
    print("-"*80)
    
    for rank, sid in enumerate(neighbors, 1):
        sl, sr = id2lr[sid]
        text = " ".join(tokens[sl:sr+1])
        
        flag = ""
        if sl == l and sr == r:
            flag = "★ SELF"
        elif gate_scores and sid in gate_scores:
            score = gate_scores[sid]
            flag = f"{score:.4f}"
        else:
            flag = "GEOM"
        
        print(f"{rank:<6}{sid:<10}[{sl},{sr}]{'':8}{text:<25}{flag:<12}")


# ============================================================
# 5. 完整测试
# ============================================================

def test_complete_pipeline():
    """完整测试流程"""
    
    print("="*80)
    print("Span Neighbor Builder - Complete Test")
    print("="*80)
    
    # ===== 测试数据 =====
    tokens = ["The","quick","brown","fox","jumps","over","the","lazy","dog"]
    n = len(tokens)
    
    parents_np = np.array([3,3,3,4,-1,8,8,8,4])
    pos = ["DET","ADJ","ADJ","NOUN","VERB","ADP","DET","ADJ","NOUN"]
    deprel = ["det","amod","amod","nsubj","root","case","det","amod","obl"]
    
    print(f"\nSentence: {' '.join(tokens)}")
    print(f"Parents:  {parents_np.tolist()}")
    print(f"POS:      {pos}")
    print(f"DepRel:   {deprel}")
    
    # ===== 准备输入 =====
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    bsz = 1
    dim = 64
    
    # 依存父节点
    dep_parents = torch.from_numpy(parents_np).unsqueeze(0).to(device)  # (1, 9)
    
    # Token embeddings (随机)
    token_embeds = torch.randn(bsz, dim, n, device=device)
    
    # 批次化的 POS 和 DepRel
    pos_batch = [pos]
    deprel_batch = [deprel]
    
    # ===== 计算依存距离 =====
    print("\n" + "-"*80)
    print("Computing dependency distance...")
    dist_matrix = compute_dep_distance(parents_np)
    dist = dist_matrix.unsqueeze(0).to(device)  # (1, 9, 9)
    
    print("\nDependency Distance Matrix:")
    print(dist_matrix.numpy())
    
    # ===== 计算 Soft Heads =====
    print("\n" + "-"*80)
    print("Computing soft heads...")
    
    computer = SoftHeadComputer(
        w_coverage=2.0,
        w_degree=1.5,
        w_pos=0.8,
        w_deprel=0.5,
        w_medoid=0.5,
        w_direction=0.3,
        temperature=0.7,
        k=3
    )
    
    soft_heads, head_weights, head_indices = computer.compute_soft_heads(
        dep_parents, token_embeds, pos_batch, deprel_batch
    )
    
    print(f"\nSoft heads shape: {soft_heads.shape}")      # (1, 9, 9, 64)
    print(f"Head weights shape: {head_weights.shape}")    # (1, 9, 9, 3)
    print(f"Head indices shape: {head_indices.shape}")    # (1, 9, 9, 3)
    
    # 示例：查看 span [4,6] "jumps over the" 的 heads
    target_l, target_r = 4, 6
    print(f"\nSpan [{target_l},{target_r}] \"{' '.join(tokens[target_l:target_r+1])}\":")
    print(f"  Top heads indices: {head_indices[0, target_l, target_r].tolist()}")
    print(f"  Top heads weights: {head_weights[0, target_l, target_r].tolist()}")
    
    # ===== 构建 Span 邻居 =====
    print("\n" + "-"*80)
    print("Building span neighbors...")
    
    builder = SpanNeighborBuilder(
        K=10,
        Ktok=3,
        d=2,
        gamma=1.0,
        max_width=None,
    )
    
    N_idx, N_mask, span_maps = builder(head_indices, head_weights, dist)
    
    print(f"\nNeighbor indices shape: {N_idx.shape}")  # (1, S, 10)
    print(f"Neighbor mask shape: {N_mask.shape}")
    
    # ===== 分析目标 Span =====
    span2id, id2lr = span_maps[0]
    
    # 找到目标 span 的 ID
    target_sid = span2id[target_l, target_r].item()
    print(f"\nTarget span ID: {target_sid}")
    
    # 获取邻居
    neighbors_mask = N_mask[0, target_sid]
    neighbors_ids = N_idx[0, target_sid][neighbors_mask].tolist()
    
    print(f"Number of neighbors: {len(neighbors_ids)}")
    
    # 打印邻居详情
    print_neighbor_details(
        tokens, (target_l, target_r), neighbors_ids,
        [(id2lr[i, 0].item(), id2lr[i, 1].item()) for i in range(id2lr.size(0))]
    )
    
    # ===== 可视化 =====
    print("\n" + "-"*80)
    print("Generating visualizations...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 依存树
    visualize_dependency_tree(tokens, parents_np, ax1)
    
    # Span 邻居图
    id2lr_list = [(id2lr[i, 0].item(), id2lr[i, 1].item()) 
                  for i in range(id2lr.size(0))]
    visualize_span_neighbors(
        tokens, (target_l, target_r), neighbors_ids, id2lr_list, ax2
    )
    
    plt.tight_layout()
    plt.savefig('span_neighbors_test.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to: span_neighbors_test.png")
    
    # ===== 统计信息 =====
    print("\n" + "="*80)
    print("Statistics")
    print("="*80)
    
    total_spans = id2lr.size(0)
    avg_neighbors = N_mask[0].float().sum(-1).mean().item()
    
    print(f"Total spans: {total_spans}")
    print(f"Average neighbors per span: {avg_neighbors:.2f}")
    
    # 邻居类型分布
    self_loops = 0
    for sid in range(total_spans):
        if sid in N_idx[0, sid][N_mask[0, sid]]:
            self_loops += 1
    
    print(f"Spans with self-loop: {self_loops} / {total_spans} ({100*self_loops/total_spans:.1f}%)")
    
    plt.show()
    
    return {
        'tokens': tokens,
        'target_span': (target_l, target_r),
        'neighbors': neighbors_ids,
        'N_idx': N_idx,
        'N_mask': N_mask,
        'span_maps': span_maps,
    }


if __name__ == "__main__":
    results = test_complete_pipeline()
