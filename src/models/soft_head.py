import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from typing import List, Tuple, Optional, Dict

class SoftHeadComputer:
    def __init__(self, 
                 init_w_coverage: float = 2.0,       # coverage_in 权重
                 init_w_degree: float = 1.5,         # degree_out 惩罚权重
                 init_w_pos: float = 0.8,            # degree_out 惩罚权重
                 init_w_deprel: float = 0.5,         # DEPREL先验权重
                 init_w_medoid: float = 0.5,         # tree medoid权重
                 init_w_direction: float = 0.3,      # 方向性权重
                 init_temperature: float = 0.7,      # softmax温度， 值越小，权重越尖锐
                 k: int = 3,                         # Top-K
                 learnable: bool = False):           # learnable: 是否可学习
        self.k = k
        self.learnable = learnable
        if learnable:                   # 定义为可学习参数
            self.w_coverage = nn.Parameter(torch.tensor(init_w_coverage))
            self.w_degree = nn.Parameter(torch.tensor(init_w_degree))
            self.w_pos = nn.Parameter(torch.tensor(init_w_pos))
            self.w_deprel = nn.Parameter(torch.tensor(init_w_deprel))
            self.w_medoid = nn.Parameter(torch.tensor(init_w_medoid))
            self.w_direction = nn.Parameter(torch.tensor(init_w_direction)) 
            # temperature 需要保持正值，用log参数化
            self.log_temperature = nn.Parameter(torch.tensor(init_temperature).log())             
        else:                           # 固定值
            self.register_buffer('w_coverage', torch.tensor(init_w_coverage))
            self.register_buffer('w_degree', torch.tensor(init_w_degree))
            self.register_buffer('w_pos', torch.tensor(init_w_pos))
            self.register_buffer('w_deprel', torch.tensor(init_w_deprel))
            self.register_buffer('w_medoid', torch.tensor(init_w_medoid))
            self.register_buffer('w_direction', torch.tensor(init_w_direction))
            self.register_buffer('log_temperature', torch.tensor(init_temperature).log())
        # 先验分数
        self.pos_prior = {
            "PROPN": 1.0, "NOUN": 0.8, "ADJ": 0.3, "VERB": 0.2, "ADV": 0.1,
            "NUM": 0.1, "PRON": 0.0, "PART": -0.2,
            "ADP": -0.8, "DET": -0.8, "AUX": -0.5, 
            "CCONJ": -0.8, "SCONJ": -0.8, "PUNCT": -1.0
        }
        
        self.deprel_prior = {
            "name": 0.6, "flat": 0.6, "compound": 0.6, "appos": 0.5, 
            "nmod": 0.4, "amod": 0.3, "nn": 0.3, "nsubj": 0.4, "obj": 0.4,
            "case": -0.8, "det": -0.8, "cc": -0.8, "mark": -0.8, 
            "aux": -0.8, "punct": -0.9
        }
    
    def precompute_tree_distances(self, dep_parents: torch.Tensor) -> torch.Tensor:
        """
        预计算依存树距离矩阵（采用高效BFS方法）
        Args:
            dep_parents: (bsz, L) 每个词的父节点索引
        Returns:
            dist: (bsz, L, L) 树上的最短路距离
        """
        bsz, L = dep_parents.shape
        device = dep_parents.device
        
        # 构建邻接表
        dist = torch.full((bsz, L, L), float('inf'), device=device)
        
        for b in range(bsz):
            # 构建无向邻接矩阵
            adj = torch.zeros(L, L, dtype=torch.bool, device=device)
            for i in range(L):
                parent = dep_parents[b, i].item()
                if parent >= 0:
                    adj[i, parent] = True
                    adj[parent, i] = True
            
            # BFS求每个点到其他点的距离
            for s in range(L):
                dist[b, s, s] = 0
                queue = [s]
                visited = {s}
                
                while queue:
                    v = queue.pop(0)
                    for u in range(L):
                        if adj[v, u] and u not in visited:
                            dist[b, s, u] = dist[b, s, v] + 1
                            visited.add(u)
                            queue.append(u)
        
        return dist
    
    def build_children_list(self, dep_parents: torch.Tensor) -> List[List[List[int]]]:
        """构建children列表"""
        bsz, L = dep_parents.shape
        children = [[[] for _ in range(L)] for _ in range(bsz)]
        
        for b in range(bsz):
            for u in range(L):
                p = dep_parents[b, u].item()
                if p >= 0:
                    children[b][p].append(u)
        
        return children
    
    def find_candidates(self, b: int, l: int, r: int, 
                       dep_parents: torch.Tensor) -> List[int]:
        """找候选head集合"""
        candidates = []
        for t in range(l, r + 1):
            parent = dep_parents[b, t].item()
            if parent == -1 or parent < l or parent > r:
                candidates.append(t)
        return candidates
    
    def coverage_in(self, b: int, t: int, l: int, r: int,
                   children: List[List[List[int]]]) -> int:
        """计算t的子树中落在[l,r]的词数"""
        cnt = 0
        stack = [t]
        visited = set()
        
        while stack:
            v = stack.pop()
            if v in visited:
                continue
            visited.add(v)
            if l <= v <= r:
                cnt += 1
            for u in children[b][v]:
                stack.append(u)
        
        return cnt
    
    def degree_out(self, b: int, t: int, l: int, r: int,
                  dep_parents: torch.Tensor,
                  children: List[List[List[int]]]) -> int:
        """计算t与span外的连接数"""
        deg = 0
        parent = dep_parents[b, t].item()
        
        # 父在外
        if parent >= 0 and not (l <= parent <= r):
            deg += 1
        
        # 子在外
        for u in children[b][t]:
            if not (l <= u <= r):
                deg += 1
        
        return deg
    
    def tree_medoid_score(self, t: int, l: int, r: int,
                         dist: torch.Tensor, b: int) -> float:
        """计算tree medoid分数"""
        span_len = r - l + 1
        total_dist = 0.0
        for u in range(l, r + 1):
            total_dist += dist[b, t, u].item()
        return -total_dist / max(1, span_len)
    
    def direction_score(self, t: int, l: int, r: int) -> float:
        """方向性分数（英语偏右）"""
        if r == l:
            return 0.0
        return (t - l) / (r - l)
    
    def compute_head_score(self, b: int, t: int, l: int, r: int,
                          dep_parents: torch.Tensor,
                          children: List[List[List[int]]],
                          dist: torch.Tensor,
                          pos_tags: Optional[List[List[str]]],
                          deprel_tags: Optional[List[List[str]]]) -> float:
        """计算候选head的综合得分（采用用户的归一化方法）"""
        # 1. Coverage (归一化)
        cov = self.coverage_in(b, t, l, r, children)
        cov_normalized = cov / (r - l + 1)
        # 2. Degree out
        deg = self.degree_out(b, t, l, r, dep_parents, children)
        # 3. POS prior
        pos_score = 0.0
        if pos_tags is not None:
            pos_score = self.pos_prior.get(pos_tags[b][t], 0.0)
        # 4. DEPREL prior
        rel_score = 0.0
        if deprel_tags is not None:
            rel_score = self.deprel_prior.get(deprel_tags[b][t], 0.0)
        # 5. Tree medoid
        medoid_score = self.tree_medoid_score(t, l, r, dist, b)
        # 6. Direction
        dir_score = self.direction_score(t, l, r)
        # 综合得分
        score = (self.w1 * cov_normalized - 
                self.w2 * deg + 
                self.w3 * pos_score + 
                self.w4 * rel_score + 
                self.w5 * medoid_score + 
                self.w6 * dir_score)
        
        return score
    
    def compute_soft_heads(self, token_embeds: torch.Tensor, dep_parents: torch.Tensor, dep_dist: torch.Tensor,
                          pos_tags: Optional[List[List[str]]] = None,
                          deprel_tags: Optional[List[List[str]]] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算所有span的软head
        Returns:
            soft_heads: (bsz, L, L, dim) 软head向量
            head_weights: (bsz, L, L, k) Top-K权重
            head_indices: (bsz, L, L, k) Top-K索引
        """
        bsz, L, dim = token_embeds.shape
        device = token_embeds.device
        
        # 预计算距离矩阵
        # dist = self.precompute_tree_distances(dep_parents)
        
        # 构建children列表
        children = self.build_children_list(dep_parents)
        
        # 初始化输出
        soft_heads = torch.zeros(bsz, L, L, dim, device=device)
        head_weights = torch.zeros(bsz, L, L, self.k, device=device)
        head_indices = torch.full((bsz, L, L, self.k), -1, 
                                 dtype=torch.long, device=device)
        
        # 遍历每个span
        for b in range(bsz):
            for l in range(L):
                for r in range(l, L):
                    # 找候选
                    candidates = self.find_candidates(b, l, r, dep_parents)
                    
                    if len(candidates) == 0:
                        continue
                    
                    # 打分
                    scores = []
                    for t in candidates:
                        score = self.compute_head_score(
                            b, t, l, r, dep_parents, children, dep_dist,
                            pos_tags, deprel_tags
                        )
                        scores.append(score)
                    
                    scores = torch.tensor(scores, device=device, dtype=torch.float)
                    
                    # Top-K
                    actual_k = min(self.k, len(candidates))
                    topk_scores, topk_idx = torch.topk(scores, actual_k)
                    
                    # Softmax
                    logits = topk_scores / self.tau
                    logits = logits - logits.max()  # 数值稳定
                    weights = torch.exp(logits)
                    weights = weights / weights.sum()
                    
                    # 保存
                    for i in range(actual_k):
                        head_indices[b, l, r, i] = candidates[topk_idx[i]]
                        head_weights[b, l, r, i] = weights[i]
                    
                    # 聚合embedding
                    for i in range(actual_k):
                        t_idx = candidates[topk_idx[i]]
                        soft_heads[b, l, r] += weights[i] * token_embeds[b, t_idx]
        
        return soft_heads, head_weights, head_indices


# ============= 测试代码 =============

def test_comparison():
    # 测试数据
    tokens = ["The","quick","brown","fox","jumps","over","the","lazy","dog"]
    n = len(tokens)
    
    parents_np = np.array([3,3,3,4,-1,8,8,8,4])
    pos = ["DET","ADJ","ADJ","NOUN","VERB","ADP","DET","ADJ","NOUN"]
    deprel = ["det","amod","amod","nsubj","root","case","det","amod","obl"]
    
    # 转为PyTorch
    dep_parents = torch.tensor([parents_np], dtype=torch.long)  # (1, 9)
    token_embeds = torch.randn(1, n, 128)  # (1, 9, 128)
    pos_batch = [pos]
    deprel_batch = [deprel]
    
    # 创建计算器
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
    
    # 计算软head
    soft_heads, head_weights, head_indices = computer.compute_soft_heads(
        dep_parents, token_embeds, pos_batch, deprel_batch
    )
    
    print(f"\n句子: {' '.join(tokens)}")
    print(f"依存结构: {parents_np.tolist()}")
    
    # 测试几个有趣的span
    test_spans = [
        (0, 3, "The quick brown fox"),
        (1, 2, "quick brown"),
        (3, 6, "fox jumps over the"),
        (5, 8, "over the lazy dog"),
        (6, 8, "the lazy dog"),
        (3, 3, "fox"),
        (8, 8, "dog"),
    ]
    
    print("\n" + "=" * 70)
    print("详细结果:")
    print("=" * 70)
    
    for l, r, text in test_spans:
        print(f"\n Span [{l},{r}]: '{text}'")
        
        # 候选集
        candidates = computer.find_candidates(0, l, r, dep_parents)
        print(f"   候选集: {[tokens[c] for c in candidates]}")
        
        # Top-K结果
        indices = head_indices[0, l, r]
        weights = head_weights[0, l, r]
        
        valid_k = (indices >= 0).sum().item()
        if valid_k > 0:
            print(f"   Top-{valid_k} 软Head:")
            for i in range(valid_k):
                idx = indices[i].item()
                w = weights[i].item()
                print(f"      • {tokens[idx]:8s} (idx={idx}, α={w:.4f})")
            
            # 软head向量
            h_norm = soft_heads[0, l, r].norm().item()
            print(f"   软Head向量范数: {h_norm:.4f}")
        else:
            print("   ⚠️  无有效head")
    
    # 权重分析
    print("\n" + "=" * 70)
    print("权重分布分析:")
    print("=" * 70)
    
    for l, r, text in [(4, 7, "jumps over the lazy"), (3, 6, "fox jumps over the")]:
        print(f"\nSpan '{text}':")
        indices = head_indices[0, l, r]
        weights = head_weights[0, l, r]
        valid_k = (indices >= 0).sum().item()
        
        if valid_k > 0:
            total_weight = weights[:valid_k].sum().item()
            print(f"  权重和: {total_weight:.4f} (应为1.0)")
            
            for i in range(valid_k):
                idx = indices[i].item()
                w = weights[i].item()
                print(f"    {tokens[idx]}: {w:.4f} ({w*100:.1f}%)")
    
    print("\n" + "=" * 70)
    print("✅ 测试完成！")
    print("=" * 70)


def test_numpy_version():
    """运行用户的NumPy版本（作为对照）"""
    
    print("\n" + "=" * 70)
    print("NumPy原版输出（作为对照）:")
    print("=" * 70 + "\n")
    
    # 用户的代码
    tokens = ["The","quick","brown","fox","jumps","over","the","lazy","dog"]
    n = len(tokens)
    parents = np.array([3,3,3,4,-1,8,8,8,4])
    pos = ["DET","ADJ","ADJ","NOUN","VERB","ADP","DET","ADJ","NOUN"]
    deprel = ["det","amod","amod","nsubj","root","case","det","amod","obl"]
    
    children = [[] for _ in range(n)]
    for u, p in enumerate(parents):
        if p != -1:
            children[p].append(u)
    
    # 距离矩阵
    adj = [[0]*n for _ in range(n)]
    for i in range(n):
        if parents[i] != -1:
            p = parents[i]
            adj[i][p] = adj[p][i] = 1
    
    dist = np.full((n,n), 1e9, dtype=np.float32)
    for s in range(n):
        dist[s,s] = 0
        q = [s]
        while q:
            v = q.pop(0)
            for u in range(n):
                if adj[v][u] and dist[s,u] > dist[s,v] + 1:
                    dist[s,u] = dist[s,v] + 1
                    q.append(u)
    
    pos_prior = {
        "PROPN": 1.0, "NOUN": 0.8, "ADJ": 0.3, "VERB": 0.2, "ADV": 0.1,
        "NUM": 0.1, "PRON": 0.0, "PART": -0.2,
        "ADP": -0.8, "DET": -0.8, "AUX": -0.5, "CCONJ": -0.8, "SCONJ": -0.8, "PUNCT": -1.0
    }
    deprel_prior = {
        "name": 0.6, "flat": 0.6, "compound": 0.6, "appos": 0.5, "nmod": 0.4,
        "amod": 0.3, "nn": 0.3,
        "case": -0.8, "det": -0.8, "cc": -0.8, "mark": -0.8, "aux": -0.8, "punct": -0.9
    }
    
    w1, w2, w3, w4, w5, w6 = 2.0, 1.5, 0.8, 0.5, 0.5, 0.3
    tau = 0.7
    
    def candidates(l, r, parents):
        return [t for t in range(l, r+1) if parents[t] == -1 or parents[t] < l or parents[t] > r]
    
    def coverage_in(t, l, r, children):
        cnt = 0
        stack = [t]
        visited = set()
        while stack:
            v = stack.pop()
            if v in visited: continue
            visited.add(v)
            if l <= v <= r: cnt += 1
            for u in children[v]: stack.append(u)
        return cnt
    
    def degree_out(t, l, r, parents, children):
        deg = 0
        if parents[t] != -1 and not (l <= parents[t] <= r): deg += 1
        for u in children[t]:
            if not (l <= u <= r): deg += 1
        return deg
    
    def tree_medoid_score(t, l, r, dist):
        L = r - l + 1
        s = 0.0
        for u in range(l, r+1):
            s += dist[t, u]
        return - s / max(1, L)
    
    def dir_score(t, l, r):
        if r == l: return 0.0
        return (t - l) / (r - l)
    
    def score_t(t, l, r):
        cov = coverage_in(t, l, r, children)
        cov = cov / (r - l + 1)
        deg = degree_out(t, l, r, parents, children)
        pos_s = pos_prior.get(pos[t], 0.0)
        rel_s = deprel_prior.get(deprel[t], 0.0)
        med = tree_medoid_score(t, l, r, dist)
        dsc = dir_score(t, l, r)
        f = w1*cov - w2*deg + w3*pos_s + w4*rel_s + w5*med + w6*dsc
        return f
    
    def select_heads_for_span(l, r, K=2):
        H = candidates(l, r, parents)
        if not H:
            return [], []
        fs = np.array([score_t(t, l, r) for t in H], dtype=np.float32)
        idx = np.argsort(-fs)[:K]
        chosen = [H[i] for i in idx]
        logits = fs[idx] / tau
        alpha = np.exp(logits - logits.max())
        alpha = alpha / alpha.sum()
        return chosen, alpha
    
    # 测试相同的span
    test_spans = [(0,3), (1,2), (5,8), (6,8), (3,3), (8,8)]
    
    for l, r in test_spans:
        heads, alpha = select_heads_for_span(l, r, K=2)
        span_text = ' '.join(tokens[l:r+1])
        print(f"span[{l},{r}] '{span_text}'")
        print(f"  heads: {[tokens[t] for t in heads]}")
        print(f"  weights: {np.round(alpha, 4).tolist()}")


if __name__ == "__main__":
    # 先运行NumPy版本
    # test_numpy_version()
    # 再运行融合版本
    print("\n")
    test_comparison()