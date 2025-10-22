import sys
import os

# 获取当前文件所在目录的父目录的父目录，即 HiSA/ 目录
# span_softhead.py -> model/ -> src/ -> HiSA/
hisa_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 将 HiSA/ 目录添加到 sys.path
sys.path.insert(0, hisa_root_dir)


import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
from src.utils.data_utils import build_pos_score_table, build_deprel_score_table

class SoftHeadComputer(nn.Module):
    """基于依存语法的 Span Soft Head 计算器"""
    
    def __init__(self, 
                 w_coverage: float = 2.0,       # coverage_in 权重
                 w_degree: float = 1.5,         # degree_out 惩罚权重
                 w_pos: float = 0.8,            # POS 先验权重
                 w_deprel: float = 0.5,         # DEPREL 先验权重
                 w_medoid: float = 0.5,         # tree medoid 权重
                 w_direction: float = 0.3,      # 方向性权重
                 temperature: float = 0.7,      # softmax 温度
                 k: int = 3,                    # Top-K
                 learnable: bool = False,       # 是否可学习
                 pos2id: Optional[Dict[str, int]] = None,    # POS 标签映射
                 deprel2id: Optional[Dict[str, int]] = None
                 ): # DEPREL 标签映射
        super().__init__()
        
        self.k = k
        self.learnable = learnable
        self.r_block = 32
        # 权重参数（可学习 or 固定）
        if learnable:
            self.w_coverage = nn.Parameter(torch.tensor(w_coverage))
            self.w_degree = nn.Parameter(torch.tensor(w_degree))
            self.w_pos = nn.Parameter(torch.tensor(w_pos))
            self.w_deprel = nn.Parameter(torch.tensor(w_deprel))
            self.w_medoid = nn.Parameter(torch.tensor(w_medoid))
            self.w_direction = nn.Parameter(torch.tensor(w_direction))
            self.log_temperature = nn.Parameter(torch.tensor(temperature).log())
        else:
            self.register_buffer('w_coverage', torch.tensor(w_coverage))
            self.register_buffer('w_degree', torch.tensor(w_degree))
            self.register_buffer('w_pos', torch.tensor(w_pos))
            self.register_buffer('w_deprel', torch.tensor(w_deprel))
            self.register_buffer('w_medoid', torch.tensor(w_medoid))
            self.register_buffer('w_direction', torch.tensor(w_direction))
            self.register_buffer('log_temperature', torch.tensor(temperature).log())
        
        # 保存映射
        self.pos2id = pos2id
        self.deprel2id = deprel2id
        
        # Penn Treebank POS 标签先验分数（完整覆盖）
        # self.pos_prior = self._init_pos_prior()
        
        # Universal Dependencies 依存关系先验分数（完整覆盖）
        # self.deprel_prior = self._init_deprel_prior()
        
        # 构建 ID -> score 的查找表（用于快速访问）
        if pos2id is not None:
            pos_score_table = build_pos_score_table(pos2id)
            self.register_buffer('pos_score_table', pos_score_table)
        else:
            self.pos_score_table = None
            
        if deprel2id is not None:
            deprel_score_table = build_deprel_score_table(deprel2id)
            self.register_buffer('deprel_score_table', deprel_score_table)
        else:
            self.deprel_score_table = None
    
    def _init_pos_prior(self) -> Dict[str, float]:
        """初始化 Penn Treebank POS 标签先验分数（完整覆盖所有标签）"""
        return {
            # ===== 名词类 (高分 - 倾向作为 head) =====
            "NN": 0.8, "NNS": 0.8, "NNP": 1.0, "NNPS": 1.0, "NN|SYM": 0.7,
            
            # ===== 形容词类 (中等分) =====
            "JJ": 0.3, "JJR": 0.3, "JJS": 0.3,
            
            # ===== 动词类 (中低分) =====
            "VB": 0.2, "VBD": 0.2, "VBG": 0.2, "VBN": 0.2, "VBP": 0.2, "VBZ": 0.2,
            
            # ===== 副词类 (低分) =====
            "RB": 0.1, "RBR": 0.1, "RBS": 0.1, "WRB": -0.3,
            
            # ===== 数词 =====
            "CD": 0.1,
            
            # ===== 代词 =====
            "PRP": 0.0, "PRP$": 0.0, "WP": 0.0, "WP$": 0.0,
            
            # ===== 限定词 (负分 - 不倾向作为 head) =====
            "DT": -0.8, "PDT": -0.8, "WDT": -0.5,
            
            # ===== 介词/连词 (负分) =====
            "IN": -0.8, "TO": -0.8, "CC": -0.8,
            
            # ===== 助动词 =====
            "MD": -0.5,
            
            # ===== 其他功能词 =====
            "EX": -0.5, "POS": -0.2, "RP": -0.2,
            
            # ===== 标点符号 (最低分) =====
            '"': -1.0, "$": -1.0, "''": -1.0, "(": -1.0, ")": -1.0,
            ",": -1.0, ".": -1.0, ":": -1.0, "SYM": -1.0,
            
            # ===== 其他 =====
            "LS": -0.5, "FW": 0.0, "UH": -0.3,
        }
    
    def _init_deprel_prior(self) -> Dict[str, float]:
        """初始化 UD 依存关系先验分数（完整覆盖所有标签）"""
        return {
            # ===== 高分 (倾向作为 head 的关系) =====
            "flat": 0.6, "compound": 0.6, "appos": 0.5, "nmod": 0.4, "amod": 0.3,
            "nsubj": 0.4, "obj": 0.4, "nummod": 0.3,
            
            # ===== 中等分 =====
            "acl": 0.3, "acl:relcl": 0.3, "advcl": 0.2, "advcl:relcl": 0.2,
            "ccomp": 0.2, "xcomp": 0.2, "obl": 0.2, "obl:tmod": 0.2,
            "obl:npmod": 0.2, "obl:agent": 0.2,
            
            # ===== 中低分 =====
            "conj": 0.1, "parataxis": 0.1, "list": 0.1, "iobj": 0.3,
            "csubj": 0.3, "csubj:outer": 0.3, "nsubj:pass": 0.3, "nsubj:outer": 0.3,
            
            # ===== 修饰语 (0分) =====
            "advmod": 0.0, "nmod:poss": 0.0, "nmod:tmod": 0.0,
            "nmod:npmod": 0.0, "nmod:desc": 0.0,
            
            # ===== 低分 (功能词) =====
            "det": -0.8, "det:predet": -0.8, "case": -0.8, "cc": -0.8,
            "cc:preconj": -0.8, "mark": -0.8, "aux": -0.8, "aux:pass": -0.8,
            "cop": -0.8, "expl": -0.8,
            
            # ===== 最低分 =====
            "punct": -0.9,
            
            # ===== 其他 =====
            "fixed": -0.5, "compound:prt": 0.0, "discourse": -0.3,
            "dislocated": -0.3, "vocative": -0.3, "dep": 0.0,
            "orphan": -0.2, "reparandum": -0.5,
            
            # ===== Root (特殊情况) =====
            "root": 1.0,
        }
    
    def _build_pos_score_table(self, pos2id: Dict[str, int]) -> torch.Tensor:
        """构建 POS ID -> score 的查找表"""
        num_tags = len(pos2id)
        score_table = torch.zeros(num_tags)
        
        for tag, tag_id in pos2id.items():
            score_table[tag_id] = self.pos_prior.get(tag, 0.0)
        
        return score_table
    
    def _build_deprel_score_table(self, deprel2id: Dict[str, int]) -> torch.Tensor:
        """构建 DEPREL ID -> score 的查找表"""
        num_rels = len(deprel2id)
        score_table = torch.zeros(num_rels)
        
        for rel, rel_id in deprel2id.items():
            score_table[rel_id] = self.deprel_prior.get(rel, 0.0)
        
        return score_table
    
    @property
    def temperature(self):
        """获取当前温度值"""
        return self.log_temperature.exp()
    
    def _ensure_cache_block(self, W: int, device, dtype):
        # 缓存与 W 相关的常量
        if not hasattr(self, "_base_cache_blk"):
            self._base_cache_blk = {}
        key = (W, device, dtype)
        if key in self._base_cache_blk:
            return self._base_cache_blk[key]
        tri = torch.triu(torch.ones(W, W, dtype=torch.bool, device=device))  # i<=j
        t_rel = torch.arange(W, device=device, dtype=dtype)                  # (W,)
        inv_span = 1.0 / (torch.arange(1, W + 1, device=device, dtype=dtype))# (W,)
        dir_denom = torch.arange(W, device=device, dtype=dtype).clamp_min(1) # (W,)
        # direction 矩阵 (W,W): t_rel / j
        direction = (t_rel.view(W, 1) / dir_denom.view(1, W)).to(dtype)      # (W,W)
        cache = dict(tri=tri, inv_span=inv_span, direction=direction)
        self._base_cache_blk[key] = cache
        return cache

    def get_score_tables(self):
        """获取先验分数表（供 DataLoader 使用）"""
        return {
            'pos_score_table': self.pos_score_table,
            'deprel_score_table': self.deprel_score_table 
        }
    
    @torch.no_grad()
    def compute_soft_heads(self, 
                          token_embeds: torch.Tensor,       # (bsz, L, dim)
                          dep_parents: torch.Tensor,        # (bsz, L)
                          precomputed: Optional[Dict[str, torch.Tensor]] = None,
                          l_block: int = 256                # l 维分块大小，可调
                          ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算所有 span 的软 head（仅在 span 内索引，O(W)），含列块化以降低中间张量规模。返回张量第三维为 W（最大 span 宽度），非全 L。
        Args:
            token_embeds: (bsz, L, dim) token embeddings
            dep_parents: (bsz, L) 依存父节点索引   
            precomputed: 预计算特征字典，包含：
                - subtree_prefix: (bsz, L, L) 子树前缀和
                - neighbors_prefix: (bsz, L, L) 邻接前缀和
                - deg_total: (bsz, L) 节点总度数
                - dist_sum_prefix: (bsz, L, L) 距离和前缀
                - dist_cnt_prefix: (bsz, L, L) 距离计数前缀
                - pos_scores: (bsz, L) POS 先验分数
                - deprel_scores: (bsz, L) DEPREL 先验分数     
        Returns:
            soft_heads:   (bsz, L, W, dim)
            head_weights: (bsz, L, W, k)
            head_indices: (bsz, L, W, k)
        """
        bsz, L, dim = token_embeds.shape
        device = token_embeds.device
        compute_dtype = torch.float32
        out_dtype = token_embeds.dtype
        W = precomputed.get('max_span_width', getattr(self, 'max_span_width', 32))
        W = min(W, L)
        k = self.k
        k_eff = min(k, W)
        T = self.temperature
        r_block = W # W<=32 时直接全宽处理，减少循环和索引开销, min(getattr(self, 'r_block', 16), W)

        subtree_prefix = precomputed['subtree_prefix'].to(compute_dtype).contiguous()
        neighbors_prefix = precomputed['neighbors_prefix'].to(compute_dtype).contiguous() 
        deg_total = precomputed['deg_total'].to(compute_dtype).contiguous()               
        dist_sum_prefix = precomputed['dist_sum_prefix'].to(compute_dtype).contiguous()
        dist_cnt_prefix = precomputed['dist_cnt_prefix'].to(compute_dtype).contiguous()
        pos_score = precomputed['pos_scores'].to(compute_dtype).contiguous()
        deprel_score = precomputed['deprel_scores'].to(compute_dtype).contiguous()
        # 展平成 (B, L*L)，便于线性 gather
        subtree_flat = subtree_prefix.view(bsz, L * L)
        neighbors_flat = neighbors_prefix.view(bsz, L * L)
        dist_sum_flat = dist_sum_prefix.view(bsz, L * L)
        dist_cnt_flat = dist_cnt_prefix.view(bsz, L * L)
        # 输出缓冲
        out_heads   = torch.zeros(bsz, L, W, dim, device=device, dtype=out_dtype)
        out_weights = torch.zeros(bsz, L, W, k,   device=device, dtype=out_dtype)
        out_indices = torch.full((bsz, L, W, k), -1, dtype=torch.long, device=device)
        # ===== 特征开关 =====
        use_coverage  = abs(self.w_coverage)  > 1e-9
        use_degree    = abs(self.w_degree)    > 1e-9
        use_pos       = abs(self.w_pos)       > 1e-9
        use_deprel    = abs(self.w_deprel)    > 1e-9
        use_medoid    = abs(self.w_medoid)    > 1e-9
        use_direction = abs(self.w_direction) > 1e-9
        # 基础缓存
        base = self._ensure_cache_block(W, device, out_dtype)
        tri = base["tri"]                                                # (W, W), 上三角 bool
        inv_span = base["inv_span"].to(compute_dtype).view(1, 1, 1, W)   # (1,1,1,W)
        direction = base["direction"].to(compute_dtype).view(1, 1, W, W) # (1,1,W,W)
        neg_inf = torch.tensor(float("-inf"), device=device, dtype=compute_dtype)
        parents = dep_parents.to(device)        # (bsz, L), long
        # t_rel = torch.arange(W, device=device)  # (W,)
        # r_rel = torch.arange(W, device=device)  # (W,)
        w_rel = torch.arange(W, device=device)  # 0..W-1
        # 预备批次索引 view（避免循环内重复创建）
        # b3_idx = torch.arange(bsz, device=device).view(bsz, 1, 1)       # (bsz,1,1)
        # b4_idx = torch.arange(bsz, device=device).view(bsz, 1, 1, 1)    # (bsz,1,1,1)

        for l0 in range(0, L, l_block):
            LB = min(l_block, L - l0)
            l_vals = l0 + torch.arange(LB, device=device)  # (LB,)
            t_idx2d = l_vals.view(LB, 1) + w_rel.view(1, W)  # (LB, W)
            r_idx2d = l_vals.view(LB, 1) + w_rel.view(1, W)  # (LB, W)
            t_valid_2d = (t_idx2d < L)
            r_valid_2d = (r_idx2d < L)
             # (B,LB,W) 的 t 绝对下标
            T3c = t_idx2d.view(1, LB, W).expand(bsz, LB, W).clamp(0, L - 1).long()
            # parents/deg/pos/deprel 仅与 t 相关：块内复用一次
            idx_flat = T3c.reshape(bsz, -1)                                   # (B,LB*W)
            p_sub = torch.gather(parents, 1, idx_flat).view(bsz, LB, W)       # long
            deg_tot = torch.gather(deg_total, 1, idx_flat).view(bsz, LB, W)   # fp32
            pos = torch.gather(pos_score, 1, idx_flat).view(bsz, LB, W)       # fp32
            dr = torch.gather(deprel_score, 1, idx_flat).view(bsz, LB, W)     # fp32
            # l-1 列（用于前缀差分）
            l_prev = (l_vals - 1).clamp_min(0)                                     # (LB,)
            Lm1_3c  = l_prev.view(1, LB, 1).expand(bsz, LB, W).clamp(0, L - 1).long()
            l_pos3 = (l_vals > 0).view(1, LB, 1)                                    # (1,LB,1)
            base_t = torch.zeros(bsz, LB, W, dtype=compute_dtype, device=device)
            if use_coverage:
                base_t.add_(self.w_pos * pos)       # (B,LB,W)
            if use_deprel:
                base_t.add_(self.w_deprel * dr)
            if use_degree:
                base_t.add_(-self.w_degree * deg_tot)  # -w*deg_tot，后续再 +w*deg_in
            # T4c/R4c 形状 (B,LB,W,W)
            T4c = T3c.unsqueeze(-1).expand(bsz, LB, W, W)
            R4c = r_idx2d.view(1, LB, 1, W).expand(bsz, LB, W, W).clamp(0, L - 1).long()
            lin_idx_r = (T4c * L + R4c).reshape(bsz, -1)  # (B,LB*W*W)
            # l-1 的线性索引（无 r 维）
            lin_idx_lm1 = (T3c * L + Lm1_3c).reshape(bsz, -1)  # (B,LB*W)
            # 覆盖度/邻居度/距离的前缀值
            # r 维列块化处理
            if use_coverage:
                cov_r = torch.gather(subtree_flat, 1, lin_idx_r).view(bsz, LB, W, W)             # fp32
                cov_lm1 = torch.gather(subtree_flat, 1, lin_idx_lm1).view(bsz, LB, W)            # fp32
            if use_degree:
                deg_in_r = torch.gather(neighbors_flat, 1, lin_idx_r).view(bsz, LB, W, W)        # fp32
                deg_in_lm1 = torch.gather(neighbors_flat, 1, lin_idx_lm1).view(bsz, LB, W)       # fp32
            if use_medoid:
                ds_r = torch.gather(dist_sum_flat, 1, lin_idx_r).view(bsz, LB, W, W)             # fp32
                ds_lm1 = torch.gather(dist_sum_flat, 1, lin_idx_lm1).view(bsz, LB, W)            # fp32
                dc_r = torch.gather(dist_cnt_flat, 1, lin_idx_r).view(bsz, LB, W, W)             # int
                dc_lm1 = torch.gather(dist_cnt_flat, 1, lin_idx_lm1).view(bsz, LB, W)            # int
            # ---- 评分张量（B,LB,W,W） ----
            score = base_t.unsqueeze(-1).expand(bsz, LB, W, W).clone()  # 初始为只依赖 t 的项
            if use_coverage:
                cov = torch.where(l_pos3.unsqueeze(-1), cov_r - cov_lm1.unsqueeze(-1), cov_r)
                score.add_(self.w_coverage * cov * inv_span)  # inv_span: (1,1,1,W)
            if use_degree:
                deg_in = torch.where(l_pos3.unsqueeze(-1), deg_in_r - deg_in_lm1.unsqueeze(-1), deg_in_r)
                score.add_(self.w_degree * deg_in)  # 已包含 -w*deg_tot，故加上 +w*deg_in 即 -w*(deg_tot-deg_in)
            if use_medoid:
                ds = torch.where(l_pos3.unsqueeze(-1), ds_r - ds_lm1.unsqueeze(-1), ds_r)        # fp32
                dc = torch.where(l_pos3.unsqueeze(-1), dc_r - dc_lm1.unsqueeze(-1), dc_r).to(compute_dtype)
                dc.clamp_min_(1)
                score.add_(self.w_medoid * (-(ds / dc)))
            # ---- 有效掩码并置 -inf ----
            t_valid_blk = t_valid_2d.view(1, LB, W, 1).expand(bsz, LB, W, W)
            r_valid_blk = r_valid_2d.view(1, LB, 1, W).expand(bsz, LB, W, W)
            p4 = p_sub.unsqueeze(-1)  # (B,LB,W,1)
            l4 = l_vals.view(1, LB, 1, 1)
            r4 = r_idx2d.view(1, LB, 1, W)
            cand_mask = (p4 < l4) | (p4 > r4) | (p4 < 0)
            valid_mask = cand_mask & t_valid_blk & r_valid_blk & tri.view(1, 1, W, W)
            score.masked_fill_(~valid_mask, neg_inf)
            # ---- 在 t 维做 top-k（dim=2）并求权重 ----
            topk_vals, topk_rows = torch.topk(score, k=k_eff, dim=2)               # (B,LB,k,W)
            logits = topk_vals / T
            logits.sub_(logits.max(dim=2, keepdim=True).values)                    # 稳定化
            weights = torch.softmax(logits, dim=2)                                 # (B,LB,k,W)
            has_cand = valid_mask.any(dim=2)                                       # (B,LB,W)
            weights = weights * has_cand.unsqueeze(2)
            # 绝对行下标（token 索引），无效置 -1
            rows_abs = topk_rows + l_vals.view(1, LB, 1, 1)                        # (B,LB,k,W)
            good_slot = torch.isfinite(topk_vals) & has_cand.unsqueeze(2)
            rows_out = torch.where(good_slot, rows_abs, torch.full_like(rows_abs, -1))
            # ---- 选择 token_embeds：2D gather（扁平化后再 reshape）----
            rows_safe = rows_out.clamp(0, L - 1)
            flat_idx = rows_safe.reshape(bsz, -1)                                  # (B,LB*k*W)
            sel_flat = torch.gather(
                token_embeds, 1, flat_idx.unsqueeze(-1).expand(bsz, flat_idx.size(1), dim))  # (B,LB*k*W,dim)
            sel = sel_flat.view(bsz, LB, k_eff, W, dim)
            # ---- 加权求和并写回输出 ----
            soft_blk = (weights.unsqueeze(-1) * sel.to(compute_dtype)).sum(dim=2)  # (B,LB,W,dim)
            out_heads[:,   l0:l0+LB, :W] = soft_blk.to(out_dtype)
            out_weights[:, l0:l0+LB, :W, :k_eff] = weights.permute(0, 1, 3, 2).contiguous().to(out_dtype)
            out_indices[:, l0:l0+LB, :W, :k_eff] = rows_out.permute(0, 1, 3, 2).contiguous()
        return out_heads, out_weights, out_indices
    
    def forward(self, *args, **kwargs):
        """前向传播，调用 compute_soft_heads"""
        return self.compute_soft_heads(*args, **kwargs)

    def expand_to_full_matrix(self, soft_heads, head_weights, head_indices):
        # soft_heads: (bsz, L, W, *rest)
        # head_weights: (bsz, L, W, Dw)
        # head_indices: (bsz, L, W, Ki)
        bsz, L, W = soft_heads.shape[:3]
        rest = list(soft_heads.shape[3:])
        device = soft_heads.device
        dtype = soft_heads.dtype
        # 构造每个 (l,r) 对应的 w = r - l
        l = torch.arange(L, device=device).view(L, 1)          # (L,1)
        r = torch.arange(L, device=device).view(1, L)          # (1,L)
        w_full = r - l                                         # (L,L)
        mask = (w_full >= 0) & (w_full < W)                    # 有效位置
        w_full = w_full.clamp(min=0, max=W-1).to(torch.long)   # 越界先夹到合法，后面再用 mask 置零/置-1
        # soft_heads
        if rest:
            idx_soft = w_full.view(1, L, L, *([1]*len(rest))).expand(bsz, L, L, *rest)
            mask_soft = mask.view(1, L, L, *([1]*len(rest))).expand(bsz, L, L, *rest)
        else:
            idx_soft = w_full.view(1, L, L).expand(bsz, L, L)
            mask_soft = mask.view(1, L, L).expand(bsz, L, L)
        full_soft_heads = torch.gather(soft_heads, 2, idx_soft)
        full_soft_heads = full_soft_heads * mask_soft  # 越界位置置零
        # head_weights
        Dw = head_weights.shape[-1]
        idx_w = w_full.view(1, L, L, 1).expand(bsz, L, L, Dw)
        mask_w = mask.view(1, L, L, 1).expand(bsz, L, L, Dw)
        full_head_weights = torch.gather(head_weights, 2, idx_w)
        full_head_weights = full_head_weights * mask_w          # 越界置零
        # head_indices（长整型，越界置 -1）
        Ki = head_indices.shape[-1]
        idx_i = w_full.view(1, L, L, 1).expand(bsz, L, L, Ki)
        mask_i = mask.view(1, L, L, 1).expand(bsz, L, L, Ki)
        full_head_indices = torch.gather(head_indices, 2, idx_i)
        full_head_indices = torch.where(mask_i, full_head_indices, torch.full_like(full_head_indices, -1))
        return full_soft_heads.contiguous(), full_head_weights.contiguous(), full_head_indices.contiguous()

# ===== 辅助函数：模拟 DataLoader 的预计算（用于测试） =====
def mock_precompute_features(dep_parents, dep_dist, pos_tags, deprel_tags, 
                             pos_score_table, deprel_score_table):
    """
    模拟 DataLoader 的预计算过程（仅用于测试）
    
    Args:
        dep_parents: (bsz, L) 依存父节点
        dep_dist: (bsz, L, L) 依存距离矩阵
        pos_tags: (bsz, L) POS 标签 ID
        deprel_tags: (bsz, L) 依存关系 ID
        pos_score_table: (num_pos,) POS 先验分数表
        deprel_score_table: (num_deprel,) 依存关系先验分数表
    """
    bsz, L = dep_parents.shape
    
    # 初始化
    neighbors_prefix = torch.zeros(bsz, L, L, dtype=torch.int32)
    deg_total = torch.zeros(bsz, L, dtype=torch.int32)
    subtree_prefix = torch.zeros(bsz, L, L, dtype=torch.int32)
    dist_sum_prefix = torch.zeros(bsz, L, L, dtype=torch.float32)
    dist_cnt_prefix = torch.zeros(bsz, L, L, dtype=torch.int32)
    pos_scores = torch.zeros(bsz, L, dtype=torch.float32)
    deprel_scores = torch.zeros(bsz, L, dtype=torch.float32)
    
    # 逐样本处理（模拟 CPU 上的 DataLoader）
    for b in range(bsz):
        parents = dep_parents[b].cpu().tolist()
        
        # 1. 邻接与度数
        neighbors = torch.zeros(L, L, dtype=torch.int32)
        for i, p in enumerate(parents):
            if 0 <= p < L:
                neighbors[i, p] = 1
                neighbors[p, i] = 1
        deg = neighbors.sum(dim=-1)
        neigh_prefix = neighbors.cumsum(dim=-1)
        
        deg_total[b] = deg
        neighbors_prefix[b] = neigh_prefix
        
        # 2. 子树掩码（DFS）
        children = [[] for _ in range(L)]
        for u, p in enumerate(parents):
            if 0 <= p < L:
                children[p].append(u)
        
        visited = [False] * L
        subtree_mask = torch.zeros(L, L, dtype=torch.uint8)
        
        def dfs(t):
            visited[t] = True
            mask = torch.zeros(L, dtype=torch.uint8)
            mask[t] = 1
            for c in children[t]:
                if not visited[c]:
                    child_mask = dfs(c)
                else:
                    child_mask = subtree_mask[c]
                mask |= child_mask
            subtree_mask[t] = mask
            return mask
        
        for u in range(L):
            if parents[u] < 0 and not visited[u]:
                dfs(u)
        for u in range(L):
            if not visited[u]:
                dfs(u)
        
        sub_prefix = subtree_mask.to(torch.int32).cumsum(dim=-1)
        subtree_prefix[b] = sub_prefix
        
        # 3. 距离前缀
        dist_tensor = dep_dist[b].cpu().float()
        valid = dist_tensor < 999
        dist_sum = (dist_tensor * valid).cumsum(dim=-1)
        dist_cnt = valid.to(torch.int32).cumsum(dim=-1)
        
        dist_sum_prefix[b] = dist_sum
        dist_cnt_prefix[b] = dist_cnt
        
        # 4. POS 和 DEPREL 分数
        if pos_score_table is not None:
            for t in range(L):
                pid = pos_tags[b, t].item()
                if 0 <= pid < len(pos_score_table):
                    pos_scores[b, t] = pos_score_table[pid]
        
        if deprel_score_table is not None:
            for t in range(L):
                rid = deprel_tags[b, t].item()
                if 0 <= rid < len(deprel_score_table):
                    deprel_scores[b, t] = deprel_score_table[rid]
    
    return {
        'neighbors_prefix': neighbors_prefix,
        'deg_total': deg_total,
        'subtree_prefix': subtree_prefix,
        'dist_sum_prefix': dist_sum_prefix,
        'dist_cnt_prefix': dist_cnt_prefix,
        'pos_scores': pos_scores,
        'deprel_scores': deprel_scores,
    }
# ===== 测试函数 =====
def test_soft_head_computer():
    """测试 SoftHeadComputer（使用预计算特征）"""
    
    print("=" * 60)
    print("测试 SoftHeadComputer（使用预计算特征）")
    print("=" * 60)
    
    # 标签映射
    postag2id = {"NN": 0, "VB": 1, "DT": 2, "IN": 3}
    deplabel2id = {"nsubj": 0, "root": 1, "det": 2, "case": 3}
    
    # 初始化
    computer = SoftHeadComputer(
        w_coverage=2.0,
        w_degree=1.5,
        w_pos=0.8,
        w_deprel=0.5,
        w_medoid=0.5,
        w_direction=0.3,
        temperature=0.7,
        k=3, pos2id=postag2id, deprel2id=deplabel2id
    )
    
    # 获取先验分数表
    score_tables = computer.get_score_tables()
    
    # 输入数据
    bsz, L, dim = 2, 5, 64
    token_embeds = torch.randn(bsz, L, dim)
    dep_parents = torch.tensor([[3, 3, 3, 4, -1], [2, 2, -1, 2, 2]])
    dep_dist = torch.randint(0, 5, (bsz, L, L)).long()
    pos_tags = torch.randint(0, 4, (bsz, L))
    deprel_tags = torch.randint(0, 4, (bsz, L))
    
    # 模拟 DataLoader 的预计算
    print("\n[1/3] 模拟 DataLoader 预计算特征...")
    precomputed = mock_precompute_features(
        dep_parents, dep_dist, pos_tags, deprel_tags,
        score_tables['pos_score_table'], 
        score_tables['deprel_score_table']
    )
    print(f"  ✓ 预计算完成，包含 {len(precomputed)} 个特征张量")
    
    # 计算 Soft Heads
    print("\n[2/3] 计算 Soft Heads...")
    soft_heads, head_weights, head_indices = computer(
        token_embeds=token_embeds,
        dep_parents=dep_parents,
        precomputed=precomputed
    )
    
    print(f"  ✓ Soft heads shape: {soft_heads.shape}")
    print(f"  ✓ Head weights shape: {head_weights.shape}")
    print(f"  ✓ Head indices shape: {head_indices.shape}")
    
    soft_heads_expand, head_weights_expand, head_indices_expand = \
    computer.expand_to_full_matrix(soft_heads=soft_heads, head_weights=head_weights,head_indices=head_indices)
    # 验证
    print("\n[3/3] 验证输出...")
    assert soft_heads_expand.shape == (bsz, L, L, dim), f"期望 {(bsz, L, L, dim)}，实际 {soft_heads_expand.shape}"
    assert head_weights_expand.shape == (bsz, L, L, 3), f"期望 {(bsz, L, L, 3)}，实际 {head_weights_expand.shape}"
    assert head_indices_expand.shape == (bsz, L, L, 3), f"期望 {(bsz, L, L, 3)}，实际 {head_indices_expand.shape}"
    
    # 检查权重和为 1
    for b in range(bsz):
        for l in range(L):
            for r in range(l, L):
                w_sum = head_weights_expand[b, l, r].sum().item()
                if w_sum > 0:  # 有候选的 span
                    assert abs(w_sum - 1.0) < 1e-5, f"Span [{l}, {r}] 权重和 = {w_sum}，期望 1.0"
    
    print("  ✓ 权重归一化正确")
    
    # 打印一个示例 span 的详细信息
    print("\n" + "=" * 60)
    print("示例：Batch 0, Span [1, 3]")
    print("=" * 60)
    b, l, r = 0, 1, 3
    print(f"Top-K indices: {head_indices_expand[b, l, r].tolist()}")
    print(f"Top-K weights: {head_weights_expand[b, l, r].tolist()}")
    print(f"Soft head norm: {soft_heads_expand[b, l, r].norm().item():.4f}")
    
    print("\n" + "=" * 60)
    print("✓ 所有测试通过！")
    print("=" * 60)
if __name__ == "__main__":
    test_soft_head_computer()