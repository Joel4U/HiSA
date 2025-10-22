from torch import nn
import torch
import torch.nn.functional as F
from .multi_head_biaffine import MultiHeadBiaffine
from .embedder import PLMEmbedder
from .span_softhead import SoftHeadComputer
from .span_neighbor import SpanNeighborBuilder
from .span_attn import SpanScoreAttnStack
from .span_attn_gates import SpanDistanceBias, build_multihead_gates, pack_span_maps


class SpanAttn(nn.Module):
    def __init__(self, bert_name , num_rel_tag, num_ner_tag, hidden_dim=200, biaffine_size=200,
                 size_embed_dim=0, logit_drop=0, n_head=4, num_span_attn_layers=3, max_span_width=32):
        super(SpanAttn, self).__init__()
        assert hidden_dim % n_head == 0, f"hidden_dim ({hidden_dim}) must be divisible by n_head ({n_head})"
        # ==================== 基础模块 ====================
        self.embedder = PLMEmbedder(encoder_name=bert_name)
        # self.rel_embedding = nn.Embedding(num_rel_tag + 1, embedding_dim=25, padding_idx=-2) # 51个位置：50个关系 + 1个padding
        emb_dim = self.embedder.get_output_dim()
        # ==================== Size Embedding ====================
        if size_embed_dim!=0:
            n_pos = 50
            self.size_embedding = torch.nn.Embedding(n_pos, size_embed_dim)
            _span_size_ids = torch.arange(512) - torch.arange(512).unsqueeze(-1)
            _span_size_ids.masked_fill_(_span_size_ids < -n_pos/2, -n_pos/2)
            _span_size_ids = _span_size_ids.masked_fill(_span_size_ids >= n_pos/2, n_pos/2-1) + n_pos/2
            self.register_buffer('span_size_ids', _span_size_ids.long())
            hsz = biaffine_size*2 + size_embed_dim + 2
        else:
            hsz = biaffine_size*2+2
        # ==================== Biaffine 通路 ====================
        self.head_mlp = nn.Sequential(nn.Dropout(0.4), nn.Linear(emb_dim, biaffine_size), nn.GELU())
        self.tail_mlp = nn.Sequential(nn.Dropout(0.4), nn.Linear(emb_dim, biaffine_size), nn.GELU())
        self.dropout = nn.Dropout(0.4)
        if n_head>0:
            self.multi_head_biaffine = MultiHeadBiaffine(biaffine_size, hidden_dim, n_head=n_head)
        else:
            self.U = nn.Parameter(torch.randn(hidden_dim, biaffine_size, biaffine_size))
            torch.nn.init.xavier_normal_(self.U.data)
        # ==================== W 投影矩阵 ====================
        self.W = torch.nn.Parameter(torch.empty(hidden_dim, hsz))
        torch.nn.init.xavier_normal_(self.W.data)
        # ==================== Soft Heads 分支 ====================
        self.use_soft_heads = True
        self.get_softheads = SoftHeadComputer(w_coverage=2.0, w_degree=1.5, w_pos=0.8, w_deprel=0.5, w_medoid=0.5, w_direction=0.3, 
                                              temperature=0.7, k=3, learnable=False)  # 如果需要学习权重，设为 True
        self.softhead_proj = nn.Linear(emb_dim, hsz, bias=False)
        self.softhead_fuse_ln = nn.LayerNorm(hsz)
        self.softhead_fuse_dropout = nn.Dropout(logit_drop)
        self.softhead_alpha = nn.Parameter(torch.tensor(0.0))  # 可学习缩放，初始 0 更稳
        # ==================== Span Adjacency ====================
        # K:每个 span 的邻居数; Ktok:每个 span 的 soft_head 数量; d:依存距离阈值; gamma:门控衰减系数; 
        K_nei = 10
        self.get_spanadj = SpanNeighborBuilder(K=K_nei, Ktok=3, d=2, gamma=1.0, max_width=max_span_width)
        # ==================== Span Attention ====================
        self.span_attn = SpanScoreAttnStack(K=hidden_dim, num_layers=num_span_attn_layers, num_heads=n_head)
        # 可选: 在 head1 上叠加可学习的距离偏置（替代或增强乘法 gate）
        # self.dist_bias_layer = SpanDistanceBias(num_heads=n_head, head_idx=1, lambda_bias=0.15)
        # ==================== 下游分类器 ====================
        self.down_fc = nn.Linear(hidden_dim, num_ner_tag)

    def forward(self, input_ids, attention_mask, orig_to_tok_index, heads, rels, precomputed, matrix=None): 
        word_rep = self.embedder(input_ids, orig_to_tok_index, attention_mask) # (bsz, L, dim)
        # rel_emb = self.rel_embedding(rels)
        # word_rep = torch.cat((word_rep, rel_emb), dim=-1).contiguous()
        # ==================== 2. Biaffine 特征通路 ====================
        head_state = self.head_mlp(word_rep)
        tail_state = self.tail_mlp(word_rep)
        if hasattr(self, 'U'):
            biaf_scores = torch.einsum('bxi, oij, byj -> boxy', head_state, self.U, tail_state)
        else:
            biaf_scores = self.multi_head_biaffine(head_state, tail_state)
        # 构造初始 span 表示
        head_state = torch.cat([head_state, torch.ones_like(head_state[..., :1])], dim=-1)
        tail_state = torch.cat([tail_state, torch.ones_like(tail_state[..., :1])], dim=-1)
        affined_cat = torch.cat([self.dropout(head_state).unsqueeze(2).expand(-1, -1, tail_state.size(1), -1),
                                 self.dropout(tail_state).unsqueeze(1).expand(-1, head_state.size(1), -1, -1)], dim=-1) # (B,L,L,H)

        if hasattr(self, 'size_embedding'):
            size_embedded = self.size_embedding(self.span_size_ids[:word_rep.size(1), :word_rep.size(1)])
            affined_cat = torch.cat([affined_cat,
                                     self.dropout(size_embedded).unsqueeze(0).expand(word_rep.size(0), -1, -1, -1)], dim=-1) # (B, L, L, hsz)
        # ==================== 3. Soft Heads 特征注入 ====================
        # with timer.section("3a.softheads_compute"):
        max_width_soft_heads, max_width_head_weights, max_width_head_indices = self.get_softheads(word_rep, heads, precomputed)
        # timer.report(prefix=f"3a.softheads_compute", reset=True)
        # head_weights:(B, L, L, Ktok); head_indices:(B, L, L, Ktok);
        soft_heads, head_weights, head_indices = self.get_softheads.expand_to_full_matrix(max_width_soft_heads, max_width_head_weights, max_width_head_indices)
        if getattr(self, 'use_soft_heads', False):
            soft_feat = self.softhead_proj(soft_heads)  # (B, L, L, hsz)
            affined_cat = self.softhead_fuse_ln(affined_cat + self.softhead_alpha * self.softhead_fuse_dropout(soft_feat))  # (B, L, L, hsz)
        
        softhead_scores = torch.einsum('bmnh,kh->bkmn', affined_cat, self.W)    # (bsz, hidden_dim, L, L)
        scores = softhead_scores + biaf_scores
        # ==================== 4. 构图 ====================
        dist = self.get_spanadj._recover_dist_from_prefix(precomputed['dist_sum_prefix'], precomputed['dist_cnt_prefix'])   # dist:(B,L,L)
        # with timer.section("4b.build_neighbors"):
        N_idx, N_mask, span_maps = self.get_spanadj(head_indices, head_weights, dist)
        # timer.report(prefix=f"4b.build_neighbors", reset=True)
        #   N_idx: (B, S, K) - S 为有效 span 数量（动态）
        #   N_mask: (B, S, K)
        #   span_maps: List[(span2id, id2lr)] 长度为 B
        # ==================== 5. Span Attention 基于 soft head 重叠门控 依存先验 ====================
        Smax = N_idx.size(1)
        id2lr_pad, S_row_mask = pack_span_maps(span_maps, Smax=Smax, device=word_rep.device)
        # # with timer.section(f"5.attn_layer"):
        head_gate = build_multihead_gates(id2lr_pad=id2lr_pad, N_idx=N_idx, N_mask=N_mask, S_row_mask=S_row_mask,
                    head_indices=head_indices, head_weights=head_weights, num_heads=self.span_attn.num_heads, 
                    use_syntax_prior=True,               # head0 使用句法先验
                    use_distance_prior=True,             # head1 使用距离先验
                    gamma=0.3,                           # 距离衰减系数
                    d0=1,                                # 距离偏置
                    Ktok_limit=3)                        # (B, S, h, Knei)
        # head_bias = self.dist_bias_layer(id2lr_pad, N_idx, N_mask)  # (B, S, h, Knei)
        scores = self.span_attn(scores=scores, N_idx=N_idx, N_mask=N_mask, id2lr_pad=id2lr_pad, S_row_mask=S_row_mask, 
                                head_gate=head_gate, head_bias=None)  # (B, hidden_dim, L, L)
        # timer.report(prefix=f"5.attn_layer", reset=True)
        # ==================== 6. 下游分类 ====================
        scores = self.down_fc(scores.permute(0, 2, 3, 1))               # (B,L,L,num_ner_tag)
        if self.training:
            assert scores.size(-1) == matrix.size(-1)
            flat_scores = scores.reshape(-1)
            flat_matrix = matrix.reshape(-1)
            mask = flat_matrix.ne(-100).float().view(input_ids.size(0), -1)
            flat_loss = F.binary_cross_entropy_with_logits(flat_scores, flat_matrix.float(), reduction='none')
            loss = ((flat_loss.view(input_ids.size(0), -1)*mask).sum(dim=-1)).mean()
            return loss
        return scores