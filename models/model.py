from torch import nn
import torch
import torch.nn.functional as F
from .cnn import MaskCNN
from .gsda import GlobalSDAware
from .multi_head_biaffine import MultiHeadBiaffine
from .embedder import PLMEmbedder
# from .fusion import CGAFusion
from .gate_fusion import UnifiedSpanFusion

class CNNNer(nn.Module):
    def __init__(self, bert_name , num_rel_tag, num_ner_tag, cnn_dim=200, biaffine_size=200,
                 size_embed_dim=0, logit_drop=0, n_head=4, cnn_depth=3, attn_dropout=0.15, use_tri_bias=True):
        super(CNNNer, self).__init__()
        self.embedder = PLMEmbedder(encoder_name=bert_name)
        # self.rel_embedding = nn.Embedding(num_rel_tag + 1, embedding_dim=25, padding_idx=-2) # 51个位置：50个关系 + 1个padding
        hidden_size = self.embedder.get_output_dim()

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
        
        self.cnn_dim = cnn_dim
        biaffine_input_size = hidden_size

        self.head_mlp = nn.Sequential(nn.Dropout(0.4), nn.Linear(biaffine_input_size, biaffine_size), 
                                      nn.GELU() # nn.LeakyReLU(),
                                      )
        self.tail_mlp = nn.Sequential(nn.Dropout(0.4), nn.Linear(biaffine_input_size, biaffine_size), 
                                      nn.GELU() # nn.LeakyReLU(),
                                      )

        self.dropout = nn.Dropout(0.4)
        if n_head>0:
            self.multi_head_biaffine = MultiHeadBiaffine(biaffine_size, cnn_dim, n_head=n_head)
        else:
            self.U = nn.Parameter(torch.randn(cnn_dim, biaffine_size, biaffine_size))
            torch.nn.init.xavier_normal_(self.U.data)
        self.W = torch.nn.Parameter(torch.empty(cnn_dim, hsz))
        torch.nn.init.xavier_normal_(self.W.data)
        if cnn_depth>0:
            self.cnn = MaskCNN(cnn_dim, cnn_dim, kernel_size=3, depth=cnn_depth)

        self.down_fc = nn.Linear(cnn_dim, num_ner_tag)
        self.logit_drop = logit_drop
        self.num_ner_tag = num_ner_tag
        self.gsda = GlobalSDAware(channel=cnn_dim, use_dep_mask=True)
        self.fusion = UnifiedSpanFusion(hidden_dim=cnn_dim, 
            fusion_type="adaptive",  # 自动学习CNN和GSDA的重要性 # 高性能需求 多视角融合 multihead # simple
            dropout=0.1)

    def forward(self, input_ids, attention_mask, orig_to_tok_index, rels, head_mat=None, matrix=None): 
        word_rep = self.embedder(input_ids, orig_to_tok_index, attention_mask)
        # rel_emb = self.rel_embedding(rels)
        # word_rep = torch.cat((word_rep, rel_emb), dim=-1).contiguous()

        head_state = self.head_mlp(word_rep)
        tail_state = self.tail_mlp(word_rep)

        if hasattr(self, 'U'):
            scores1 = torch.einsum('bxi, oij, byj -> boxy', head_state, self.U, tail_state)
        else:
            scores1 = self.multi_head_biaffine(head_state, tail_state)

        head_state = torch.cat([head_state, torch.ones_like(head_state[..., :1])], dim=-1)
        tail_state = torch.cat([tail_state, torch.ones_like(tail_state[..., :1])], dim=-1)
        affined_cat = torch.cat([self.dropout(head_state).unsqueeze(2).expand(-1, -1, tail_state.size(1), -1),
                                 self.dropout(tail_state).unsqueeze(1).expand(-1, head_state.size(1), -1, -1)], dim=-1)

        if hasattr(self, 'size_embedding'):
            size_embedded = self.size_embedding(self.span_size_ids[:word_rep.size(1), :word_rep.size(1)])
            affined_cat = torch.cat([affined_cat,
                                     self.dropout(size_embedded).unsqueeze(0).expand(word_rep.size(0), -1, -1, -1)], dim=-1)
        scores2 = torch.einsum('bmnh,kh->bkmn', affined_cat, self.W)  # bsz x dim x L x L
        scores = scores2 + scores1
        g_scores = self.gsda(scores, head_dep_mat=head_mat)
        # wo fusion + lsde
        scores = g_scores + scores
        # CNN
        if hasattr(self, 'cnn'):
            lengths = (orig_to_tok_index != 0).sum(dim=-1)
            mask = torch.arange(lengths.max(), device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1) # bsz x length x length
            mask = mask[:, None] * mask.unsqueeze(-1)
            pad_mask = mask[:, None].eq(0)
            u_scores = scores.masked_fill(pad_mask, 0)
            if self.logit_drop != 0:
                u_scores = F.dropout(u_scores, p=self.logit_drop, training=self.training)
            # bsz, num_label, max_len, max_len = u_scores.size()
            u_scores = self.cnn(u_scores, pad_mask)
            scores = self.fusion(u_scores, g_scores, scores)
            # wo fusion
            # scores = u_scores + g_scores + scores
            # wo fusion + gsda
            # scores = u_scores + scores
        scores = self.down_fc(scores.permute(0, 2, 3, 1))
        if self.training:
            assert scores.size(-1) == matrix.size(-1)
            flat_scores = scores.reshape(-1)
            flat_matrix = matrix.reshape(-1)
            mask = flat_matrix.ne(-100).float().view(input_ids.size(0), -1)
            flat_loss = F.binary_cross_entropy_with_logits(flat_scores, flat_matrix.float(), reduction='none')
            loss = ((flat_loss.view(input_ids.size(0), -1)*mask).sum(dim=-1)).mean()
            return loss
        return scores
