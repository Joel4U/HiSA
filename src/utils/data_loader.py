import json
import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
from src.utils.data_utils import bio_to_spans, compute_dep_distance, build_pos_score_table, build_deprel_score_table

class InputFeatures(object):
    def __init__(self, input_ids, attention_mask, orig_to_tok_index, postags=None, matrix=None, depheads=None, rel_ids=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.orig_to_tok_index = orig_to_tok_index
        self.postags= postags           # 词性标签序列
        self.matrix = matrix            # 实体矩阵标签
        self.depheads = depheads        # 依存头序列
        self.rel_ids = rel_ids          # 依存标签ID序列

class EntDataset(Dataset):
    def __init__(self, data, tokenizer, postag2id, deplabel2id, ent2id, model_name='bert-base-cased', 
                 max_span_width=32, max_len=512, is_train=True, json=False):
        self.tokenizer = tokenizer
        self.postag2id = postag2id
        self.ent2id = ent2id
        self.deplabel2id = deplabel2id
        self.max_len = max_len
        self.max_span_width = max_span_width
        self.train_stride = 1
        self.is_train = is_train
        if 'roberta' in model_name:
            self.add_prefix_space = True
            self.pad = self.tokenizer.pad_token_id
            self.cls = self.tokenizer.cls_token_id
            self.sep = self.tokenizer.sep_token_id
        elif 'deberta' in model_name:
            self.add_prefix_space = False
            self.pad = self.tokenizer.pad_token_id
            self.cls = self.tokenizer.bos_token_id
            self.sep = self.tokenizer.eos_token_id
        elif 'bert' in model_name:
            self.add_prefix_space = False
            self.pad = self.tokenizer.pad_token_id
            self.cls = self.tokenizer.cls_token_id
            self.sep = self.tokenizer.sep_token_id
        else:
            raise RuntimeError(f"Unsupported {model_name}")
        self.mlm_probability = 0.15
        # self.window = window
        if json:
            self.data = self.convert_json(data)
        else:
            self.data = self.convert_conllx(data)
        
        self.pos_score_table = build_pos_score_table(postag2id) if postag2id else None
        self.deprel_score_table = build_deprel_score_table(deplabel2id) if deplabel2id else None

    def __len__(self):
        return len(self.data)
    
    def get_new_ins(self, bpes, spans, attention_mask, orig_to_tok_index, postags=None, depheads=None, rel_ids=None):
            cur_word_idx = len(orig_to_tok_index)
            ent_target = []
            if self.is_train:
                # matrix = np.zeros((cur_word_idx, 2*self.window+1, len(self.ent2id)), dtype=np.int8)
                # 构造矩阵，使用 int8 类型，矩阵 shape 为 (cur_word_idx, cur_word_idx, num_labels)
                matrix = np.zeros((cur_word_idx, cur_word_idx, len(self.ent2id)), dtype=np.int8)
                for (s, e, t) in spans:
                    matrix[s, e, t] = 1
                    matrix[e, s, t] = 1
                    ent_target.append((s, e, t))
                # matrix = sparse.COO.from_numpy(matrix)
                assert len(bpes) <= self.max_len, f"超长了：{len(bpes)}"
                new_ins = InputFeatures(input_ids=bpes, attention_mask=attention_mask, orig_to_tok_index=orig_to_tok_index, postags=postags, matrix=matrix, depheads=depheads, rel_ids=rel_ids)
            else:
                for _ner in spans:
                    s, e, t = _ner
                    ent_target.append((s, e, t))
                assert len(bpes)<=self.max_len, len(bpes)
                new_ins = InputFeatures(input_ids=bpes, attention_mask=attention_mask, orig_to_tok_index=orig_to_tok_index, postags=postags, depheads=depheads, rel_ids=rel_ids)
            return new_ins

    def sequence_padding(self, inputs, length=None, value=0, seq_dims=1, mode='post', square_matrix=False):

        if square_matrix: # 对于方阵，找到最大的n
            if length is None:
                length = max([x.shape[0] for x in inputs])
            
            outputs = []
            for x in inputs:
                n = x.shape[0]
                pad_width = [(0, 0) for _ in x.shape]
                
                # 前两个维度使用相同的padding
                if mode == 'post':
                    pad_width[0] = (0, length - n)
                    pad_width[1] = (0, length - n)
                else:  # pre
                    pad_width[0] = (length - n, 0)
                    pad_width[1] = (length - n, 0)
                
                x = np.pad(x, pad_width, 'constant', constant_values=value)
                outputs.append(x)
            
            return np.array(outputs)

        if length is None:
            length = np.max([np.shape(x)[:seq_dims] for x in inputs], axis=0)
        elif not hasattr(length, '__getitem__'):
            length = [length]
        slices = [np.s_[:length[i]] for i in range(seq_dims)]
        slices = tuple(slices) if len(slices) > 1 else slices[0]
        pad_width = [(0, 0) for _ in np.shape(inputs[0])]

        outputs = []
        for x in inputs:
            x = x[slices]
            for i in range(seq_dims):
                if mode == 'post':
                    pad_width[i] = (0, length[i] - np.shape(x)[i])
                elif mode == 'pre':
                    pad_width[i] = (length[i] - np.shape(x)[i], 0)
                else:
                    raise ValueError('"mode" argument must be "post" or "pre".')
            x = np.pad(x, pad_width, 'constant', constant_values=value)
            outputs.append(x)

        return np.array(outputs)


    def convert_json(self, path):
        ins_lst = []
        word2bpes = {}
        
        with open(path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                if line.strip():
                    entry = json.loads(line.strip())
                    raw_words = entry.get("sentence", entry.get("text", []))
                    
                    # 处理依存关系
                    depheads = None
                    if "dephead" in entry:
                        depheads = [head - 1 for head in entry["dephead"]]
                    
                    deplabels = entry.get("deplabel", None)
                    
                    # 解析 NER 信息
                    raw_ents = []
                    entities = entry.get("entities", [])
                    for entity in entities:
                        start = entity["start"]
                        end = entity["end"]
                        entity_type = entity["type"]
                        
                        if start <= end and entity_type in self.ent2id:
                            raw_ents.append((start, end, self.ent2id[entity_type]))
                    
                    # 获取词性标签（如果有）
                    postags = entry.get("postags", None)
                    
                    # Tokenize
                    bpes = []
                    indexes = []
                    orig_to_tok_index = []
                    
                    for idx, word in enumerate(raw_words):
                        if word in word2bpes:
                            _bpes = word2bpes[word]
                        else:
                            _bpes = self.tokenizer.encode(
                                ' ' + word if self.add_prefix_space else word,
                                add_special_tokens=False
                            )
                            word2bpes[word] = _bpes
                        
                        orig_to_tok_index.append(len(bpes) + 1)  # +1 for [CLS]
                        indexes.extend([idx] * len(_bpes))
                        bpes.extend(_bpes)
                    
                    # 处理长句子
                    new_bpes = [[self.cls] + bpes[i:i+self.max_len-2] for i in range(0, len(bpes), self.max_len-self.train_stride-1)]
                    new_indexes = [indexes[i:i+self.max_len-2] for i in range(0, len(indexes), self.max_len-self.train_stride-1)]
                    
                    rel_ids = None
                    if deplabels:
                        rel_ids = [self.deplabel2id[rel] for rel in deplabels]
                    
                    for _bpes, _indexes in zip(new_bpes, new_indexes):
                        _bpes = _bpes + [self.sep]
                        attention_mask = [1] * len(_bpes)
                        
                        # 调整 orig_to_tok_index
                        offset = _indexes[0] if _indexes else 0
                        _orig_to_tok_index = [0] + [i - offset + 1 for i in orig_to_tok_index if offset <= i <= (_indexes[-1] if _indexes else 0)]
                        
                        # 调整 spans
                        spans = []
                        for s, e, t in raw_ents:
                            if _indexes and _indexes[0] <= s <= e <= _indexes[-1]:
                                spans.append((s - _indexes[0], e - _indexes[0], t))
                        
                        new_ins = self.get_new_ins(_bpes, spans, attention_mask, _orig_to_tok_index, postags=postags, depheads=depheads, rel_ids=rel_ids)
                        ins_lst.append(new_ins)
        return ins_lst
        
    def convert_conllx(self, path):
        ins_lst = []
        max_entity_width = 0 
        with open(path, 'r', encoding='utf-8') as f:         # 先读取所有行
            all_lines = f.readlines()
            raw_words, orig_to_tok_index, postags, raw_labels, depheads, deplabels  = [], [], [], [], [], []
            for line in tqdm(all_lines, desc="Processing file"):
                line = line.strip()
                # 跳过文档开始标记
                if line.startswith("-DOCSTART"):
                    continue
                # 遇到空行，处理当前句子
                if line == "" and len(raw_words) != 0:
                    # 将 BIO 标签转换为实体spans
                    raw_ents = bio_to_spans(self.ent2id, raw_labels)
                    for s, e, t in raw_ents:
                        current_width = e - s + 1
                        if current_width > max_entity_width:
                            max_entity_width = current_width # 计算并更新最大实体宽度
                    postag_ids = [self.postag2id[pos] for pos in postags]
                    rel_ids = [self.deplabel2id[rel] for rel in deplabels]
                    res = self.tokenizer.encode_plus(raw_words, is_split_into_words=True)
                    subword_idx2word_idx = res.word_ids(batch_index=0)
                    prev_word_idx = -1
                    for i, mapped_word_idx in enumerate(subword_idx2word_idx):
                        if mapped_word_idx is None:                         # cls and sep token
                            continue
                        if mapped_word_idx != prev_word_idx:
                            orig_to_tok_index.append(i)
                            prev_word_idx = mapped_word_idx
                    assert len(orig_to_tok_index) == len(raw_words)
                    # 直接使用原始的spans（不需要坐标转换）
                    spans = [(s, e, t) for s, e, t in raw_ents]
                    new_ins = self.get_new_ins(res['input_ids'], spans, res['attention_mask'], orig_to_tok_index, postags=postag_ids, depheads=depheads, rel_ids=rel_ids)
                    ins_lst.append(new_ins)
                    # 重置变量
                    raw_words, orig_to_tok_index, postags, raw_labels, depheads, deplabels  = [], [], [], [], [], []
                    continue
                elif line == "" and len(raw_words) == 0:
                    continue
                # 解析每一行数据
                ls = line.split('\t')  # 使用制表符分割
                if len(ls) >= 9:  # 确保有足够的列
                    word = ls[1]               # 词
                    postag = ls[3]             # 词性
                    head = int(ls[6]) - 1      # 依存头（转换为从0开始，root为-1）
                    dep_label = ls[7]          # 依存关系标签
                    ner_label = ls[-1]         # NER标签
                    
                    raw_words.append(word)
                    postags.append(postag)
                    raw_labels.append(ner_label)
                    depheads.append(head)
                    deplabels.append(dep_label)
            
            # 处理文件末尾的最后一个句子（如果存在）
            if len(raw_words) != 0:
                raw_ents = bio_to_spans(self.ent2id, raw_labels)
                for s, e, t in raw_ents:
                    current_width = e - s + 1
                    if current_width > max_entity_width:
                        max_entity_width = current_width # 计算并更新最大实体宽度
                postag_ids = [self.postag2id[pos] for pos in postags]
                rel_ids = [self.deplabel2id[rel] for rel in deplabels]
                res = self.tokenizer.encode_plus(raw_words, is_split_into_words=True)
                subword_idx2word_idx = res.word_ids(batch_index=0)
                prev_word_idx = -1
                for i, mapped_word_idx in enumerate(subword_idx2word_idx):
                    if mapped_word_idx is None:                         # cls and sep token
                        continue
                    if mapped_word_idx != prev_word_idx:
                        orig_to_tok_index.append(i)
                        prev_word_idx = mapped_word_idx
                assert len(orig_to_tok_index) == len(raw_words)

                spans = [(s, e, t) for s, e, t in raw_ents]
                new_ins = self.get_new_ins(res['input_ids'], spans, res['attention_mask'], orig_to_tok_index, postags=postag_ids, depheads=depheads, rel_ids=rel_ids)
                ins_lst.append(new_ins)
        # postag_set = set()
        # for ins in ins_lst:
        #     if ins.postags is not None:
        #         postag_set.update(ins.postags)
        # print(f"词性标签种类数: {len(postag_set)}")
        # print(f"所有词性标签: {sorted(postag_set)}")
        print(f"\n文件 {path} 中检测到的最大实体宽度是: {max_entity_width}")
        return ins_lst
    
    def _precompute_soft_head_features(self, batch_samples, batch_maxL, max_span_width=32):
        """预计算 Soft Head 所需的所有特征（在 CPU 上并行完成）
        Args:
            batch_samples: list of dict，每个样本的原始信息
            batch_maxL: batch 最大长度（padding 后）
            max_span_width: 最大实体宽度（默认32）, 只计算宽度 <= max_span_width 的 span
            使用紧凑存储 (bsz, maxL, maxW) 而非 (bsz, maxL, maxL)
        Returns:
            dict: 预计算的特征字典（全部在 CPU 上，便于后续 pin_memory 转移）
        """
        bsz = len(batch_samples)
        # W = min(max_span_width + 1, batch_maxL)  # 实际窗口大小
        # === 初始化输出张量（全部在 CPU 上）紧凑存储：(bsz, maxL, W) ===
        # neighbors_prefix[b, i, w] 表示 span [i, i+w] 的邻居累积计数
        neighbors_prefix = torch.zeros(bsz, batch_maxL, batch_maxL, dtype=torch.int32)
        deg_total = torch.zeros(bsz, batch_maxL, dtype=torch.int32)
        subtree_prefix = torch.zeros(bsz, batch_maxL, batch_maxL, dtype=torch.int32)
        dist_sum_prefix = torch.zeros(bsz, batch_maxL, batch_maxL, dtype=torch.float32)
        dist_cnt_prefix = torch.zeros(bsz, batch_maxL, batch_maxL, dtype=torch.int32)
        pos_scores = torch.zeros(bsz, batch_maxL, dtype=torch.float32)
        deprel_scores = torch.zeros(bsz, batch_maxL, dtype=torch.float32)
        
        # 逐样本计算
        for b, sample_info in enumerate(batch_samples):
            L = sample_info['length']
            parents = sample_info['depheads']  # list
            dist_matrix = sample_info['dist_matrix']  # list of list
            # ---------- 1. 邻接与度数 ----------
            neighbors = torch.zeros(L, L, dtype=torch.int32)
            for i, p in enumerate(parents):
                if 0 <= p < L:
                    neighbors[i, p] = 1
                    neighbors[p, i] = 1
        
            deg_total[b, :L] = neighbors.sum(dim=-1).to(torch.int32)
            # ========== 2. 邻接前缀和（限制范围 + 向量化） ==========
            neigh_prefix = neighbors.cumsum(dim=-1)  # (L, L)
            neighbors_prefix[b, :L, :L] = neigh_prefix
            # ---------- 3. 子树掩码 + 前缀和 ----------
            children = [[] for _ in range(L)]
            for u, p in enumerate(parents):
                if 0 <= p < L:
                    children[p].append(u)
            
            visited = [False] * L
            subtree_mask = torch.zeros(L, L, dtype=torch.uint8)
            
            def dfs(t: int) -> torch.Tensor:
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
            
            # DFS遍历所有根节点
            for u in range(L):
                if parents[u] < 0 and not visited[u]:
                    dfs(u)
            # 处理未连通节点
            for u in range(L):
                if not visited[u]:
                    dfs(u)
            # 子树前缀和（标准累积和）
            sub_prefix = subtree_mask.to(torch.int32).cumsum(dim=-1)  # (L, L)
            subtree_prefix[b, :L, :L] = sub_prefix
            # ---------- 4. 距离前缀和（限制范围 + 向量化） ----------
            dist_tensor = torch.tensor(dist_matrix, dtype=torch.float32)  # (L, L)
            valid = (dist_tensor < 999).float()
            # 距离和累积
            dist_sum = (dist_tensor * valid).cumsum(dim=-1)  # (L, L)
            dist_sum_prefix[b, :L, :L] = dist_sum
            # 有效节点计数累积
            dist_cnt = valid.to(torch.int32).cumsum(dim=-1)  # (L, L)
            dist_cnt_prefix[b, :L, :L] = dist_cnt
            # ---------- 5. POS 和 DEPREL 先验分数 ----------
            if self.pos_score_table is not None:
                pos_ids = sample_info['postags']
                for t in range(min(L, len(pos_ids))):
                    pid = pos_ids[t]
                    if 0 <= pid < len(self.pos_score_table):
                        pos_scores[b, t] = self.pos_score_table[pid]
            
            if self.deprel_score_table is not None:
                rel_ids = sample_info['rel_ids']
                for t in range(min(L, len(rel_ids))):
                    rid = rel_ids[t]
                    if 0 <= rid < len(self.deprel_score_table):
                        deprel_scores[b, t] = self.deprel_score_table[rid]
        return {
            'neighbors_prefix': neighbors_prefix,      # (bsz, maxL, W) int32
            'deg_total': deg_total,                    # (bsz, maxL) int32
            'subtree_prefix': subtree_prefix,          # (bsz, maxL, W) int32
            'dist_sum_prefix': dist_sum_prefix,        # (bsz, maxL, W) float32
            'dist_cnt_prefix': dist_cnt_prefix,        # (bsz, maxL, W) int32
            'pos_scores': pos_scores,                  # (bsz, maxL) float32
            'deprel_scores': deprel_scores,            # (bsz, maxL) float32
            'max_span_width': max_span_width,          # 标记最大宽度
        }

    def collate(self, examples):
        if self.is_train:
            batch_input_id, batch_input_mask, batch_orig_to_tok_index, batch_heads, batch_rels, batch_samples, batch_matrix = [], [], [], [], [], [], []
            batch_samples = []
            for item in examples:
                batch_input_id.append(item.input_ids)
                batch_input_mask.append(item.attention_mask)
                batch_orig_to_tok_index.append(item.orig_to_tok_index)
                batch_matrix.append(item.matrix)
                batch_heads.append(item.depheads)
                batch_rels.append(item.rel_ids)

                # ===== 计算依存距离矩阵 =====
                dist_matrix = compute_dep_distance(item.depheads)
                batch_samples.append({'depheads': item.depheads,'dist_matrix': dist_matrix,
                                      'postags': item.postags, 'rel_ids': item.rel_ids, 'length': len(item.depheads)})

            batch_input_ids = torch.tensor(self.sequence_padding(batch_input_id, value=0)).long()
            batch_input_mask = torch.tensor(self.sequence_padding(batch_input_mask, value=0)).long() 
            batch_orig_to_tok_index = torch.tensor(self.sequence_padding(batch_orig_to_tok_index, value=-1)).long() 
            batch_heads = torch.tensor(self.sequence_padding(batch_heads, value=-2)).long()
            batch_rels = torch.tensor(self.sequence_padding(batch_rels, value=len(self.deplabel2id))).long()
            batch_labels = torch.tensor(self.sequence_padding(batch_matrix, square_matrix=True)).long()
            # ===== 预计算 Soft Head 所需特征 =====
            batch_maxL = batch_heads.shape[1]
            precomputed = self._precompute_soft_head_features(batch_samples, batch_maxL, self.max_span_width)

            return batch_input_ids, batch_input_mask, batch_orig_to_tok_index, batch_heads, batch_rels, precomputed, batch_labels
        else:
            batch_input_id, batch_input_mask, batch_orig_to_tok_index, batch_heads, batch_rels, batch_samples = [], [], [], [], [], []
            for item in examples:
                batch_input_id.append(item.input_ids)
                batch_input_mask.append(item.attention_mask)
                batch_orig_to_tok_index.append(item.orig_to_tok_index)
                batch_heads.append(item.depheads)
                batch_rels.append(item.rel_ids) 
                # ===== 计算依存距离矩阵 =====
                dist_matrix = compute_dep_distance(item.depheads)
                batch_samples.append({'depheads': item.depheads,'dist_matrix': dist_matrix,
                                      'postags': item.postags, 'rel_ids': item.rel_ids, 'length': len(item.depheads)})
            batch_input_ids = torch.tensor(self.sequence_padding(batch_input_id, value=0)).long()
            batch_input_mask = torch.tensor(self.sequence_padding(batch_input_mask, value=0)).long() 
            batch_orig_to_tok_index = torch.tensor(self.sequence_padding(batch_orig_to_tok_index, value=0)).long() 
            batch_heads = torch.tensor(self.sequence_padding(batch_heads, value=-2)).long()
            batch_rels = torch.tensor(self.sequence_padding(batch_rels, value=len(self.deplabel2id))).long()
            # ===== 预计算 Soft Head 所需特征 =====
            batch_maxL = batch_heads.shape[1]
            precomputed = self._precompute_soft_head_features(batch_samples,  batch_maxL, self.max_span_width)
            
            return batch_input_ids, batch_input_mask, batch_orig_to_tok_index, batch_heads, batch_rels, precomputed

    def __getitem__(self, index):
        item = self.data[index]
        return item
    