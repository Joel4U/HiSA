import json
import torch
from torch.utils.data import Dataset
import numpy as np
import random
from tqdm import tqdm
from utils.data_utils import bio_to_spans
from utils.span_head_extractor import SpanHeadExtractor

class InputFeatures(object):
    def __init__(self, input_ids, attention_mask, orig_to_tok_index, matrix=None, rel_ids=None, spanhead_matrix=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.orig_to_tok_index = orig_to_tok_index
        self.matrix = matrix
        # self.depheads = depheads      # 完整的依存头序列
        self.rel_ids = rel_ids          # 依存标签ID序列
        self.spanhead_matrix = spanhead_matrix      # span的head词矩阵

class EntDataset(Dataset):
    def __init__(self, data, tokenizer, deplabel2id, ent2id, model_name='bert-base-cased', max_len=512, is_train=True, json=False):
        self.tokenizer = tokenizer
        self.ent2id = ent2id
        self.deplabel2id = deplabel2id
        self.max_len = max_len
        self.train_stride = 1
        self.is_train = is_train
        if 'roberta' in model_name:
            self.add_prefix_space = True
            self.cls = self.tokenizer.cls_token_id
            self.sep = self.tokenizer.sep_token_id
        elif 'deberta' in model_name:
            self.add_prefix_space = False
            self.cls = self.tokenizer.bos_token_id
            self.sep = self.tokenizer.eos_token_id
        elif 'bert' in model_name:
            self.add_prefix_space = False
            self.cls = self.tokenizer.cls_token_id
            self.sep = self.tokenizer.sep_token_id
        else:
            raise RuntimeError(f"Unsupported {model_name}")
        self.span_head_extractor = SpanHeadExtractor(model_name=model_name)
        self.mlm_probability = 0.15
        # self.window = window
        if json:
            self.data = self.convert_json(data)
        else:
            self.data = self.convert_conllx(data)

    def __len__(self):
        return len(self.data)
    
    def get_new_ins(self, bpes, spans, attention_mask, orig_to_tok_index, rel_ids=None, spanhead_matrix=None):
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
                new_ins = InputFeatures(input_ids=bpes, attention_mask=attention_mask, orig_to_tok_index=orig_to_tok_index, matrix=matrix, rel_ids=rel_ids, spanhead_matrix=spanhead_matrix)
            else:
                for _ner in spans:
                    s, e, t = _ner
                    ent_target.append((s, e, t))
                assert len(bpes)<=self.max_len, len(bpes)
                new_ins = InputFeatures(input_ids=bpes, attention_mask=attention_mask, orig_to_tok_index=orig_to_tok_index, rel_ids=rel_ids, spanhead_matrix=spanhead_matrix)
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
        # 修改的部分开始
        with open(path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                if line.strip():                        # 忽略空行
                    entry = json.loads(line.strip())    # 逐行解析 JSON
                    # 适配字段名称
                    raw_words = entry.get("sentence", entry.get("text", []))  # 兼容两种字段名
                    # 处理依存关系（如果存在）
                    depheads, deplabels = None, None
                    if "dephead" in entry:
                        depheads = [head - 1 for head in entry["dephead"]]  # 转换为 root is -1 not 0
                    if "deplabel" in entry:
                        deplabels = entry["deplabel"]

                    # 解析 NER 信息 - 新格式
                    raw_ents = []
                    entities = entry.get("entities", [])
                    for entity in entities:
                        start = entity["start"]         # 实体起始位置
                        end = entity["end"]             # 实体结束位置
                        entity_type = entity["type"]    # 实体类型
                        
                        if start <= end and entity_type in self.ent2id:
                            raw_ents.append((start, end, self.ent2id[entity_type]))
                    # 处理关系信息（如果存在）
                    # relations = entry.get("relations", [])

                    bpes = []
                    indexes = []
                    cand_indexes = []
                    for idx, word in enumerate(raw_words, start=0):
                        if word in word2bpes:
                            _bpes = word2bpes[word]
                        else:
                            _bpes = self.tokenizer.encode(' '+word if self.add_prefix_space else word,
                                                        add_special_tokens=False)
                            word2bpes[word] = _bpes
                        cand_indexes.append(list(range(len(bpes)+1, len(bpes)+len(_bpes)+1)))
                        indexes.extend([idx]*len(_bpes))
                        bpes.extend(_bpes)
                    
                    new_bpes = [[self.cls] + bpes[i:i+self.max_len-2] for i in range(0, len(bpes), self.max_len-self.train_stride-1)]
                    new_indexes = [indexes[i:i+self.max_len-2] for i in range(0, len(indexes), self.max_len-self.train_stride-1)]
                    rel_ids = [self.deplabel2id[rel] for rel in deplabels]
                    for _bpes, _indexes in zip(new_bpes, new_indexes):
                        spans = []
                        offset = _indexes[0]        # 短句子，不超过self.max_len的句子，offset 将是 0
                        for s, e, t in raw_ents:
                            if _indexes[0] <= s <= e <= _indexes[-1]:
                                spans += [(s-_indexes[0], e-_indexes[0], t)]
                        
                        if self.is_train:
                            _indexes = [0] + [i - offset + 1 for i in _indexes]
                            new_ins = self.get_new_ins(_bpes, spans, _indexes, orig_to_tok_index, cand_indexes, depheads=depheads, rel_ids=rel_ids)
                            ins_lst.append(new_ins)
                        else:
                            _indexes = [0] + [i - offset + 1 for i in _indexes]
                            new_ins = self.get_new_ins(_bpes, spans, _indexes, orig_to_tok_index, depheads=depheads, rel_ids=rel_ids)
                            ins_lst.append(new_ins)
        return ins_lst
        
    def convert_conllx(self, path):
        ins_lst = []
        with open(path, 'r', encoding='utf-8') as f:         # 先读取所有行
            all_lines = f.readlines()
            raw_words, orig_to_tok_index, raw_labels, depheads, deplabels  = [], [], [], [], []
            for line in tqdm(all_lines, desc="Processing file"):
                original_line = line  # 保存原始行
                line = line.strip()
                # 跳过文档开始标记
                if line.startswith("-DOCSTART"):
                    continue
                # 遇到空行，处理当前句子
                if line == "" and len(raw_words) != 0:
                    # 将 BIO 标签转换为实体spans
                    raw_ents = bio_to_spans(self.ent2id, raw_labels)
                    rel_ids = [self.deplabel2id[rel] for rel in deplabels]
                    # 计算注意力权重
                    attention_weights = self.span_head_extractor.get_attention_weights(raw_words)
                    # 计算span head矩阵
                    span_head_matrix = self.span_head_extractor.extract_span_heads(raw_words, depheads, attention_weights)
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
                    new_ins = self.get_new_ins(res['input_ids'], spans, res['attention_mask'], orig_to_tok_index, rel_ids=rel_ids, spanhead_matrix=span_head_matrix)
                    ins_lst.append(new_ins)
                    # 重置变量
                    raw_words, orig_to_tok_index, raw_labels, depheads, deplabels, span_head_matrix  = [], [], [], [], [], []
                    continue
                elif line == "" and len(raw_words) == 0:
                    continue
                # 解析每一行数据
                ls = line.split('\t')  # 使用制表符分割
                if len(ls) >= 9:  # 确保有足够的列
                    word = ls[1]               # 词
                    head = int(ls[6]) - 1      # 依存头（转换为从0开始，root为-1）
                    dep_label = ls[7]          # 依存关系标签
                    ner_label = ls[-1]         # NER标签
                    
                    raw_words.append(word)
                    raw_labels.append(ner_label)
                    depheads.append(head)
                    deplabels.append(dep_label)
            
            # 处理文件末尾的最后一个句子（如果存在）
            if len(raw_words) != 0:
                raw_ents = bio_to_spans(self.ent2id, raw_labels)
                rel_ids = [self.deplabel2id[rel] for rel in deplabels]
                # 计算注意力权重
                attention_weights = self.span_head_extractor.get_attention_weights(raw_words)
                # 计算span head矩阵
                span_head_matrix = self.span_head_extractor.extract_span_heads(raw_words, depheads, attention_weights)
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
                new_ins = self.get_new_ins(res['input_ids'], spans, res['attention_mask'], orig_to_tok_index, rel_ids=rel_ids, head_matrix=span_head_matrix)
                ins_lst.append(new_ins)
        return ins_lst
    
    def collate(self, examples):
        if self.is_train:
            batch_input_id, batch_input_mask, batch_orig_to_tok_index, batch_spanhead_matrix, batch_rels, batch_matrix = [], [], [], [], [], []
            for item in examples:
                batch_input_id.append(item.input_ids)
                batch_input_mask.append(item.attention_mask)
                batch_orig_to_tok_index.append(item.orig_to_tok_index)
                batch_matrix.append(item.matrix)
                # batch_heads.append(item.depheads)
                batch_spanhead_matrix.append(item.spanhead_matrix)
                batch_rels.append(item.rel_ids) 

            batch_input_ids = torch.tensor(self.sequence_padding(batch_input_id, value=self.tokenizer.pad_token_id)).long()
            batch_input_mask = torch.tensor(self.sequence_padding(batch_input_mask, value=0)).long() 
            batch_orig_to_tok_index = torch.tensor(self.sequence_padding(batch_orig_to_tok_index, value=0)).long() 
            batch_labels = torch.tensor(self.sequence_padding(batch_matrix, square_matrix=True)).long()
            # batch_heads = torch.tensor(self.sequence_padding(batch_heads, value=-2)).long()
            batch_spanhead_matrix = torch.tensor(self.sequence_padding(batch_spanhead_matrix, value=-1, square_matrix=True)).long()
            batch_rels = torch.tensor(self.sequence_padding(batch_rels, value=len(self.deplabel2id))).long()
            return batch_input_ids, batch_input_mask, batch_orig_to_tok_index, batch_spanhead_matrix, batch_rels, batch_labels
        else:
            batch_input_id, batch_input_mask, batch_orig_to_tok_index, batch_spanhead_matrix, batch_rels = [], [], [], [], []
            for item in examples:
                batch_input_id.append(item.input_ids)
                batch_input_mask.append(item.attention_mask)
                batch_orig_to_tok_index.append(item.orig_to_tok_index)
                # batch_heads.append(item.depheads)
                batch_spanhead_matrix.append(item.spanhead_matrix)
                batch_rels.append(item.rel_ids) 

            batch_input_ids = torch.tensor(self.sequence_padding(batch_input_id, value=self.tokenizer.pad_token_id)).long()
            batch_input_mask = torch.tensor(self.sequence_padding(batch_input_mask, value=0)).long() 
            batch_orig_to_tok_index = torch.tensor(self.sequence_padding(batch_orig_to_tok_index, value=0)).long() 
            # batch_heads = torch.tensor(self.sequence_padding(batch_heads, value=-2)).long()
            batch_spanhead_matrix = torch.tensor(self.sequence_padding(batch_spanhead_matrix, value=-1, square_matrix=True)).long()
            batch_rels = torch.tensor(self.sequence_padding(batch_rels, value=len(self.deplabel2id))).long()

            return batch_input_ids, batch_input_mask, batch_orig_to_tok_index, batch_spanhead_matrix, batch_rels

    def __getitem__(self, index):
        item = self.data[index]
        return item
    