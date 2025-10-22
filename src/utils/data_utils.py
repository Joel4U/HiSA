import json
import torch
import numpy as np
from typing import Dict

def move_to_device(obj, device, non_blocking=True):
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=non_blocking)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device, non_blocking) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(move_to_device(v, device, non_blocking) for v in obj)
    return obj 

def bio_to_spans(ent2id, bio_labels):
    """
    将BIO标签序列转换为实体spans
    """
    spans = []
    current_entity = None
    
    for i, label in enumerate(bio_labels):
        if label == 'O':
            if current_entity is not None:
                # 结束当前实体
                start, entity_type = current_entity
                if entity_type in ent2id:
                    spans.append((start, i-1, ent2id[entity_type]))
                current_entity = None
        elif label.startswith('B-'):
            # 开始新实体
            if current_entity is not None:
                # 先结束之前的实体
                start, entity_type = current_entity
                if entity_type in ent2id:
                    spans.append((start, i-1, ent2id[entity_type]))
            
            entity_type = label[2:]  # 去掉 'B-' 前缀
            current_entity = (i, entity_type)
        elif label.startswith('I-'):
            # 继续当前实体
            entity_type = label[2:]  # 去掉 'I-' 前缀
            if current_entity is None:
                # 如果没有对应的B标签，将I标签视为B标签
                current_entity = (i, entity_type)
            elif current_entity[1] != entity_type:
                # 实体类型不匹配，结束之前的实体，开始新实体
                start, prev_entity_type = current_entity
                if prev_entity_type in ent2id:
                    spans.append((start, i-1, ent2id[prev_entity_type]))
                current_entity = (i, entity_type)
    
    # 处理最后一个实体
    if current_entity is not None:
        start, entity_type = current_entity
        if entity_type in ent2id:
            spans.append((start, len(bio_labels)-1, ent2id[entity_type]))
    
    return spans

def compute_dep_distance(parents):
    """从父节点数组计算依存树距离矩阵
    Args:
        parents: 可以是 list, np.ndarray 或 torch.Tensor
    Returns:
        dist: np.ndarray (L, L) - 距离矩阵
    """
    # 统一转换为 numpy array
    if isinstance(parents, torch.Tensor):
        parents = parents.cpu().numpy()
    elif isinstance(parents, list):
        parents = np.array(parents)
    
    L = len(parents)
    dist = np.full((L, L), L, dtype=np.int32)
    
    # Floyd-Warshall 算法
    # 初始化直接边
    for i in range(L):
        dist[i, i] = 0
        if parents[i] >= 0:
            p = int(parents[i])
            dist[i, p] = 1
            dist[p, i] = 1
    
    # 迭代更新最短路径
    for k in range(L):
        for i in range(L):
            for j in range(L):
                dist[i, j] = min(dist[i, j], dist[i, k] + dist[k, j])
    
    return dist

# ===== 先验分数字典定义 =====
def get_pos_prior() -> Dict[str, float]:
    """Penn Treebank POS 标签先验分数"""
    return {
        # 名词类 (高分 - 倾向作为 head)
        "NN": 0.8, "NNS": 0.8, "NNP": 1.0, "NNPS": 1.0, "NN|SYM": 0.7,
        # 形容词类 (中等分)
        "JJ": 0.3, "JJR": 0.3, "JJS": 0.3,
        # 动词类 (中低分)
        "VB": 0.2, "VBD": 0.2, "VBG": 0.2, "VBN": 0.2, "VBP": 0.2, "VBZ": 0.2,
        # 副词类 (低分)
        "RB": 0.1, "RBR": 0.1, "RBS": 0.1, "WRB": -0.3,
        # 数词
        "CD": 0.1,
        # 代词
        "PRP": 0.0, "PRP$": 0.0, "WP": 0.0, "WP$": 0.0,
        # 限定词 (负分 - 不倾向作为 head)
        "DT": -0.8, "PDT": -0.8, "WDT": -0.5,
        # 介词/连词 (负分)
        "IN": -0.8, "TO": -0.8, "CC": -0.8,
        # 助动词
        "MD": -0.5,
        # 其他功能词
        "EX": -0.5, "POS": -0.2, "RP": -0.2,
        # 标点符号 (最低分)
        '"': -1.0, "$": -1.0, "''": -1.0, "(": -1.0, ")": -1.0,
        ",": -1.0, ".": -1.0, ":": -1.0, "SYM": -1.0,
        # 其他
        "LS": -0.5, "FW": 0.0, "UH": -0.3,
    }

def get_deprel_prior() -> Dict[str, float]:
    """Universal Dependencies 依存关系先验分数"""
    return {
        # 高分 (倾向作为 head 的关系)
        "flat": 0.6, "compound": 0.6, "appos": 0.5, "nmod": 0.4, "amod": 0.3,
        "nsubj": 0.4, "obj": 0.4, "nummod": 0.3,
        # 中等分
        "acl": 0.3, "acl:relcl": 0.3, "advcl": 0.2, "advcl:relcl": 0.2,
        "ccomp": 0.2, "xcomp": 0.2, "obl": 0.2, "obl:tmod": 0.2,
        "obl:npmod": 0.2, "obl:agent": 0.2,
        # 中低分
        "conj": 0.1, "parataxis": 0.1, "list": 0.1, "iobj": 0.3,
        "csubj": 0.3, "csubj:outer": 0.3, "nsubj:pass": 0.3, "nsubj:outer": 0.3,
        # 修饰语 (0分)
        "advmod": 0.0, "nmod:poss": 0.0, "nmod:tmod": 0.0,
        "nmod:npmod": 0.0, "nmod:desc": 0.0,
        # 低分 (功能词)
        "det": -0.8, "det:predet": -0.8, "case": -0.8, "cc": -0.8,
        "cc:preconj": -0.8, "mark": -0.8, "aux": -0.8, "aux:pass": -0.8,
        "cop": -0.8, "expl": -0.8,
        # 最低分
        "punct": -0.9,
        # 其他
        "fixed": -0.5, "compound:prt": 0.0, "discourse": -0.3,
        "dislocated": -0.3, "vocative": -0.3, "dep": 0.0,
        "orphan": -0.2, "reparandum": -0.5,
        # Root (特殊情况)
        "root": 1.0,
    }

def build_pos_score_table(pos2id: Dict[str, int]) -> torch.Tensor:
    """构建 POS ID -> score 的查找表
    
    Args:
        pos2id: POS 标签到 ID 的映射字典
    
    Returns:
        score_table: (num_tags,) 先验分数张量
    """
    pos_prior = get_pos_prior()
    num_tags = len(pos2id)
    score_table = torch.zeros(num_tags, dtype=torch.float32)
    
    for tag, tag_id in pos2id.items():
        score_table[tag_id] = pos_prior.get(tag, 0.0)
    
    return score_table

def build_deprel_score_table(deprel2id: Dict[str, int]) -> torch.Tensor:
    """构建 DEPREL ID -> score 的查找表
    
    Args:
        deprel2id: 依存关系到 ID 的映射字典
    
    Returns:
        score_table: (num_rels,) 先验分数张量
    """
    deprel_prior = get_deprel_prior()
    num_rels = len(deprel2id)
    score_table = torch.zeros(num_rels, dtype=torch.float32)
    
    for rel, rel_id in deprel2id.items():
        score_table[rel_id] = deprel_prior.get(rel, 0.0)
    
    return score_table

def load_conll_data(path, ent2id):
    """
    加载CoNLL格式的NER数据
    """
    D = {"entities": [], "text": []}
    
    with open(path, 'r', encoding='utf-8') as f:
        current_words = []
        current_labels = []
        
        for line in f:
            line = line.strip()
            
            # 跳过文档开始标记
            if line.startswith("-DOCSTART"):
                continue
            
            # 遇到空行，处理当前句子
            if line == "" and len(current_words) > 0:
                # 将当前句子的词连接成文本
                sentence_text = ' '.join(current_words)
                D["text"].append(sentence_text)
                
                # 使用已有的bio_to_spans函数
                entities = bio_to_spans(ent2id, current_labels)
                D["entities"].append(entities)
                
                # 重置当前句子
                current_words = []
                current_labels = []
                continue
            
            elif line == "" and len(current_words) == 0:
                continue
            
            # 解析每一行数据
            parts = line.split('\t')
            if len(parts) >= 9:  # 确保有足够的列
                word = parts[1]        # 词
                ner_label = parts[-1]  # NER标签
                
                current_words.append(word)
                current_labels.append(ner_label)
        
        # 处理文件末尾的最后一个句子（如果存在）
        if len(current_words) > 0:
            sentence_text = ' '.join(current_words)
            D["text"].append(sentence_text)
            
            entities = bio_to_spans(ent2id, current_labels)
            D["entities"].append(entities)
    
    return D

def load_json_data(path, ent2id):
    D = {"entities": [], "text": []}
    for data in open(path):
        d = json.loads(data)
        D["text"].append(' '.join(d['sentence']))
        D["entities"].append([])
        for e in d["entities"]:
            start = e["start"]
            end = e["end"] 
            label = e["type"]
            if start <= end:
                D["entities"][-1].append((start, end, ent2id[label]))
    return D

def load_data(path, ent2id, json_flag=False):
    if json_flag == True:
        return load_json_data(path, ent2id)  # 原来的JSON加载函数
    else:
        return load_conll_data(path, ent2id)