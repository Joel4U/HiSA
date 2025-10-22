import json
import os
from tqdm import tqdm
from dataclasses import dataclass
from typing import List

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@dataclass
class Instance:
    words: List[str]
    depheads: List[int]
    deplabels: List[str]
    labels: List[str] = None

class Vocabulary(object):

    def __init__(self, dataset):
        self.dataset = dataset
        self.label2id = {}
        self.id2label = {}
        self.deplabel2id = {}
        self.id2deplabel = {}

    def get_chunk_type(self, tok):
        tag_class = tok.split('-')[0]
        tag_type = '-'.join(tok.split('-')[1:])
        return tag_class, tag_type

    def get_chunks(self, seq):
        default = 'O'
        chunks = []
        chunk_type, chunk_start = None, None
        for i, tok in enumerate(seq):
            # End of a chunk 1
            if tok == default and chunk_type is not None: # Add a chunk.
                chunk = ((chunk_start, i-1), chunk_type)
                chunks.append(chunk)
                chunk_type, chunk_start = None, None
            # End of a chunk + start of a chunk!
            elif tok != default:
                tok_chunk_class, tok_chunk_type = self.get_chunk_type(tok)
                if chunk_type is None:
                    chunk_type, chunk_start = tok_chunk_type, i
                elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                    chunk = ((chunk_start, i-1), chunk_type)
                    chunks.append(chunk)
                    chunk_type, chunk_start = tok_chunk_type, i
            else:
                pass
        # end condition
        if chunk_type is not None:
            chunk = ((chunk_start, len(seq)-1), chunk_type)
            chunks.append(chunk)
        return chunks

    def read_json(self, file: str) -> List[Instance]:
        print(f"[Data Info] Reading file: {file}")
        insts = []                                      # 存储所有的实例
        max_sent_length = 0
        with open(file, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                if line.strip():                        # 忽略空行
                    entry = json.loads(line.strip())    # 逐行解析 JSON
                    words = entry["sentence"]
                    sent_len = len(words)
                    max_sent_length = max(max_sent_length, sent_len)
                    depheads = [head - 1 for head in entry["dephead"]]  # 转换为 root is -1 not 0
                    deplabels = entry["deplabel"]       # 依存树的依存关系标签
                    # 解析 NER 信息
                    chunks = []
                    for entity in entry["entities"]:
                        start = entity["start"]         # 实体起始位置
                        end = entity["end"]             # 实体结束位置
                        entity_type = entity["type"]
                        chunks.append(((start, end), entity_type))
                    # 创建一个 Instance 对象
                    insts.append(Instance(words=words, depheads=depheads, deplabels=deplabels, labels=chunks))
        return insts, max_sent_length
    
    def read_file(self, file):
        insts = []
        max_sent_length = 0
        with open(file, 'r', encoding='utf-8') as f:
            words, heads, rels, labels = [], [], [], []
            for line in tqdm(f.readlines()):
                line = line.rstrip()
                if line.startswith("-DOCSTART"):
                    continue
                if line == "" and len(words) != 0:
                    insts.append(Instance(words=words, depheads=heads, deplabels=rels, labels=labels))
                    sent_len = len(words)
                    max_sent_length = max(max_sent_length, sent_len)
                    words, heads, rels, labels = [], [], [], []
                    continue
                elif line == "" and len(words) == 0:
                    continue
                ls = line.split()
                word, head, rel, label = ls[1], int(ls[6])-1, ls[7], ls[-1]
                words.append(word)
                heads.append(head)
                rels.append(rel)
                labels.append(label)
        print(f"{file} 最大句子长度: {max_sent_length}")
        return insts, max_sent_length
    
    def load_label(self):
        if os.path.exists('./data/{}/ontology.json'.format(self.dataset)):
            with open('./data/{}/ontology.json'.format(self.dataset), 'r', encoding='utf-8') as f:
                infos = json.load(f)
                self.id2label = infos["id2label"]
                self.label2id = infos["label2id"]
                self.id2deplabel = infos["id2deplabel"]
                self.deplabel2id = infos["deplabel2id"]
            return True
        else:
            return False

    def build_label(self, train_data, dev_data, test_data, max_seqlen, json_flag=False):
        if json_flag:
            train_ent_num = self.count_json(train_data)
            dev_ent_num = self.count_json(dev_data)
            test_ent_num = self.count_json(test_data)
            with open('./data/{}/ontology.json'.format(self.dataset), 'w', encoding='utf-8') as f:
                infos = {
                    "train": [len(train_data), train_ent_num], "dev": [len(dev_data), dev_ent_num], "test": [len(test_data), test_ent_num],
                    "max_seqlen": max_seqlen,
                    "deplabel2id": self.deplabel2id, "id2deplabel": self.id2deplabel,
                    "label2id": self.label2id, "id2label": self.id2label
                }
                f.write(json.dumps(infos, ensure_ascii=False))
            return
        else:
            train_ent_num = self.count_conllx(train_data)
            dev_ent_num = self.count_conllx(dev_data)
            test_ent_num = self.count_conllx(test_data)
            with open('./data/{}/ontology.json'.format(self.dataset), 'w', encoding='utf-8') as f:
                infos = {
                    "train": [len(train_data), train_ent_num], "dev": [len(dev_data), dev_ent_num], "test": [len(test_data), test_ent_num],
                    "max_seqlen": max_seqlen,
                    "deplabel2id": self.deplabel2id,
                    "id2deplabel": self.id2deplabel,
                    "label2id": self.label2id,
                    "id2label": self.id2label
                }
                f.write(json.dumps(infos, ensure_ascii=False))

    def count_json(self, samples): 
        entity_counter = {}
        deplabel_counter = {}
        total_entities = 0
        
        for sample in samples:
            # 统计实体
            for entity in sample.labels:
                entity_type = entity[1]
                entity_counter[entity_type] = entity_counter.get(entity_type, 0) + 1
                self.add_label(entity_type)
                total_entities += 1
            
            # 统计依存标签
            for deplabel in sample.deplabels:
                deplabel_counter[deplabel] = deplabel_counter.get(deplabel, 0) + 1
                self.add_deplabel(deplabel)
        
        print(f"Total entities: {total_entities}")
        print("Entity types:", entity_counter)
        print("Dependency labels:", deplabel_counter)
        
        return total_entities

    def count_conllx(self, samples):
        entity_num = 0
        for instance in samples:
            # 使用现有的get_chunks方法从labels中提取实体
            chunks = self.get_chunks(instance.labels)  # instance.labels 是 BIO 标注序列
            
            # 统计每个实体类型
            for chunk in chunks:
                span, entity_type = chunk  # chunk格式: ((start, end), type)
                self.add_label(entity_type)
                entity_num += 1
            
            for deplabel in instance.deplabels:
                self.add_deplabel(deplabel)
                
        return entity_num

    def add_label(self, label):
        if label not in self.label2id:
            self.label2id[label] = len(self.label2id)
            self.id2label[self.label2id[label]] = label
        assert label == self.id2label[self.label2id[label]]

    def label_to_id(self, label):
        return self.label2id[label]

    def id_to_label(self, i):
        return self.id2label[i]
    
    def add_deplabel(self, deplabel):
        if deplabel not in self.deplabel2id:
            self.deplabel2id[deplabel] = len(self.deplabel2id)
            self.id2deplabel[self.deplabel2id[deplabel]] = deplabel
        assert deplabel == self.id2deplabel[self.deplabel2id[deplabel]]

    def deplabel_to_id(self, deplabel):
        return self.deplabel2id[deplabel]

    def id_to_deplabel(self, i):
        return self.id2deplabel[i]

    def __len__(self):
        return len(self.label2id)
    
if __name__ == "__main__":
    dataset = 'genia'
    vocab = Vocabulary(dataset)
    train_data, train_max_seqlen  = vocab.read_json(f'./data/{dataset}/train.txt')
    dev_data, dev_max_seqlen = vocab.read_json(f'./data/{dataset}/dev.txt')
    test_data, test_max_seqlen = vocab.read_json(f'./data/{dataset}/test.txt')
    # train_data, train_max_seqlen  = vocab.read_file(f'./data/{dataset}/train.txt')
    # dev_data, dev_max_seqlen = vocab.read_file(f'./data/{dataset}/dev.txt')
    # test_data, test_max_seqlen = vocab.read_file(f'./data/{dataset}/test.txt')
    max_seqlen = max(max(train_max_seqlen, dev_max_seqlen), test_max_seqlen)
    if not vocab.load_label():
        vocab.build_label(train_data, dev_data, test_data, max_seqlen, json_flag=True)