import json

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