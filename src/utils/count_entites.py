from collections import defaultdict
import json
import os

def count_bmes_entities(file_path):
    """
    按照 BMES 格式统计标注数据中的实体个数
    :param file_path: 标注数据文件路径
    :return: 实体总数量和每种实体类型的计数
    """
    entity_count = 0
    entity_type_counts = {}  # 记录不同类型实体的数量
    inside_entity = False  # 是否处于实体内部
    current_entity_type = None  # 当前实体类型

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            line = line.strip()
            if line == "":  # 跳过空行
                inside_entity = False  # 重置状态
                current_entity_type = None
                continue

            parts = line.split()  # 分割行内容
            if len(parts) >= 2:  # 确保行中有两个字段（单词和实体标签）
                entity_label = parts[-1]
                if entity_label.startswith("B-"):  # 实体开始
                    entity_count += 1
                    inside_entity = True
                    current_entity_type = entity_label.split("-")[-1]  # 提取实体类型
                    if current_entity_type not in entity_type_counts:
                        entity_type_counts[current_entity_type] = 1
                    else:
                        entity_type_counts[current_entity_type] += 1
                elif entity_label.startswith("S-"):  # 单独实体
                    entity_count += 1
                    entity_type = entity_label.split("-")[-1]
                    if entity_type not in entity_type_counts:
                        entity_type_counts[entity_type] = 1
                    else:
                        entity_type_counts[entity_type] += 1
                elif entity_label.startswith("E-") and inside_entity:  # 实体结束
                    inside_entity = False  # 结束当前实体
                    current_entity_type = None
                elif entity_label.startswith("M-") and inside_entity:  # 实体中间部分
                    continue
                else:  # 遇到 O 标签或其他非实体标签
                    inside_entity = False
                    current_entity_type = None

    return entity_count, entity_type_counts

def count_bio_entities(file_path):
    """
    统计 BIO 格式标注数据中的实体数量和类型。
    :param file_path: BIO 格式标注数据文件路径
    :return: 实体总数和每种实体类型的统计
    """
    entity_count = 0
    max_sent_length =0 
    entity_type_counts = {}  # 存储每种实体类型的计数
    inside_entity = False  # 是否处于实体内部
    current_entity_type = None

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            line = line.strip()
            if line == "":
                inside_entity = False  # 如果遇到空行，表示句子结束，重置实体状态
                continue

            parts = line.split()  # 分割行内容
            if len(parts) >= 2:  # 确保行中有至少两个字段（单词和实体标签）
                entity_label = parts[-1]  # 实体标签在最后一列
                if entity_label.startswith("B-"):  # 遇到实体的开始
                    entity_count += 1  # 实体数量 +1
                    entity_type = entity_label[2:]  # 提取实体类型
                    entity_type_counts[entity_type] = entity_type_counts.get(entity_type, 0) + 1  # 类型计数累加
                    inside_entity = True  # 标记进入实体
                    current_entity_type = entity_type
                elif entity_label.startswith("I-"):
                    # 如果是 I- 标签，且当前实体类型与之前一致，则继续统计
                    if inside_entity and entity_label[2:] == current_entity_type:
                        continue
                    else:
                        # I- 出现但未对应 B-，说明标注不规范，忽略这种情况
                        inside_entity = False
                else:
                    # 非实体标签，结束当前实体
                    inside_entity = False
                    current_entity_type = None

    return entity_count, entity_type_counts

def count_ace05(data):
    """
    统计 ACE05 数据格式中的实体数量
    :param data: ACE05 数据，解析后的 JSON 数据
    :return: 实体总数和每种实体类型的数量
    """
    entity_counts = defaultdict(int)  # 用于统计每种实体类型的数量
    total_count = 0  # 实体总数
    total_relations = 0  # 关系总数
    total_sentences = 0  # 句子总数

    with open(file_path, 'r', encoding='utf-8') as f:
      for line in f:
        if line.strip():  # 跳过空行
          data = json.loads(line.strip())
          # 遍历数据中的所有句子
          for ner_list in data["ner"]:
              for entity in ner_list:
                  entity_type = entity[2]  # 第三个字段是实体类型
                  entity_counts[entity_type] += 1
                  total_count += 1
          # 统计关系数量
          for relation_list in data["relations"]:
            total_relations += len(relation_list)

          # 统计句子数量
          total_sentences += len(data["sentences"])

    return total_count, dict(entity_counts), total_relations, total_sentences


import json
from tqdm import tqdm

def count_ace04(file_path):
    """
    Reads a JSON file and counts the total number of entities, relations, and tokens.

    :param file_path: Path to the JSON file
    :return: Total counts for entities, relations, and tokens
    """
    batch_entities = 0
    batch_relations = 0
    batch_sentences = 0
    entity_type_counts = {}
    relation_type_counts = {}

    with open(file_path, "r", encoding="utf-8") as f:
        for line in tqdm(f):
        # Read the JSON data as a list of dictionaries
            data = json.loads(line.strip())
            sentence = data["sentence"]
            # Count sentences (each dictionary entry represents one sentence)
            if len(sentence) > 0:
                batch_sentences += 1

            # Count entities
            entities = data.get("entities", [])
            batch_entities += len(entities)
            for entity in entities:
                entity_type = entity['type']  # The type of the entity is the last element
                if entity_type not in entity_type_counts:
                    entity_type_counts[entity_type] = 0
                entity_type_counts[entity_type] += 1

            # Count relations
            relations = data.get("relations", [])
            batch_relations += len(relations)
            for relation in relations:
                relation_type = relation[-1]  # The type of the relation is the last element
                if relation_type not in relation_type_counts:
                    relation_type_counts[relation_type] = 0
                relation_type_counts[relation_type] += 1

    return batch_entities, batch_relations, batch_sentences, entity_type_counts, relation_type_counts


if __name__ == "__main__":

    total_entities = 0
    total_relations = 0
    total_sentences = 0
    script_dir = os.path.dirname(os.path.abspath(__file__)) # utils目录
    parent_dir = os.path.dirname(script_dir)                # 父目录
    data_dir = os.path.join(parent_dir, "data")             # data目录
    dataset = "catalan"                                     # 或其他数据集
    files = [
        (os.path.join(data_dir, dataset, "train.txt")),
        (os.path.join(data_dir, dataset, "dev.txt")),
        (os.path.join(data_dir, dataset, "test.txt")),
        ]

    for file_path in files:
    #   cout, entity_type_counts = count_bmes_entities(file_path)
      cout, entity_type_counts = count_bio_entities(file_path)
    #   entities, entity_type_counts, relations, sentences = count_ace05(file_path)
    #   entities, relations, sentences, entity_type_counts, relation_type_counts = count_ace04(file_path) # genia
      print(json.dumps(entity_type_counts, indent=4, ensure_ascii=False))
    #   total_entities += entities
    #   total_relations += relations
    #   total_sentences += sentences

    print(f"Total sentences: {total_sentences}")
    print(f"Total Entities: {total_entities}")
    print(f"Total Relations: {total_relations}")