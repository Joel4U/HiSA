import torch
from collections import Counter
import collections

class MetricsCalculator(object):
    def __init__(self, ent_thres, id2ent, allow_nested=True):
        super().__init__()
        self.allow_nested = allow_nested
        self.ent_thres = ent_thres
        self.id2ent = id2ent

    def get_evaluate_fpr(self, examples, scores, word_lens):
        """使用标准的上三角矩阵解码方式"""
        X, Y, Z = [], [], []
        
        for idx, (example, score, word_len) in enumerate(zip(examples["entities"], scores, word_lens)):
            # 真实标注
            example_annotations = set(map(tuple, example))
            # 预测处理 - score应该是 [seq_len, seq_len, num_classes]
            ent_scores = score.sigmoid()
            ent_scores = (ent_scores + ent_scores.transpose(0, 1)) / 2
            # 获取每个位置的最大概率和对应的类别
            span_pred = ent_scores.max(dim=-1)[0]
            # 包装成批量格式传递给_decode_standard
            batch_span_pred = span_pred.unsqueeze(0)  # [1, seq_len, seq_len]
            batch_word_len = torch.tensor([word_len.item()]) if hasattr(word_len, 'item') else torch.tensor([word_len])
            # 使用最大概率进行解码
            span_ents = self._decode_standard(batch_span_pred, batch_word_len, allow_nested=self.allow_nested, thres=self.ent_thres)
            example_predictions = set()
            for s, e, l in span_ents[0]:
                score_at_pos = ent_scores[s, e]  # 使用对称化后的分数
                ent_type = score_at_pos.argmax().item()
                if score_at_pos[ent_type] >= self.ent_thres:
                    example_predictions.add((s, e, ent_type))
        
            # 统计结果
            X.extend(example_annotations)
            Y.extend(example_predictions)
            Z.extend([pre_entity for pre_entity in example_predictions if pre_entity in example_annotations])
        
        return X, Y, Z

    def _decode_standard(self, scores, length, allow_nested=False, thres=0.5):
        """标准的上三角矩阵解码"""
        batch_chunks = []
        for idx, (curr_scores, curr_len) in enumerate(zip(scores, length.cpu().tolist())):
            curr_non_mask = scores.new_ones(curr_len, curr_len, dtype=bool).triu()
            tmp_scores = curr_scores[:curr_len, :curr_len][curr_non_mask].cpu().numpy()

            confidences, label_ids = tmp_scores, tmp_scores >= thres
            labels = [i for i in label_ids]
            chunks = [(label, start, end) for label, (start, end) in 
                     zip(labels, self._spans_from_upper_triangular(curr_len)) if label != 0]
            confidences = [conf for label, conf in zip(labels, confidences) if label != 0]

            assert len(confidences) == len(chunks)
            chunks = [ck for _, ck in sorted(zip(confidences, chunks), reverse=True)]
            chunks = self._filter_clashed_by_priority(chunks, allow_nested=allow_nested)
            if len(chunks):
                batch_chunks.append(set([(s, e, l) for l, s, e in chunks]))
            else:
                batch_chunks.append(set())
        return batch_chunks

    def _spans_from_upper_triangular(self, seq_len: int):
        """生成上三角区域的所有span"""
        for start in range(seq_len):
            for end in range(start, seq_len):
                yield (start, end)

    def _filter_clashed_by_priority(self, chunks, allow_nested: bool=True):
        """过滤冲突的chunks"""
        filtered_chunks = []
        for ck in chunks:
            if all(not self._is_clashed(ck, ex_ck, allow_nested=allow_nested) for ex_ck in filtered_chunks):
                filtered_chunks.append(ck)
        return filtered_chunks

    def _is_overlapped(self, chunk1: tuple, chunk2: tuple):
        """检查两个span是否重叠"""
        (_, s1, e1), (_, s2, e2) = chunk1, chunk2
        return (s1 < e2 and s2 < e1)

    def _is_nested(self, chunk1: tuple, chunk2: tuple):
        """检查两个span是否嵌套"""
        (_, s1, e1), (_, s2, e2) = chunk1, chunk2
        return (s1 <= s2 and e2 <= e1) or (s2 <= s1 and e1 <= e2)

    def _is_clashed(self, chunk1: tuple, chunk2: tuple, allow_nested: bool=True):
        """检查两个span是否冲突"""
        if allow_nested:
            return self._is_overlapped(chunk1, chunk2) and not self._is_nested(chunk1, chunk2)
        else:
            return self._is_overlapped(chunk1, chunk2)

    def compute(self, origin, found, right):
        """计算precision, recall, f1"""
        recall = 0 if origin == 0 else (right / origin)
        precision = 0 if found == 0 else (right / found)
        f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
        return recall, precision, f1

    def result(self, origins, founds, rights):
        """计算整体和每个类别的指标"""
        class_info = {}
        origin_counter = Counter([self.id2ent[x[-1]] for x in origins])
        found_counter = Counter([self.id2ent[x[-1]] for x in founds])
        right_counter = Counter([self.id2ent[x[-1]] for x in rights])
        
        for type_, count in origin_counter.items():
            origin = count
            found = found_counter.get(type_, 0)
            right = right_counter.get(type_, 0)
            recall, precision, f1 = self.compute(origin, found, right)
            class_info[type_] = {
                "acc": precision, 'recall': recall, 'f1': f1,
                'origin': origin, 'found': found, 'right': right
            }
            
        origin = len(origins)
        found = len(founds)
        right = len(rights)
        recall, precision, f1 = self.compute(origin, found, right)
        
        return {
            'acc': precision, 'recall': recall, 'f1': f1,
            'origin': origin, 'found': found, 'right': right
        }, class_info