import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel, AutoConfig

class SpanHeadExtractor:
    def __init__(self, model_name='bert-base-cased', layer_strategy='last', head_strategy='mean'):
        super().__init__()
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, output_attentions=True)
        self.config = AutoConfig.from_pretrained(model_name)
        self.model_type = self.config.model_type
        self.layer_strategy = layer_strategy
        self.head_strategy = head_strategy

    def _extract_attention_from_outputs(self, outputs) -> torch.Tensor:
        """
        从模型输出中提取注意力权重
        """
        # 获取注意力权重
        attention_weights = None
        
        if hasattr(outputs, 'attentions') and outputs.attentions is not None:
            attention_weights = outputs.attentions
        elif hasattr(outputs, 'encoder_attentions') and outputs.encoder_attentions is not None:
            attention_weights = outputs.encoder_attentions
        else:
            available_attrs = [attr for attr in dir(outputs) if not attr.startswith('_')]
            raise ValueError(f"Cannot find attention weights. Available: {available_attrs}")
        
        # 选择层
        attention = self._select_layer(attention_weights)
        
        # 移除batch维度
        if attention.dim() == 4:  # (batch, heads, seq, seq)
            attention = attention.squeeze(0)
        
        # 聚合注意力头
        if attention.dim() == 3:  # (heads, seq, seq)
            attention = self._aggregate_heads(attention, self.head_strategy)
        
        return attention
    
    def _select_layer(self, attention_weights: tuple) -> torch.Tensor:
        """
        选择要使用的层
        """
        num_layers = len(attention_weights)
        
        if self.layer_strategy == 'last':
            return attention_weights[-1]
        elif self.layer_strategy == 'first':
            return attention_weights[0]
        elif self.layer_strategy == 'middle':
            return attention_weights[num_layers // 2]
        elif self.layer_strategy == 'second_to_last':
            return attention_weights[-2] if num_layers > 1 else attention_weights[-1]
        elif self.layer_strategy == 'mean':
            # 所有层的平均
            return torch.stack(attention_weights).mean(dim=0)
        elif self.layer_strategy == 'weighted':
            # 给后面的层更高的权重
            weights = torch.arange(1, num_layers + 1, dtype=torch.float32)
            weights = weights / weights.sum()
            weighted_attention = torch.zeros_like(attention_weights[0])
            for i, (layer_attention, weight) in enumerate(zip(attention_weights, weights)):
                weighted_attention += weight * layer_attention
            return weighted_attention
        else:
            return attention_weights[-1]
    
    def _aggregate_heads(self, attention: torch.Tensor, method: str = 'mean') -> torch.Tensor:
        """
        聚合多个注意力头
        """
        if method == 'mean':
            return attention.mean(dim=0)
        elif method == 'max':
            return attention.max(dim=0)[0]
        elif method == 'sum':
            return attention.sum(dim=0)
        elif method == 'first':
            return attention[0]
        elif method == 'last':
            return attention[-1]
        elif method == 'entropy_weighted':
            # 根据熵加权（熵低的头权重更高，因为更确定）
            entropies = self._compute_attention_entropy(attention)
            weights = 1.0 / (1.0 + entropies)
            weights = weights / weights.sum()
            weighted = torch.sum(attention * weights.unsqueeze(-1).unsqueeze(-1), dim=0)
            return weighted
        else:
            return attention.mean(dim=0)
    
    def _compute_attention_entropy(self, attention: torch.Tensor) -> torch.Tensor:
        """
        计算每个注意力头的熵
        """
        # attention: (num_heads, seq_len, seq_len)
        # 计算每个头的平均熵
        epsilon = 1e-9
        attention_probs = attention + epsilon
        entropy = -(attention_probs * torch.log(attention_probs)).sum(dim=-1).mean(dim=-1)
        return entropy
    
    def _tokenize_for_roberta(self, tokens: List[str]) -> dict:
        """
        RoBERTa的特殊tokenization处理
        """
        # RoBERTa期望词之间有空格
        text = ' '.join(tokens)
        # 使用特殊的tokenization方法保持对齐
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,  # 获取字符偏移
            padding=False,
            truncation=True
        )
        
        # 手动创建word_ids
        offsets = encoded['offset_mapping'][0]
        word_ids = []
        current_word = 0
        last_end = 0
        
        for start, end in offsets:
            if start == end:  # 特殊token
                word_ids.append(None)
            elif start > last_end + 1:  # 新词（有空格）
                current_word += 1
                word_ids.append(current_word)
            else:
                word_ids.append(current_word)
            last_end = end
        
        # 添加word_ids到encoded
        encoded['word_ids'] = lambda batch_idx=0: word_ids
        
        return encoded
    
    def _validate_word_ids(self, word_ids: List[int], expected_num_words: int) -> bool:
        """
        验证word_ids的正确性
        """
        if not word_ids:
            return False
        
        # 过滤掉None（特殊标记）
        actual_word_ids = [w for w in word_ids if w is not None]
        
        if not actual_word_ids:
            return False
        
        # 检查最大word_id
        max_word_id = max(actual_word_ids)
        
        # word_id应该从0到expected_num_words-1
        if max_word_id >= expected_num_words:
            print(f"Warning: max_word_id ({max_word_id}) >= expected_num_words ({expected_num_words})")
            return False
        
        # 检查是否有所有的word_id
        unique_word_ids = set(actual_word_ids)
        expected_ids = set(range(expected_num_words))
        
        if unique_word_ids != expected_ids:
            missing = expected_ids - unique_word_ids
            if missing:
                print(f"Warning: missing word_ids: {missing}")
            return False
        
        return True
    
    def _get_fallback_attention(self, tokens: List[str]) -> torch.Tensor:
        """
        降级方案：基于距离的简单注意力
        """
        n = len(tokens)
        attention = torch.zeros(n, n)
        
        for i in range(n):
            for j in range(n):
                # 距离衰减
                distance = abs(i - j)
                attention[i, j] = 1.0 / (1.0 + distance)
        # 行归一化
        attention = attention / attention.sum(dim=1, keepdim=True)
        
        return attention

    def get_attention_weights(self, tokens: List[str]) -> torch.Tensor:
        """
        获取任意模型的注意力权重,对大多数模型通用的tokenization，RoBERTa需要特殊处理空格
        """
        if self.model_type in ['roberta', 'xlm-roberta']:
            inputs = self._tokenize_for_roberta(tokens)
        else:
            # 标准处理
            inputs = self.tokenizer(tokens, is_split_into_words=True, return_tensors="pt", padding=False, truncation=True)
        
        # 获取模型输出
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # 提取注意力
        attention = self._extract_attention_from_outputs(outputs)
        
        # 获取word映射
        word_ids = inputs.word_ids(batch_index=0)
        
        # 验证word_ids
        if not self._validate_word_ids(word_ids, len(tokens)):
            print(f"Warning: word_ids validation failed for {self.model_type}")
        
        # 聚合到单词级别
        word_attention = self.aggregate_to_words(attention, word_ids, len(tokens), model_type=self.model_type)
        
        return word_attention
    
    def aggregate_to_words(self, attention, word_ids, num_words, model_type=None):
        """
        通用的aggregate_to_words，适用于所有模型
        """
        # 基础实现与之前相同
        word_attention = torch.zeros(num_words, num_words)
        
        # 创建映射
        word_to_tokens = defaultdict(list)
        for token_idx, word_idx in enumerate(word_ids):
            if word_idx is not None:
                word_to_tokens[word_idx].append(token_idx)
        
        # 聚合
        for word_i in range(num_words):
            for word_j in range(num_words):
                tokens_i = word_to_tokens[word_i]
                tokens_j = word_to_tokens[word_j]
                
                if tokens_i and tokens_j:
                    # 可以根据model_type选择不同的聚合策略
                    if model_type == 'gpt2' and word_j > word_i:
                        # GPT2的因果注意力
                        continue
                    
                    attn_values = []
                    for tok_i in tokens_i:
                        for tok_j in tokens_j:
                            if tok_i < attention.shape[0] and tok_j < attention.shape[1]:
                                attn_values.append(attention[tok_i, tok_j].item())
                    
                    if attn_values:
                        # 可以选择不同的聚合方式
                        word_attention[word_i, word_j] = sum(attn_values) / len(attn_values)
        
        # 归一化
        for i in range(num_words):
            row_sum = word_attention[i].sum()
            if row_sum > 0:
                word_attention[i] = word_attention[i] / row_sum
        
        return word_attention
        
    def build_dependency_tree(self, dep_heads: List[int]) -> Dict[int, List[int]]:
        """
        构建依存树结构
        Returns:
            children字典，key是父节点，value是子节点列表
        """
        children = defaultdict(list)
        for i, head in enumerate(dep_heads):
            if head != -1:  # -1表示根节点
                children[head].append(i)
        return dict(children)

    def compute_dependency_levels(self, dep_heads: List[int]) -> List[int]:
        """
        计算每个节点到根节点的距离（层级）
        """
        n = len(dep_heads)
        levels = [-1] * n
        
        # 找到所有根节点
        roots = [i for i, head in enumerate(dep_heads) if head == -1]
        
        # BFS计算层级
        from collections import deque
        queue = deque()
        
        # 根节点层级为0
        for root in roots:
            levels[root] = 0
            queue.append(root)
        
        # 逆向构建children关系
        children = defaultdict(list)
        for i, head in enumerate(dep_heads):
            if head != -1:
                children[head].append(i)
        
        # BFS遍历
        while queue:
            node = queue.popleft()
            for child in children[node]:
                levels[child] = levels[node] + 1
                queue.append(child)
        
        return levels

    def find_span_head_by_dependency(self, dep_heads: List[int], span: Tuple[int, int]) -> Optional[List[int]]:
        """
        通过依存句法找到span中的候选head（返回所有最高层级的节点）
        Returns:
            最高依存层级的所有节点列表，如果无法确定唯一head则返回多个
        """
        start, end = span
        span_tokens = list(range(start, end))
        
        # 计算层级
        levels = self.compute_dependency_levels(dep_heads)
        
        # 找到最小层级
        span_levels = [(i, levels[i]) for i in span_tokens]
        min_level = min(level for _, level in span_levels)
        
        # 返回所有最小层级的节点
        candidates = [i for i, level in span_levels if level == min_level]
        
        return candidates if len(candidates) > 1 else None

    def select_span_head_by_attention(self, candidates: List[int], attention_weights: torch.Tensor, span: Tuple[int, int]) -> int:
        """
        在候选词中使用注意力权重选择head
        
        策略：选择在span内获得最多注意力的候选词
        """
        start, end = span
        best_score = -1
        best_candidate = candidates[0]
        
        for candidate in candidates:
            # 计算候选词在span内获得的注意力总和
            attention_score = 0
            
            # 方法1：计算span内其他词对该候选词的注意力
            for i in range(start, end):
                if i != candidate:
                    attention_score += attention_weights[i, candidate].item()
            
            # 方法2：也可以考虑该候选词对span内其他词的注意力（可选）
            # for j in range(start, end):
            #     if j != candidate:
            #         attention_score += attention_weights[candidate, j].item()
            
            # 方法3：双向注意力的平均（可选）
            # attention_score = (incoming_attention + outgoing_attention) / 2
            
            if attention_score > best_score:
                best_score = attention_score
                best_candidate = candidate
        
        return best_candidate

    def find_span_head(self, levels: List[int],  sentence_roots: List[int],
                            attention_weights: Optional[torch.Tensor], span: Tuple[int, int]) -> int:
        """
        综合依存句法和注意力权重找到span的head
        
        策略：
        1. 如果span包含句子根节点，直接返回根节点
        2. 找到依存层级最高的节点
        3. 如果有多个相同层级的节点，使用注意力权重决策
        """
        start, end = span
        span_tokens = list(range(start, end))
        
        # Step 1: 检查是否包含句子根节点
        for root in sentence_roots:
            if start <= root < end:
                return root
        
        # Step 2: 找span中level最小者
        span_levels = [(i, levels[i]) for i in span_tokens]
        min_level = min(level for _, level in span_levels)
        candidates = [i for i, level in span_levels if level == min_level]
        
        # 如果只有一个候选，直接返回
        if len(candidates) == 1:
            return candidates[0]
        
        # Step 3: 有多个相同层级的候选，使用注意力权重策略决策
        return self.select_span_head_by_attention(candidates, attention_weights, span)

    def extract_span_heads(self, tokens: List[str], dep_heads: List[int],
                          attention_weights: Optional[torch.Tensor] = None) -> np.ndarray:
        """
        提取所有span的head词
        Args:
            tokens: 词列表
            dep_heads: 依存句法头词索引列表
            attention_weights: 注意力权重矩阵
        Returns:
            head矩阵，-1表示无效位置（下三角）
        """
        n = len(tokens)
        head_matrix = np.full((n, n), -1, dtype=int)
        # 构建依存树
        levels = self.compute_dependency_levels(dep_heads)
        sentence_roots = [i for i, head in enumerate(dep_heads) if head == -1]

        # 枚举所有span    
        for start in range(n):
            for end in range(start + 1, n + 1):
                span = (start, end)
                
                # 综合使用依存句法和注意力权重
                head_idx = self.find_span_head(levels, sentence_roots, attention_weights, span)
                # 填充head矩阵
                head_matrix[start, end-1] = head_idx

        return head_matrix
    
    def visualize_matrices(self, tokens: List[str], head_matrix: np.ndarray):
        """
        可视化span矩阵和head矩阵
        Args:
            tokens: 词列表
            head_matrix: head矩阵
        """
        n = len(tokens)
        
        print("Tokens:", " ".join(tokens))
        print("\nSpan Matrix (i,j represents span from i to j+1):")
        print("   ", "  ".join(f"{i:2d}" for i in range(n)))
        for i in range(n):
            row = []
            for j in range(n):
                if j >= i:
                    row.append(f"({i},{j+1})")
                else:
                    row.append("   -  ")
            print(f"{i:2d}:", " ".join(f"{cell:^6}" for cell in row))
        
        print("\nHead Matrix (value is the head token index):")
        print("   ", "  ".join(f"{i:2d}" for i in range(n)))
        for i in range(n):
            row = []
            for j in range(n):
                if head_matrix[i, j] >= 0:
                    row.append(str(head_matrix[i, j]))
                else:
                    row.append("-")
            print(f"{i:2d}:", " ".join(f"{cell:^4}" for cell in row))


# 使用示例
if __name__ == "__main__":
    # 示例句子
    tokens = ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
    # -1表示根，其他数字表示头词在句子中的索引
    dep_heads = [3, 3, 3, 4, -1, 8, 8, 8, 4] # jumps是根，quick修饰fox等
    dep_labels = ["det", "amod", "amod", "nsubj", "root", "obl", "case", "det", "amod"]
    
    # 如果使用-1表示根，根-1转换为0-based索引的位置序号
    # dep_heads = [h if h != -1 else i for i, h in enumerate(dep_heads)]
    
    # 创建提取器
    extractor = SpanHeadExtractor(model_name='bert-base-cased')
    
    # 计算真实的注意力权重
    attention_weights = extractor.get_attention_weights(tokens)

    # 显示一些高注意力的词对
    top_k = 5
    values, indices = attention_weights.flatten().topk(top_k)
    print(f"\n  Top {top_k} attention pairs:")
    for val, idx in zip(values, indices):
        i = idx // len(tokens)
        j = idx % len(tokens)
        print(f"    {tokens[i]} -> {tokens[j]}: {val:.4f}")

    # 提取head矩阵
    head_matrix = extractor.extract_span_heads(tokens, dep_heads, attention_weights)
    
    # 可视化结果
    extractor.visualize_matrices(tokens, head_matrix)
    
    # 打印一些具体的span和它们的head
    # print("\nSome example spans and their heads:")
    # spans = [(0, 2), (1, 4), (0, 5)]
    # for start, end in spans:
    #     head_idx = head_matrix[start, end-1]
    #     if head_idx >= 0:
    #         span_text = " ".join(tokens[start:end])
    #         head_token = tokens[head_idx]
    #         print(f"Span '{span_text}' -> Head: '{head_token}' (index {head_idx})")
