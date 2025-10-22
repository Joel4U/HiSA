import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel, AutoConfig

class SpanHeadExtractor:
    def __init__(self, model_name='bert-base-cased', layer_strategy='last', head_strategy='mean', device='cuda:2'):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, output_attentions=True).to(device)
        self.config = AutoConfig.from_pretrained(model_name)
        self.model_type = self.config.model_type
        self.layer_strategy = layer_strategy
        self.head_strategy = head_strategy

        self.model.eval()

    def _extract_attention_from_outputs(self, outputs) -> torch.Tensor:
        """从模型输出中提取注意力权重"""
        if hasattr(outputs, 'attentions') and outputs.attentions is not None:
            attention_weights = outputs.attentions
        else:
            raise ValueError(f"Cannot find attention weights. Available")
        
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
        """选择要使用的层"""
        num_layers = len(attention_weights)
        
        if self.layer_strategy == 'last':
            return attention_weights[-1]
        elif self.layer_strategy == 'first':
            return attention_weights[0]
        elif self.layer_strategy == 'middle':
            return attention_weights[num_layers // 2]
        elif self.layer_strategy == 'second_to_last':
            return attention_weights[-2] if num_layers > 1 else attention_weights[-1]
        elif self.layer_strategy == 'mean':             # 所有层的平均
            return torch.stack(attention_weights).mean(dim=0)
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
        else:
            return attention.mean(dim=0)
    
    def _get_word_mappings(self, tokens: List[str]) -> Tuple[dict, List[Optional[int]], int]:
        """
        获取词到子词的映射关系
        返回: (inputs, word_ids, num_subwords)
        """
        if self.model_type in ['roberta', 'xlm-roberta']:  # RoBERTa系列模型的特殊处理
            return self._get_roberta_word_mappings(tokens)
        else: # BERT和其他模型
            return self._get_bert_word_mappings(tokens)
        
    def _get_bert_word_mappings(self, tokens: List[str]) -> Tuple[dict, List, List]:
        """BERT模型的词映射"""
        inputs = self.tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            padding=False,
            truncation=True
        )
        
        # 获取word_ids
        word_ids = inputs.word_ids(batch_index=0)
        
        # 获取子词tokens
        subword_tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        return inputs, word_ids, subword_tokens
    
    def _get_roberta_word_mappings(self, tokens: List[str]) -> Tuple[dict, List, List]:
        """RoBERTa模型的词映射 - 改进版本"""
        word_ids = []
        all_subwords = []
        all_input_ids = []
        
        # 添加起始token
        all_input_ids.append(self.tokenizer.bos_token_id)
        all_subwords.append(self.tokenizer.bos_token)
        word_ids.append(None)
        
        # 对每个词单独tokenize
        for word_idx, word in enumerate(tokens):
            # RoBERTa需要在词前加空格（除了第一个词）
            if word_idx == 0:
                word_to_tokenize = word
            else:
                word_to_tokenize = ' ' + word
            
            # Tokenize单个词
            word_tokens = self.tokenizer.tokenize(word_to_tokenize)
            word_input_ids = self.tokenizer.convert_tokens_to_ids(word_tokens)
            
            # 记录映射关系
            for token, token_id in zip(word_tokens, word_input_ids):
                all_subwords.append(token)
                all_input_ids.append(token_id)
                word_ids.append(word_idx)
        
        # 添加结束token
        all_input_ids.append(self.tokenizer.eos_token_id)
        all_subwords.append(self.tokenizer.eos_token)
        word_ids.append(None)
        
        # 创建输入张量
        inputs = {
            'input_ids': torch.tensor([all_input_ids]),
            'attention_mask': torch.ones(1, len(all_input_ids), dtype=torch.long)
        }
        
        return inputs, word_ids, all_subwords
    
    def get_attention_weights(self, tokens: List[str]) -> torch.Tensor:
        """获取词级别的注意力权重"""
        # 获取词映射
        inputs, word_ids, subword_tokens = self._get_word_mappings(tokens)
        
        # 将输入移到正确的设备
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in inputs.items()}
        
        # 获取模型输出
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True, return_dict=True)
        
        # 提取注意力
        attention = self._extract_attention_from_outputs(outputs)
        
        # 聚合到单词级别
        word_attention = self.aggregate_to_words(attention, word_ids, len(tokens))
        
        return word_attention
    
    def aggregate_to_words(self, attention: torch.Tensor, word_ids: List, num_words: int) -> torch.Tensor:
        """将子词级别的注意力聚合到词级别"""
        word_attention = torch.zeros(num_words, num_words)
        
        # 创建词到token的映射
        word_to_tokens = defaultdict(list)
        for token_idx, word_idx in enumerate(word_ids):
            if word_idx is not None:
                word_to_tokens[word_idx].append(token_idx)
        
        # 确保所有词都有映射
        for word_idx in range(num_words):
            if word_idx not in word_to_tokens:
                print(f"Warning: Word {word_idx} has no token mappings")
        
        # 聚合注意力
        for word_i in range(num_words):
            for word_j in range(num_words):
                tokens_i = word_to_tokens.get(word_i, [])
                tokens_j = word_to_tokens.get(word_j, [])
                
                if tokens_i and tokens_j:
                    attn_values = []
                    for tok_i in tokens_i:
                        for tok_j in tokens_j:
                            if tok_i < attention.shape[0] and tok_j < attention.shape[1]:
                                attn_values.append(attention[tok_i, tok_j].item())
                    
                    if attn_values:
                        # 使用平均值聚合
                        word_attention[word_i, word_j] = sum(attn_values) / len(attn_values)
        
        # 行归一化
        for i in range(num_words):
            row_sum = word_attention[i].sum()
            if row_sum > 0:
                word_attention[i] = word_attention[i] / row_sum
            else:
                # 如果行和为0，使用均匀分布
                word_attention[i] = torch.ones(num_words) / num_words
        
        return word_attention
    
    def extract_word_importance(self, tokens: List[str], 
                               aggregation: str = 'sum') -> np.ndarray:
        """
        提取每个词的重要性分数
        
        Args:
            tokens: 词列表
            aggregation: 如何聚合注意力 ('sum', 'mean', 'max')
        
        Returns:
            每个词的重要性分数
        """
        attention = self.get_attention_weights(tokens)
        
        if aggregation == 'sum':
            # 每个词被其他词关注的总和
            importance = attention.sum(dim=0)
        elif aggregation == 'mean':
            importance = attention.mean(dim=0)
        elif aggregation == 'max':
            importance = attention.max(dim=0)[0]
        else:
            importance = attention.sum(dim=0)
        
        # 归一化
        importance = importance / importance.sum()
        
        return importance.cpu().numpy()
        
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
    extractor = SpanHeadExtractor(model_name='/home/user/1019_wp/HiSA/roberta-base')

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
     # 获取词重要性
    importance = extractor.extract_word_importance(tokens)
    print("Word importance:")
    for token, score in zip(tokens, importance):
        print(f"  {token}: {score:.4f}")
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
