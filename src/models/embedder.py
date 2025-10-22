import torch
import torch.nn as nn
from transformers import AutoModel
import logging

logger = logging.getLogger(__name__)

class PLMEmbedder(nn.Module):
    """
    Encode the input with transformers model such as
    BERT, Roberta, and so on.
    """

    def __init__(self, encoder_name: str):
        super(PLMEmbedder, self).__init__()
        output_hidden_states = False ## to use all hidden states or not
        logger.info(f"[Model Info] Loading pretrained language model {encoder_name}")
        self.model = AutoModel.from_pretrained(encoder_name, output_hidden_states=output_hidden_states, return_dict=True)
        """
        use the following line if you want to freeze the model, 
        but don't forget also exclude the parameters in the optimizer
        """
        # self.model.requires_grad = False

    def get_output_dim(self):
        ## use differnet model may have different attribute
        ## for example, if you are using GPT, it should be self.model.config.n_embd
        ## Check out https://huggingface.co/transformers/model_doc/gpt.html
        ## But you can directly write it as 768 as well.
        return self.model.config.hidden_size

    def forward(self, subword_input_ids: torch.Tensor,
                orig_to_token_index: torch.LongTensor,  ## batch_size * max_seq_leng
                attention_mask: torch.LongTensor) -> torch.Tensor:
        """

        :param subword_input_ids: (batch_size x max_wordpiece_len x hidden_size) the input id tensor
        :param orig_to_token_index: (batch_size x max_sent_len x hidden_size) the mapping from original word id map to subword token index
        :param attention_mask: (batch_size x max_wordpiece_len)
        :return:
        """
        subword_rep = self.model(**{"input_ids": subword_input_ids, "attention_mask": attention_mask}).last_hidden_state
        # 检查BERT输出
        # print(f"\nBERT output stats:")
        # print(f"  - Shape: {subword_rep.shape}")
        # print(f"  - Min: {subword_rep.min():.4f}, Max: {subword_rep.max():.4f}")
        # print(f"  - Mean: {subword_rep.mean():.4f}, Std: {subword_rep.std():.4f}")

        batch_size, _, rep_size = subword_rep.size()
        _, max_sent_len = orig_to_token_index.size()
        # 创建有效词的mask（-1表示padding位置）
        valid_word_mask = orig_to_token_index >= 0
        # 将-1替换为0，避免gather索引错误
        # 使用0是安全的，因为后面会用mask将这些位置置零
        safe_indices = torch.where(valid_word_mask, orig_to_token_index, 0)
        # select the word index. 使用gather获取对应位置的表示
        # word_rep = torch.gather(subword_rep[:, :, :], 1, orig_to_token_index.unsqueeze(-1).expand(batch_size, max_sent_len, rep_size))
        word_rep = torch.gather(subword_rep, 1, safe_indices.unsqueeze(-1).expand(batch_size, max_sent_len, rep_size))
        # 将padding位置（原本是-1的位置）的表示设为0
        word_rep = word_rep * valid_word_mask.unsqueeze(-1).float()
        return word_rep