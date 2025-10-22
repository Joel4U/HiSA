from dataclasses import dataclass
from typing import List, Dict
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast
B_PREF = "B-"
I_PREF = "I-"
S_PREF = "S-"
E_PREF = "E-"
O = "O"

@dataclass
class Instance:
    words: List[str]
    labels: List[str] = None
    prediction: List[str] = None

def convert_iobes(labels: List[str]) -> List[str]:
    for pos in range(len(labels)):
        curr_entity = labels[pos]
        if pos == len(labels) - 1:
            if curr_entity.startswith(B_PREF):
                labels[pos] = curr_entity.replace(B_PREF, S_PREF)
            elif curr_entity.startswith(I_PREF):
                labels[pos] = curr_entity.replace(I_PREF, E_PREF)
        else:
            next_entity = labels[pos + 1]
            if curr_entity.startswith(B_PREF):
                if next_entity.startswith(O) or next_entity.startswith(B_PREF):
                    labels[pos] = curr_entity.replace(B_PREF, S_PREF)
            elif curr_entity.startswith(I_PREF):
                if next_entity.startswith(O) or next_entity.startswith(B_PREF):
                    labels[pos] = curr_entity.replace(I_PREF, E_PREF)
    return labels

def convert_instances_to_feature_tensors(
    instances: List[Instance],
    tokenizer: PreTrainedTokenizerFast,
    label2idx: Dict[str, int],
) -> List[Dict]:
    features = []
    for idx, inst in enumerate(instances):
        words = inst.words
        orig_to_tok_index = []
        res = tokenizer.encode_plus(words, is_split_into_words=True)
        subword_idx2word_idx = res.word_ids(batch_index=0)
        prev_word_idx = -1
        for i, mapped_word_idx in enumerate(subword_idx2word_idx):
            if mapped_word_idx is None:  ## cls and sep token
                continue
            if mapped_word_idx != prev_word_idx:
                orig_to_tok_index.append(i)
                prev_word_idx = mapped_word_idx
        assert len(orig_to_tok_index) == len(words)
        labels = inst.labels
        label_ids = (
            [label2idx[label] for label in labels] if labels else [-100] * len(words)
        )
        segment_ids = [0] * len(res["input_ids"])

        features.append(
            {
                "input_ids": res["input_ids"],
                "attention_mask": res["attention_mask"],
                "orig_to_tok_index": orig_to_tok_index,
                "token_type_ids": segment_ids,
                "word_seq_len": len(orig_to_tok_index),
                "label_ids": label_ids,
            }
        )
    return features

class TransformersNERDataset(Dataset):
    def __init__( self, file: str, tokenizer, ent2id, model_name='bert'):
        """
        sents: we use sentences if we want to build dataset from sentences directly instead of file
        """
        self.tokenizer = tokenizer
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
        self.insts = self.read_file(file=file)
        self.insts_ids = convert_instances_to_feature_tensors(self.insts, self.tokenizer, ent2id)
        return self.insts

    def read_file(self, file: str, number: int = -1) -> List[Instance]:
        insts = []
        with open(file, "r", encoding="utf-8") as f:
            words = []
            labels = []
            for line in tqdm(f.readlines()):
                line = line.rstrip()
                if line == "":
                    labels = convert_iobes(labels)
                    insts.append(
                        Instance(words=words, labels=labels)
                    )
                    words = []
                    labels = []
                    if len(insts) == number:
                        break
                    continue
                ls = line.split()
                word, label = ls[1], ls[-1]
                words.append(word)
                labels.append(label)
        return insts

    def __len__(self):
        return len(self.insts_ids)

    def __getitem__(self, index):
        return self.insts_ids[index]

    def collate_fn(self, batch: List[Dict]):
        word_seq_len = [len(feature["orig_to_tok_index"]) for feature in batch]
        max_seq_len = max(word_seq_len)
        max_wordpiece_length = max([len(feature["input_ids"]) for feature in batch])
        for i, feature in enumerate(batch):
            padding_length = max_wordpiece_length - len(feature["input_ids"])
            input_ids = (
                feature["input_ids"] + [self.tokenizer.pad_token_id] * padding_length
            )
            mask = feature["attention_mask"] + [0] * padding_length
            type_ids = (
                feature["token_type_ids"]
                + [self.tokenizer.pad_token_type_id] * padding_length
            )
            padding_word_len = max_seq_len - len(feature["orig_to_tok_index"])
            orig_to_tok_index = feature["orig_to_tok_index"] + [0] * padding_word_len
            label_ids = feature["label_ids"] + [0] * padding_word_len

            batch[i] = {
                "input_ids": input_ids,
                "attention_mask": mask,
                "token_type_ids": type_ids,
                "orig_to_tok_index": orig_to_tok_index,
                "word_seq_len": feature["word_seq_len"],
                "label_ids": label_ids,
            }
        encoded_inputs = {
            key: [example[key] for example in batch] for key in batch[0].keys()
        }
        results = BatchEncoding(encoded_inputs, tensor_type="pt")
        return results