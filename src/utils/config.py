import json
import torch

class Config:
    def __init__(self, args):
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)

        self.task = config["task"]
        self.json_flag = config["json_flag"]
        self.ENT_CLS_NUM = config["ENT_CLS_NUM"]
        self.ent2id = config["ent2id"]
        self.postag2id = config["postag2id"]
        self.deplabel2id = config["deplabel2id"]
        self.id2deplabel =config["id2deplabel"]
        self.bert_name = config["bert_name"]
        self.max_span_width= config["max_span_width"]
        self.lr = config["bert_learning_rate"]
        self.n_head = config["n_head"]
        self.batch_size = config["batch_size"]
        self.n_epochs = config["n_epochs"]
        self.device = config["device"]
        self.warmup = args.warmup
        # self.cnn_depth = args.cnn_depth
        self.num_span_attn_layers = args.num_span_attn_layers
        self.hidden_dim = args.hidden_dim
        self.size_embed_dim = args.size_embed_dim
        self.logit_drop = args.logit_drop
        self.biaffine_size = args.biaffine_size

    def __repr__(self):
        return "{}".format(self.__dict__.items())
    
from collections import defaultdict
import time

class CudaTimer:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.stats_ms = defaultdict(float)

    def section(self, name):
        return _TimerSection(self, name)

    def reset(self):
        self.stats_ms.clear()

    def report(self, topk=None, reset=False, prefix=""):
        items = sorted(self.stats_ms.items(), key=lambda x: x[1], reverse=True)
        if topk is not None:
            items = items[:topk]
        for k, v in items:
            print(f"{prefix}{k}: {v:.2f} ms")
        if reset:
            self.reset()

class _TimerSection:
    def __init__(self, timer: "CudaTimer", name: str):
        self.timer = timer
        self.name = name
        self.start = None
    
    def __enter__(self):
        if not self.timer.enabled:
            return self
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc, tb):
        if not self.timer.enabled:
            return False
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - self.start) * 1000.0
        self.timer.stats_ms[self.name] += dt_ms
        return False  # 不吞异常

# 一个空计时器，方便在不想计时时传参：
class NullTimer:
    def __init__(self):
        pass
    def section(self, name: str):
        return _NullSec()
    
class _NullSec:
    def __enter__(self):
        return self
    def __exit__(self, a, b, c):
        return False