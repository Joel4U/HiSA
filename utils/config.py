import json


class Config:
    def __init__(self, args):
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)

        self.task = config["task"]
        self.json_flag = config["json_flag"]
        self.ENT_CLS_NUM = config["ENT_CLS_NUM"]
        self.ent2id = config["ent2id"]
        self.deplabel2id = config["deplabel2id"]
        self.id2deplabel =config["id2deplabel"]
        self.bert_name = config["bert_name"]
        self.lr = config["bert_learning_rate"]
        self.n_head = config["n_head"]
        self.batch_size = config["batch_size"]
        self.n_epochs = config["n_epochs"]
        self.device = config["device"]
        self.warmup = args.warmup
        self.cnn_depth = args.cnn_depth
        self.cnn_dim = args.cnn_dim
        self.size_embed_dim = args.size_embed_dim
        self.logit_drop = args.logit_drop
        self.biaffine_size = args.biaffine_size

    def __repr__(self):
        return "{}".format(self.__dict__.items())