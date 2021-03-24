class Config(object):
    def __init__(self):
        self.data_dir = '/home/hezoujie/Competition/Pytorch_NER/data/ccks'
        self.vocab_file = '/home/hezoujie/Models/roberta_pytorch/vocab.txt'
        self.bert_path = '/home/hezoujie/Models/roberta_pytorch'
        self.checkpoint = '/home/hezoujie/Competition/Pytorch_NER/output/ccks/model'
        self.output = '/home/hezoujie/Competition/Pytorch_NER/output/ccks/result'
        self.seed = 121
        self.full_finetuning = True
        self.sequence_length = 360
        self.epoch_num = 20
        self.min_epoch_num = 5
        self.batch_size = 4

        self.device = 'cuda'  # cpu
        self.n_gpu = 3
        self.multi_gpu = True

        self.clip_grad = 2
        self.warmup_proportion = 0.1
        self.rnn_hidden = 100
        self.bert_embedding = 1024
        self.dropout = 0.5
        self.rnn_layer = 1
        self.learning_rate = 5e-5
        self.embed_learning_rate = 2e-5  # BERT的微调学习率 3e-5 5e-5

        # pretrain_embed_file
        self.use_pretrained_embedding = True
        self.pretrain_embed_file = '/home/hezoujie/Models/Embedding/sgns.merge.word'
        # self.pretrain_embed_file = '/home/hezoujie/Models/Embedding/sgns.baidubaike.bigram-char'
        self.pretrain_embed_pkl = '/home/hezoujie/Competition/Pytorch_NER/data/ccks/baidubaike_pretrain_word_embeddings.pkl'

        self.word_vocab_size = None
        self.word_embedding_dim = None
        self.requires_grad = True
        # self.lr_decay = 0.00001
        self.weight_decay = 0.01
        # self.loss_scale = 0
        # "Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
        # "0 (default value): dynamic loss scaling.\n"
        # "Positive power of 2: static loss scaling value.\n")

        self.load_model = False
        self.load_path = None
        self.restore_file = "1599465012"

        # self.gradient_accumulation_steps = 1
        self.patience = 0.02
        self.patience_num = 4

    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __str__(self):
        return '\n'.join(['%s:%s' % item for item in self.__dict__.items()])
