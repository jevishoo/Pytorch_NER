# coding: utf-8

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from model import CRF
from model import BaseModel


class BiLSTM(BaseModel):

    def __init__(self, config, pretrained_word_embed=None):
        super(BiLSTM, self).__init__()

        self.config = config
        self.num_labels = len(config.tag2idx)
        self.max_seq_length = config.sequence_length
        self.hidden_dim = config.rnn_hidden
        self.use_cuda = torch.cuda.is_available()

        # word embedding layer
        self.word_embedding = nn.Embedding(num_embeddings=config.word_vocab_size,
                                           embedding_dim=config.word_embedding_dim)
        if self.config.use_pretrained_embedding:
            self.word_embedding.weight.data.copy_(torch.from_numpy(pretrained_word_embed))
            self.word_embedding.weight.requires_grad = config.requires_grad
        # dropout layer
        self.dropout_embed = nn.Dropout(config.dropout)
        self.dropout_rnn = nn.Dropout(config.dropout)
        # rnn layer
        self.rnn_layers = config.rnn_layer
        self.lstm = nn.LSTM(input_size=config.word_embedding_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=self.rnn_layers,
                            batch_first=True,
                            bidirectional=True)
        self.hidden2label = nn.Linear(self.hidden_dim * 2, self.num_labels)
        self.loss_function = CrossEntropyLoss()

    def rand_init_hidden(self, batch_size):
        """
        random initialize hidden variable
        双向是2，单向是1
        """
        if self.use_cuda:
            return (torch.zeros(2 * self.rnn_layers, batch_size, self.hidden_dim).cuda(),
                    torch.zeros(2 * self.rnn_layers, batch_size, self.hidden_dim).cuda())
        else:
            return (torch.zeros(2 * self.rnn_layers, batch_size, self.hidden_dim),
                    torch.zeros(2 * self.rnn_layers, batch_size, self.hidden_dim))

    def get_lstm_outputs(self, input_ids):
        word_embeds = self.word_embedding(input_ids)
        word_embeds = self.dropout_embed(word_embeds)

        batch_size = input_ids.size(0)
        hidden = self.rand_init_hidden(batch_size)
        self.lstm.flatten_parameters()
        lstm_outputs, hidden = self.lstm(word_embeds, hidden)
        # lstm_outputs = lstm_outputs.contiguous().view(-1, self.hidden_dim * 2)
        return lstm_outputs

    def forward(self, input_ids, input_mask, segment_ids=None):
        lstm_outputs = self.get_lstm_outputs(input_ids)
        logits = self.hidden2label(lstm_outputs)
        # return logits.view(-1, self.max_seq_length, self.num_labels)
        return logits

    def loss_fn(self, feats, mask, labels):
        loss_value = self.loss_function(feats.view(-1, self.num_labels), labels.view(-1))
        return loss_value

    def predict(self, feats, mask=None):
        return feats.argmax(-1)


class BiLSTMCRF(BaseModel):

    def __init__(self, config, pretrained_word_embed=None):
        super(BiLSTMCRF, self).__init__()

        self.config = config
        self.num_labels = len(config.tag2idx)
        self.max_seq_length = config.sequence_length
        self.use_cuda = torch.cuda.is_available()

        self.bilstm = BiLSTM(config, pretrained_word_embed)
        self.crf = CRF(target_size=self.num_labels,
                       use_cuda=self.use_cuda,
                       average_batch=False)
        self.hidden2label = nn.Linear(self.bilstm.hidden_dim * 2, self.num_labels + 2)

    def forward(self, input_ids, labels=None, attention_mask=None):
        lstm_outputs = self.bilstm.get_lstm_outputs(input_ids)
        logits = self.hidden2label(lstm_outputs)
        # return logits.view(-1, self.max_seq_length, self.num_labels + 2)

        if labels is not None:
            loss_value = self.crf.neg_log_likelihood_loss(logits, attention_mask, labels)
            batch_size = logits.size(0)
            loss_value /= float(batch_size)
            return loss_value
        else:
            return logits

    def loss_fn(self, feats, mask, labels):
        """
        Args:
            feats: size=(batch_size, seq_len, tag_size)
            mask: size=(batch_size, seq_len)
            tags: size=(batch_size, seq_len)
        """
        batch_size = feats.size(0)
        loss_value = self.crf.neg_log_likelihood_loss(feats, mask, labels) / float(batch_size)
        return loss_value

    def predict(self, feats, mask):
        path_score, best_path = self.crf(feats, mask.byte())
        return best_path
