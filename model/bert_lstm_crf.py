# coding=utf-8
import torch.nn as nn
from bert.modeling import BertModel
from model import CRF
import torch


class BERT_LSTM_CRF(nn.Module):
    """
    bert_lstm_crf model
    """

    def __init__(self, bert_config, num_labels, embedding_dim, hidden_dim, rnn_layers, dropout, use_cuda=False):
        super(BERT_LSTM_CRF, self).__init__()

        self.rnn_layers = rnn_layers
        self.hidden_dim = hidden_dim

        self.bert_embed = BertModel.from_pretrained(bert_config)
        # self.fc = nn.Sequential(
        #     nn.Linear(embedding_dim * 2, embedding_dim),
        #     nn.ReLU(inplace=True),
        # )

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=rnn_layers, bidirectional=True, batch_first=True)
        out_dim = hidden_dim * 2
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(out_dim, num_labels + 2)
        self.crf = CRF(target_size=num_labels, use_cuda=use_cuda)

    def rand_init_hidden(self, batch_size):
        """
        random initialize hidden variable
        """
        return torch.randn(2 * self.rnn_layers, batch_size, self.hidden_dim), \
               torch.randn(2 * self.rnn_layers, batch_size, self.hidden_dim)

    def forward(self, input_ids, labels=None, attention_mask=None):
        """
        args:
            input_ids (word_seq_len, batch_size) : word-level representation of sentence
            hidden: initial hidden state

        return:
            crf loss
            output
        """
        batch_size = input_ids.size(0)
        hidden = self.rand_init_hidden(batch_size)

        sequence_output, _ = self.bert_embed(input_ids, attention_mask=attention_mask, output_all_encoded_layers=False)
        # sequence_output = self.fc(sequence_output)

        if sequence_output.is_cuda:
            hidden = [i.cuda() for i in hidden]

        self.lstm.flatten_parameters()
        # lstm_out, _ = self.lstm(sequence_output)
        lstm_out, _ = self.lstm(sequence_output, hidden)
        dropout_lstm_out = self.dropout(lstm_out)
        logits = self.classifier(dropout_lstm_out)

        if labels is not None:
            loss_value = self.crf.neg_log_likelihood_loss(logits, attention_mask, labels)
            batch_size = logits.size(0)
            loss_value /= float(batch_size)
            return loss_value
        else:
            return logits

    def predict(self, logits, attention_mask):
        return self.crf.decode(logits, attention_mask)
