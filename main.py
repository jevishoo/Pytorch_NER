"""
    @author: Jevis_Hoo
    @Date: 2020/7/11 12:50
    @Description:
"""

import random
import logging
import time
import torch
from torch.optim import Adam
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from tqdm import trange
from config import Config
from model import BERT_LSTM_CRF, BiLSTMCRF
from opt import OpenAIAdam
from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule
from pytorch_pretrained_bert import BertTokenizer
from bert.modeling import BertForTokenClassification
from data_loader import DataLoader
from evaluate import evaluate
from embedding import build_word_embed
import utils
import os
import math

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"


def train(model, data_iterator, optimizer, scheduler, config, normal_optimizer=None):
    """Train the model on `steps` batches"""
    # set model to training mode
    model.train()

    # a running average object for loss
    loss_avg = utils.RunningAverage()

    # Use tqdm for progress bar
    t = trange(config.train_steps)
    for _ in t:
        # fetch the next training batch
        batch_ids, batch_labels = next(data_iterator)
        batch_masks = batch_ids.gt(0)

        # compute model output and los
        loss = model(batch_ids, labels=batch_labels, attention_mask=batch_masks)

        if config.n_gpu > 1 and config.multi_gpu:
            loss = loss.mean()  # mean() to average on multi-gpu

        # clear previous gradients, compute gradients of all variables wrt loss
        model.zero_grad()

        loss.backward()

        # gradient clipping
        nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=config.clip_grad)

        # performs updates using calculated gradients
        optimizer.step()
        if normal_optimizer:
            normal_optimizer.step()
        scheduler.step()

        # update the average loss
        loss_avg.update(loss.item())
        t.set_postfix(loss='{:05.3f}'.format(loss_avg()), lr='{}'.format(scheduler.get_last_lr()[0]))
        # show_lr = scheduler.get_last_lr()
        # t.set_postfix(loss='{:05.3f}'.format(loss_avg()),
        #               lr='{.6f}{.6f}{.6f}{.6f}'.format(show_lr[0], show_lr[1], show_lr[2], show_lr[3]))


def train_and_evaluate(model, train_data, val_data, optimizer, scheduler, config, timestamp, normal_optimizer=None,
                       load_path=None):
    """Train the model and evaluate every epoch."""
    # reload weights from restore_file if specified
    if load_path is not None:
        restore_path = os.path.join(load_path + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    best_val_f1 = 0.0
    patience_counter = 0

    # Train!
    print("***** Running training *****")
    print("  Num train data size = {}".format(train_data['size']))
    print("  Num Epochs = {}".format(config.epoch_num))

    for epoch in range(1, config.epoch_num + 1):
        # Run one epoch
        print("Epoch {}/{}".format(epoch, config.epoch_num))

        # Compute number of batches in one epoch
        config.train_steps = config.train_size // config.batch_size
        config.val_steps = config.val_size // config.batch_size
        print("train_steps: {}".format(config.train_steps))
        print("val_steps: {}".format(config.val_steps))

        # data iterator for training
        # train_data_iterator = DataLoader(train_data, batch_size=config.batch_size)

        train_data_iterator = data_loader.data_iterator(train_data, shuffle=True)
        # Train for one epoch on training set
        train(model, train_data_iterator, optimizer, scheduler, config, normal_optimizer=normal_optimizer)

        # data iterator for evaluation
        val_data_iterator = data_loader.data_iterator(val_data, shuffle=False)
        tokens = val_data['token']

        # Evaluate for one epoch on training set and validation set
        config.eval_steps = config.train_steps

        # train_metrics = evaluate(model, train_data_iterator, config, mark='Train')
        config.eval_steps = config.val_steps

        val_metrics, val_result = evaluate(model, val_data_iterator, tokens, config, mark='Val')

        val_f1 = val_metrics['f1']
        improve_f1 = val_f1 - best_val_f1

        # Save weights of the network
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model_to_save.state_dict(),
                               },
                              is_best=improve_f1 > 0,
                              checkpoint=config.checkpoint,
                              timestamp=timestamp)

        if improve_f1 > 0:
            print("- Found new best F1")
            val_result.to_csv(config.output + timestamp + '_dev_result.csv', encoding='utf-8')
            best_val_f1 = val_f1
            if improve_f1 < config.patience:
                patience_counter += 1
            else:
                patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping and logging best f1
        if (patience_counter >= config.patience_num and epoch > config.min_epoch_num) or epoch == config.epoch_num:
            logging.info("Best val f1: {:05.2f}".format(best_val_f1))
            break


if __name__ == '__main__':
    config = Config()

    # Use GPUs if available
    # config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if config.device == 'cuda':
        use_cuda = True
    else:
        use_cuda = False

    # Multi_GPU or not
    if config.multi_gpu:
        config.n_gpu = torch.cuda.device_count()

    # Set the logger
    timestamp = str(int(time.time()))
    print(timestamp)
    if not os.path.exists(os.path.join(config.checkpoint)):
        os.makedirs(os.path.join(config.checkpoint))
    utils.set_logger(os.path.join(config.checkpoint, timestamp + '.train.log'))
    print("device: {}, n_gpu: {}".format(config.device, config.n_gpu))

    # Create the input data pipeline
    print("Loading the dataset...")

    # Initialize the DataLoader
    data_loader = DataLoader(config.data_dir, config.bert_path, config, token_pad_idx=0)
    # Load training data and test data

    train_data = data_loader.load_data('train')
    val_data = data_loader.load_data('val')

    # Specify the training and validation dataset sizes
    config.train_size = train_data['size']
    config.val_size = val_data['size']
    config.num_train_steps = (config.train_size // config.batch_size) * config.epoch_num

    # Set the random seed for reproducible experiments
    random.seed(config.seed)
    torch.manual_seed(config.seed)  # For CPU

    if config.multi_gpu:
        torch.cuda.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)  # set random seed for all GPUs

    # Prepare model
    """
        param: len(config.tag2idx) 实体种类
    """
    tokenizer = BertTokenizer.from_pretrained(config.bert_path, do_lower_case=True)

    if config.use_pretrained_embedding:
        pretrained_word_embed = build_word_embed(tokenizer,
                                                 pretrain_embed_file=config.pretrain_embed_file,
                                                 pretrain_embed_pkl=config.pretrain_embed_pkl)
        config.word_vocab_size = pretrained_word_embed.shape[0]
        config.word_embedding_dim = pretrained_word_embed.shape[1]
        model = BiLSTMCRF(config, pretrained_word_embed)
    else:
        model = BERT_LSTM_CRF(config.bert_path, len(config.tag2idx), config.bert_embedding, config.rnn_hidden,
                              config.rnn_layer, dropout=config.dropout, use_cuda=use_cuda)
    # model = BertForTokenClassification.from_pretrained(config.bert_path, num_labels=len(config.tag2idx))
    model.to(config.device)

    # Prepare optimizer
    if config.use_pretrained_embedding:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': config.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=config.learning_rate,
                          correct_bias=False)

        scheduler = WarmupLinearSchedule(optimizer,
                                         warmup_steps=int(config.num_train_steps * config.warmup_proportion),
                                         t_total=config.num_train_steps)
    else:
        if config.full_finetuning:
            param_optimizer = list(model.named_parameters())

            bert_grouped_parameters = [
                {'params': [value for key, value in param_optimizer if 'word_embedding' in key],
                 'lr': config.embed_learning_rate}
            ]
            normal_grouped_parameters = [
                {'params': [value for key, value in param_optimizer if 'word_embedding' not in key],
                 'lr': config.learning_rate}
            ]
        else:
            param_optimizer = list(model.classifier.named_parameters())
            bert_grouped_parameters = []
            normal_grouped_parameters = [{'params': [key for key, value in param_optimizer]}]

        # logging.info(optimizer_grouped_parameters)
        lambda1 = lambda epoch: 1 / (1 + 0.005 * epoch)
        lambda2 = lambda epoch: math.pow(0.1, epoch // 300)  # 300步减小lr

        optimizer = OpenAIAdam(bert_grouped_parameters,
                               schedule="warmup_cosine",
                               warmup=0.05,
                               t_total=config.num_train_steps,
                               l2=0.01,
                               max_grad_norm=1)
        normal_optimizer = Adam(normal_grouped_parameters)
        # scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (1 + 0.005 * epoch))
        # scheduler = LambdaLR(normal_optimizer, lr_lambda=[lambda1])
        scheduler = WarmupLinearSchedule(optimizer,
                                         warmup_steps=int(config.num_train_steps * config.warmup_proportion),
                                         t_total=config.num_train_steps)

    if config.multi_gpu:
        model = torch.nn.DataParallel(model)

    # Train and evaluate the model
    logging.info(config)
    if config.use_pretrained_embedding:
        train_and_evaluate(model, train_data, val_data, optimizer, scheduler, config, timestamp,
                           load_path=config.load_path)
    else:
        train_and_evaluate(model, train_data, val_data, optimizer, scheduler, config, timestamp,
                           normal_optimizer=normal_optimizer, load_path=config.load_path)
