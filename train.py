"""
    @author: Jevis_Hoo
    @Date: 2020/7/11 12:50
    @Description:
"""

import random
import logging
import os
import torch
from torch.optim import Adam
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from config import Config
from model import BERT_LSTM_CRF
from bert.modeling import BertForTokenClassification
from bert.tokenization import BertTokenizer
from torch.utils.data import DataLoader, RandomSampler
from apex import amp
import utils
from metrics import f1_score
from metrics import classification_report
import numpy as np


def evaluate(model, data_iterator, config, mark='Eval', verbose=False):
    """Evaluate the model on `steps` batches."""
    # set model to evaluation mode
    model.eval()

    idx2tag = config.idx2tag

    true_tags = []
    pred_tags = []

    # a running average object for loss
    loss_avg = utils.RunningAverage()

    # Use tqdm for progress bar
    for step, batch in enumerate(tqdm(data_iterator, desc="Iteration")):
        # fetch the next training batch
        batch_ids, batch_masks, segment_ids, batch_labels = batch

        loss = model(batch_ids, labels=batch_labels, attention_mask=batch_masks)

        if config.n_gpu > 1 and config.multi_gpu:
            loss = loss.mean()
        loss_avg.update(loss.item())
        batch_output = model(batch_ids, attention_mask=batch_masks)

        batch_output = batch_output.detach().cpu().numpy()
        batch_labels = batch_labels.to('cpu').numpy()

        print("================================")
        print(batch_output)
        print(len(np.array(batch_output)[0]))
        print(len(np.array(batch_output)[0][0]))
        print("================================")
        labels = [idx for indices in np.argmax(batch_output, axis=2) for idx in indices]
        print(labels)
        # print(true_labels)
        print("================================")

        pred_tags.extend([idx2tag.get(idx) for indices in np.argmax(batch_output, axis=2) for idx in indices])
        pred_tags.extend([idx2tag.get(idx) for indices in batch_output for idx in indices])
        true_tags.extend([idx2tag.get(idx) for indices in batch_labels for idx in indices])
    assert len(pred_tags) == len(true_tags)

    # logging loss, f1 and report
    metrics = {}
    f1 = f1_score(true_tags, pred_tags)
    metrics['loss'] = loss_avg()
    metrics['f1'] = f1
    metrics_str = "; ".join("{}: {:05.2f}".format(k, v) for k, v in metrics.items())
    logging.info("- {} metrics: ".format(mark) + metrics_str)

    if verbose:
        report = classification_report(true_tags, pred_tags)
        logging.info(report)
    return metrics


def train(model, data_iterator, optimizer, scheduler, config):
    """Train the model on `steps` batches"""
    # set model to training mode
    model.train()

    # a running average object for loss
    loss_avg = utils.RunningAverage()

    # Use tqdm for progress bar
    for step, batch in enumerate(tqdm(data_iterator, desc="Iteration")):
        # fetch the next training batch
        batch_ids, batch_masks, segment_ids, batch_labels = batch


        # compute model output and loss
        loss = model(batch_ids, labels=batch_labels, attention_mask=batch_masks)

        if config.n_gpu > 1 and config.multi_gpu:
            loss = loss.mean()  # mean() to average on multi-gpu

        # clear previous gradients, compute gradients of all variables wrt loss
        model.zero_grad()
        if config.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            # optimizer.backward(loss)
        else:
            loss.backward()

        # gradient clipping
        nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=config.clip_grad)

        # performs updates using calculated gradients
        optimizer.step()
        scheduler.step()

        # update the average loss
        loss_avg.update(loss.item())
        tqdm.set_postfix(loss='{:05.3f}'.format(loss_avg()), lr='{}'.format(scheduler.get_last_lr()))


def train_and_evaluate(model, train_data, val_data, optimizer, scheduler, config, load_path=None):
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
    print("  Num examples = {}".format(len(train_data)))
    print("  Num Epochs = {}".format(config.epoch_num))

    for epoch in range(1, config.epoch_num + 1):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch, config.epoch_num))

        # Train for one epoch on training set
        train(model, train_data, optimizer, scheduler, config)

        val_metrics = evaluate(model, val_data, config, mark='Eval')

        val_f1 = val_metrics['f1']
        improve_f1 = val_f1 - best_val_f1

        # Save weights of the network
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        optimizer_to_save = optimizer.optimizer if config.fp16 else optimizer
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model_to_save.state_dict(),
                               'optim_dict': optimizer_to_save.state_dict()},
                              is_best=improve_f1 > 0,
                              checkpoint=config.checkpoint)
        if improve_f1 > 0:
            logging.info("- Found new best F1")
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
    utils.set_logger(os.path.join(config.checkpoint, 'train.log'))
    logging.info("device: {}, n_gpu: {}, 16-bits training: {}".format(config.device, config.n_gpu, config.fp16))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    processor = utils.NerProcessor()
    label_list = processor.get_labels(config)
    num_labels = len(label_list)
    config.label_list = label_list
    config.tokenizer_name = ''

    tokenizer = BertTokenizer.from_pretrained(config.bert_path, do_lower_case=True)

    # Initialize the DataLoader
    train_examples, train_features, train_data = utils.get_Dataset(config, processor, tokenizer, mode="train")
    eval_examples, eval_features, eval_data = utils.get_Dataset(config, processor, tokenizer, mode="eval")

    train_sampler = RandomSampler(train_data)
    train_data_loader = DataLoader(train_data, sampler=train_sampler, batch_size=config.batch_size,
                                   num_workers=config.n_gpu)
    eval_data_loader = DataLoader(eval_data, batch_size=config.batch_size, num_workers=config.n_gpu)

    # Set the random seed for reproducible experiments
    random.seed(config.seed)
    torch.manual_seed(config.seed)  # For CPU

    if config.multi_gpu:
        torch.cuda.manual_seed_all(config.seed)  # set random seed for all GPUs

    # Prepare model
    model = BERT_LSTM_CRF(config.bert_path, config.relation_num, config.bert_embedding, config.rnn_hidden,
                          config.rnn_layer, dropout=config.dropout, use_cuda=use_cuda)
    # model = BertForTokenClassification.from_pretrained(config.bert_path, num_labels=config.relation_num)
    model.to(config.device)

    # Prepare optimizer
    if config.full_finetuning:
        param_optimizer = list(model.named_parameters())
        # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [value for key, value in param_optimizer if 'word_embedding' in key],
             'lr': config.learning_rate},
            # 'weight_decay_rate': 0.01},
            {'params': [value for key, value in param_optimizer if 'word_embedding' not in key],
             'lr': config.embed_learning_rate}
            # 'weight_decay_rate': 0.0}
        ]
    else:
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{'params': [key for key, value in param_optimizer]}]

    # logging.info(optimizer_grouped_parameters)
    lambda1 = lambda epoch: 1 / (1 + 0.005 * epoch)
    lambda2 = lambda epoch: (3000 + epoch) // 300 * 0.1

    if config.fp16:
        optimizer = Adam(optimizer_grouped_parameters)
        # optimizer = FusedAdam(optimizer_grouped_parameters,
        #                       lr=config.learning_rate,
        #                       bias_correction=False)
        # scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (1 + 0.05 * epoch))
        scheduler = LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])
        model, optimizer = amp.initialize(model, optimizer, opt_level='O2')

    else:
        optimizer = Adam(optimizer_grouped_parameters, lr=config.learning_rate)
        scheduler = LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])

    if config.multi_gpu:
        model = torch.nn.DataParallel(model)

    # Train and evaluate the model
    logging.info(config)
    logging.info("Starting training for {} epoch(s)".format(config.epoch_num))
    train_and_evaluate(model, train_data, eval_data, optimizer, scheduler, config, config.load_path)
