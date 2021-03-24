"""
    @author: Jevis_Hoo
    @Date: 2020/7/21 16:58
    @Description: 
"""

"""
    @author: Jevis_Hoo
    @Date: 2020/7/11 13:16
    @Description: 
"""

import logging
import os
import random
import pandas as pd
import numpy as np
from tqdm import trange
from data_loader import DataLoader
from config import Config
from model import BERT_LSTM_CRF, CRF
import torch
import utils

logger = logging.getLogger(__name__)
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6"


def get_label(num):
    if num in [2, 3]:
        return "试验要素"
    elif num in [4, 5]:
        return "性能指标"
    elif num in [6, 7]:
        # if num in [2, 3, 4, 5]:
        #     return "试验要素"
        # elif num in [6, 7, 8, 9]:
        #     return "性能指标"
        # elif num in [10, 11, 12, 13]:
        return "系统组成"
    else:
        return "任务场景"


def get_labels_index(y_label_list, tokens):
    y_entity_list = []  # 标签
    start_pos_list = []  # 开始位置索引
    end_pos_list = []  # 结束位置索引

    len_list = [len(labels) for labels in y_label_list]
    print("label list's length: {}".format(len(len_list)))

    for i, input_tokens in enumerate(tokens):
        ys = y_label_list[i]  # 每条数据对应的数字标签列表
        temp = []
        label_list = []

        s_list = []
        e_list = []
        is_start = False

        for index, num in enumerate(ys):
            if (num in [1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16]) and len(temp) == 0:
                s_list.append(index)
                is_start = True
                temp.append(input_tokens[index])
            elif (num in [1, 3, 5, 7, 9, 12, 13, 15]) and len(temp) > 0:
                temp.append(input_tokens[index])
            elif len(temp) > 0:
                if is_start:
                    e_list.append(index - 1)
                is_start = False

                label_list.append("".join(temp))
                temp = []

        y_entity_list.append(";".join(label_list))
        start_pos_list.append(s_list)
        end_pos_list.append(e_list)

    return y_label_list, y_entity_list, start_pos_list, end_pos_list


def predict(model, data_iterator, config):
    """Evaluate the model on `steps` batches."""
    # set model to evaluation mode
    model.eval()
    print("***** Running test *****")
    logger.info("test step: {}".format(config.test_steps))

    predict_labels = []

    # a running average object for loss
    t = trange(config.test_steps)
    for _ in t:
        # fetch the next evaluation batch
        batch_ids, batch_labels = next(data_iterator)
        batch_masks = batch_ids.gt(0)

        # shape: (batch_size, max_len, num_labels)
        batch_output = model(batch_ids, attention_mask=batch_masks)

        batch_output = crf_func.decode(batch_output, batch_masks)
        batch_output = batch_output.detach().cpu().numpy()
        for l in batch_output:
            pred_label = []
            for idx in l:
                pred_label.append(idx)
            predict_labels.append(pred_label)
        # predict_labels.extend([idx for indices in np.argmax(batch_output, axis=2) for idx in indices])

    return predict_labels


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

    # Set the random seed for reproducible experiments
    random.seed(config.seed)
    torch.manual_seed(config.seed)  # For CPU

    # Set the logger
    utils.set_logger(os.path.join(config.checkpoint, 'predict.log'))

    if config.multi_gpu:
        torch.cuda.manual_seed_all(config.seed)  # set random seed for all GPUs

    # Create the input data pipeline
    logging.info("Loading the dataset...")

    # Initialize the DataLoader
    data_loader = DataLoader(config.data_dir, config.bert_path, config, token_pad_idx=0)
    # Load training data and test data

    test_data = data_loader.load_data('test')
    config.test_size = test_data['size']
    config.test_steps = config.test_size // config.batch_size

    tokens = test_data['token']
    sentences = test_data['text']

    test_data_iterator = data_loader.data_iterator(test_data, shuffle=False)
    crf_func = CRF(target_size=len(config.tag2idx), use_cuda=True)

    model = BERT_LSTM_CRF(config.bert_path, len(config.tag2idx), config.bert_embedding, config.rnn_hidden,
                          config.rnn_layer, dropout=config.dropout, use_cuda=use_cuda)
    # model = BertForTokenClassification.from_pretrained(config.bert_path, num_labels=len(config.tag2idx))
    model.to(config.device)

    utils.load_checkpoint(os.path.join(config.checkpoint, config.restore_file + '.best.pth.tar'), model)

    if config.multi_gpu:
        model = torch.nn.DataParallel(model)

    logging.info("Starting prediction...")

    labels = predict(model, test_data_iterator, config)
    y_pred_list, y_pred_entity_list, pred_start_pos_list, pred_end_pos_list = get_labels_index(labels, tokens)

    y_pred_label_list = [i for i in y_pred_list if i != []]

    label_type_list = []
    for i in range(len(pred_start_pos_list)):
        label_list = []
        for j in range(len(pred_start_pos_list[i])):
            label_type_num = y_pred_label_list[i][pred_start_pos_list[i][j]]
            # print(label_type_num)
            label_type = get_label(label_type_num)
            label_list.append(label_type)
        label_type_list.append(label_list)

    dict_data = {
        'label_type': label_type_list,
        'start_pos': pred_start_pos_list,
        'end_pos': pred_end_pos_list,
        'entities': y_pred_entity_list,
        'originalText': sentences,
    }
    df = pd.DataFrame(dict_data)
    df = df.fillna("0")
    df.to_csv(config.output + '/test_result.csv', encoding='utf-8')
