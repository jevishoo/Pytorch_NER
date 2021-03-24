"""
    @author: Jevis_Hoo
    @Date: 2020/7/11 13:16
    @Description:
"""

import logging
import pandas as pd
import numpy as np
from tqdm import trange
from metrics import f1_score
from metrics import classification_report
import utils

logger = logging.getLogger(__name__)


def get_P_R_F(dev_pd):
    dev_pd = dev_pd.fillna("0")
    y_true_entity_list = list(dev_pd['y_true_entity'])
    y_pred_entity_list = list(dev_pd['y_pred_entity'])
    y_true_entity_type_list = list(dev_pd['y_true_entity_type'])
    y_pred_entity_type_list = list(dev_pd['y_pred_entity_type'])

    TP = 0
    FP = 0
    FN = 0

    class_dict_data = {
        "ELE": [0, 0, 0],
        "SIT": [0, 0, 0],
        "CON": [0, 0, 0],
        "IND": [0, 0, 0],
    }

    y_not_pred = []
    y_pred_true = []
    y_pred_false = []
    for i, y_true_entity in enumerate(y_true_entity_list):
        y_pred_entity = y_pred_entity_list[i].split('|')
        y_true_entity = y_true_entity.split('|')

        if y_pred_entity == ['']:
            continue

        current_TP = 0
        current_class_dict_data = {
            "ELE": 0,
            "SIT": 0,
            "CON": 0,
            "IND": 0,
        }
        y_pred_true_list = []
        temp_true, temp_false, temp_not = [], [], []

        for j, y_pred in enumerate(y_pred_entity):
            if y_pred in y_true_entity:
                # current_TP += 1  # 粗

                if y_pred_entity_type_list[i][j] == y_true_entity_type_list[i][y_true_entity.index(y_pred)]:
                    current_class_dict_data[y_pred_entity_type_list[i][j]] += 1
                    current_TP += 1  # 细

                    class_dict_data[y_pred_entity_type_list[i][j]][0] += 1  # class TP
                    y_pred_true_list.append(y_pred)
                    temp_true.append("".join(y_pred))
                else:
                    temp_false.append("".join(y_pred))
                    class_dict_data[y_true_entity_type_list[i][y_true_entity.index(y_pred)]][1] += 1  # class FP
                    FP += 1  # 细
            else:
                temp_false.append("".join(y_pred))
                class_dict_data[y_pred_entity_type_list[i][j]][1] += 1  # class FP
                FP += 1

        TP += current_TP
        FN += (len(y_true_entity) - current_TP)

        from collections import Counter
        # print(current_class_dict_data)
        # print(Counter(y_true_entity_type_list[i]))

        for pred_type in set(y_pred_entity_type_list[i]):
            class_dict_data[pred_type][2] += Counter(y_true_entity_type_list[i])[pred_type] - current_class_dict_data[
                pred_type]  # class FN

        for y_true in y_true_entity:
            if y_true not in y_pred_true_list:
                temp_not.append("".join(y_true))

        y_pred_true.append(";".join(temp_true))
        y_pred_false.append(";".join(temp_false))
        y_not_pred.append(";".join(temp_not))

    dict_data = {
        "y_not_pred": y_not_pred,
        "y_pred_true": y_pred_true,
        "y_pred_false": y_pred_false
    }

    dev_data = pd.DataFrame.from_dict(dict_data, orient='index')
    class_data = pd.DataFrame.from_dict(class_dict_data, orient='index', columns=['TP', 'FP', 'FN'])
    print(class_data)

    class_dict_prf_data = class_dict_data.copy()
    for key, values in class_dict_prf_data.items():
        try:
            p = class_dict_data[key][0] / (class_dict_data[key][0] + class_dict_data[key][1])
        except:
            p = 0

        try:
            r = class_dict_data[key][0] / (class_dict_data[key][0] + class_dict_data[key][2])
        except:
            r = 0

        try:
            f = 2 * p * r / (p + r)
        except:
            f = 0

        values[0] = p
        values[1] = r
        values[2] = f

    class_prf_data = pd.DataFrame.from_dict(class_dict_prf_data, orient='index', columns=['P', 'R', 'F1'])
    print(class_prf_data)

    P = TP / (TP + FP)
    R = TP / (TP + FN)
    try:
        F = 2 * P * R / (P + R)
    except:
        F = 0

    return P, R, F, dev_data


def get_text_and_entity(y_label_list, input_tokens_list):
    y_entity_list = []  # 标签
    start_pos_list = []  # 开始位置索引
    end_pos_list = []  # 结束位置索引
    entity_type_list = []

    for i, input_tokens in enumerate(input_tokens_list):
        ys = y_label_list[i]  # 每条数据对应的数字标签列表
        temp = []
        label_list = []

        s_list = []
        e_list = []
        e_type = []
        is_start = False

        for index, num in enumerate(ys):
            if (num in [2, 4, 6, 8, 10, 12, 14, 16]) \
                    and len(temp) == 0:  # B S (标签开头及单独)
                s_list.append(index)
                type_num = int((num - 1) / 2)
                # type_num = int((num - 1) / 4)
                if type_num == 0:
                    e_type.append("ELE")
                elif type_num == 1:
                    e_type.append("IND")
                elif type_num == 2:
                    e_type.append("CON")
                else:
                    e_type.append("SIT")

                is_start = True
                temp.append(input_tokens[index])
            elif (num in [1, 3, 5, 7, 9, 11, 13, 15]) \
                    and len(temp) > 0:  # I/M E (标签中间及结尾)
                temp.append(input_tokens[index])
            elif len(temp) > 0:
                if is_start:
                    e_list.append(index - 1)
                is_start = False

                label_list.append("".join(temp))
                temp = []

        if len(s_list) != len(e_list):
            print(ys)

        y_entity_list.append("|".join(label_list))
        start_pos_list.append(s_list)
        end_pos_list.append(e_list)
        entity_type_list.append(e_type)

    return y_label_list, y_entity_list, entity_type_list, start_pos_list, end_pos_list


def evaluate(model, data_iterator, tokens, config, mark='Eval', verbose=False):
    """Evaluate the model on `steps` batches."""
    # set model to evaluation mode
    model.eval()
    print("***** Running eval *****")

    idx2tag = config.idx2tag
    true_labels = []
    pred_labels = []

    # a running average object for loss
    loss_avg = utils.RunningAverage()
    t = trange(config.eval_steps)
    for _ in t:
        # fetch the next evaluation batch
        batch_ids, batch_labels = next(data_iterator)
        batch_masks = batch_ids.gt(0)
        loss = model(batch_ids, labels=batch_labels, attention_mask=batch_masks)

        if config.n_gpu > 1 and config.multi_gpu:
            loss = loss.mean()
        loss_avg.update(loss.item())

        # shape: (batch_size, max_len, num_labels)
        batch_output = model(batch_ids, attention_mask=batch_masks)

        batch_output = batch_output.detach().cpu().numpy()
        batch_labels = batch_labels.to('cpu').numpy()

        for l in np.argmax(batch_output, axis=2):
            pred_label = []
            for idx in l:
                pred_label.append(idx)
            pred_labels.append(pred_label)

        for l in batch_labels:
            true_label = []
            for idx in l:
                true_label.append(idx)
            true_labels.append(true_label)

        # pred_labels.append([idx2tag.get(idx) for indices in np.argmax(batch_output, axis=2) for idx in indices])
        # true_labels.append([idx2tag.get(idx) for indices in batch_labels for idx in indices])

        # pred_labels.append([idx for indices in np.argmax(batch_output, axis=2) for idx in indices])
        # true_labels.append([idx for indices in batch_labels for idx in indices])

    assert len(pred_labels) == len(true_labels)

    predict_labels = []
    for i in range(len(tokens)):
        length = len(tokens[i])
        predict_labels.append(pred_labels[i][:length])

        assert len(tokens[i]) == len(predict_labels[i])

    _, y_pred_entity_list, y_pred_entity_type_list, _, _ = get_text_and_entity(predict_labels, tokens)
    _, y_true_entity_list, y_true_entity_type_list, _, _ = get_text_and_entity(true_labels, tokens)

    dict_data = {
        'y_true_entity': y_true_entity_list,
        'y_pred_entity': y_pred_entity_list,
        'y_pred_entity_type': y_pred_entity_type_list,
        'y_true_entity_type': y_true_entity_type_list,
    }
    df = pd.DataFrame(dict_data)

    precision, recall, f1, val_result = get_P_R_F(df)

    # logging loss, f1 and report
    metrics = {}
    # print(pred_labels)
    # f1 = f1_score(true_labels, pred_labels)
    metrics['loss'] = loss_avg()
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1'] = f1
    metrics_str = "; ".join("{}: {:05.2f}".format(k, v) for k, v in metrics.items())
    logging.info("- {} metrics: ".format(mark) + metrics_str)

    if verbose:
        report = classification_report(true_labels, pred_labels)
        logging.info(report)

    return metrics, val_result
