# coding:utf-8
import os
import json
import random

random.seed(1000)

label_dict = {
    '试验要素': 'ELE',
    '性能指标': 'IND',
    '系统组成': 'CON',
    '任务场景': 'SIT',
}


def find_all(sub, s):
    index_list = []
    index = s.find(sub)
    while index != -1:
        index_list.append(index)
        index = s.find(sub, index + 1)

    if len(index_list) > 1:
        return index_list
    else:
        return -1


def transfer_data(data, type):
    g = open(type + '.txt', 'w', encoding='utf-8')
    f1 = open(type + '_sentences.txt', 'w', encoding='utf-8')
    f2 = open(type + '_tags.txt', 'w', encoding='utf-8')
    for file in data:
        filename = os.path.join('./data', file)
        with open(filename, encoding='gbk') as f:
            data = json.load(f)
            text = data['originalText'].rstrip().lower()  # 去掉末尾的 \r\n
            text = text.replace('“', '"').replace('”', '"').replace('—', '-')  # 引号中文里面没有  ——没有
            text = text[:-1] + '。'
            if '?' in text:
                text = text.replace('?', ',')

            tokens = list(text)
            assert len(text) == len(tokens)

            entities = []
            for e in data['entities']:
                a = e['start_pos']  #
                b = e['end_pos']
                label = e['label_type']
                overlap = e["overlap"]
                cut = 0
                for t in range(len(tokens)):
                    if t == e['start_pos'] - 1:  # 因为是从1开始记索引 到了a位
                        a -= cut
                    if t == e['end_pos'] - 1:
                        b -= cut
                        break  # 对应的每个实体位置在这里重新校正完毕
                    tt = tokens[t]
                    if tt == " ":  # 如果是空格则对应的标签索引前面减去1 对应位置取消
                        cut += 1
                        # tokens[t] = ''
                entities.append({
                    "label_type": label,
                    "overlap": overlap,
                    "start_pos": a,
                    "end_pos": b
                })

            tokens = [i for i in tokens if i != ' ']  # 最后把空格筛除
            text = ''.join(tokens)  # 相当于自动把空格部分去除

            sub_ent_list = []
            # 提取所有实体
            for j in range(len(entities)):
                start_pos = entities[j]['start_pos']
                end_pos = entities[j]['end_pos']
                entity = text[start_pos - 1:end_pos]
                sub_ent_list.append("".join(entity))
            # print(sub_ent_list)

            rep_entities = []
            for e in entities:
                # tmp_tokens = tokens  #保证处理过程中tokens不被修改
                a = e['start_pos']  #
                b = e['end_pos']
                label = e['label_type']
                overlap = e["overlap"]

                # 重复实体位置的加入
                entity = text[a - 1:b]
                index = find_all(entity, text)
                # print(entity)
                # print(a, index)

                # 防止实体是另外实体的子串
                is_sub_entity = False
                for k in range(len(sub_ent_list)):
                    if sub_ent_list[k].find(entity) != -1 and sub_ent_list[k] != entity:
                        is_sub_entity = True
                        break

                if index != -1 and (not is_sub_entity):  # 有重复实体，不是子串，全部加入
                    length = b - a
                    for en_index in range(len(index)):
                        # print(text[index[en_index] + 1:index[en_index] + 1 + length])
                        rep_entities.append({
                            "label_type": label,
                            "overlap": overlap,
                            "start_pos": index[en_index] + 1,
                            "end_pos": index[en_index] + 1 + length
                        })
                else:
                    rep_entities.append({
                        "label_type": label,
                        "overlap": overlap,
                        "start_pos": a,
                        "end_pos": b
                    })

            s, l, t = [], [], []
            # 提取所有实体
            for j in range(len(rep_entities)):
                label = rep_entities[j]['label_type']
                start_pos = rep_entities[j]['start_pos']
                end_pos = rep_entities[j]['end_pos']
                entity = text[start_pos - 1:end_pos]
                t.append("".join(entity))
                l.append(label)
            s.append(t)
            s.append(l)
            print(s)

            bio = ['O'] * len(text)
            tmp = []
            for ent in rep_entities:
                bio[ent['start_pos'] - 1] = label_dict[ent['label_type']] + '-B'
                bio[ent['start_pos']:ent['end_pos']] = [label_dict[ent['label_type']] + '-I'] * (
                        ent['end_pos'] - ent['start_pos'])
                # bio[ent['start_pos'] - 1] = 'B-' + label_dict[ent['label_type']]
                # bio[ent['start_pos']:ent['end_pos']] = ['I-' + label_dict[ent['label_type']]] * (
                #         ent['end_pos'] - ent['start_pos'])
                tmp.append((ent['start_pos'], ent['end_pos']))

            assert len(text) == len(bio)
            for v in range(len(text)):
                g.write(text[v] + ' ' + bio[v] + '\n')
                if v == len(text) - 1:
                    f1.write(text[v])
                    f2.write(bio[v])
                    break
                f1.write(text[v] + ' ')
                f2.write(bio[v] + ' ')
            g.write('\n')
            f1.write('\n')
            f2.write('\n')


def load_sentences(path):
    """
    加载训练样本，一句话就是一个样本。
    训练样本中，每一行是这样的：长 B-Dur，即字和对应的标签
    句子之间使用空行隔开的
    return : sentences: [[[['无', 'O'], ['长', 'B-Dur'], ['期', 'I-Dur'],...]]
    """

    sentences = []
    sentence = []

    for line in open(path, 'r', encoding='utf8'):

        """ 如果包含有数字，就把每个数字用0替换 """
        # line = line.rstrip()
        # line = self.zero_digits(line)

        """ 如果不是句子结束的换行符，就继续添加单词到句子中 """
        if line != "\n":
            word_pair = ["<unk>", line[2:]] if line[0] == " " else line.split()
            assert len(word_pair) == 2
            sentence.append(word_pair)

        else:
            """ 如果遇到换行符，说明一个句子处理完毕 """
            if len(sentence) > 0:
                sentences.append(sentence)
                sentence = []

    return sentences


def iob_iobes(tags):
    """
    IOB -> IOBES
    """
    new_tags = []

    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        elif tag.split('-')[-1] == 'B':
            if i + 1 != len(tags) and tags[i + 1].split('-')[-1] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('-B', '-S'))
        elif tag.split('-')[-1] == 'I':
            if i + 1 < len(tags) and tags[i + 1].split('-')[-1] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('-I', '-E'))
        else:
            raise Exception('Invalid IOB format!')
    return new_tags


def update_tag_scheme(doc_path, out_path):
    sentences = load_sentences(doc_path)
    """ 将IOB格式转化为IOBES格式 """

    f1 = open(out_path + "sentences.txt", "w")
    f2 = open(out_path + "tags.txt", "w")
    for i, s in enumerate(sentences):
        char = [w[0] for w in s]
        tags = [w[-1] for w in s]
        new_tags = iob_iobes(tags)
        for j in range(len(new_tags)):
            f1.write(char[j] + ' ')
            f2.write(new_tags[j] + ' ')
        f1.write('\n')
        f2.write('\n')


if __name__ == '__main__':
    files = os.listdir('./data')  # 目前重新标注了111个精确标注
    random.shuffle(files)

    train_data = files[:int(0.8 * len(files))]
    val_data = files[int(0.8 * len(files)):]

    transfer_data(train_data, "train")
    print("train data finish")

    transfer_data(val_data, "val")
    print("val data finish")

    train_iob_output_path = 'train.txt'
    train_iobes_output_path = '/home/hezoujie/Competition/Pytorch_NER/data/ccks/train/'
    update_tag_scheme(train_iob_output_path, train_iobes_output_path)

    val_iob_output_path = 'val.txt'
    val_iobes_output_path = '/home/hezoujie/Competition/Pytorch_NER/data/ccks/val/'
    update_tag_scheme(val_iob_output_path, val_iobes_output_path)

    print("update_tag_scheme done")
