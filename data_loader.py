"""
    @author: Jevis_Hoo
    @Date: 2020/7/11 12:19
    @Description: 
"""

"""Data loader"""

import random
import numpy as np
import os
import torch
from pytorch_pretrained_bert import BertTokenizer
from config import Config


class DataLoader(object):
    def __init__(self, data_dir, bert_model_dir, config, token_pad_idx=0):
        self.data_dir = data_dir
        self.batch_size = config.batch_size
        self.sequence_length = config.sequence_length
        self.device = config.device
        self.seed = config.seed
        self.token_pad_idx = 0

        tags = self.load_tags()
        self.tag2idx = {tag: idx for idx, tag in enumerate(tags)}
        self.idx2tag = {idx: tag for idx, tag in enumerate(tags)}
        config.tag2idx = self.tag2idx
        config.idx2tag = self.idx2tag
        self.tag_pad_idx = self.tag2idx['O']
        print(self.tag2idx)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_dir, do_lower_case=True)

    def load_tags(self):
        tags = []
        file_path = os.path.join(self.data_dir, 'tags.txt')
        with open(file_path, 'r') as file:
            for tag in file:
                tags.append(tag.strip())
        return tags

    def load_sentences_tags(self, sentences_file, tags_file, d):
        """Loads sentences and tags.txt from their corresponding files.
            Maps tokens and tags.txt to their indices and stores them in the provided dict d.
        """
        sentences = []
        tags = []
        token_list = []
        text = []

        with open(sentences_file, 'r') as file:
            for line in file:
                # replace each token by its index
                text.append(''.join(line[:-1].split(' ')))
                tokens = self.tokenizer.tokenize(line.strip())
                token_list.append(tokens)
                sentences.append(self.tokenizer.convert_tokens_to_ids(tokens))

                # while len(token_list) < config.sequence_length:
                #     pass

        with open(tags_file, 'r') as file:
            for line in file:
                # replace each tag by its index
                tag_seq = [self.tag2idx.get(tag) for tag in line.strip().split(' ')]
                tags.append(tag_seq)
        # checks to ensure there is a tag for each token
        assert len(sentences) == len(tags)
        # print(tags_file.find("test"))
        for s in range(len(sentences)):
            if tags_file.find("test") >= 0:
                tags[s] = [0] * len(sentences[s])
            # if len(tags[s]) != len(sentences[s]):
            # print(tags[s])
            # print(sentences[s])
            # print(len(tags[s]))
            # print(len(sentences[s]))
            assert len(tags[s]) == len(sentences[s])

        # storing sentences and tags.txt in dict d

        d['data'] = sentences
        d['text'] = text
        d['tags.txt'] = tags
        d['size'] = len(sentences)
        d['token'] = token_list

    def load_data(self, data_type):
        """Loads the data for each type in types from data_dir.
        Args:
            data_type: (str) has one of 'train', 'val', 'test' depending on which data is required.
        Returns:
            data: (dict) contains the data with tags.txt for each type in types.
        """
        data = {}

        if data_type in ['train', 'val', 'test']:
            sentences_file = os.path.join(self.data_dir, data_type, 'sentences.txt')
            tags_path = os.path.join(self.data_dir, data_type, 'tags.txt')
            self.load_sentences_tags(sentences_file, tags_path, data)
        else:
            raise ValueError("data type not in ['train', 'val', 'test']")

        return data

    def data_iterator(self, data, shuffle=False):
        """Returns a generator that yields batches data with tags.txt.
        Args:
            data: (dict) contains data which has keys 'data', 'tags.txt' and 'size'
            shuffle: (bool) whether the data should be shuffled

        Yields:
            batch_data: (tensor) shape: (batch_size, sequence_length)
            batch_tags: (tensor) shape: (batch_size, sequence_length)
        """

        # make a list that decides the order in which we go over the data- this avoids explicit shuffling of data
        order = list(range(data['size']))
        if shuffle:
            random.seed(self.seed)
            random.shuffle(order)

        # one pass over data
        for i in range(data['size'] // self.batch_size):
            # fetch sentences and tags.txt
            sentences = [data['data'][idx] for idx in order[i * self.batch_size:(i + 1) * self.batch_size]]
            tags = [data['tags.txt'][idx] for idx in order[i * self.batch_size:(i + 1) * self.batch_size]]

            # batch length
            batch_len = len(sentences)

            # compute length of longest sentence in batch
            batch_sequence_length = max([len(s) for s in sentences])
            sequence_length = min(batch_sequence_length, self.sequence_length)

            # prepare a numpy array with the data, initialising the data with pad_idx
            batch_data = self.token_pad_idx * np.ones((batch_len, sequence_length))
            batch_tags = self.tag_pad_idx * np.ones((batch_len, sequence_length))

            # copy the data to the numpy array
            for j in range(batch_len):
                cur_len = len(sentences[j])
                if cur_len <= sequence_length:
                    batch_data[j][:cur_len] = sentences[j]
                    batch_tags[j][:cur_len] = tags[j]
                else:
                    batch_data[j] = sentences[j][:sequence_length]
                    batch_tags[j] = tags[j][:sequence_length]

            # since all data are indices, we convert them to torch LongTensors
            batch_data = torch.tensor(batch_data, dtype=torch.long)
            batch_tags = torch.tensor(batch_tags, dtype=torch.long)

            # shift tensors to GPU if available
            batch_data, batch_tags = batch_data.to(self.device), batch_tags.to(self.device)
            yield batch_data, batch_tags


if __name__ == '__main__':
    config = Config()
    # Initialize the DataLoader
    data_loader = DataLoader(config.data_dir, config.bert_path, config, token_pad_idx=0)

    # Load training data and test data
    train_data = data_loader.load_data('test')
    train_data_iterator = data_loader.data_iterator(train_data, shuffle=True)

    from tqdm import trange

    t = trange(12)
    for i in t:
        # fetch the next training batch
        batch_data, batch_tags = next(train_data_iterator)
        print(batch_data)
