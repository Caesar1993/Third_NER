# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import random
import jieba
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        # self.vocab = load_vocab(config["vocab_path"])
        # self.config["vocab_size"] = len(self.vocab)
        self.tokenizer = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.tokenizer.vocab)
        self.sentences = []
        self.schema = self.load_schema(config["schema_path"])
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            segments = f.read().split("\n\n")#将traindata切割成一句句的话
            for segment in segments:#取出第一句话
                sentence = []
                labels = []
                for line in segment.split("\n"):#将第一句话切割，取出第一个字和label
                    if line.strip() == "":
                        continue
                    char, label = line.split()#把每一行读进来，分成字和label
                    sentence.append(char)
                    labels.append(self.schema[label])#通过schema.json转成对应的标号
                self.sentences.append("".join(sentence))#把一句话的字和标点符号存储下来
                input_ids = self.tokenizer.encode(sentence, max_length=self.config["max_length"],
                                                 pad_to_max_length=True)
                # input_ids = self.encode_sentence(sentence)#把这句话按照char转换为数字表示
                labels = self.padding(labels, -1)#将train data中的一句话对应的label，补全，都补成-1
                self.data.append([torch.LongTensor(input_ids), torch.LongTensor(labels)])#将一句话的数字表示，和label表示存入此处
        return

    def encode_sentence(self, text, padding=True):
        input_id = []
        if self.config["vocab_path"] == "words.txt":
            for word in jieba.cut(text):
                input_id.append(self.vocab.get(word, self.vocab["[UNK]"]))
        else:
            for char in text:
                input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        if padding:
            input_id = self.padding(input_id)
        return input_id

    #补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id, pad_token=0):
        input_id = input_id[:self.config["max_length"]]
        input_id += [pad_token] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def load_schema(self, path):
        with open(path, encoding="utf8") as f:
            # return json.load(f)
            return json.loads(f.read())

#加载字表或词表
def load_vocab(vocab_path):
    # token_dict = {}
    # with open(vocab_path, encoding="utf8") as f:
    #     for index, line in enumerate(f):
    #         token = line.strip()
    #         token_dict[token] = index + 1  #0留给padding位置，所以从1开始
    # return token_dict
    tokenizer = BertTokenizer(vocab_path)
    return tokenizer

#用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl





