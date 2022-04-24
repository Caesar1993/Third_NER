# -*- coding: utf-8 -*-

import torch#名字是pytorch，import是torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torchcrf import CRF#名字是pytorch-crf
from transformers import BertModel
"""
建立网络模型结构
"""

class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        # vocab_size = config["vocab_size"] + 1
        # max_length = config["max_length"]
        class_num = config["class_num"]
        #----使用lstm+CRF-----
        # hidden_size = config["hidden_size"]
        # self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        # self.layer = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True, num_layers=1)
        # self.classify = nn.Linear(hidden_size * 2, class_num)#9分类。因为是双向lstm，所以此处是hidden*2
        #----使用bert+crf----
        self.bert=BertModel.from_pretrained(config["pretrain_model_path"])
        hidden_size=self.bert.config.hidden_size
        self.classify = nn.Linear(hidden_size , class_num)

        self.crf_layer = CRF(class_num, batch_first=True)
        self.use_crf = config["use_crf"]
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)  #loss采用交叉熵损失，index为-1的不参与loss计算

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, target=None):
        # x = self.embedding(x)  #输入64*150，输出64 150 256 。input shape:(batch_size, sen_len)
        # x, _ = self.layer(x)   #输出 64 150 512   #input shape:(batch_size, sen_len, input_dim)

        x=self.bert(x)[0]#x变为32*150*768
        predict = self.classify(x) #输出predict为 32 150 9 #ouput:(batch_size, sen_len, num_tags)
        if target is not None:
            if self.use_crf:
                mask = target.gt(-1)#以-1为界限，-1为false，其他为true，为true参与计算crf
                return - self.crf_layer(predict, target, mask, reduction="mean")#过完crf取得负号，把此当做loss
            else:
                return self.loss(predict.view(-1, predict.shape[-1]), target.view(-1))
        else:
            if self.use_crf:
                return self.crf_layer.decode(predict)
            else:
                return predict


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)


