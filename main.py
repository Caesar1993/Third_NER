# -*- coding: utf-8 -*-

import torch
import os
import random
import os
import numpy as np
import logging
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
模型训练主程序
1-加载训练数据，训练数据是竖的，一句话中每个字都对应有是否为实体的label。
  将一句话中的字编码，将每个字的实体也转换为schema中的数字
2-将64句话和实体label传入模型，先基于每句话进行编码，进行9分类，然后使用CRF，将64*150*9，实体label的64*150等参数传入CRF，调整CRF中的转移矩阵
3-预测阶段，数据形式与train阶段相同。包括输入语句和每个字的真实label。
  输入语句，通过模型预测9分类情况，再通过CRF。
  然后将输入语句和预测每个字的label，进行解码，存入预测结果字典
  将输入语句和真实的每个字的label，进行解码，存入真实结果字典
4-最后统计，真实字典中有多个实体存在预测字典中，得到准确率，召回率，F1值
"""


def main(config):
    # 创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    # 加载训练数据
    train_data = load_data(config["train_data_path"], config)
    # 加载模型
    model = TorchModel(config)
    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    # 加载优化器
    optimizer = choose_optimizer(config, model)
    # 加载效果测试类
    evaluator = Evaluator(config, model, logger)
    # 训练
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        for index, batch_data in enumerate(train_data):  # 1412个train数据，每次64个，共23批
            optimizer.zero_grad()
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            input_id, labels = batch_data  # 输入变化时这里需要修改，比如多输入，多输出的情况。
            loss = model(input_id, labels)  # input_ids和labels都是64*150
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:#一共23轮数据，0轮时输出一次，12轮时输出一次loss，23轮时输出一次loss
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        evaluator.eval(epoch)
    model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    # torch.save(model.state_dict(), model_path)
    return model, train_data


if __name__ == "__main__":
    model, train_data = main(Config)
