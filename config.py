# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "model_output",
    "schema_path": "ner_data/schema.json",
    "train_data_path": "ner_data/train.txt",
    "valid_data_path": "ner_data/test.txt",
    # "vocab_path":"chars.txt",
    "vocab_path": r"D:\NLP_BaDou_AI\bert-base-chinese\vocab.txt",
    "pretrain_model_path": r"D:\NLP_BaDou_AI\bert-base-chinese",
    "max_length": 150,
    "hidden_size": 256,
    "epoch": 20,
    "batch_size": 32,
    "optimizer": "adam",
    "learning_rate": 1e-5,
    "use_crf": True,
    "class_num": 9
}
