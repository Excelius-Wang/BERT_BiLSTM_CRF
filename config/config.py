"""
  @FileName：config.py
  @Author：Excelius
  @CreateTime：2024/11/9 21:19
  @Company: None
  @Description：
"""
import os
from pathlib import Path

import json
from src.utils.log_config import logger


class CommonConfig:
    current_path = Path.cwd()
    bert_large_dir = os.path.join(current_path.parent.parent, "model_hub", "chinese_roberta_wwm_large_ext_pytorch")
    ck_bert_dir = os.path.join(current_path.parent.parent, "model_hub", "pai_ckbert_bert")
    bert_wwn_dir = os.path.join(current_path.parent.parent, "model_hub", "chinese_wwm_ext_pytorch")
    data_dir = os.path.join(current_path.parent.parent, "data")
    output_dir = os.path.join(current_path.parent.parent, "output")
    entities_count_dir = os.path.join(data_dir, 'pre_data')


class NerConfig:
    def __init__(self, data_name, model_name):
        cf = CommonConfig()
        self.model_name = model_name
        # 根据 data_name：bert_wwn_ext_base, bert_wwn_ext_large, ck_bert 选择 bert_dir
        if model_name == "bert_wwn_ext_base":
            self.bert_dir = cf.bert_wwn_dir
        elif model_name == "bert_wwn_ext_large":
            self.bert_dir = cf.bert_large_dir
        elif model_name == "ck_bert":
            self.bert_dir = cf.ck_bert_dir
        else:
            self.bert_dir = cf.bert_wwn_dir

        self.output_dir = cf.output_dir
        self.output_dir = os.path.join(self.output_dir, data_name)
        self.entities_count_dir = cf.entities_count_dir

        if self.bert_dir.endswith("chinese_wwm_ext_pytorch"):
            self.output_dir = os.path.join(self.output_dir, "chinese_wwm_ext_pytorch")

            if not os.path.exists(self.output_dir):
                logger.info(f"Create directory: {self.output_dir}")
                os.mkdir(self.output_dir)

        if self.bert_dir.endswith("chinese_wwm_ext_pytorch"):
            self.output_dir = os.path.join(self.output_dir, "chinese_wwm_ext_pytorch")

            if not os.path.exists(self.output_dir):
                logger.info(f"Create directory: {self.output_dir}")
                os.mkdir(self.output_dir)

        if self.bert_dir.endswith("chinese_roberta_wwm_large_ext_pytorch"):
            self.output_dir = os.path.join(self.output_dir, "chinese_roberta_wwm_large_ext_pytorch")

            if not os.path.exists(self.output_dir):
                logger.info(f"Create directory: {self.output_dir}")
                os.mkdir(self.output_dir)

        if self.bert_dir.endswith("pai_ckbert_bert"):
            self.output_dir = os.path.join(self.output_dir, "pai_ckbert_bert")

            if not os.path.exists(self.output_dir):
                logger.info(f"Create directory: {self.output_dir}")
                os.mkdir(self.output_dir)

        self.data_dir = cf.data_dir

        self.data_path = os.path.join(self.data_dir, data_name)
        with open(os.path.join(self.data_path, "labels.txt"), "r", encoding='utf-8') as fp:
            self.labels = fp.read().strip().split("\n")
        self.bio_labels = ["O"]
        for label in self.labels:
            self.bio_labels.append("B-{}".format(label))
            self.bio_labels.append("I-{}".format(label))
        # print(self.bio_labels)
        self.num_labels = len(self.bio_labels)
        self.label2id = {label: i for i, label in enumerate(self.bio_labels)}
        # print(self.label2id)
        self.id2label = {i: label for i, label in enumerate(self.bio_labels)}

        # 打开 entities_count.txt 读取实体数量，保存为 dict 格式
        self.entities_count_data = {}
        with open(os.path.join(self.entities_count_dir, "entities_count.txt"), "r", encoding='utf-8') as fp:
            self.entities_count_data = json.load(fp)

        self.max_seq_len = 128
        self.epochs = 15
        if model_name == 'bert_wwn_ext_large':
            self.train_batch_size = 48
            self.dev_batch_size = 24
        else:
            self.train_batch_size = 64
            self.dev_batch_size = 32
        # 学习率 2e-5, 3e-5, 4e-5
        self.bert_learning_rate = 2e-5
        self.crf_learning_rate = 1e-3
        self.adam_epsilon = 1e-8
        self.weight_decay = 0.001

        self.warmup_proportion = 0.1
        self.save_step = 500
