"""
  @FileName：predict.py
  @Author：Excelius
  @CreateTime：2024/11/10 19:11
  @Company: None
  @Description：
"""
import os
import json
from collections import namedtuple
from pathlib import Path

import torch
import numpy as np
from seqeval.metrics.sequence_labeling import get_entities
from transformers import BertTokenizer

from src.models.model import BertNer
from src.utils.log_config import logger


def get_args(args_path, args_name=None):
    """
    从文件中加载参数
    :param args_path: 参数文件路径
    :param args_name: 参数文件名
    :return: 参数字典
    """
    with open(args_path, "r", encoding='utf-8') as fp:
        args_dict = json.load(fp)
    args = namedtuple(args_name, args_dict.keys())(*args_dict.values())
    return args


class Predictor:
    """
    预测器类
    """

    def __init__(self, _data_name):
        self.data_name = data_name
        # 加载配置参数
        _file_path = os.path.join(Path.cwd().parent.parent, "output", _data_name, _data_name + "_args.json")
        self.ner_args = get_args(_file_path, "ner_args")
        # 将id和标签的映射关系存储在字典中
        self.ner_id2label = {int(k): v for k, v in self.ner_args.id2label.items()}
        # 初始化BERT分词器
        self.tokenizer = BertTokenizer.from_pretrained(self.ner_args.bert_dir)
        # 最大序列长度
        self.max_seq_len = self.ner_args.max_seq_len
        # 确定设备，优先使用GPU
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        # 初始化NER模型
        self.ner_model = BertNer(self.ner_args)
        # 加载模型权重
        self.ner_model.load_state_dict(
            torch.load(
                os.path.join(self.ner_args.output_dir, "best_model.bin"),
                map_location=torch.device('cpu')
            ),
            strict=False
        )
        self.ner_model.to(self.device)
        self.data_name = _data_name

    def ner_tokenizer(self, text):
        # 截断文本以满足最大序列长度要求
        text = text[:self.max_seq_len - 2]
        # 将文本转换为BERT的输入格式
        text = ["[CLS]"] + [i for i in text] + ["[SEP]"]
        tmp_input_ids = self.tokenizer.convert_tokens_to_ids(text)
        input_ids = tmp_input_ids + [0] * (self.max_seq_len - len(tmp_input_ids))
        attention_mask = [1] * len(tmp_input_ids) + [0] * (self.max_seq_len - len(tmp_input_ids))
        # 将输入转换为PyTorch张量
        input_ids = torch.tensor(np.array([input_ids]))
        attention_mask = torch.tensor(np.array([attention_mask]))
        return input_ids, attention_mask

    def ner_predict(self, text):
        # 使用分词器处理文本
        input_ids, attention_mask = self.ner_tokenizer(text)
        # 将输入数据移动到指定设备
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        # 进行预测
        output = self.ner_model(input_ids, attention_mask)
        attention_mask = attention_mask.detach().cpu().numpy()
        length = sum(attention_mask[0])
        logits = output.logits
        logits = logits[0][1:length - 1]
        logits = [self.ner_id2label[i] for i in logits]
        # 从模型输出中提取实体
        entities = get_entities(logits)
        result = {}
        for ent in entities:
            ent_name = ent[0]
            ent_start = ent[1]
            ent_end = ent[2]
            # 将实体信息存储在字典中
            if ent_name not in result:
                result[ent_name] = [("".join(text[ent_start:ent_end + 1]), ent_start, ent_end)]
            else:
                result[ent_name].append(("".join(text[ent_start:ent_end + 1]), ent_start, ent_end))
        return result


if __name__ == "__main__":
    data_name = "pre_data"
    predictor = Predictor(data_name)
    # 根据data_name选择不同的文本列表
    test_file_path = os.path.join(Path.cwd().parent.parent, 'data', 'ori_data', 'test.json')
    with open(test_file_path, 'r', encoding='utf-8') as train_f:
        logger.info(f"Train data path: {test_file_path}")
        _train_data = json.load(train_f)
        logger.info(f"Train data size: {len(_train_data)}")

    # 获取文本列表
    texts = []
    for item in _train_data:
        texts.append(item.get("sentence", []))

    count = 0
    res = []
    # 遍历每一条文本，进行预测
    for text in texts:
        ner_result = predictor.ner_predict(text)
        entities = []
        for key, value in ner_result.items():
            for v in value:
                item = {"name": v[0], "type": key, "pos": [v[1], v[2] + 1]}
                entities.append(item)
        # 按照实体的起始位置排序
        entities = sorted(entities, key=lambda x: x["pos"][0])
        res.append({"sentence": text, "entities": entities})
        count += 1
        if count % 100 == 0:
            logger.info(f"Processed {count} samples.")

    # 保存预测结果
    out_put_path = os.path.join(Path.cwd().parent.parent, 'output', data_name, 'predict_result.json')
    with open(out_put_path, 'w', encoding='utf-8') as fp:
        json.dump(res, fp, ensure_ascii=False)
