"""
  @FileName：data_loader.py
  @Author：Excelius
  @CreateTime：2024/11/10 08:24
  @Company: None
  @Description：
"""
import torch
import numpy as np

from torch.utils.data import Dataset


class NerDataset(Dataset):
    """
    NER 数据集类
    """

    def __init__(self, data, args, tokenizer):
        """
        初始化数据集
        :param data: 包含文本和标签的列表，每个元素都是字典，如 {"text": [...], "labels": [...]}
        :param args: 包含配置参数，例如 label2id（标签到 ID 的映射）和 max_seq_len（最大序列长度）
        :param tokenizer: 分词器，用于将文本转化为模型输入的 token ID
        """
        self.data = data
        self.args = args
        self.tokenizer = tokenizer
        self.label2id = args.label2id
        self.max_seq_len = args.max_seq_len

    def __len__(self):
        """
        获取数据集大小
        :return: 数据集的长度（样本数量）
        """
        return len(self.data)

    def __getitem__(self, item):
        """
        获取数据集中的一个样本，并将其转化为模型输入格式
        :param item: 样本的索引
        :return: 一个字典，包括 input_ids、attention_mask 和 labels（所有为张量）
        """
        # 获取文本和标签
        text = self.data[item]["text"]
        labels = self.data[item]["labels"]
        # 如果文本长度超过最大序列长度，则截断，截断为 max_seq_len - 2，因为需要加上 [CLS] 和 [SEP]
        if len(text) > self.max_seq_len - 2:
            text = text[:self.max_seq_len - 2]
            labels = labels[:self.max_seq_len - 2]

        # 添加特殊标记 [CLS] 和 [SEP]，并将其转化为 token ID
        tmp_input_ids = self.tokenizer.convert_tokens_to_ids(["[CLS]"] + text + ["[SEP]"])

        # attention_mask: 标记实际有效的 token 位置（1 为有效，0 为填充）文本所在的位置则为有效位置，剩余空白直到 max_seq_len 为填充
        attention_mask = [1] * len(tmp_input_ids)

        # 将 input_ids 和 attention_mask 补充到 max_seq_len 长度，使用 0 进行填充
        input_ids = tmp_input_ids + [0] * (self.max_seq_len - len(tmp_input_ids))
        attention_mask = attention_mask + [0] * (self.max_seq_len - len(tmp_input_ids))

        # 将标签转化为 ID，并在首尾添加 0（[CLS] 和 [SEP] 的标签为 0），并将其补充到 max_seq_len 长度
        labels = [self.label2id[label] for label in labels]
        labels = [0] + labels + [0] + [0] * (self.max_seq_len - len(tmp_input_ids))

        # 将 input_ids、attention_mask 和 labels 转化为张量
        input_ids = torch.tensor(np.array(input_ids))
        attention_mask = torch.tensor(np.array(attention_mask))
        labels = torch.tensor(np.array(labels))

        # 返回数据
        data = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        return data
