"""
  @FileName：preprocess.py
  @Author：Excelius
  @CreateTime：2024/11/8 10:47
  @Company: None
  @Description：
"""
import os
import json
import random
import re
from pathlib import Path

import opencc
from matplotlib import rcParams
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from src.utils.log_config import logger


class ProcessSanWenData:
    def __init__(self):
        # 载入数据集
        self.colab_cwd = '/content'
        # 当前路径
        self.current_path = Path.cwd()
        if os.getcwd() == self.colab_cwd:
            # 处理好的文本存储在谷歌云盘
            self.file_path = os.path.join(os.getcwd(), "drive", "MyDrive", "data", "ori_data")
        else:
            # 本地的文件路径
            self.file_path = os.path.join(self.current_path.parent.parent, "data", "ori_data")

    def load_train_dev_data(self):
        """
        载入数据集
        :param self: str, 数据集路径
        :return: pd.DataFrame, 数据集
        """
        with open(os.path.join(self.file_path, 'train.json'), 'r', encoding='utf-8') as train_f:
            # with open(os.path.join(self.file_path, 'train_aug.json'), 'r', encoding='utf-8') as train_f:
            logger.info(f"Train data path: {os.path.join(self.file_path, 'train.ori_data')}")
            _train_data = json.load(train_f)
            logger.info(f"Train data size: {len(_train_data)}")

        with open(os.path.join(self.file_path, 'dev.json'), 'r', encoding='utf-8') as dev_f:
            logger.info(f"Dev data path: {os.path.join(self.file_path, 'dev.ori_data')}")
            _dev_data = json.load(dev_f)
            logger.info(f"Dev data size: {len(_dev_data)}")
        return _train_data, _dev_data

    def convert_to_bio_format_txt(self, _data, _file_name, max_seq_len=128):
        """
        将输入数据转换为 BIO 格式，并保存到本地
        样例数据格式：
        {
            "sentence": "母亲特别强调，女孩子贪吃的种种后果。",
            "entities": [
                {
                    "name": "母亲",
                    "type": "人物",
                    "pos": [
                        0,
                        2
                    ]
                },
                {
                    "name": "女孩子",
                    "type": "人物",
                    "pos": [
                        7,
                        10
                    ]
                }
            ]
        }
        转换为：
        {
            "text": ["母", "亲", "特", "别", "强", "调", "，", "女", "孩", "子", "贪", "吃", "的", "种", "种", "后", "果", "。"],
            "labels": ["B-人物", "I-人物", "O", "O", "O", "O", "O", "B-人物", "I-人物", "I-人物", "O", "O", "O", "O", "O", "O", "O", "O"],
        }
        :param self: 自身对象
        :param _data: ori_data 格式的原始数据
        :param _file_name: str, 保存的文件名
        :param max_seq_len: int, 句子的最大长度, 默认为 128
        :return: None
        """
        _txt_data = []
        _replace_en_sign = 0
        for _item in _data:

            _sentence = _item.get('sentence', '')
            _entities = _item.get('entities', [])

            # 繁体转简体
            converter = opencc.OpenCC('t2s')  # 繁体转简体
            _sentence = converter.convert(_sentence)

            # 英文部分字符转换为中文字符
            punctuation_map = {
                ',': '，',
                ':': '：',
                ';': '；',
                '!': '！',
                '?': '？',
                '(': '（',
                ')': '）',
                '[': '【',
                ']': '】'
            }

            # 替换文本中的英文标点符号为中文标点符号
            for en_punct, zh_punct in punctuation_map.items():
                _sentence = _sentence.replace(en_punct, zh_punct)
                _replace_en_sign += 1
            # logger.info(f"Replace English punctuation to Chinese punctuation: {_replace_en_sign}")

            if len(_sentence) > max_seq_len:
                # 分割句子
                merged_sentences, start_indices = cut_sent(_sentence, max_seq_len)
                # 获得句子长度列表
                sent_len_list = [len(sent) for sent in merged_sentences]
                sent_index = 0
                all_entity = []
                cur_entity = []
                for _entity in _entities:
                    _entity_pos = _entity['pos']
                    # 遍历实体，获取实体的开始和结束位置
                    entity_start = _entity_pos[0]
                    entity_end = _entity_pos[1]

                    # 获取句子的开始和结束位置
                    sent_start = start_indices[sent_index]
                    sent_end = start_indices[sent_index] + len(merged_sentences[sent_index])

                    # 如果实体的开始和结束位置在句子的开始和结束位置之间，加入到当前句子的实体列表中
                    if entity_start >= sent_start and entity_end <= sent_end:
                        cur_entity.append(_entity)
                    # 如果实体的开始位置大于了句子的结束位置之间，将当前句子的实体列表加入到所有实体列表中，
                    # 重置当前句子的实体列表，更新句子索引，加入当前实体
                    else:
                        all_entity.append(cur_entity)
                        cur_entity = []
                        sent_index += 1
                        cur_entity.append(_entity)

                # 将最后一个句子的实体列表加入到所有实体列表中
                all_entity.append(cur_entity)

                # 累加的句子长度
                _sum_sent_len = 0
                for index, _entity in enumerate(all_entity):
                    # 从下标为 1 开始，累加句子长度
                    if index > 0:
                        _sum_sent_len += sent_len_list[index - 1]
                    for _index, item in enumerate(_entity):
                        # 从下标为 1 开始，调整实体的位置
                        item['pos'][0] -= _sum_sent_len
                        item['pos'][1] -= _sum_sent_len

                linked_sent_entities = []
                for sent, _entity in zip(merged_sentences, all_entity):
                    linked_sent_entities.append({'sentence': sent, 'entities': _entity})

                for linked_sent_entity in linked_sent_entities:

                    _sent = linked_sent_entity.get('sentence', '')

                    _sent = [i for i in _sent]
                    _entities = linked_sent_entity.get('entities', [])
                    # print(_entities)
                    _labels = ["O"] * len(_sent)

                    for _entity in _entities:
                        start, end = _entity["pos"]
                        entity_type = _entity["type"]
                        _labels[start] = f"B-{entity_type}"
                        for i in range(start + 1, end):
                            _labels[i] = f"I-{entity_type}"

                    _each_data = {
                        "text": _sent,
                        "labels": _labels
                    }

                    _txt_data.append(_each_data)
            else:
                _sent = [i for i in _sentence]
                _labels = ["O"] * len(_sent)
                for _entity in _entities:
                    start, end = _entity["pos"]
                    entity_type = _entity["type"]
                    _labels[start] = f"B-{entity_type}"
                    for i in range(start + 1, end):
                        _labels[i] = f"I-{entity_type}"

                _each_data = {
                    "text": _sent,
                    "labels": _labels
                }

                _txt_data.append(_each_data)

        _pre_path = os.path.join(os.path.dirname(self.file_path), 'pre_data')
        # 如果文件路径不存在，则创建路径
        if not os.path.exists(_pre_path):
            os.makedirs(_pre_path)
            logger.info(f"Create directory: {_pre_path}")
        else:
            logger.info(f"Directory exists: {_pre_path}")
        # 保存到本地
        with open(os.path.join(_pre_path, _file_name), 'w', encoding='utf-8') as train_f:
            logger.info(f"Save data to: {os.path.join(self.file_path, _file_name)}")
            train_f.write("\n".join([json.dumps(d, ensure_ascii=False) for d in _txt_data]))

            logger.info(f"Save complete! Save txt \"{_file_name}\", size: {len(_txt_data)}")


def cut_sentences_v1(sent):
    """
    第一层句子切分，处理单字符断句符、英文省略号和中文省略号
    :param sent: 输入的句子文本
    :return: 切分后的句子列表
    """
    # 使用正则表达式处理各种断句符
    sent = re.sub('([。！？\?])([^”’])', r"\1\n\2", sent)  # 单字符断句符
    sent = re.sub('(\.{6})([^”’])', r"\1\n\2", sent)  # 英文省略号
    sent = re.sub('(\…{2})([^”’])', r"\1\n\2", sent)  # 中文省略号
    sent = re.sub('([。！？\?][”’])([^，。！？\?])', r"\1\n\2", sent)  # 双引号后的处理
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后
    return sent.split("\n")  # 用换行符切分成句子列表


def cut_sentences_v2(sent):
    """
    第二层句子切分，处理分隔符 '；' | ';'
    :param sent: 输入的句子文本
    :return: 切分后的句子列表
    """
    sent = re.sub('([；;])([^”’])', r"\1\n\2", sent)  # 处理分隔符
    return sent.split("\n")  # 用换行符切分成句子列表


def cut_sent(text, max_seq_len):
    """
    切分文本为句子，合并句子并确保每个句子长度不超过最大限制
    :param text: 原始文本
    :param max_seq_len: 句子的最大长度
    :return: 合并后的句子列表和每个句子的起始下标
    """
    sentences = []  # 存储所有切分的句子
    start_indices = []  # 存储每个句子的起始下标

    # 细粒度划分
    sentences_v1 = cut_sentences_v1(text)
    current_index = 0  # 当前字符索引

    for sent_v1 in sentences_v1:
        if len(sent_v1) > max_seq_len - 2:
            # 如果句子超长，进行二次切分
            sentences_v2 = cut_sentences_v2(sent_v1)
            for s in sentences_v2:
                sentences.append(s)
                start_indices.append(current_index)  # 记录起始位置
                current_index += len(s)  # 更新当前索引
        else:
            sentences.append(sent_v1)
            start_indices.append(current_index)  # 记录起始位置
            current_index += len(sent_v1)  # 更新当前索引

    assert ''.join(sentences) == text  # 确保切分后能重新拼接成原文本

    # 合并句子
    merged_sentences = []  # 存储合并后的句子
    merged_start_indices = []  # 存储合并句子的起始下标
    start_index_ = 0  # 当前合并句子的起始索引

    while start_index_ < len(sentences):
        tmp_text = sentences[start_index_]  # 当前合并的句子
        tmp_start_index = start_indices[start_index_]  # 当前句子的起始下标

        end_index_ = start_index_ + 1  # 合并的下一个句子索引

        # 根据最大长度合并句子
        while end_index_ < len(sentences) and \
                len(tmp_text) + len(sentences[end_index_]) <= max_seq_len - 2:
            tmp_text += sentences[end_index_]  # 添加到当前句子
            end_index_ += 1

        start_index_ = end_index_  # 更新合并的开始索引

        merged_sentences.append(tmp_text)  # 添加合并后的句子
        merged_start_indices.append(tmp_start_index)  # 添加起始下标

    return merged_sentences, merged_start_indices  # 返回合并后的句子和起始下标


def combine_and_split(_train_data, _dev_data, train_ratio=0.95):
    """
    将 train_data 和 dev_data 组合到一起，并按照比例重新划分
    :param _train_data: list, 原始训练数据
    :param _dev_data: list, 原始验证数据
    :param train_ratio: float, 训练集的比例，默认为 95%
    :return: tuple, (重新划分的训练集, 验证集)
    """
    # 将 train_data 和 dev_data 合并
    combined_data = _train_data + _dev_data
    logger.info(f"Combined data size: {len(combined_data)}")

    # 使用 sklearn 的 train_test_split 按比例划分
    new_train_data, new_dev_data = train_test_split(combined_data, test_size=1 - train_ratio, random_state=42)

    logger.info(f"New train data size: {len(new_train_data)}")
    logger.info(f"New dev data size: {len(new_dev_data)}")

    return new_train_data, new_dev_data


if __name__ == "__main__":
    processSanWenData = ProcessSanWenData()
    train_data, dev_data = processSanWenData.load_train_dev_data()
    # 重新划分数据集
    # train_data, dev_data = combine_and_split(train_data, dev_data, train_ratio=0.95)

    # 初始化标签计数器
    label_counter = dict()

    cent_size_dict = dict()
    # 遍历_txt_data
    for data in train_data:
        entities = data.get("entities", [])
        sentence = data.get("sentence", '')
        cent_size_dict[len(sentence)] = cent_size_dict.get(len(sentence), 0) + 1
        for entity in entities:
            label = entity.get("type")
            if label not in label_counter:
                label_counter[label] = 1
            else:
                label_counter[label] += 1

    all_count = sum(cent_size_dict.values())
    label_counter = dict(sorted(label_counter.items(), key=lambda item: item[1], reverse=True))
    # 打印各标签的数量
    for label, count in label_counter.items():
        print(f"{label}: {count}")

    # 将标签数量以 json 格式保存到本地 entities_count.txt 中
    with open(os.path.join(processSanWenData.current_path.parent.parent, "data", "pre_data", 'entities_count.txt'), 'w',
              encoding='utf-8') as fp:
        json.dump(label_counter, fp, ensure_ascii=False, indent=4)

    # 数据
    categories = label_counter.keys()
    values = label_counter.values()

    # 指定支持中文的字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 创建柱状图
    plt.figure(figsize=(10, 6))
    plt.bar(categories, values, color='#7195A6')  # 创建条形图，指定颜色

    # 添加标题和标签
    plt.title('实体类别分布图', fontsize=16)
    plt.xlabel('类别', fontsize=12)
    plt.ylabel('数量', fontsize=12)

    # 在每个条形上显示数量
    for i, v in enumerate(values):
        plt.text(i, v + 500, str(v), ha='center', fontsize=10)

    # 调整布局
    plt.tight_layout()

    # 保存图表到本地
    plt.savefig('distribution_chart.png', dpi=300)  # 保存为PNG文件，设置分辨率为300 dpi
    plt.show()

    cent_size_dict = sorted(cent_size_dict.items(), key=lambda item: item[1], reverse=True)
    over_128 = 0
    for size, count in cent_size_dict:
        if size > 128:
            over_128 += count

    print(f"句子长度大于 128 的句子数量: {over_128}")

    cent_size_dict = dict(cent_size_dict)

    print(f"Total count: {all_count}")

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    keys = list(cent_size_dict.keys())
    values = list(cent_size_dict.values())

    keys = list(cent_size_dict.keys())
    values = list(cent_size_dict.values())
    # 创建直方图
    plt.figure(figsize=(8, 4))  # 设置图形的大小
    plt.bar(keys, values, color='#7195A6')  # 创建条形图，指定颜色

    # 添加标题和标签
    plt.title('句子长度分布图')
    plt.xlabel('长度')
    plt.ylabel('数量')

    plt.savefig('distribution_sent_chart.png', dpi=300)  # 保存为PNG文件，设置分辨率为300 dpi
    # 显示图形
    plt.show()

    # 计算中位数
    n = len(keys)
    if n % 2 == 1:
        median_length = keys[n // 2]
    else:
        median_length = (keys[n // 2 - 1] + keys[n // 2]) / 2

    # 打印中位数
    print("中位数是:", median_length)

    # 计算 95% 的阈值
    percentile_95 = np.percentile(keys, 95)
    print(f"95%的句子长度小于或等于 {percentile_95}")

    processSanWenData.convert_to_bio_format_txt(train_data, 'train.txt', 128)
    processSanWenData.convert_to_bio_format_txt(dev_data, 'dev.txt', 128)
