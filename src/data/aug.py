"""
  @FileName：aug.py.py
  @Author：Excelius
  @CreateTime：2024/11/19 16:39
  @Company: None
  @Description：
"""
import copy
import glob
import json
import os
import random
from pathlib import Path

from tqdm import tqdm

from src.utils.log_config import logger

current_path = Path.cwd()
main_path = current_path.parent.parent
train_file = os.path.join(main_path, "data", "ori_data", "train.json")
labels_file = os.path.join(main_path, "data", "ori_data", "label.json")

output_dir = os.path.join(main_path, "data", "pre_data")
aug_dir = os.path.join(main_path, "data", "aug_data")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if not os.path.exists(aug_dir):
    os.makedirs(aug_dir)


def get_data():
    # ["PRO", "ORG", "CONT", "RACE", "NAME", "EDU", "LOC", "TITLE"]
    # 定义函数获取数据

    # 打开并读取训练数据文件
    with open(train_file, "r", encoding="utf-8") as fp:
        data = fp.read()

    # 打开并读取标签数据文件，转换为JSON对象
    with open(labels_file, "r", encoding="utf-8") as fp:
        labels = json.loads(fp.read())

    # 创建一个字典，键为标签名称，值为空列表，用于存储对应标签的实体
    entities = {v: [] for v in labels.values()}

    # 初始化一个列表，用于存储处理后的文本
    texts = []

    # 将读取的数据字符串转换为JSON对象
    data = json.loads(data)
    # 遍历数据集中的每条数据
    for d in data:
        text = d['sentence']  # 获取文本内容
        labels = d['entities']  # 获取标签列表
        # 遍历每条数据的标签
        for label in labels:
            label_name = label.get('name')  # 获取标签名称
            label_type = label.get('type')  # 获取标签类型
            # 将文本中的实体替换为特殊标记，格式为"#;#标签类型#;#"
            text = text.replace(label_name, "#;#{}#;#".format(label_type))
            # 向实体列表中添加实体
            entities[label_type].append(label_name)
        # 将处理后的文本添加到文本列表中
        texts.append(text)

    # 遍历每种实体类型及其对应的实体列表
    for k, v in entities.items():
        # 将每种类型的实体列表去重并保存到对应的文件中
        with open(os.path.join(aug_dir, k + ".txt"), "w", encoding="utf-8") as fp:
            fp.write("\n".join(list(set(v))))

    # 将处理后的所有文本保存到一个文件中
    with open(os.path.join(aug_dir, "texts.txt"), 'w', encoding="utf-8") as fp:
        fp.write("\n".join(texts))


def aug_by_template(text_repeat=2):
    """ 基于模板的文本数据增强方法。
    参数:
    text_repeat (int): 每条原始文本重复处理的次数，用以增强数据。
    """
    # 读取处理后的文本数据
    with open(os.path.join(aug_dir, "texts.txt"), 'r', encoding="utf-8") as fp:
        texts = fp.read().strip().split('\n')

    # 初始化一个字典来存储每种实体类型的具体实例
    entities = {}
    for ent_txt in glob.glob(aug_dir + "/*.txt"):
        # 跳过文本
        if "texts.txt" in ent_txt:
            continue

        with open(ent_txt, 'r', encoding="utf-8") as fp:
            # 读取实体列表
            label = fp.read().strip().split("\n")
            # 获取实体类型名称
            ent_txt = ent_txt.replace("\\", "/")
            # 获取实体类型名称
            label_name = ent_txt.split("/")[-1].split(".")[0]
            # 将实体列表添加到实体字典中
            entities[label_name] = label

    # 创建实体数据的副本，用于实体的不放回抽样
    entities_copy = copy.deepcopy(entities)

    # 读取原始的训练数据
    with open(train_file, "r", encoding="utf-8") as fp:
        ori_data = json.loads(fp.read())

    res = []
    # 遍历每一条文本，进行增强
    for text in tqdm(texts, ncols=100):
        # 将文本按照"#;#"分割，获取文本中的实体
        text = text.split("#;#")
        # 初始化一个列表，用于存储处理后的文本
        text_tmp = []
        # 初始化一个列表，用于存储处理后的标签
        labels_tmp = []
        # 对每条文本重复处理指定次数
        for i in range(text_repeat):
            # 遍历文本中的每个词
            for t in text:
                # 如果词为空，则跳过
                if t == "":
                    continue
                # 如果词在实体字典中
                if t in entities:
                    # 不放回抽样，为了维持实体的多样性
                    if not entities[t]:
                        # 如果实体列表为空，则将实体列表重置为副本
                        entities[t] = copy.deepcopy(entities_copy[t])
                    # 随机选择一个实体
                    ent = random.choice(entities[t])
                    # 从实体列表中移除该实体
                    entities[t].remove(ent)
                    # 计算实体的长度
                    length = len("".join(text_tmp))
                    # 将实体添加到文本中
                    text_tmp.append(ent)
                    # 将实体的标签添加到标签列表中
                    labels_tmp.append({"name": ent, "type": t, "pos": [length, length + len(ent)]})
                else:
                    # 如果词不在实体字典中，则直接添加到文本中
                    text_tmp.append(t)
            # 将处理后的文本和标签添加到结果列表中
            tmp = {
                "sentence": "".join(text_tmp),
                "entities": labels_tmp
            }
            # 重置文本和标签列表
            text_tmp = []
            # 重置标签列表
            labels_tmp = []
            # 将处理后的文本添加到结果列表中
            res.append(tmp)
    # 加上原始的
    res = ori_data + res

    logger.info("增强后的数据量: {}".format(len(res)))

    with open(os.path.join(main_path, "data", "ori_data", "train_aug.json"), "w", encoding="utf-8") as fp:
        json.dump(res, fp, ensure_ascii=False)


if __name__ == '__main__':
    # 1、第一步：获取基本数据
    get_data()
    # 2、第二步：进行模板类数据增强
    aug_by_template(text_repeat=1)
