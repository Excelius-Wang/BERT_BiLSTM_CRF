"""
  @FileName：train.py.py
  @Author：Excelius
  @CreateTime：2024/11/9 21:18
  @Company: None
  @Description：
"""
import os
import json
import torch

from config.config import NerConfig
from model import BertNer
from src.data.data_loader import NerDataset

from tqdm import tqdm
from seqeval.metrics import classification_report
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, BertTokenizer, RoCBertTokenizer
from torch.optim import AdamW

from src.utils.log_config import logger

# 指定要使用的 GPU ID
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # P100 的 ID 是 0 和 1
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


class Trainer:
    """
    训练器类
    """

    def __init__(self,
                 output_dir=None,
                 model=None,
                 train_loader=None,
                 save_step=500,
                 dev_loader=None,
                 test_loader=None,
                 optimizer=None,
                 schedule=None,
                 epochs=1,
                 device="cpu",
                 id2label=None,
                 patience=3,
                 ):
        """
        初始化训练器
        :param output_dir: 模型保存的输出目录
        :param model: 需要训练的模型
        :param train_loader: 训练数据加载器
        :param save_step: 模型保存的间隔
        :param dev_loader: 验证数据加载器
        :param test_loader: 测试数据加载器
        :param optimizer: 优化器
        :param schedule: 学习率调度器
        :param epochs: 训练的轮数
        :param device: 训练设备
        :param id2label: ID 到标签的映射，用于解码预测的信息
        :param patience: 早停的耐心度（在多少轮没有验证集改善时停止）
        """
        self.output_dir = output_dir
        self.model = model
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.device = device
        self.optimizer = optimizer
        self.schedule = schedule
        self.id2label = id2label
        self.save_step = save_step
        self.patience = patience  # 早停的耐心度
        self.best_weighted_f1 = 0  # 最好的验证损失
        self.no_improvement_epochs = 0  # 记录没有改善的轮数
        self.best_model_path = None  # 最好的模型路径
        # 计算总的训练步数
        self.total_step = len(self.train_loader) * self.epochs

    def train(self):
        """
        训练函数
        :return: None
        """
        early_stopping_counter = 0  # 早停计数器
        # 初始化全局步数，用于记录训练轮数
        global_step = 1
        # 遍历所有训练周期
        for epoch in range(1, self.epochs + 1):
            # 设置模型为训练模式
            self.model.train()
            epoch_size = len(self.train_loader)
            # 遍历所有训练数据
            for step, batch_data in enumerate(self.train_loader):
                # 将批次数据加载到设备上（如 GPU）
                for key, value in batch_data.items():
                    batch_data[key] = value.to(self.device)
                # 获取输入数据、注意力掩码和标签
                input_ids = batch_data["input_ids"]
                attention_mask = batch_data["attention_mask"]
                labels = batch_data["labels"]
                # 模型前向传播
                output = self.model(input_ids, attention_mask, labels)
                # 获取损失
                loss = output.loss
                # 梯度清零
                self.optimizer.zero_grad()
                # 反向传播
                loss.backward()
                # 更新模型参数
                self.optimizer.step()
                # 更新学习率
                self.schedule.step()
                # 打印训练信息
                # if step % 100 == 0:
                logger.info(f"【Train】Epoch {epoch}, Step {step}/{epoch_size}, Loss: {loss.item()}")

                # 定期保存模型
                if global_step % self.save_step:
                    torch.save(self.model.state_dict(), os.path.join(self.output_dir, "pytorch_model_ner.bin"))
                # 更新全局步数
                global_step += 1

            current_weighted_f1 = self.evaluate()
            if current_weighted_f1 > self.best_weighted_f1:
                self.best_weighted_f1 = current_weighted_f1
                early_stopping_counter = 0
                self.best_model_path = "best_model.bin"
                torch.save(
                    self.model.state_dict(),
                    os.path.join(
                        self.output_dir,
                        self.best_model_path
                    )
                )
                logger.info(f"New best weighted F1: {self.best_weighted_f1:.4f}")
            else:
                early_stopping_counter += 1
                logger.info(f"No improvement, current F1: {current_weighted_f1}. Counter: {early_stopping_counter}")

            if early_stopping_counter >= self.patience:  # 连续patience次没有提升则停止训练
                logger.info("Early stopping...")
                return

    def evaluate(self):
        self.model.eval()  # 设置模型为评估模式

        # 初始化预测和真实标签列表
        preds = []
        trues = []
        with torch.no_grad():  # 关闭梯度计算
            for step, batch_data in enumerate(tqdm(self.dev_loader)):
                # 将批次数据加载到设备上（如 GPU）
                for key, value in batch_data.items():
                    batch_data[key] = value.to(self.device)
                # 获取输入数据、注意力掩码和标签
                input_ids = batch_data["input_ids"]
                attention_mask = batch_data["attention_mask"]
                labels = batch_data["labels"]
                # 模型前向传播，计算结果
                output = self.model(input_ids, attention_mask, labels)
                # 获取预测结果
                logits = output.logits
                # 将预测结果和标签转移到 CPU 上
                attention_mask = attention_mask.detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()

                # 获取批次大小
                batch_size = input_ids.size(0)
                # 遍历批次里的每个样本
                for i in range(batch_size):
                    # 获取真实标签的长度
                    length = sum(attention_mask[i])
                    # 获取预测结果，去除 [CLS] 和 [SEP] 标记
                    logit = logits[i][1:length]
                    # 将预测结果转换为标签
                    logit = [self.id2label[i] for i in logit]
                    # 获取真实标签
                    label = labels[i][1:length]
                    # 将真实标签 ID 转换为标签
                    label = [self.id2label[i] for i in label]
                    # 存储预测结果和真实标签
                    preds.append(logit)
                    trues.append(label)

            # 生成分类报告，包括精确度、召回率和 F1 分数等指标
            report = classification_report(trues, preds, digits=4, output_dict=True)

            # 提取 weighted F1
            weighted_f1 = report["micro avg"]["f1-score"]

            return weighted_f1

    def test(self):
        """
        测试函数，评估模型在测试集上的性能
        :return: 返回分类报告，包括精确度、召回率和F1分数等指标
        """
        self.best_model_path = "best_model.bin"
        # 加载模型
        self.model.load_state_dict(torch.load(os.path.join(self.output_dir, self.best_model_path)))
        # 设置模型为评估模式
        self.model.eval()
        # 初始化预测和真实标签列表
        preds = []
        trues = []
        for step, batch_data in enumerate(tqdm(self.test_loader)):
            # 将批次数据加载到设备上（如 GPU）
            for key, value in batch_data.items():
                batch_data[key] = value.to(self.device)
            # 获取输入数据、注意力掩码和标签
            input_ids = batch_data["input_ids"]
            attention_mask = batch_data["attention_mask"]
            labels = batch_data["labels"]
            # 模型前向传播
            output = self.model(input_ids, attention_mask, labels)
            # 获取预测结果
            logits = output.logits
            # 将预测结果和标签转移到 CPU 上
            attention_mask = attention_mask.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()

            # 获取批次大小
            batch_size = input_ids.size(0)
            # 遍历批次里的每个样本
            for i in range(batch_size):
                # 获取真实标签的长度
                length = sum(attention_mask[i])
                # 获取预测结果，去除 [CLS] 和 [SEP] 标记
                logit = logits[i][1:length]
                # 将预测结果转换为标签
                logit = [self.id2label[i] for i in logit]
                # 获取真实标签
                label = labels[i][1:length]
                # 将真实标签 ID 转换为标签
                label = [self.id2label[i] for i in label]
                # 存储预测结果和真实标签
                preds.append(logit)
                trues.append(label)

        # 生成分类报告，包括精确度、召回率和 F1 分数等指标
        report = classification_report(trues, preds, digits=4)
        return report


def build_optimizer_and_scheduler(args, model, t_total):
    """
    构建优化器和学习率调度器
    :param args: 配置参数
    :param model: NER 模型
    :param t_total: 总的训练步数
    :return: 优化器和学习率调度器
    """
    # 确保模型是正确的模块，如果是DataParallel模型则获取其module属性
    module = (model.module if hasattr(model, "module") else model)

    # 差分学习率
    no_decay = ["bias", "LayerNorm.weight"]
    # 获取模型的所有参数
    model_param = list(module.named_parameters())

    # 存储 BERT 和其他参数
    bert_param_optimizer = []
    other_param_optimizer = []

    # 遍历模型参数，根据参数名称分配到 BERT 或其他参数列表
    for name, para in model_param:
        space = name.split('.')
        # print(name)
        if space[0] == 'bert_module' or space[0] == "bert":
            bert_param_optimizer.append((name, para))
        else:
            other_param_optimizer.append((name, para))

    # 构建优化器参数组
    optimizer_grouped_parameters = [
        # BERT模块参数组，不包含不衰减的参数
        {"params": [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay, 'lr': args.bert_learning_rate},
        # BERT 模块参数，包含不衰减的参数
        {"params": [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': args.bert_learning_rate},

        # 其他模块，差分学习率
        {"params": [p for n, p in other_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay, 'lr': args.crf_learning_rate},
        # 其他模块，不包含不衰减的参数
        {"params": [p for n, p in other_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': args.crf_learning_rate},
    ]

    # 使用 AdamW 优化器
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.bert_learning_rate, eps=args.adam_epsilon)
    # 使用线性预热调度器
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(args.warmup_proportion * t_total), num_training_steps=t_total
    )

    return optimizer, scheduler


def main(_data_name, _model_name):
    """
    主函数，用于训练NER模型
    :param _data_name: 数据集名称
    :param _model_name: BERT 名称
    :return: None
    """
    # 读取配置参数
    args = NerConfig(_data_name, _model_name)
    # 加载 BERT 分词器
    logger.info(f"Loading BERT tokenizer from {args.bert_dir}")
    if _model_name == "roc_bert":
        tokenizer = RoCBertTokenizer.from_pretrained("weiweishi/roc-bert-base-zh")
    else:
        tokenizer = BertTokenizer.from_pretrained(args.bert_dir)

    # 保存模型参数配置
    with open(os.path.join(args.output_dir, _data_name + "_args.json"), "w", encoding='utf-8') as fp:
        json.dump(vars(args), fp, ensure_ascii=False, indent=2)

    # 读取训练数据
    with open(os.path.join(args.data_path, "train.txt"), "r", encoding='utf-8') as fp:
        train_data = fp.read().split("\n")
    train_data = [json.loads(d) for d in train_data]

    # 读取验证数据
    with open(os.path.join(args.data_path, "dev.txt"), "r", encoding='utf-8') as fp:
        dev_data = fp.read().split("\n")
    dev_data = [json.loads(d) for d in dev_data]

    # 创建训练数据集和验证数据集
    train_dataset = NerDataset(train_data, args, tokenizer)
    dev_dataset = NerDataset(dev_data, args, tokenizer)
    # 创建训练数据加载器和验证数据加载器
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.train_batch_size, num_workers=2)
    dev_loader = DataLoader(dev_dataset, shuffle=False, batch_size=args.dev_batch_size, num_workers=2)

    # 初始化 NER 模型，并将模型加载到设备上
    model = BertNer(args)

    # 检查CUDA是否可用并获取设备
    if torch.cuda.is_available():
        # 获取 GPU 数量
        n_gpu = torch.cuda.device_count()
        print(f"Number of GPUs available: {n_gpu}")

        # 选择用于后续操作的设备
        device = torch.device("cuda:0")  # 可以使用设备 0 作为主设备
        model.to(device)  # 将模型移动到主设备

        # 遍历所有 GPU 并打印信息
        for i in range(n_gpu):
            gpu_name = torch.cuda.get_device_name(i)
            print(f"GPU {i}: {gpu_name}")
    else:
        print("CUDA is not available. No GPU detected.")
        device = torch.device("cpu")  # 如果没有 GPU，则使用 CPU

    # 增加以下代码以使用 DataParallel
    if n_gpu > 1:
        logger.info(f"Using {n_gpu} GPUs")
        # 将模型包装在 DataParallel 中
        model = torch.nn.DataParallel(model, device_ids=range(n_gpu))  # 使用所有可用的 GPU
    else:
        logger.info(f"Using device: {device}")

    # 获取总的训练步数
    t_toal = len(train_loader) * args.epochs
    # 构建优化器和学习率调度器
    optimizer, schedule = build_optimizer_and_scheduler(args, model, t_toal)

    # 初始化训练器
    train = Trainer(
        output_dir=args.output_dir,
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        test_loader=dev_loader,
        optimizer=optimizer,
        schedule=schedule,
        epochs=args.epochs,
        device=device,
        id2label=args.id2label,
        # fold=fold_idx
    )

    # 训练模型
    train.train()

    # 测试模型
    report = train.test()
    print(report)


if __name__ == "__main__":
    data_name = "pre_data"
    # model_name = "roc_bert"
    model_name = "bert_wwn_ext_large"
    # model_name = "bert_wwn_ext_base"
    main(data_name, model_name)
