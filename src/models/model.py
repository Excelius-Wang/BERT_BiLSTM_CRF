"""
  @FileName：model.py
  @Author：Excelius
  @CreateTime：2024/11/8 10:42
  @Company: None
  @Description：
"""
# 解决无法加载 'https://huggingface.co' 问题
import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch.nn as nn

from torchcrf import CRF
from transformers import BertModel, BertConfig, AutoModel, AutoConfig

from src.utils.log_config import logger


class ModelOutput:
    """
    模型输出类
        logits：模型的解码输出（预测结果）。
        labels：真实标签。
        loss：如果提供了标签，计算的损失值。
    """

    def __init__(self, logits, labels=None, loss=None):
        self.logits = logits
        self.labels = labels
        self.loss = loss

    # 重写 __iter__ 方法，使得 ModelOutput 对象可以迭代，方便并行训练
    def __iter__(self):
        return iter((self.logits, self.labels, self.loss))


class BertNer(nn.Module):
    """
    主模型类，模型构成：Bert + BiLSTM + CRF
    """

    def __init__(self, args):
        super(BertNer, self).__init__()
        # 加载预训练的 Bert 模型
        if args.model_name == "roc_bert":
            self.bert = AutoModel.from_pretrained("weiweishi/roc-bert-base-zh")
            self.bert_config = AutoConfig.from_pretrained("weiweishi/roc-bert-base-zh")
        else:
            self.bert = BertModel.from_pretrained(args.bert_dir)
            # 加载 Bert 配置(如隐藏层大小等)
            self.bert_config = BertConfig.from_pretrained(args.bert_dir)
        # Bert 隐藏层大小
        hidden_size = self.bert_config.hidden_size
        # BiLSTM 隐藏层大小
        if args.model_name == "bert_wwn_ext_base":
            self.lstm_hiden = 512
        else:
            self.lstm_hiden = 256
        # 最大序列长度
        self.max_seq_len = args.max_seq_len
        # BiLSTM 层, 输入大小为 hidden_size，输出大小为 2* self.lstm_hiden
        self.bilstm = nn.LSTM(hidden_size, self.lstm_hiden, 1, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.1)
        # 线性层，将 BiLSTM 输出转换为标签数量大小
        self.linear = nn.Linear(self.lstm_hiden * 2, args.num_labels)
        # CRF 层，用于优化序列预测，batch_first=True 表示输入的 batch_size 在第一维
        self.crf = CRF(args.num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        """
        前向传播
        :param input_ids: 输入序列的编码表示
        :param attention_mask: 指示哪些位置是有效的输入
        :param labels: 标签，可选，如果提供，将用于计算损失
        :return: 模型输出
        """
        # Bert 输入为 input_ids 和 attention_mask
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # 输出 bert_output[0]，即最后一层隐藏状态。
        seq_out = bert_output[0]  # [batchsize, max_len, 768]
        batch_size = seq_out.size(0)
        # 将上一层（BERT）的输出 seq_out 传入 BiLSTM 层
        # seq_out：形状为 [batch_size, max_seq_len, lstm_hidden * 2]，代表每个时间步的隐藏状态（双向，因此是 lstm_hidden * 2）
        # _：LSTM的最后一个时间步的隐藏状态，这里没有用到
        seq_out, _ = self.bilstm(seq_out)
        seq_out = self.dropout(seq_out)
        # 将张量展平，为了将数据展平，以适应全连接层的输入要求
        seq_out = seq_out.contiguous().view(-1, self.lstm_hiden * 2)
        # 在全连接层处理后，将数据重塑回原始的序列形状，以便 CRF 层可以正确处理序列数据
        seq_out = seq_out.contiguous().view(batch_size, self.max_seq_len, -1)
        # 线性层处理 BiLSTM 输出，将 LSTM 的输出转换为适合 CRF 层的形状
        seq_out = self.linear(seq_out)
        # 使用CRF解码预测的序列标签（logits），使用 attention_mask 来忽略无效的位置
        logits = self.crf.decode(seq_out, mask=attention_mask.bool())
        loss = None
        # 如果提供了真实标签 labels，计算负对数似然损失 loss
        if labels is not None:
            loss = -self.crf(seq_out, labels, mask=attention_mask.bool(), reduction='mean')
        # 将预测结果、真实标签和损失包装成 ModelOutput
        model_output = ModelOutput(logits, labels, loss)
        return model_output
