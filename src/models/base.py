"""
RNN 模型抽象基类模块

定义 BaseRNN 抽象基类，封装 Embedding → Dropout → RNN → Attention → Classifier 的
统一前向流程。子类只需实现 _build_rnn() 方法即可得到特定类型的 RNN 模型。
"""

from typing import Literal

import torch
import torch.nn as nn

from src.models.attention import Attention


class BaseRNN(nn.Module):
    """
    RNN 模型抽象基类

    架构: Embedding → Dropout → RNN → [Attention | Last Valid Step] → Dropout → Linear → Sigmoid

    支持 LSTM / RNN / GRU 三种类型切换，支持双向和 Attention 可选。

    Args:
        vocabulary_size: 词表大小
        pad_index: PAD 标记的索引
        embedding_dimension: 词嵌入向量的维度
        hidden_dimension: RNN 隐藏状态的维度
        number_of_layers: RNN 层数
        dropout: Dropout 概率
        bidirectional: 是否使用双向 RNN
        use_attention: 是否使用 Attention 池化（否则取最后有效时刻）
        rnn_type: RNN 类型，可选 "LSTM" / "RNN" / "GRU"
    """

    def __init__(
        self,
        vocabulary_size: int,
        pad_index: int,
        embedding_dimension: int = 300,
        hidden_dimension: int = 256,
        number_of_layers: int = 2,
        dropout: float = 0.5,
        bidirectional: bool = True,
        use_attention: bool = True,
        rnn_type: Literal["LSTM", "RNN", "GRU"] = "LSTM",
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocabulary_size, embedding_dimension, padding_idx=pad_index
        )
        self.dropout = nn.Dropout(dropout)
        self.use_attention = use_attention
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type

        # 单层 RNN 不接受 dropout 参数
        rnn_dropout = dropout if number_of_layers > 1 else 0.0
        rnn_class = {"LSTM": nn.LSTM, "RNN": nn.RNN, "GRU": nn.GRU}[rnn_type]
        self.rnn = rnn_class(
            input_size=embedding_dimension,
            hidden_size=hidden_dimension,
            num_layers=number_of_layers,
            dropout=rnn_dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )

        # 计算 RNN 输出维度（双向则翻倍）
        self.hidden_size = hidden_dimension * 2 if bidirectional else hidden_dimension

        if use_attention:
            self.attention = Attention(self.hidden_size)

        self.classifier = nn.Linear(self.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            input_ids: 文本索引序列，形状为 (batch_size, sequence_length)
            mask: Attention Mask，1=有效 0=PAD，形状为 (batch_size, sequence_length)

        Returns:
            output: 预测概率，形状为 (batch_size, 1)，值域 [0, 1]
        """
        # Embedding + Dropout, 形状 (batch_size, sequence_length, embedding_dimension)
        embedded = self.dropout(self.embedding(input_ids))

        # RNN 前向, rnn_output 形状 (batch_size, sequence_length, hidden_size)
        rnn_output, _ = self.rnn(embedded)

        if self.use_attention:
            # Attention 池化, 形状 (batch_size, hidden_size)
            context = self.attention(rnn_output, mask)
        else:
            # 不使用 Attention 时取最后有效时刻
            if self.bidirectional:
                # 双向：前向取最后，反向取最前（第一时刻的反向输出）
                forward_last = rnn_output[:, -1, : self.hidden_size // 2]
                backward_first = rnn_output[:, 0, self.hidden_size // 2 :]
                context = torch.cat((forward_last, backward_first), dim=1)
            else:
                context = rnn_output[:, -1, :]

        # Dropout → Linear → Sigmoid
        context = self.dropout(context)
        logits = self.classifier(context)
        output = self.sigmoid(logits)

        return output
