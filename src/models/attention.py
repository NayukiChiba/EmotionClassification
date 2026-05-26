"""
Attention 注意力机制模块

实现 Bahdanau 加性注意力（Additive Attention），
对 RNN 输出的所有时刻隐藏状态进行加权求和，生成上下文向量。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """
    Bahdanau 加性注意力层

    通过一个可学习的得分网络（Linear -> Tanh -> Linear）为每个时刻
    的隐藏状态打分，经过 softmax 归一化后对隐藏状态加权求和。
    PAD 位置的得分被设为极小值 (-1e9)，使其权重趋近于 0。

    Args:
        hidden_size: RNN 输出的隐藏状态维度
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.score_network = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1, bias=False),
        )

    def forward(self, rnn_output: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            rnn_output: RNN 所有时刻的输出，形状为 (batch_size, sequence_length, hidden_size)
            mask: Attention Mask，1=有效位置 0=PAD，形状为 (batch_size, sequence_length)

        Returns:
            context_vector: 加权求和后的上下文向量，形状为 (batch_size, hidden_size)
        """
        # 计算每个时刻的注意力得分, 形状 (batch_size, sequence_length)
        scores = self.score_network(rnn_output).squeeze(-1)

        # 将 PAD 位置的得分设为极小值
        scores = scores.masked_fill(mask == 0, -1e9)

        # softmax 归一化得到注意力权重, 形状 (batch_size, sequence_length)
        attention_weights = F.softmax(scores, dim=1)

        # 加权求和, 形状 (batch_size, hidden_size)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), rnn_output).squeeze(
            1
        )

        return context_vector
