"""
GRU 模型模块

实现 GRU (Gated Recurrent Unit) 模型，继承 BaseRNN 抽象基类。
"""

from src.models.base import BaseRNN


class GRU(BaseRNN):
    """
    GRU 模型

    使用 nn.GRU 作为序列编码器，继承 BaseRNN 的完整前向流程：
    Embedding -> Dropout -> GRU -> [Attention | Last Step] -> Classifier -> Sigmoid

    GRU 是 LSTM 的简化变体，将遗忘门和输入门合并为更新门，
    参数更少，训练更快，效果与 LSTM 接近。
    """

    def __init__(self, **kwargs):
        """
        初始化 GRU

        Args 参考 BaseRNN.__init__，rnn_type 固定为 "GRU"
        """
        kwargs["rnn_type"] = "GRU"
        super().__init__(**kwargs)
