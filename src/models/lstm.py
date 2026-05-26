"""
LSTM 模型模块

实现 LSTM 模型，继承 BaseRNN 抽象基类。
"""

from src.models.base import BaseRNN


class LSTM(BaseRNN):
    """
    LSTM 模型

    使用 nn.LSTM 作为序列编码器，继承 BaseRNN 的完整前向流程：
    Embedding -> Dropout -> LSTM -> [Attention | Last Step] -> Classifier -> Sigmoid

    LSTM 相比 RNN 引入了输入门、遗忘门和输出门机制，
    能够更好地捕捉长距离依赖关系。
    """

    def __init__(self, **kwargs):
        """
        初始化 LSTM

        Args 参考 BaseRNN.__init__，rnn_type 固定为 "LSTM"
        """
        kwargs["rnn_type"] = "LSTM"
        super().__init__(**kwargs)
