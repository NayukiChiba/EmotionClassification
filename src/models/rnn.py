"""
SimpleRNN 模型模块

实现标准 RNN (Vanilla RNN) 模型，继承 BaseRNN 抽象基类。
"""

from src.models.base import BaseRNN


class SimpleRNN(BaseRNN):
    """
    标准 RNN 模型

    使用 nn.RNN 作为序列编码器，继承 BaseRNN 的完整前向流程：
    Embedding → Dropout → RNN → [Attention | Last Step] → Classifier → Sigmoid
    """

    def __init__(self, **kwargs):
        """
        初始化 SimpleRNN

        Args 参考 BaseRNN.__init__，rnn_type 固定为 "RNN"
        """
        kwargs["rnn_type"] = "RNN"
        super().__init__(**kwargs)
