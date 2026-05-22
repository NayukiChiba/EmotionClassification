"""
模型模块

提供模型注册表和工厂函数，支持通过名称字符串切换 RNN/LSTM/GRU。
统一导出所有模型组件。

模型注册表:
    MODEL_REGISTRY = {"rnn": SimpleRNN, "lstm": LSTM, "gru": GRU}
"""

from typing import Dict, Type

from src.models.attention import Attention
from src.models.base import BaseRNN
from src.models.gru import GRU
from src.models.lstm import LSTM
from src.models.rnn import SimpleRNN

# 模型注册表，按名称映射到具体模型类
MODEL_REGISTRY: Dict[str, Type[BaseRNN]] = {
    "rnn": SimpleRNN,
    "lstm": LSTM,
    "gru": GRU,
}


def build_model(vocabulary_size: int, pad_index: int, **kwargs) -> BaseRNN:
    """
    模型工厂函数

    根据 kwargs 中的 rnn_type 参数从注册表中选取对应模型类并实例化。
    默认使用 LSTM。

    Args:
        vocabulary_size: 词表大小
        pad_index: PAD 标记的索引
        **kwargs: 传递给模型构造函数的其他参数，包括:
            - rnn_type (str): 模型类型 "lstm" / "rnn" / "gru"
            - embedding_dimension (int): 词嵌入维度
            - hidden_dimension (int): 隐藏层维度
            - number_of_layers (int): RNN 层数
            - dropout (float): Dropout 概率
            - bidirectional (bool): 是否双向
            - use_attention (bool): 是否使用 Attention

    Returns:
        实例化后的模型（SimpleRNN / LSTM / GRU）

    Raises:
        ValueError: 当 rnn_type 不在注册表中时抛出
    """
    rnn_type = kwargs.get("rnn_type", "lstm")
    model_class = MODEL_REGISTRY.get(rnn_type)
    if model_class is None:
        raise ValueError(
            f"未知的模型类型: '{rnn_type}'，请从 {list(MODEL_REGISTRY.keys())} 中选择"
        )
    return model_class(vocabulary_size=vocabulary_size, pad_index=pad_index, **kwargs)


__all__ = [
    "Attention",
    "BaseRNN",
    "SimpleRNN",
    "LSTM",
    "GRU",
    "MODEL_REGISTRY",
    "build_model",
]
