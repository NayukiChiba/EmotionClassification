"""
训练工具函数模块

提供随机种子设置、设备选择、参数量统计和时间格式化等通用工具函数。
"""

import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from config.default import DefaultParams


def set_seed(seed: int = DefaultParams.SEED) -> None:
    """
    设置全局随机种子，确保实验可复现

    同时设置 Python random、NumPy、PyTorch CPU 和 CUDA 的随机种子，
    并启用 cudnn deterministic 模式。

    Args:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_string: str = DefaultParams.DEVICE) -> torch.device:
    """
    解析设备字符串并返回对应的 torch.device 对象

    如果指定 cuda 但 CUDA 不可用，则自动降级为 CPU 并打印提示。

    Args:
        device_string: 设备字符串，如 "cuda" 或 "cpu"

    Returns:
        解析后的 torch.device 对象
    """
    if device_string == "cuda" and not torch.cuda.is_available():
        print("警告: CUDA 不可用，已自动降级为 CPU")
        return torch.device("cpu")
    return torch.device(device_string)


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    统计模型的参数数量

    Args:
        model: PyTorch 模型

    Returns:
        (total_parameters, trainable_parameters): 总参数量和可训练参数量
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def format_time(elapsed_seconds: float) -> str:
    """
    将秒数格式化为人类可读的时间字符串

    Args:
        elapsed_seconds: 经过的秒数

    Returns:
        格式化后的时间字符串，如 "2h 30m 15s"
    """
    minutes, seconds = divmod(int(elapsed_seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes > 0:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"
