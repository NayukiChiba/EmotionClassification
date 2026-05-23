"""
学习率调度器模块

提供 build_scheduler 函数，用于创建 ReduceLROnPlateau 调度器。
"""

import torch


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    reduce_factor: float = 0.5,
    patience: int = 3,
) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
    """
    构建 ReduceLROnPlateau 学习率调度器

    当验证损失在 patience 轮内未改善时，学习率乘以 reduce_factor。
    配合早停机制使用，避免学习率过低时继续无效训练。

    Args:
        optimizer: PyTorch 优化器
        reduce_factor: 学习率缩减因子（新学习率 = 旧学习率 × reduce_factor）
        patience: 容忍轮数，即验证损失可连续不改善的最大 epoch 数

    Returns:
        配置好的 ReduceLROnPlateau 调度器
    """
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=reduce_factor,
        patience=patience,
    )
