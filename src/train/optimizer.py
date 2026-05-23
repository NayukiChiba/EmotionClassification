"""
优化器构建模块

提供 build_optimizer 函数，用于创建 AdamW 优化器。
"""

import torch


def build_optimizer(
    model: torch.nn.Module,
    learning_rate: float = 1e-2,
    weight_decay: float = 1e-5,
) -> torch.optim.AdamW:
    """
    构建 AdamW 优化器

    Args:
        model: PyTorch 模型实例
        learning_rate: 学习率
        weight_decay: 权重衰减系数（L2 正则化强度）

    Returns:
        配置好的 torch.optim.AdamW 优化器
    """
    return torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
