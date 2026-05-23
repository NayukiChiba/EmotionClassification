"""
Checkpoint 管理模块

提供 save_checkpoint 和 load_checkpoint 函数，
用于保存和恢复训练状态（模型权重、优化器状态、调度器状态、词表等）。
"""

from pathlib import Path
from typing import Dict, Optional

import torch

from src.data.mapping import VocabMapping


def save_checkpoint(
    filepath: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    validation_loss: float,
    vocabulary: VocabMapping,
) -> None:
    """
    保存完整训练状态到 checkpoint 文件

    将模型权重、优化器状态、调度器状态、当前 epoch、
    验证损失和词表打包保存为单个 .pth 文件。

    Args:
        filepath: checkpoint 文件的保存路径
        model: PyTorch 模型
        optimizer: 优化器
        scheduler: 学习率调度器
        epoch: 当前 epoch 编号
        validation_loss: 当前验证集损失
        vocabulary: 词表映射实例
    """
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "validation_loss": validation_loss,
            "vocabulary": vocabulary.to_dict(),
        },
        filepath,
    )


def load_checkpoint(
    filepath: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    device: str = "cpu",
) -> Dict:
    """
    从 checkpoint 文件加载训练状态

    恢复模型权重，可选择性恢复优化器和调度器状态。

    Args:
        filepath: checkpoint 文件的路径
        model: 用于加载权重的 PyTorch 模型
        optimizer: 可选，用于恢复状态的优化器
        scheduler: 可选，用于恢复状态的调度器
        device: 加载设备字符串

    Returns:
        包含 epoch、validation_loss、vocabulary 等字段的 checkpoint 字典
    """
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint
