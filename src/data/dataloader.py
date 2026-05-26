"""
DataLoader 工厂模块

提供 SentimentDataset 类和 create_data_loaders 函数，
封装 PyTorch Dataset/DataLoader 的创建逻辑。
"""

from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset


class SentimentDataset(Dataset):
    """
    情感分类 Dataset

    封装输入索引、Attention Mask 和标签，供 PyTorch DataLoader 使用。

    输入张量形状:
        input_ids: (N, max_sequence_length)
        masks:     (N, max_sequence_length)
        labels:    (N, 1)

    返回形状:
        __getitem__: (max_sequence_length,), (max_sequence_length,), (1,)
    """

    def __init__(
        self,
        input_ids: torch.Tensor,
        masks: torch.Tensor,
        labels: torch.Tensor,
    ):
        """
        初始化数据集

        Args:
            input_ids: 填充后的文本索引张量，形状为 (N, max_sequence_length)
            masks: Attention Mask 张量，形状为 (N, max_sequence_length)
            labels: 标签张量，形状为 (N,)，将自动转换为 (N, 1) float32
        """
        self.input_ids = input_ids
        self.masks = masks
        self.labels = labels.float().unsqueeze(1)  # (N,) -> (N, 1)

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取单个样本

        Args:
            index: 样本索引

        Returns:
            (input_ids[index], masks[index], labels[index])
            形状分别为 (max_sequence_length,), (max_sequence_length,), (1,)
        """
        return self.input_ids[index], self.masks[index], self.labels[index]

    def __len__(self) -> int:
        """返回数据集样本总数"""
        return len(self.labels)


def create_data_loaders(
    train_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    valid_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    test_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    batch_size: int = 64,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建训练集、验证集和测试集的 DataLoader

    训练集 shuffle=True，验证集和测试集 shuffle=False。

    Args:
        train_data: (input_ids, masks, labels) 元组，各张量形状为 (N_train, ...)
        valid_data: (input_ids, masks, labels) 元组，各张量形状为 (N_valid, ...)
        test_data:  (input_ids, masks, labels) 元组，各张量形状为 (N_test, ...)
        batch_size: 批大小
        num_workers: DataLoader 的工作进程数（Windows 下建议设为 0）

    Returns:
        (train_loader, valid_loader, test_loader)
    """
    train_dataset = SentimentDataset(*train_data)
    valid_dataset = SentimentDataset(*valid_data)
    test_dataset = SentimentDataset(*test_data)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, valid_loader, test_loader
