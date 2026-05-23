"""
训练器模块

提供 Trainer 类，封装完整训练循环，包含：
- 单 epoch 训练与验证
- 梯度裁剪
- tqdm 进度条
- 学习率调度
- 早停检查
- checkpoint 自动保存
- 训练历史记录
"""

import time
from typing import Dict, List, Literal

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.paths import BEST_MODEL_PATH, LAST_MODEL_PATH
from src.data.mapping import VocabMapping
from src.train.checkpoint import save_checkpoint
from src.train.early_stopping import EarlyStopping
from src.train.logger import Logger


class Trainer:
    """
    训练器

    封装模型训练的完整流程，包括训练循环、验证、日志输出、
    学习率调度、早停和 checkpoint 管理。

    Args:
        model: 待训练的 PyTorch 模型
        train_loader: 训练集 DataLoader
        valid_loader: 验证集 DataLoader
        optimizer: PyTorch 优化器
        scheduler: 学习率调度器
        criterion: 损失函数（BCELoss）
        early_stopping: 早停实例
        vocabulary: 词表映射（用于保存 checkpoint）
        device: 训练设备
        epochs: 最大训练轮数
        gradient_clip_value: 梯度裁剪阈值
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
        criterion: nn.Module,
        early_stopping: EarlyStopping,
        vocabulary: VocabMapping,
        device: torch.device,
        epochs: int = 50,
        gradient_clip_value: float = 1.0,
    ):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.early_stopping = early_stopping
        self.vocabulary = vocabulary
        self.device = device
        self.epochs = epochs
        self.gradient_clip_value = gradient_clip_value
        self.logger = Logger()

        # 训练历史记录
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_accuracy": [],
            "valid_loss": [],
            "valid_accuracy": [],
        }
        self.best_valid_loss = float("inf")

    def train_epoch(self, epoch: int) -> tuple[float, float]:
        """
        执行训练模式下的一个 epoch

        Args:
            epoch: 当前 epoch 编号（从 0 开始）

        Returns:
            (average_loss, accuracy): 该 epoch 的平均损失和准确率
        """
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        description = f"[Train] Epoch {epoch + 1}/{self.epochs}"
        progress_bar = tqdm(self.train_loader, desc=description, unit="batch")

        for input_ids, masks, labels in progress_bar:
            input_ids = input_ids.to(self.device)
            masks = masks.to(self.device)
            labels = labels.to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            predictions = self.model(input_ids, masks)
            loss = self.criterion(predictions, labels)

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.gradient_clip_value
            )
            self.optimizer.step()

            # 统计
            batch_size = input_ids.size(0)
            total_loss += loss.item() * batch_size
            predicted_labels = (predictions >= 0.5).float()
            total_correct += (predicted_labels == labels).sum().item()
            total_samples += batch_size

            progress_bar.set_postfix(
                loss=f"{total_loss / total_samples:.4f}",
                accuracy=f"{total_correct / total_samples:.4f}",
            )

        return total_loss / total_samples, total_correct / total_samples

    @torch.no_grad()
    def valid_epoch(
        self, epoch: int, mode: Literal["Val", "Test"] = "Val"
    ) -> tuple[float, float]:
        """
        执行验证/测试模式下的一个 epoch

        Args:
            epoch: 当前 epoch 编号
            mode: 评估模式 "Val" 或 "Test"

        Returns:
            (average_loss, accuracy): 该 epoch 的平均损失和准确率
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        description = f"[{mode}] Epoch {epoch + 1}/{self.epochs}"
        progress_bar = tqdm(self.valid_loader, desc=description, unit="batch")

        for input_ids, masks, labels in progress_bar:
            input_ids = input_ids.to(self.device)
            masks = masks.to(self.device)
            labels = labels.to(self.device)

            # 仅前向传播
            predictions = self.model(input_ids, masks)
            loss = self.criterion(predictions, labels)

            # 统计
            batch_size = input_ids.size(0)
            total_loss += loss.item() * batch_size
            predicted_labels = (predictions >= 0.5).float()
            total_correct += (predicted_labels == labels).sum().item()
            total_samples += batch_size

            progress_bar.set_postfix(
                loss=f"{total_loss / total_samples:.4f}",
                accuracy=f"{total_correct / total_samples:.4f}",
            )

        return total_loss / total_samples, total_correct / total_samples

    def train(self) -> Dict[str, List[float]]:
        """
        执行完整训练流程

        每个 epoch 依次执行训练→验证→学习率调整→checkpoint 保存→早停检查。
        训练过程中同时保存最佳模型（best_model.pth）和最新模型（last_model.pth）。

        Returns:
            history: 包含 train_loss、train_accuracy、valid_loss、valid_accuracy 的字典
        """
        print(f"开始训练: epochs={self.epochs}, device={self.device}")
        self.logger.start()
        start_time = time.time()

        for epoch in range(self.epochs):
            # 训练和验证
            train_loss, train_accuracy = self.train_epoch(epoch)
            valid_loss, valid_accuracy = self.valid_epoch(epoch, mode="Val")

            # 记录历史
            self.history["train_loss"].append(train_loss)
            self.history["train_accuracy"].append(train_accuracy)
            self.history["valid_loss"].append(valid_loss)
            self.history["valid_accuracy"].append(valid_accuracy)

            # 学习率调度
            self.scheduler.step(valid_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch + 1}/{self.epochs} - "
                f"train_loss: {train_loss:.4f}, train_acc: {train_accuracy:.4f}, "
                f"valid_loss: {valid_loss:.4f}, valid_acc: {valid_accuracy:.4f}, "
                f"lr: {current_lr:.6f}"
            )

            # 日志记录（TensorBoard）
            self.logger.log(
                metrics={
                    "[Train] Loss": train_loss,
                    "[Train] Acc": train_accuracy,
                    "[Valid] Loss": valid_loss,
                    "[Valid] Acc": valid_accuracy,
                    "[Train] LR": current_lr,
                },
                step=epoch,
            )

            # 保存 checkpoint
            if valid_loss < self.best_valid_loss:
                self.best_valid_loss = valid_loss
                save_checkpoint(
                    BEST_MODEL_PATH,
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch,
                    valid_loss,
                    self.vocabulary,
                )
                print(f"  -> 最佳模型已保存 (valid_loss={valid_loss:.4f})")

            save_checkpoint(
                LAST_MODEL_PATH,
                self.model,
                self.optimizer,
                self.scheduler,
                epoch,
                valid_loss,
                self.vocabulary,
            )

            # 早停检查
            if self.early_stopping(valid_loss):
                print(
                    f"早停触发: 验证损失在 {self.early_stopping.patience} 轮内未显著改善"
                )
                break

        elapsed_time = time.time() - start_time
        minutes, seconds = divmod(int(elapsed_time), 60)
        print(
            f"训练完成, 总耗时: {minutes}m {seconds}s, "
            f"最佳验证损失: {self.best_valid_loss:.4f}"
        )

        self.logger.close()
        return self.history
