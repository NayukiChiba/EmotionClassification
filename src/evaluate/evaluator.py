"""
评估器模块

提供 Evaluator 类，在测试集上执行完整评估：
收集预测 → 计算指标 → 生成可视化图表 → 打印评估报告
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.paths import FIGURES_DIR
from src.evaluate.metrics import MetricsReport, compute_metrics
from src.evaluate.visualize import plot_confusion_matrix, plot_roc_curve


class Evaluator:
    """
    测试集评估器

    加载模型后在测试集上执行推理，计算全部评估指标并生成可视化图表。

    Args:
        model: 已训练的 PyTorch 模型
        test_loader: 测试集 DataLoader
        device: 推理设备
    """

    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        device: torch.device,
    ):
        self.model = model
        self.test_loader = test_loader
        self.device = device

    @torch.no_grad()
    def _collect_predictions(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        在测试集上收集所有预测结果

        Returns:
            y_true: 真实标签, 形状 (N,)
            y_prediction: 预测标签 (0/1), 形状 (N,)
            y_probability: 预测概率, 形状 (N,)
        """
        self.model.eval()
        all_predictions = []
        all_probabilities = []
        all_labels = []

        for input_ids, masks, labels in tqdm(self.test_loader, desc="Evaluating"):
            input_ids = input_ids.to(self.device)
            masks = masks.to(self.device)

            predictions = self.model(input_ids, masks)
            probabilities = predictions.cpu().numpy()
            predicted_labels = (probabilities >= 0.5).astype(int)

            all_predictions.extend(predicted_labels.flatten())
            all_probabilities.extend(probabilities.flatten())
            all_labels.extend(labels.numpy().flatten())

        return (
            np.array(all_labels),
            np.array(all_predictions),
            np.array(all_probabilities),
        )

    def evaluate(self) -> MetricsReport:
        """
        执行完整评估流程

        1. 在测试集上收集预测
        2. 计算评估指标
        3. 生成混淆矩阵和 ROC 曲线图表
        4. 打印评估报告

        Returns:
            包含全部评估指标的 MetricsReport 实例
        """
        print("正在评估模型...")

        # 收集预测结果
        y_true, y_prediction, y_probability = self._collect_predictions()

        # 计算指标
        report = compute_metrics(y_true, y_prediction, y_probability)

        # 生成图表
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        plot_confusion_matrix(
            y_true, y_prediction, FIGURES_DIR / "confusion_matrix.png"
        )
        plot_roc_curve(y_true, y_probability, report.auc, FIGURES_DIR / "roc_curve.png")
        print(f"图表已保存到 {FIGURES_DIR}")

        # 打印报告
        print(f"""
{"=" * 50}
  测试集评估结果
{"=" * 50}
  Accuracy:   {report.accuracy:.4f}
  Precision:  {report.precision:.4f}
  Recall:     {report.recall:.4f}
  F1 Score:   {report.f1_score:.4f}
  AUC:        {report.auc:.4f}
{"=" * 50}

{report.classification_report}
""")

        return report
