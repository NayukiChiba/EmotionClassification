"""
评估指标模块

提供情感分类模型评估所需的全部指标计算函数。
"""

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class MetricsReport:
    """评估指标报告数据类"""

    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    auc: float = 0.0
    classification_report: str = ""


def compute_metrics(
    y_true: np.ndarray,
    y_prediction: np.ndarray,
    y_probability: np.ndarray,
    target_names: tuple = ("负面", "正面"),
) -> MetricsReport:
    """
    计算情感二分类的全部评估指标

    Args:
        y_true: 真实标签数组, 形状为 (N,), 值为 0 或 1
        y_prediction: 预测标签数组, 形状为 (N,), 值为 0 或 1
        y_probability: 预测概率数组, 形状为 (N,), 值域 [0, 1]
        target_names: 类别名称元组, (负类名, 正类名)

    Returns:
        包含 accuracy, precision, recall, f1, auc, classification_report 的 MetricsReport
    """
    report = MetricsReport()
    report.accuracy = accuracy_score(y_true, y_prediction)
    report.precision = precision_score(y_true, y_prediction, zero_division=0)
    report.recall = recall_score(y_true, y_prediction, zero_division=0)
    report.f1_score = f1_score(y_true, y_prediction, zero_division=0)
    report.auc = roc_auc_score(y_true, y_probability)
    report.classification_report = classification_report(
        y_true, y_prediction, target_names=list(target_names), zero_division=0
    )
    return report


def format_classification_report_string(
    y_true: np.ndarray, y_prediction: np.ndarray
) -> str:
    """
    生成 sklearn 格式的分类报告

    Args:
        y_true: 真实标签数组
        y_prediction: 预测标签数组

    Returns:
        sklearn classification_report 文本
    """
    return classification_report(
        y_true, y_prediction, target_names=["负面", "正面"], zero_division=0
    )
