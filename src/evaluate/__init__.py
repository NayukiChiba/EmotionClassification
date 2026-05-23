"""
评估模块

统一导出评估相关的所有组件：
- Evaluator: 测试集评估器
- compute_metrics / MetricsReport: 评估指标计算
- plot_confusion_matrix / plot_roc_curve / plot_training_history: 可视化函数
"""

from src.evaluate.evaluator import Evaluator
from src.evaluate.metrics import MetricsReport, compute_metrics
from src.evaluate.visualize import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_training_history,
)

__all__ = [
    "Evaluator",
    "MetricsReport",
    "compute_metrics",
    "plot_confusion_matrix",
    "plot_roc_curve",
    "plot_training_history",
]
