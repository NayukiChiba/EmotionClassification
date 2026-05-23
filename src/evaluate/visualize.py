"""
可视化模块

提供训练过程中的可视化函数：
- 训练曲线（loss + accuracy）
- 混淆矩阵热力图
- ROC 曲线
"""

from pathlib import Path
from typing import Dict, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve

# 设置中文字体支持
matplotlib.rcParams["font.sans-serif"] = [
    "SimHei",
    "Noto Sans CJK SC",
    "WenQuanYi Micro Hei",
    "Microsoft YaHei",
]
matplotlib.rcParams["axes.unicode_minus"] = False


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Path,
    model_name: str = "LSTM",
    use_attention: bool = True,
) -> None:
    """
    绘制训练/验证 loss 和 accuracy 曲线

    使用双 Y 轴：左轴为 loss（train + valid），右轴为 accuracy（valid）。

    Args:
        history: 包含 train_loss、valid_loss、valid_accuracy 的字典
        save_path: 图表保存路径
        model_name: 模型名称（用于标题）
        use_attention: 是否使用 Attention
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    figure, axis1 = plt.subplots(figsize=(10, 5))

    axis1.set_xlabel("Epoch")
    axis1.set_ylabel("Loss", color="tab:blue")
    axis1.plot(epochs, history["train_loss"], "b-", label="Train Loss", alpha=0.6)
    axis1.plot(epochs, history["valid_loss"], "b-", label="Valid Loss", linewidth=2)
    axis1.tick_params(axis="y", labelcolor="tab:blue")
    axis1.legend(loc="upper left")

    axis2 = axis1.twinx()
    axis2.set_ylabel("Accuracy", color="tab:orange")
    axis2.plot(
        epochs,
        history["valid_accuracy"],
        "o-",
        color="tab:orange",
        label="Valid Accuracy",
        markersize=4,
    )
    axis2.tick_params(axis="y", labelcolor="tab:orange")
    axis2.legend(loc="upper right")

    attention_text = "Attention" if use_attention else "LastStep"
    plt.title(f"{model_name.upper()} + {attention_text} — Training History")
    figure.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_prediction: np.ndarray,
    save_path: Path,
    class_names: list = None,
) -> None:
    """
    绘制归一化混淆矩阵热力图

    每个单元格显示样本数（上方）和占比（下方）。
    颜色越深表示占比越高。

    Args:
        y_true: 真实标签数组
        y_prediction: 预测标签数组
        save_path: 图表保存路径
        class_names: 类别名称列表
    """
    if class_names is None:
        class_names = ["负面", "正面"]

    confusion = confusion_matrix(y_true, y_prediction)
    confusion_normalized = confusion.astype("float") / confusion.sum(
        axis=1, keepdims=True
    )

    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(confusion_normalized, cmap="Blues", vmin=0, vmax=1)

    axis.set_xticks([0, 1])
    axis.set_xticklabels(class_names)
    axis.set_yticks([0, 1])
    axis.set_yticklabels(class_names)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title("Confusion Matrix")

    for row in range(2):
        for col in range(2):
            count = confusion[row, col]
            ratio = confusion_normalized[row, col]
            text = f"{count}\n({ratio:.1%})"
            text_color = "white" if confusion_normalized[row, col] > 0.5 else "black"
            axis.text(
                col, row, text, ha="center", va="center", color=text_color, fontsize=14
            )

    plt.colorbar(image, ax=axis)
    figure.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_roc_curve(
    y_true: np.ndarray,
    y_probability: np.ndarray,
    auc_score: float,
    save_path: Path,
) -> None:
    """
    绘制 ROC 曲线

    Args:
        y_true: 真实标签数组
        y_probability: 预测概率数组
        auc_score: AUC 值
        save_path: 图表保存路径
    """
    false_positive_rate, true_positive_rate, _ = roc_curve(y_true, y_probability)

    figure, axis = plt.subplots(figsize=(6, 5))
    axis.plot(
        false_positive_rate,
        true_positive_rate,
        "b-",
        linewidth=2,
        label=f"ROC (AUC = {auc_score:.3f})",
    )
    axis.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random Guess")
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.set_title("ROC Curve")
    axis.legend(loc="lower right")
    axis.set_xlim([-0.02, 1.02])
    axis.set_ylim([-0.02, 1.02])
    figure.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
