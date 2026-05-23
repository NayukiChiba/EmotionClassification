"""
CLI 参数解析模块

提供 build_parser 函数，构造 argparse 解析器，
支持 train / eval / predict 三个子命令。
"""

import argparse


def build_parser() -> argparse.ArgumentParser:
    """
    构建命令行参数解析器

    支持三个子命令：
        train    — 训练模型
        eval     — 评估模型
        predict  — 推理预测

    Returns:
        配置好的 argparse.ArgumentParser 实例
    """
    parser = argparse.ArgumentParser(
        prog="EmotionClassification",
        description="基于 RNN 的 JDfull 情感二分类工具",
    )

    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # -------- train 子命令 --------
    train_parser = subparsers.add_parser("train", help="训练模型")
    train_parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="原始数据文件路径",
    )
    train_parser.add_argument(
        "--model",
        type=str,
        default="lstm",
        choices=["rnn", "lstm", "gru"],
        help="模型类型 (默认: lstm)",
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="训练轮数 (默认: 50)",
    )
    train_parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="批大小 (默认: 64)",
    )
    train_parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="学习率 (默认: 1e-3)",
    )
    train_parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="从 checkpoint 恢复训练的路径",
    )

    # -------- eval 子命令 --------
    eval_parser = subparsers.add_parser("eval", help="评估模型")
    eval_parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="模型 checkpoint 文件路径",
    )
    eval_parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="原始数据文件路径 (可选，默认使用预处理缓存)",
    )
    eval_parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="批大小 (默认: 64)",
    )

    # -------- predict 子命令 --------
    predict_parser = subparsers.add_parser("predict", help="推理预测")
    predict_parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="模型 checkpoint 文件路径",
    )
    predict_parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="要预测的单条文本",
    )
    predict_parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="批量预测的输入 CSV 文件路径",
    )
    predict_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="批量预测的输出 CSV 文件路径",
    )
    predict_parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="分类阈值 (默认: 0.5)",
    )

    return parser
