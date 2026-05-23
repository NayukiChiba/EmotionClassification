"""
CLI 子命令路由模块

提供 train / eval / predict 三个子命令的实际执行逻辑。
"""

import argparse
from pathlib import Path

import torch.nn as nn

from config.paths import BEST_MODEL_PATH, RAW_DATASETS_PATH
from src.data.dataloader import create_data_loaders
from src.data.process import DataProcessor
from src.evaluate.evaluator import Evaluator
from src.evaluate.visualize import plot_training_history
from src.inference.predictor import Predictor
from src.models import build_model
from src.train.checkpoint import load_checkpoint
from src.train.early_stopping import EarlyStopping
from src.train.optimizer import build_optimizer
from src.train.scheduler import build_scheduler
from src.train.trainer import Trainer
from src.train.utils import get_device, set_seed


def run_train(args: argparse.Namespace):
    """
    执行模型训练

    流程: 数据加载 → 模型构建 → 训练器初始化 → 训练循环

    Args:
        args: 解析后的命令行参数
    """
    # 初始化
    set_seed(42)
    device = get_device("cuda")
    print(f"设备: {device}")

    # 数据预处理
    data_processor = DataProcessor()
    data_path = Path(args.data)

    train_data, valid_data, test_data, vocabulary = data_processor.run_pipeline(
        data_path
    )

    train_loader, valid_loader, test_loader = create_data_loaders(
        train_data,
        valid_data,
        test_data,
        batch_size=args.batch_size,
        num_workers=0,
    )

    # 模型构建
    model = build_model(
        vocabulary_size=vocabulary.vocabulary_size,
        pad_index=vocabulary.pad_index,
        rnn_type=args.model,
    ).to(device)

    # 训练组件
    optimizer = build_optimizer(model, learning_rate=args.learning_rate)
    scheduler = build_scheduler(optimizer)
    criterion = nn.BCELoss()
    early_stopping = EarlyStopping()

    # 训练
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        early_stopping=early_stopping,
        vocabulary=vocabulary,
        device=device,
        epochs=args.epochs,
    )

    history = trainer.train()

    # 绘制训练曲线
    plot_training_history(history, Path("outputs/figures/training_curve.png"))

    # 测试集评估
    print("\n开始测试集评估...")
    best_checkpoint = load_checkpoint(BEST_MODEL_PATH, model, device=str(device))
    evaluator = Evaluator(model, test_loader, device)
    evaluator.evaluate()


def run_eval(args: argparse.Namespace):
    """
    执行模型评估

    流程: 加载 checkpoint → 加载/缓存数据 → 评估

    Args:
        args: 解析后的命令行参数
    """
    device = get_device("cuda")
    print(f"设备: {device}")

    # 数据加载
    data_processor = DataProcessor()

    if args.data:
        data_path = Path(args.data)
    else:
        data_path = RAW_DATASETS_PATH

    train_data, valid_data, test_data, vocabulary = data_processor.run_pipeline(
        data_path
    )
    _, _, test_loader = create_data_loaders(
        train_data,
        valid_data,
        test_data,
        batch_size=args.batch_size,
        num_workers=0,
    )

    # 加载模型
    model = build_model(
        vocabulary_size=vocabulary.vocabulary_size,
        pad_index=vocabulary.pad_index,
    ).to(device)

    checkpoint_path = Path(args.checkpoint)
    load_checkpoint(checkpoint_path, model, device=str(device))

    # 评估
    evaluator = Evaluator(model, test_loader, device)
    evaluator.evaluate()


def run_predict(args: argparse.Namespace):
    """
    执行推理预测

    流程: 加载 checkpoint → 单条/批量/文件推理

    Args:
        args: 解析后的命令行参数
    """
    device = get_device("cuda")
    checkpoint_path = Path(args.checkpoint)

    predictor = Predictor.from_checkpoint(checkpoint_path, device)

    if args.text:
        # 单条预测
        label, confidence, label_name = predictor.predict(args.text)
        print(f"文本: {args.text}")
        print(f"结果: {label_name} (置信度: {confidence:.4f})")

    elif args.input and args.output:
        # 文件批量推理
        predictor.predict_file(args.input, args.output)

    else:
        print("请指定 --text (单条预测) 或 --input/--output (文件批量推理)")


def dispatch(args: argparse.Namespace):
    """
    根据子命令分发到对应的处理函数

    Args:
        args: 解析后的命令行参数
    """
    if args.command == "train":
        run_train(args)
    elif args.command == "eval":
        run_eval(args)
    elif args.command == "predict":
        run_predict(args)
    else:
        print("请指定子命令: train / eval / predict")
        print("使用 --help 查看详细帮助")
