from typing import Literal

import torch


class DefaultParams:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class DataParams:
    # 最长序列长度
    MAX_SEQ_LENGTH = 128

    # 最小频率
    MIN_FREQ = 3

    # 是否移除停用词
    REMOVE_STOPWORDS = True

    # 训练集, 验证集, 测试集的比例
    TRAIN_RATIO = 0.7
    VALID_RATIO = 0.2
    TEST_RATIO = 0.1


class ModelParams:
    # 模型类型
    MODEL_TYPE: Literal["LSTM", "RNN", "GRU"] = "LSTM"

    # 嵌入维度
    EMBEDDING_DIM = 300

    # 隐藏层维度
    HIDDEN_DIM = 256

    # 层数数量
    NUM_LAYERS = 2

    # Dropout率
    DROPOUT = 0.5

    # 是否使用双向RNN
    BIDIRECTIONAL = True

    # 使用Attention机制
    USE_ATTENTION = True


class TrainingParams:
    # Batch大小
    BATCH_SIZE = 64

    # 学习率
    LEARNING_RATE = 1e-2

    # 权重衰减
    WEIGHT_DECAY = 1e-5

    # 训练轮数
    EPOCHS = 50

    # 梯度裁剪
    GRAD_CLIP = 1.0

    # 学习率调整参数
    # 当验证集性能不再提升时，学习率乘以LR_REDUCE_FACTOR
    LR_REDUCE_FACTOR = 0.5
    # 当验证集性能不再提升时，等待LR_REDUCE_PATIENCE轮后才调整学习率
    LR_REDUCE_PATIENCE = 3

    # 早停参数
    # 当验证集性能不再提升时，等待EARLY_STOP_PATIENCE轮后停止训练
    EARLY_STOP_PATIENCE = 5
    # 当验证集性能提升小于EARLY_STOP_MIN_DELTA时，认为性能不再提升
    EARLY_STOP_MIN_DELTA = 1e-4
