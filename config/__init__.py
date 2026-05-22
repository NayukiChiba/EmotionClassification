"""
配置模块

统一导出项目的路径常量、默认超参数和数据集配置。
"""

from config.default import DataParams, DefaultParams, ModelParams, TrainingParams
from config.paths import (
    BEST_MODEL_PATH,
    CHECKPOINTS_DIR,
    DATASETS_DIR,
    FIGURES_DIR,
    LAST_MODEL_PATH,
    LOGS_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
    PROCESSED_DATASETS_DIR,
    PROCESSED_DATASETS_PATH,
    RAW_DATASETS_DIR,
    RAW_DATASETS_PATH,
    ROOT,
    STOPWORDS_PATH,
)

__all__ = [
    # 路径常量
    "ROOT",
    "DATASETS_DIR",
    "OUTPUTS_DIR",
    "RAW_DATASETS_DIR",
    "PROCESSED_DATASETS_DIR",
    "MODELS_DIR",
    "LOGS_DIR",
    "CHECKPOINTS_DIR",
    "FIGURES_DIR",
    "STOPWORDS_PATH",
    "RAW_DATASETS_PATH",
    "PROCESSED_DATASETS_PATH",
    "BEST_MODEL_PATH",
    "LAST_MODEL_PATH",
    # 默认超参数
    "DataParams",
    "ModelParams",
    "TrainingParams",
    "DefaultParams",
]
