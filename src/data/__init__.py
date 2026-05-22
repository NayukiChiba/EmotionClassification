"""
数据模块

统一导出数据处理相关的所有组件：
- DataProcessor: 预处理管线
- VocabMapping: 词表映射管理
- SentimentDataset: PyTorch Dataset 封装
- create_data_loaders: DataLoader 工厂函数
"""

from src.data.dataloader import SentimentDataset, create_data_loaders
from src.data.mapping import VocabMapping
from src.data.process import DataProcessor

__all__ = [
    "DataProcessor",
    "VocabMapping",
    "SentimentDataset",
    "create_data_loaders",
]
