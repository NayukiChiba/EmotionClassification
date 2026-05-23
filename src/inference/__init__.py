"""
推理模块

统一导出推理相关的所有组件：
- Predictor: 情感分类推理器
"""

from src.inference.predictor import Predictor

__all__ = ["Predictor"]
