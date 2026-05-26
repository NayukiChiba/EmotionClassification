"""
推理模块

提供 Predictor 类，支持从 checkpoint 恢复模型并执行单条/批量/文件级推理。
"""

from pathlib import Path
from typing import List, Tuple

import jieba
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from src.data.mapping import VocabMapping
from src.models import build_model


class Predictor:
    """
    情感分类推理器

    加载训练好的模型和词表，对新文本进行情感预测。

    Args:
        model: 已训练的 PyTorch 模型
        vocabulary: 词表映射实例
        device: 推理设备
        max_sequence_length: 最大序列长度
        threshold: 分类阈值（概率 >= threshold 判为正类）
    """

    def __init__(
        self,
        model: nn.Module,
        vocabulary: VocabMapping,
        device: torch.device = torch.device("cpu"),
        max_sequence_length: int = 128,
        threshold: float = 0.5,
    ):
        self.model = model
        self.vocabulary = vocabulary
        self.device = device
        self.max_sequence_length = max_sequence_length
        self.threshold = threshold

    @classmethod
    def from_checkpoint(
        cls, checkpoint_path: Path, device: torch.device = None
    ) -> "Predictor":
        """
        从 checkpoint 文件构建推理器

        自动从 checkpoint 中恢复模型权重和词表，
        使用 build_model 重建模型结构。

        Args:
            checkpoint_path: checkpoint 文件路径
            device: 推理设备，None 则使用 CPU

        Returns:
            可直接用于推理的 Predictor 实例
        """
        if device is None:
            device = torch.device("cpu")

        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )

        # 恢复词表
        vocabulary = VocabMapping.from_dict(checkpoint["vocabulary"])

        # 重建模型并加载权重
        model = build_model(
            vocabulary_size=vocabulary.vocabulary_size, pad_index=vocabulary.pad_index
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        return cls(model, vocabulary, device)

    def _preprocess(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对单条文本执行预处理（分词 -> 编码 -> 填充 -> 生成掩码）

        Args:
            text: 原始文本字符串

        Returns:
            input_ids: 填充后的索引张量, 形状 (1, max_sequence_length)
            mask: Attention Mask, 形状 (1, max_sequence_length)
        """
        # 分词
        words = jieba.lcut(text)
        words = [word.strip() for word in words if word.strip()]

        # 编码（OOV -> UNK）
        sequence = [self.vocabulary.word_to_index.get(word, 1) for word in words]

        # 填充/截断
        if len(sequence) > self.max_sequence_length:
            sequence = sequence[: self.max_sequence_length]
        else:
            sequence = sequence + [0] * (self.max_sequence_length - len(sequence))

        input_ids = torch.tensor([sequence], dtype=torch.long).to(self.device)
        mask = (input_ids != 0).long()

        return input_ids, mask

    @torch.no_grad()
    def predict(self, text: str) -> Tuple[int, float, str]:
        """
        对单条文本进行情感预测

        Args:
            text: 原始文本字符串

        Returns:
            (label, confidence, label_name):
                label: 预测标签 (0=负面, 1=正面)
                confidence: 预测置信度 (值域 [0, 1])
                label_name: 标签名称 ("正面" / "负面")
        """
        input_ids, mask = self._preprocess(text)

        probability = self.model(input_ids, mask).item()

        label = 1 if probability >= self.threshold else 0
        label_name = "正面" if label == 1 else "负面"

        return label, probability, label_name

    @torch.no_grad()
    def predict_batch(self, texts: List[str]) -> List[Tuple[int, float, str]]:
        """
        批量预测多条文本的情感

        Args:
            texts: 原始文本列表

        Returns:
            结果列表，每项为 (label, confidence, label_name)
        """
        results = []
        for text in tqdm(texts, desc="批量推理"):
            results.append(self.predict(text))
        return results

    def predict_file(
        self, input_csv: str, output_csv: str, text_column: str = "text"
    ) -> None:
        """
        对 CSV 文件中的文本列进行批量推理

        读取 CSV -> 逐行推理 -> 追加 pred_label / confidence / label_name -> 写入输出文件

        Args:
            input_csv: 输入 CSV 文件路径
            output_csv: 输出 CSV 文件路径
            text_column: 文本列的列名
        """
        dataframe = pd.read_csv(input_csv)
        texts = dataframe[text_column].astype(str).tolist()

        results = []
        for text in tqdm(texts, desc=f"推理 {input_csv}"):
            label, confidence, label_name = self.predict(text)
            results.append(
                {
                    "predicted_label": label,
                    "confidence": round(confidence, 4),
                    "label_name": label_name,
                }
            )

        result_dataframe = pd.DataFrame(results)
        output_dataframe = pd.concat([dataframe, result_dataframe], axis=1)
        output_dataframe.to_csv(output_csv, index=False, encoding="utf-8-sig")
        print(f"推理结果已保存到 {output_csv}")
