"""
数据预处理管线模块

提供 DataProcessor 类，封装从原始数据文件到模型就绪数据的完整 7 步预处理流程：
读取 → 分词 → 构建词表 → 编码 → 填充/截断 → 生成掩码 → 数据集划分。

支持管线缓存，避免重复处理。
"""

from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import jieba
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from config.default import DataParams
from config.paths import PROCESSED_DATASETS_DIR, STOPWORDS_PATH
from src.data.mapping import VocabMapping

# 预处理缓存文件路径
PROCESSED_CACHE_PATH = PROCESSED_DATASETS_DIR / "processed.pt"
VOCAB_CACHE_PATH = PROCESSED_DATASETS_DIR / "vocab.json"


class DataProcessor:
    """
    数据预处理管线

    将原始 JDfull 数据集（CSV 或 JSON）转换为 PyTorch 可用的张量格式，
    包含分词、词表构建、编码、填充、掩码生成和数据集划分等步骤。

    所有预处理参数均从 DataParams 类常量中读取，无需外部传参。
    """

    # ---------- Step 1: 读取原始数据 ----------
    def read_raw(self, filepath: Path) -> Tuple[List[str], List[int]]:
        """
        读取原始数据集文件

        支持 CSV 和 JSON 格式。自动过滤空文本和非法标签，
        校验文本列和标签列存在且非空。

        Args:
            filepath: 原始数据文件的路径

        Returns:
            texts: 文本列表，每条为一个字符串，形状为 [N]
            labels: 标签列表，每个为 0 或 1，形状为 [N]

        Raises:
            ValueError: 文件格式不支持时抛出
        """
        if filepath.suffix == ".csv":
            dataframe = pd.read_csv(filepath, encoding=DataParams.ENCODING)
        elif filepath.suffix == ".json":
            dataframe = pd.read_json(filepath, encoding=DataParams.ENCODING)
        else:
            raise ValueError(f"不支持的文件格式: {filepath}")

        # 过滤 NaN 和空文本，同时保留文本有效的对应标签
        texts = [
            str(text).strip()
            for text in dataframe[DataParams.TEXT_COLUMN].tolist()
            if pd.notna(text) and str(text).strip()
        ]
        labels = [
            label
            for index, label in enumerate(dataframe[DataParams.LABEL_COLUMN].tolist())
            if pd.notna(dataframe[DataParams.TEXT_COLUMN].iloc[index])
            and str(dataframe[DataParams.TEXT_COLUMN].iloc[index]).strip()
        ]

        # 过滤非法标签
        valid_labels = {DataParams.NEGATIVE_LABEL, DataParams.POSITIVE_LABEL}
        valid_pairs = [
            (text, label)
            for text, label in zip(texts, labels)
            if text and (label in valid_labels)
        ]
        texts = [pair[0] for pair in valid_pairs]
        labels = [pair[1] for pair in valid_pairs]

        print(f"读取数据: {len(texts)} 条")
        print(f"正样本: {sum(labels)}, 负样本: {len(labels) - sum(labels)}")
        return texts, labels

    # ---------- Step 2: 分词 ----------
    def tokenize(
        self,
        texts: List[str],
        labels: List[int] | None = None,
        remove_stopwords: bool | None = None,
    ) -> tuple[List[List[str]], List[int] | None]:
        """
        使用 jieba 分词器对文本进行分词

        可选择是否去除停用词。停用词表从 datasets/stopwords.txt 加载。
        分词后为空（不含有效词）的样本会被过滤，对应的 labels 也会同步过滤。

        Args:
            texts: 原始文本列表，每条为一个字符串，形状为 [N]
            labels: 标签列表，形状为 [N]，若传入则与 texts 同步过滤
            remove_stopwords: 是否移除停用词，None 则使用 config 中的默认值

        Returns:
            (tokenized_texts, filtered_labels):
                tokenized_texts — 分词后的词列表，形状为 [N', L_i]
                filtered_labels — 同步过滤后的标签列表，若未传入 labels 则为 None
        """
        if remove_stopwords is None:
            remove_stopwords = DataParams.REMOVE_STOPWORDS

        # 加载停用词表
        stopwords = set()
        if remove_stopwords and STOPWORDS_PATH.exists():
            with STOPWORDS_PATH.open(mode="r", encoding="utf-8") as file:
                stopwords = set(line.strip() for line in file if line.strip())

        result = []
        filtered_labels = [] if labels is not None else None
        for i, text in enumerate(texts):
            words = jieba.lcut(text)
            words = [word.strip() for word in words if word.strip()]
            if remove_stopwords:
                words = [word for word in words if word not in stopwords]
            if words:
                result.append(words)
                if filtered_labels is not None:
                    filtered_labels.append(labels[i])

        if result:
            lengths = [len(tokens) for tokens in result]
            print(f"分词完成: {len(result)} 条")
            print(
                f"平均长度: {np.mean(lengths):.1f}, 最大长度: {max(lengths)}, 最小长度: {min(lengths)}"
            )
        return result, filtered_labels

    # ---------- Step 3: 构建词表 ----------
    def build_vocabulary(
        self, tokenized_texts: List[List[str]], minimum_frequency: int | None = None
    ) -> VocabMapping:
        """
        基于分词结果构建词表

        统计词频，过滤低于 minimum_frequency 的低频词。
        预留 PAD=0 和 UNK=1，其余词按词频降序排列。

        Args:
            tokenized_texts: 分词后的文本列表，形状为 [N, L_i]
            minimum_frequency: 最小词频阈值，None 则使用 config 中的默认值

        Returns:
            构建好的 VocabMapping 实例
        """
        if minimum_frequency is None:
            minimum_frequency = DataParams.MIN_FREQ

        counter = Counter()
        for tokens in tokenized_texts:
            counter.update(tokens)

        word_to_index = {"<PAD>": 0, "<UNK>": 1}
        index_to_word = {0: "<PAD>", 1: "<UNK>"}

        current_index = 2
        for word, frequency in counter.most_common():
            if frequency >= minimum_frequency:
                word_to_index[word] = current_index
                index_to_word[current_index] = word
                current_index += 1

        print(f"词表大小: {len(word_to_index)} (minimum_frequency={minimum_frequency})")
        return VocabMapping(word_to_index, index_to_word)

    # ---------- Step 4: 文本 → 索引序列 ----------
    def encode(
        self, tokenized_texts: List[List[str]], word_to_index: Dict[str, int]
    ) -> List[List[int]]:
        """
        将分词结果编码为整数索引序列

        对于词表中不存在的词，映射为 UNK (索引 1)。

        Args:
            tokenized_texts: 分词后的文本列表，形状为 [N, L_i]
            word_to_index: 词到索引的映射字典

        Returns:
            编码后的索引序列列表，形状为 [N, L_i]
        """
        sequences = []
        for tokens in tokenized_texts:
            sequence = [word_to_index.get(word, 1) for word in tokens]
            sequences.append(sequence)
        return sequences

    # ---------- Step 5: 填充/截断 ----------
    def pad_or_truncate(
        self, sequences: List[List[int]], max_sequence_length: int | None = None
    ) -> torch.Tensor:
        """
        将不等长的序列统一填充或截断到固定长度

        超长序列截断尾部，不足序列在末尾填充 PAD (0)。

        Args:
            sequences: 编码后的索引序列列表，形状为 [N, L_i]
            max_sequence_length: 目标序列长度，None 则使用 config 中的默认值

        Returns:
            填充后的张量，形状为 (N, max_sequence_length)，dtype=torch.long
        """
        if max_sequence_length is None:
            max_sequence_length = DataParams.MAX_SEQ_LENGTH

        padded = []
        for sequence in sequences:
            if len(sequence) > max_sequence_length:
                # 超长则截断尾部
                padded.append(sequence[:max_sequence_length])
            else:
                # 不足则末尾填充 PAD
                padded.append(sequence + [0] * (max_sequence_length - len(sequence)))
        return torch.tensor(padded, dtype=torch.long)

    # ---------- Step 6: 生成 Attention Mask ----------
    def create_mask(self, padded: torch.Tensor) -> torch.Tensor:
        """
        根据填充后的张量生成 Attention Mask

        有效位置 (非 PAD) 标记为 1，PAD 位置标记为 0。

        Args:
            padded: 填充后的张量，形状为 (N, max_sequence_length)

        Returns:
            Attention Mask 张量，形状为 (N, max_sequence_length)，dtype=torch.long
        """
        return (padded != 0).long()

    # ---------- Step 7: 划分数据集 ----------
    def split_data(
        self,
        input_ids: torch.Tensor,
        masks: torch.Tensor,
        labels: List[int],
    ) -> Tuple[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """
        将数据分层划分为训练集、验证集和测试集

        使用 sklearn train_test_split 进行两级划分，
        通过 stratify 参数保证各类别在子集中的比例与原始数据一致，
        固定 random_state 保证划分可复现。

        Args:
            input_ids: 填充后的输入索引张量，形状为 (N, max_sequence_length)
            masks: Attention Mask 张量，形状为 (N, max_sequence_length)
            labels: 标签列表，形状为 [N]

        Returns:
            train_data: (train_input_ids, train_masks, train_labels)，形状均为 (N_train, ...)
            valid_data: (valid_input_ids, valid_masks, valid_labels)，形状均为 (N_valid, ...)
            test_data: (test_input_ids, test_masks, test_labels)，形状均为 (N_test, ...)
        """
        labels_tensor = torch.tensor(labels, dtype=torch.float32)
        temporary_ratio = DataParams.VALID_RATIO + DataParams.TEST_RATIO

        # 第一级划分：训练集 vs 临时集（验证+测试）
        (
            train_ids,
            temporary_ids,
            train_masks,
            temporary_masks,
            train_labels,
            temporary_labels,
        ) = train_test_split(
            input_ids,
            masks,
            labels_tensor,
            test_size=temporary_ratio,
            stratify=labels_tensor.numpy(),
            random_state=42,
        )

        # 第二级划分：验证集 vs 测试集
        valid_ratio_in_temporary = DataParams.VALID_RATIO / temporary_ratio
        (
            valid_ids,
            test_ids,
            valid_masks,
            test_masks,
            valid_labels,
            test_labels,
        ) = train_test_split(
            temporary_ids,
            temporary_masks,
            temporary_labels,
            test_size=1 - valid_ratio_in_temporary,
            stratify=temporary_labels.numpy(),
            random_state=42,
        )

        print(
            f"划分完成: train={len(train_ids)}, valid={len(valid_ids)}, test={len(test_ids)}"
        )
        return (
            (train_ids, train_masks, train_labels),
            (valid_ids, valid_masks, valid_labels),
            (test_ids, test_masks, test_labels),
        )

    # ---------- 完整管线 ----------
    def run_pipeline(
        self, raw_filepath: Path, force_reprocess: bool = False
    ) -> Tuple[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        VocabMapping,
    ]:
        """
        执行完整的数据预处理管线

        先检查缓存（processed.pt + vocab.json），如果存在且 force_reprocess=False
        则直接加载缓存，否则重新执行全部 7 步处理流程并保存缓存。

        Args:
            raw_filepath: 原始数据文件的路径
            force_reprocess: 是否强制重新处理（忽略缓存）

        Returns:
            train_data: (train_input_ids, train_masks, train_labels)
                输入张量形状 (N_train, max_sequence_length)
            valid_data: (valid_input_ids, valid_masks, valid_labels)
                输入张量形状 (N_valid, max_sequence_length)
            test_data: (test_input_ids, test_masks, test_labels)
                输入张量形状 (N_test, max_sequence_length)
            vocabulary: 构建好的词表映射
        """
        # 尝试从缓存加载
        if (
            not force_reprocess
            and PROCESSED_CACHE_PATH.exists()
            and VOCAB_CACHE_PATH.exists()
        ):
            print("从缓存加载预处理数据...")
            data = torch.load(
                PROCESSED_CACHE_PATH, map_location="cpu", weights_only=False
            )
            vocabulary = VocabMapping.load(VOCAB_CACHE_PATH)

            train_data = (data["train_ids"], data["train_masks"], data["train_labels"])
            valid_data = (data["valid_ids"], data["valid_masks"], data["valid_labels"])
            test_data = (data["test_ids"], data["test_masks"], data["test_labels"])

            print(
                f"缓存加载完成: train={len(train_data[0])}, valid={len(valid_data[0])}, test={len(test_data[0])}"
            )
            return train_data, valid_data, test_data, vocabulary

        # 执行完整管线
        print("开始执行预处理管线...")

        # Step 1: 读取原始数据
        texts, labels = self.read_raw(raw_filepath)
        # Step 2: 分词（含 labels 同步过滤，去除空样本）
        tokenized, labels = self.tokenize(texts, labels)
        # Step 3: 构建词表
        vocabulary = self.build_vocabulary(tokenized)
        # Step 4: 编码
        encoded = self.encode(tokenized, vocabulary.word_to_index)
        # Step 5: 填充/截断
        padded = self.pad_or_truncate(encoded)
        # Step 6: 生成 Attention Mask
        masks = self.create_mask(padded)
        # Step 7: 划分数据集
        train_data, valid_data, test_data = self.split_data(padded, masks, labels)

        # 保存缓存
        PROCESSED_DATASETS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "train_ids": train_data[0],
                "train_masks": train_data[1],
                "train_labels": train_data[2],
                "valid_ids": valid_data[0],
                "valid_masks": valid_data[1],
                "valid_labels": valid_data[2],
                "test_ids": test_data[0],
                "test_masks": test_data[1],
                "test_labels": test_data[2],
            },
            PROCESSED_CACHE_PATH,
        )
        vocabulary.save(VOCAB_CACHE_PATH)
        print(f"预处理结果已缓存到 {PROCESSED_DATASETS_DIR}")

        return train_data, valid_data, test_data, vocabulary
