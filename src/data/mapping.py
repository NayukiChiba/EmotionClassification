"""
词表映射模块

提供 VocabMapping 类，用于管理文本到索引的双向映射，
支持序列化（JSON）和反序列化，便于 checkpoint 存储和推理复用。
"""

import json
from pathlib import Path
from typing import Dict


class VocabMapping:
    """
    词表映射类

    封装 word_to_index 和 index_to_word 两个方向的映射字典，
    提供词表大小、填充索引和未知词索引的便捷属性。

    Attributes:
        word_to_index: 词 → 索引 的映射字典
        index_to_word: 索引 → 词 的映射字典
    """

    def __init__(self, word_to_index: Dict[str, int], index_to_word: Dict[int, str]):
        """
        初始化词表映射

        Args:
            word_to_index: 词到索引的映射字典，必须包含 "<PAD>"=0 和 "<UNK>"=1
            index_to_word: 索引到词的映射字典，与 word_to_index 互为反向映射
        """
        self.word_to_index = word_to_index
        self.index_to_word = index_to_word

    @property
    def vocabulary_size(self) -> int:
        """词表大小（包含 PAD 和 UNK）"""
        return len(self.word_to_index)

    @property
    def pad_index(self) -> int:
        """填充标记 (PAD) 的索引，固定为 0"""
        return 0

    @property
    def unk_index(self) -> int:
        """未知词标记 (UNK) 的索引，固定为 1"""
        return 1

    def to_dict(self) -> Dict[str, Dict]:
        """
        将词表序列化为可 JSON 序列化的字典

        Returns:
            包含 word_to_index 和 index_to_word 的字典
        """
        return {
            "word_to_index": self.word_to_index,
            "index_to_word": {str(k): v for k, v in self.index_to_word.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Dict]) -> "VocabMapping":
        """
        从序列化字典恢复 VocabMapping 实例

        Args:
            data: to_dict() 方法输出的字典

        Returns:
            重建的 VocabMapping 实例
        """
        word_to_index = data["word_to_index"]
        index_to_word = {int(k): v for k, v in data["index_to_word"].items()}
        return cls(word_to_index, index_to_word)

    def save(self, filepath: Path) -> None:
        """
        将词表保存为 JSON 文件

        Args:
            filepath: JSON 文件的保存路径
        """
        with filepath.open("w", encoding="utf-8") as file:
            json.dump(self.to_dict(), file, ensure_ascii=False, indent=4)

    @classmethod
    def load(cls, filepath: Path) -> "VocabMapping":
        """
        从 JSON 文件加载词表

        Args:
            filepath: JSON 文件的路径

        Returns:
            从文件重建的 VocabMapping 实例
        """
        with filepath.open("r", encoding="utf-8") as file:
            data = json.load(file)
            return cls.from_dict(data)
