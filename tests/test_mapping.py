"""
VocabMapping 词表映射单元测试

测试词表构建、属性访问、序列化往返等核心功能。
"""

import tempfile
from pathlib import Path

from src.data.mapping import VocabMapping


class TestVocabMapping:
    """VocabMapping 词表映射测试"""

    def build_sample_vocab(self) -> VocabMapping:
        """构建一个简单的示例词表供测试使用"""
        word_to_index = {
            "<PAD>": 0,
            "<UNK>": 1,
            "物流": 2,
            "很快": 3,
            "质量": 4,
            "不错": 5,
        }
        index_to_word = {v: k for k, v in word_to_index.items()}
        return VocabMapping(word_to_index, index_to_word)

    def test_vocabulary_size(self):
        """测试词表大小属性"""
        vocab = self.build_sample_vocab()
        assert vocab.vocabulary_size == 6

    def test_pad_index(self):
        """测试 PAD 索引始终为 0"""
        vocab = self.build_sample_vocab()
        assert vocab.pad_index == 0

    def test_unk_index(self):
        """测试 UNK 索引始终为 1"""
        vocab = self.build_sample_vocab()
        assert vocab.unk_index == 1

    def test_to_dict_from_dict_roundtrip(self):
        """测试 to_dict → from_dict 序列化往返"""
        vocab = self.build_sample_vocab()
        data = vocab.to_dict()
        restored = VocabMapping.from_dict(data)

        assert restored.vocabulary_size == vocab.vocabulary_size
        assert restored.word_to_index == vocab.word_to_index
        # index_to_word 经过 from_dict 后 int key 应一致
        assert restored.index_to_word == vocab.index_to_word

    def test_save_load_roundtrip(self):
        """测试 save → load JSON 持久化往返"""
        vocab = self.build_sample_vocab()

        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w", encoding="utf-8"
        ) as tmp:
            tmp_path = Path(tmp.name)

        try:
            vocab.save(tmp_path)
            loaded = VocabMapping.load(tmp_path)

            assert loaded.vocabulary_size == vocab.vocabulary_size
            assert loaded.word_to_index == vocab.word_to_index
            assert loaded.index_to_word == vocab.index_to_word
            assert loaded.pad_index == 0
            assert loaded.unk_index == 1
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_pad_unk_present(self):
        """测试词表必须包含 PAD 和 UNK"""
        vocab = self.build_sample_vocab()
        assert "<PAD>" in vocab.word_to_index
        assert "<UNK>" in vocab.word_to_index
        assert vocab.word_to_index["<PAD>"] == 0
        assert vocab.word_to_index["<UNK>"] == 1
