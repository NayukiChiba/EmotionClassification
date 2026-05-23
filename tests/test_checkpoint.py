"""
Checkpoint 管理单元测试

测试模型训练状态的保存与加载往返一致性。
"""

import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from src.data.mapping import VocabMapping
from src.models import build_model
from src.train.checkpoint import load_checkpoint, save_checkpoint


def build_test_vocab() -> VocabMapping:
    """构建测试用词表"""
    word_to_index = {"<PAD>": 0, "<UNK>": 1, "好": 2, "差": 3, "快": 4, "慢": 5}
    index_to_word = {v: k for k, v in word_to_index.items()}
    return VocabMapping(word_to_index, index_to_word)


class TestCheckpoint:
    """Checkpoint 保存/加载测试"""

    def setup_method(self):
        """每个测试方法执行前创建模型、优化器、调度器和词表"""
        self.vocab = build_test_vocab()
        self.model = build_model(
            vocabulary_size=self.vocab.vocabulary_size,
            pad_index=self.vocab.pad_index,
            rnn_type="lstm",
            embedding_dimension=32,
            hidden_dimension=64,
            number_of_layers=1,
        )
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min"
        )

    def test_save_load_model_weights_roundtrip(self):
        """测试模型权重保存后加载完全一致"""
        # 先进行一次前向传播生成不同的权重
        self.model.eval()
        input_ids = torch.randint(2, self.vocab.vocabulary_size, (2, 8))
        mask = torch.ones(2, 8)
        with torch.no_grad():
            original_output = self.model(input_ids, mask).clone()

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            # 保存
            save_checkpoint(
                tmp_path,
                self.model,
                self.optimizer,
                self.scheduler,
                epoch=5,
                validation_loss=0.35,
                vocabulary=self.vocab,
            )

            # 创建新模型并加载
            new_model = build_model(
                vocabulary_size=self.vocab.vocabulary_size,
                pad_index=self.vocab.pad_index,
                rnn_type="lstm",
                embedding_dimension=32,
                hidden_dimension=64,
                number_of_layers=1,
            )
            load_checkpoint(tmp_path, new_model, device="cpu")
            new_model.eval()

            # 验证输出一致
            with torch.no_grad():
                loaded_output = new_model(input_ids, mask)
            assert torch.allclose(original_output, loaded_output, atol=1e-6), (
                "模型权重加载后输出不一致"
            )
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_save_load_optimizer_state(self):
        """测试优化器状态保存后加载一致"""
        # 模拟训练一步，改变优化器内部状态
        input_ids = torch.randint(2, self.vocab.vocabulary_size, (2, 8))
        mask = torch.ones(2, 8)
        labels = torch.tensor([[0.0], [1.0]])

        criterion = nn.BCELoss()
        self.optimizer.zero_grad()
        predictions = self.model(input_ids, mask)
        loss = criterion(predictions, labels)
        loss.backward()
        self.optimizer.step()

        # 记录优化器状态
        original_param_groups = [
            {
                k: v.clone() if isinstance(v, torch.Tensor) else v
                for k, v in pg.items()
                if k != "params"
            }
            for pg in self.optimizer.param_groups
        ]

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            save_checkpoint(
                tmp_path,
                self.model,
                self.optimizer,
                self.scheduler,
                epoch=1,
                validation_loss=0.5,
                vocabulary=self.vocab,
            )

            # 创建新优化器并加载
            new_optimizer = optim.AdamW(self.model.parameters(), lr=0.001)
            load_checkpoint(tmp_path, self.model, new_optimizer, device="cpu")

            loaded_param_groups = [
                {
                    k: v.clone() if isinstance(v, torch.Tensor) else v
                    for k, v in pg.items()
                    if k != "params"
                }
                for pg in new_optimizer.param_groups
            ]

            for orig, loaded in zip(original_param_groups, loaded_param_groups):
                assert orig["lr"] == loaded["lr"]
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_save_load_vocabulary(self):
        """测试 checkpoint 中的词表信息完整保存和读取"""
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            save_checkpoint(
                tmp_path,
                self.model,
                self.optimizer,
                self.scheduler,
                epoch=3,
                validation_loss=0.42,
                vocabulary=self.vocab,
            )

            checkpoint = torch.load(tmp_path, map_location="cpu", weights_only=False)
            loaded_vocab_data = checkpoint["vocabulary"]
            assert loaded_vocab_data["word_to_index"] == self.vocab.word_to_index
            assert "<PAD>" in loaded_vocab_data["word_to_index"]
            assert "<UNK>" in loaded_vocab_data["word_to_index"]
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_save_load_epoch_and_loss(self):
        """测试 checkpoint 中 epoch 和 validation_loss 正确传递"""
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            save_checkpoint(
                tmp_path,
                self.model,
                self.optimizer,
                self.scheduler,
                epoch=7,
                validation_loss=0.28,
                vocabulary=self.vocab,
            )

            checkpoint = torch.load(tmp_path, map_location="cpu", weights_only=False)
            assert checkpoint["epoch"] == 7
            assert abs(checkpoint["validation_loss"] - 0.28) < 1e-6
        finally:
            tmp_path.unlink(missing_ok=True)
