"""
模型单元测试

测试三种 RNN 模型（SimpleRNN / LSTM / GRU）的前向传播，
覆盖有/无 Attention、有/无双向等组合，验证输出形状和值域。
"""

import pytest
import torch

from src.models import build_model

# 测试用的通用参数
VOCABULARY_SIZE = 100
PAD_INDEX = 0
BATCH_SIZE = 4
SEQUENCE_LENGTH = 16
EMBEDDING_DIMENSION = 32
HIDDEN_DIMENSION = 64


def build_dummy_inputs(
    batch_size: int = BATCH_SIZE, sequence_length: int = SEQUENCE_LENGTH
) -> tuple[torch.Tensor, torch.Tensor]:
    """构造虚拟输入：input_ids 和 mask"""
    input_ids = torch.randint(2, VOCABULARY_SIZE, (batch_size, sequence_length))
    # 随机将部分位置设为 PAD，生成有效 mask
    mask = torch.ones(batch_size, sequence_length)
    mask[input_ids == PAD_INDEX] = 0
    return input_ids, mask


class TestModelForward:
    """模型前向传播测试"""

    @pytest.mark.parametrize("rnn_type", ["rnn", "lstm", "gru"])
    def test_basic_forward_shape(self, rnn_type: str):
        """测试三种模型的基本前向传播输出形状应为 (batch_size, 1)"""
        model = build_model(
            vocabulary_size=VOCABULARY_SIZE,
            pad_index=PAD_INDEX,
            rnn_type=rnn_type,
            embedding_dimension=EMBEDDING_DIMENSION,
            hidden_dimension=HIDDEN_DIMENSION,
            number_of_layers=1,
            dropout=0.3,
            bidirectional=False,
            use_attention=True,
        )
        input_ids, mask = build_dummy_inputs()
        model.eval()
        with torch.no_grad():
            output = model(input_ids, mask)

        assert output.shape == (BATCH_SIZE, 1), (
            f"{rnn_type} 输出形状错误: {output.shape}"
        )

    @pytest.mark.parametrize("rnn_type", ["rnn", "lstm", "gru"])
    def test_output_value_range(self, rnn_type: str):
        """测试模型输出值域在 [0, 1] 范围内（sigmoid 输出）"""
        model = build_model(
            vocabulary_size=VOCABULARY_SIZE,
            pad_index=PAD_INDEX,
            rnn_type=rnn_type,
            embedding_dimension=EMBEDDING_DIMENSION,
            hidden_dimension=HIDDEN_DIMENSION,
            number_of_layers=1,
            dropout=0.3,
            bidirectional=False,
            use_attention=True,
        )
        input_ids, mask = build_dummy_inputs()
        model.eval()
        with torch.no_grad():
            output = model(input_ids, mask)

        assert torch.all(output >= 0.0) and torch.all(output <= 1.0), (
            f"{rnn_type} 输出值域异常: min={output.min().item():.4f}, max={output.max().item():.4f}"
        )

    def test_bidirectional_output_shape(self):
        """测试双向 LSTM 的前向传播输出形状"""
        model = build_model(
            vocabulary_size=VOCABULARY_SIZE,
            pad_index=PAD_INDEX,
            rnn_type="lstm",
            embedding_dimension=EMBEDDING_DIMENSION,
            hidden_dimension=HIDDEN_DIMENSION,
            number_of_layers=1,
            dropout=0.3,
            bidirectional=True,
            use_attention=True,
        )
        input_ids, mask = build_dummy_inputs()
        model.eval()
        with torch.no_grad():
            output = model(input_ids, mask)

        assert output.shape == (BATCH_SIZE, 1)

    def test_no_attention_output_shape(self):
        """测试不使用 Attention（取最后有效时刻）的输出形状"""
        model = build_model(
            vocabulary_size=VOCABULARY_SIZE,
            pad_index=PAD_INDEX,
            rnn_type="lstm",
            embedding_dimension=EMBEDDING_DIMENSION,
            hidden_dimension=HIDDEN_DIMENSION,
            number_of_layers=1,
            dropout=0.3,
            bidirectional=True,
            use_attention=False,
        )
        input_ids, mask = build_dummy_inputs()
        model.eval()
        with torch.no_grad():
            output = model(input_ids, mask)

        assert output.shape == (BATCH_SIZE, 1)
        assert torch.all(output >= 0.0) and torch.all(output <= 1.0)

    def test_multi_layer_gru(self):
        """测试多层 GRU 的前向传播"""
        model = build_model(
            vocabulary_size=VOCABULARY_SIZE,
            pad_index=PAD_INDEX,
            rnn_type="gru",
            embedding_dimension=EMBEDDING_DIMENSION,
            hidden_dimension=HIDDEN_DIMENSION,
            number_of_layers=3,
            dropout=0.3,
            bidirectional=True,
            use_attention=True,
        )
        input_ids, mask = build_dummy_inputs()
        model.eval()
        with torch.no_grad():
            output = model(input_ids, mask)

        assert output.shape == (BATCH_SIZE, 1)

    def test_invalid_model_type_raises(self):
        """测试传入无效模型类型时抛出 ValueError"""
        with pytest.raises(ValueError, match="未知的模型类型"):
            build_model(
                vocabulary_size=VOCABULARY_SIZE,
                pad_index=PAD_INDEX,
                rnn_type="transformer",
            )

    def test_all_input_padded(self):
        """测试全 PAD 输入时模型不崩溃且输出值域正常"""
        model = build_model(
            vocabulary_size=VOCABULARY_SIZE,
            pad_index=PAD_INDEX,
            rnn_type="lstm",
            embedding_dimension=EMBEDDING_DIMENSION,
            hidden_dimension=HIDDEN_DIMENSION,
            number_of_layers=1,
            use_attention=True,
        )
        input_ids = torch.zeros(BATCH_SIZE, SEQUENCE_LENGTH, dtype=torch.long)
        mask = torch.zeros(BATCH_SIZE, SEQUENCE_LENGTH)
        model.eval()
        with torch.no_grad():
            output = model(input_ids, mask)

        assert output.shape == (BATCH_SIZE, 1)
        assert torch.all(output >= 0.0) and torch.all(output <= 1.0)
