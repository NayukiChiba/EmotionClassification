"""
EarlyStopping 早停机制单元测试

测试早停触发/不触发逻辑、计数器和状态序列化往返。
"""

from src.train.early_stopping import EarlyStopping


class TestEarlyStopping:
    """EarlyStopping 早停测试"""

    def test_does_not_stop_when_improving(self):
        """测试验证损失持续改善时不触发早停"""
        early_stopping = EarlyStopping(patience=3, min_delta=1e-4)
        losses = [0.5, 0.4, 0.3, 0.2, 0.1]

        for loss in losses:
            should_stop = early_stopping(loss)
            assert not should_stop, f"损失改善时不应触发早停 (loss={loss})"

        assert early_stopping.counter == 0
        assert early_stopping.best_score == 0.1

    def test_triggers_after_patience_no_improvement(self):
        """测试连续 patience 轮未改善后触发早停"""
        early_stopping = EarlyStopping(patience=3, min_delta=1e-4)

        # 第一轮：初始化基准
        assert not early_stopping(0.5)

        # 连续 3 轮不改善（损失高于或接近 best_score）
        assert not early_stopping(0.55)
        assert not early_stopping(0.52)
        # 第 4 轮应触发
        assert early_stopping(0.51)

    def test_min_delta_threshold(self):
        """测试 min_delta 阈值：小幅改善不重置计数器"""
        early_stopping = EarlyStopping(patience=2, min_delta=0.01)

        # 基线
        early_stopping(0.5)
        # 改善幅度小于 min_delta → 视为未改善
        assert not early_stopping(0.495)
        assert early_stopping.counter == 1
        # 改善幅度大于 min_delta → 重置计数器
        assert not early_stopping(0.48)
        assert early_stopping.counter == 0

    def test_counter_resets_on_significant_improvement(self):
        """测试显著改善后计数器重置"""
        early_stopping = EarlyStopping(patience=3, min_delta=1e-4)

        early_stopping(0.5)
        # 恶化
        early_stopping(0.55)  # counter=1
        early_stopping(0.54)  # counter=2
        # 显著改善 → 重置
        early_stopping(0.40)  # counter=0
        assert early_stopping.counter == 0
        assert early_stopping.best_score == 0.40

    def test_state_dict_load_state_dict_roundtrip(self):
        """测试 state_dict / load_state_dict 序列化往返"""
        early_stopping = EarlyStopping(patience=3, min_delta=1e-4)

        # 模拟部分训练
        early_stopping(0.5)
        early_stopping(0.55)  # counter=1

        state = early_stopping.state_dict()
        assert state["counter"] == 1
        assert state["best_score"] == 0.5
        assert state["should_stop"] is False

        # 恢复到新的 EarlyStopping 实例
        restored = EarlyStopping(patience=3, min_delta=1e-4)
        restored.load_state_dict(state)
        assert restored.counter == 1
        assert restored.best_score == 0.5
        assert restored.should_stop is False

    def test_first_call_initializes_best_score(self):
        """测试首次调用使用当前损失初始化 best_score"""
        early_stopping = EarlyStopping(patience=3)
        assert early_stopping.best_score is None

        early_stopping(0.42)
        assert early_stopping.best_score == 0.42
        assert early_stopping.counter == 0
        assert not early_stopping.should_stop

    def test_exact_stop_on_patience_boundary(self):
        """测试在 patience 边界上精确触发"""
        early_stopping = EarlyStopping(patience=2, min_delta=0.0)

        early_stopping(0.5)  # baseline
        # min_delta=0.0 时，validation_loss > best_score 才计入 counter
        early_stopping(0.51)  # counter=1
        should_stop = early_stopping(0.51)  # counter=2, should stop
        assert should_stop
