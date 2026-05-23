"""
早停机制模块

提供 EarlyStopping 类，基于验证集损失监控训练的收敛状态，
在验证损失连续多轮未改善时自动触发停止信号。
"""

from typing import Dict, Optional


class EarlyStopping:
    """
    早停机制

    监控验证集损失，当连续 patience 轮验证损失的改善幅度
    小于 min_delta 时，触发停止信号。

    Args:
        patience: 容忍轮数，即验证损失可连续不改善的最大 epoch 数
        min_delta: 最小改善阈值，改善幅度小于此值视为未改善
    """

    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score: Optional[float] = None
        self.should_stop = False

    def __call__(self, validation_loss: float) -> bool:
        """
        检查并更新早停状态

        Args:
            validation_loss: 当前 epoch 的验证集损失

        Returns:
            True 表示应该停止训练，False 表示继续训练
        """
        if self.best_score is None:
            # 第一轮，记录基准分数
            self.best_score = validation_loss
        elif validation_loss > self.best_score - self.min_delta:
            # 验证损失未显著改善
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            # 验证损失显著改善，重置计数器
            self.best_score = validation_loss
            self.counter = 0
        return self.should_stop

    def state_dict(self) -> Dict:
        """
        导出早停状态（用于 checkpoint 保存）

        Returns:
            包含 counter、best_score、should_stop 的字典
        """
        return {
            "counter": self.counter,
            "best_score": self.best_score,
            "should_stop": self.should_stop,
        }

    def load_state_dict(self, state: Dict) -> None:
        """
        加载早停状态（用于 checkpoint 恢复）

        Args:
            state: state_dict() 方法输出的字典
        """
        self.counter = state["counter"]
        self.best_score = state["best_score"]
        self.should_stop = state["should_stop"]
