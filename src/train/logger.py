"""
日志记录模块

提供 Logger 类，同时支持文本日志文件和 TensorBoard 两种记录方式。
在训练开始时调用 start()，训练结束后调用 close()。
"""

import logging
import os
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

from config.paths import LOGS_DIR, TENSORBOARD_DIR


class Logger:
    """
    日志记录器

    同时将训练指标写入文本日志文件和 TensorBoard 事件文件。

    Args:
        logDir: 文本日志根目录，默认为 LOGS_DIR
        tensorboardDir: TensorBoard 事件文件目录，默认为 TENSORBOARD_DIR
        experimentName: 实验名称，用于创建子目录，默认为时间戳

    使用方法:
        logger = Logger()
        logger.start()

        for epoch in range(epochs):
            logger.log(
                metrics={"Loss/train": trainLoss, "Loss/valid": validLoss},
                step=epoch,
                message=f"Epoch {epoch}: loss={validLoss:.4f}",
            )

        logger.close()
    """

    def __init__(
        self,
        logDir: str = None,
        tensorboardDir: str = None,
        experimentName: str = None,
    ):
        self.logDir = logDir if logDir is not None else str(LOGS_DIR)
        self.tensorboardDir = (
            tensorboardDir if tensorboardDir is not None else str(TENSORBOARD_DIR)
        )
        self.experimentName = (
            experimentName
            if experimentName is not None
            else datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        self.runDir = os.path.join(self.logDir, self.experimentName)

        self.writer: SummaryWriter = None
        self.fileLogger: logging.Logger = None
        self.isActive = False

    def start(self):
        """
        启动日志记录

        创建运行目录、初始化文本日志文件和 TensorBoard writer。
        """
        os.makedirs(self.runDir, exist_ok=True)

        # 文本日志文件
        self.fileLogger = logging.getLogger(self.experimentName)
        self.fileLogger.setLevel(logging.INFO)
        self.fileLogger.handlers.clear()

        logFile = os.path.join(self.runDir, "train.log")
        fileHandler = logging.FileHandler(logFile, encoding="utf-8")
        fileHandler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        )
        self.fileLogger.addHandler(fileHandler)

        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=self.tensorboardDir)

        self.isActive = True
        self.info(f"日志记录已启动, 运行目录: {self.runDir}")

    def log(self, metrics: dict, step: int, message: str = None):
        """
        同时向 TensorBoard 写入指标和向文本日志写入消息

        Args:
            metrics: 标签到标量值的映射，如 {"Loss/train": 0.35, "Acc/train": 0.85}
            step: 全局步数（通常为 epoch 编号）
            message: 可选的文本日志消息，为 None 时自动从 metrics 生成
        """
        # TensorBoard 标量
        if self.isActive and self.writer is not None:
            for tag, value in metrics.items():
                self.writer.add_scalar(tag, value, step)

        # 文本日志
        if self.isActive and self.fileLogger is not None:
            if message:
                self.fileLogger.info(message)
            else:
                metricStr = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                self.fileLogger.info(f"Step {step}: {metricStr}")

    def info(self, message: str):
        """
        仅向文本日志写入 INFO 级别消息（不写入 TensorBoard）

        Args:
            message: 日志消息
        """
        if self.isActive and self.fileLogger is not None:
            self.fileLogger.info(message)

    def warning(self, message: str):
        """
        仅向文本日志写入 WARNING 级别消息

        Args:
            message: 日志消息
        """
        if self.isActive and self.fileLogger is not None:
            self.fileLogger.warning(message)

    def close(self):
        """
        关闭日志记录

        关闭 TensorBoard writer，清理日志 handler。
        """
        if self.writer is not None:
            self.writer.close()
            self.writer = None

        if self.fileLogger is not None:
            for handler in self.fileLogger.handlers[:]:
                handler.close()
                self.fileLogger.removeHandler(handler)
            self.fileLogger = None

        self.isActive = False
