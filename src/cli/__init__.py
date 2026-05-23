"""
CLI 模块

统一导出命令行接口相关的所有组件：
- build_parser: 参数解析器
- dispatch: 子命令路由
"""

from src.cli.menu import dispatch
from src.cli.parser import build_parser

__all__ = ["build_parser", "dispatch"]
