"""
情感分类 CLI 主入口

支持三个子命令：
    python main.py train  --data <path> [OPTIONS]
    python main.py eval   --checkpoint <path> [OPTIONS]
    python main.py predict --checkpoint <path> --text "..."

用法示例：
    python main.py train --data datasets/raw/online_shopping_10_cats.csv --model lstm --epochs 20
    python main.py eval --checkpoint outputs/models/best_model.pth
    python main.py predict --checkpoint outputs/models/best_model.pth --text "物流很快"
"""

import sys
from pathlib import Path

# 将项目根目录加入 Python 路径，确保模块导入正确
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.cli.menu import dispatch
from src.cli.parser import build_parser


def main():
    """主入口函数"""
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    dispatch(args)


if __name__ == "__main__":
    main()
