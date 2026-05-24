# EmotionClassification

基于 PyTorch RNN 系列模型的中文情感二分类项目，当前面向
JDfull / 电商评论类数据集，支持训练、评估和推理三个命令行流程。

项目目标是保持代码清晰、流程显式、便于实验复现。默认支持
`RNN`、`LSTM`、`GRU` 三类模型，并可通过配置启用双向结构和
Attention 机制。

## 功能特性

- 中文文本预处理：读取数据、jieba 分词、停用词过滤、词表构建
- 数据集划分：训练集、验证集、测试集按固定随机种子分层划分
- 模型训练：支持 RNN / LSTM / GRU，包含学习率调度和早停
- 模型评估：输出 Accuracy、Precision、Recall、F1、AUC 等指标
- 可视化输出：训练曲线、混淆矩阵、ROC 曲线
- 推理预测：支持单条文本和 CSV 文件批量预测
- 工程工具：使用 Ruff 做格式化和基础检查，使用 Pytest 做测试

## 项目结构

```text
EmotionClassification/
├── config/                 # 路径和默认参数配置
│   ├── default.py
│   └── paths.py
├── datasets/               # 数据集目录
│   ├── raw/                # 原始数据
│   └── processed/          # 预处理缓存
├── notebooks/              # 实验 notebook
├── outputs/                # 模型、日志和图表输出
│   ├── checkpoints/
│   ├── figures/
│   ├── logs/
│   └── models/
├── src/
│   ├── cli/                # 命令行入口
│   ├── data/               # 数据预处理和 DataLoader
│   ├── evaluate/           # 评估指标和可视化
│   ├── inference/          # 推理预测
│   ├── models/             # 模型定义
│   └── train/              # 训练器、优化器、checkpoint
├── tests/                  # 单元测试
├── main.py                 # CLI 主入口
├── pyproject.toml
└── README.md
```

## 环境要求

- Python >= 3.11
- uv（推荐）或 pip
- 如需 GPU 训练，请确认本机 CUDA 与 PyTorch 版本匹配

## 安装依赖

推荐使用 uv：

```bash
uv sync
```

安装开发依赖后，可运行测试和代码检查：

```bash
uv sync --group dev
```

如果使用 pip：

```bash
pip install -e ".[dev]"
```

## 数据准备

默认原始数据路径为：

```text
datasets/raw/online_shopping_10_cats.csv
```

CSV 至少需要包含以下两列：

| 列名 | 说明 |
| --- | --- |
| `text` | 评论文本 |
| `label` | 情感标签，`0` 表示负面，`1` 表示正面 |

相关配置位于 `config/default.py`：

- `DataParams.TEXT_COLUMN`
- `DataParams.LABEL_COLUMN`
- `DataParams.MAX_SEQ_LENGTH`
- `DataParams.MIN_FREQ`
- `DataParams.TRAIN_RATIO`
- `DataParams.VALID_RATIO`
- `DataParams.TEST_RATIO`

预处理结果会缓存到：

```text
datasets/processed/
```

如果修改了原始数据或预处理参数，建议删除该目录下缓存后重新训练。

## 使用方法

查看命令帮助：

```bash
python main.py --help
```

### 训练模型

```bash
python main.py train --data datasets/raw/online_shopping_10_cats.csv --model lstm --epochs 20
```

可选模型：

- `rnn`
- `lstm`
- `gru`

常用参数：

```bash
python main.py train \
  --data datasets/raw/online_shopping_10_cats.csv \
  --model lstm \
  --epochs 20 \
  --batch_size 64 \
  --learning_rate 0.01
```

训练输出默认保存到：

```text
outputs/models/
outputs/checkpoints/
outputs/figures/
```

### 评估模型

```bash
python main.py eval --checkpoint outputs/models/best_model.pth
```

指定原始数据文件：

```bash
python main.py eval \
  --checkpoint outputs/models/best_model.pth \
  --data datasets/raw/online_shopping_10_cats.csv
```

评估会输出指标报告，并在 `outputs/figures/` 中生成图表。

### 单条文本预测

```bash
python main.py predict --checkpoint outputs/models/best_model.pth --text "物流很快，包装也很好"
```

输出示例：

```text
文本: 物流很快，包装也很好
结果: 正面 (置信度: 0.9234)
```

### 批量预测

输入 CSV 需要包含 `text` 列：

```bash
python main.py predict \
  --checkpoint outputs/models/best_model.pth \
  --input datasets/raw/predict_samples.csv \
  --output outputs/predictions.csv
```

输出文件会追加：

- `predicted_label`
- `confidence`
- `label_name`

## 配置说明

主要配置文件：

- `config/default.py`：数据、模型、训练默认参数
- `config/paths.py`：项目路径和输出目录

常用模型参数：

- `ModelParams.EMBEDDING_DIM`
- `ModelParams.HIDDEN_DIM`
- `ModelParams.NUM_LAYERS`
- `ModelParams.DROPOUT`
- `ModelParams.BIDIRECTIONAL`
- `ModelParams.USE_ATTENTION`

常用训练参数：

- `TrainingParams.BATCH_SIZE`
- `TrainingParams.LEARNING_RATE`
- `TrainingParams.EPOCHS`
- `TrainingParams.GRAD_CLIP`
- `TrainingParams.EARLY_STOP_PATIENCE`

## 开发命令

运行测试：

```bash
pytest
```

格式化代码：

```bash
ruff format .
```

检查代码：

```bash
ruff check .
```

自动修复可修复问题：

```bash
ruff check . --fix
```

## 输出文件

典型输出包括：

```text
outputs/models/best_model.pth
outputs/models/last_model.pth
outputs/checkpoints/
outputs/figures/training_curve.png
outputs/figures/confusion_matrix.png
outputs/figures/roc_curve.png
```

## 注意事项

- 当前任务是二分类，标签默认约定为 `0=负面`、`1=正面`。
- 推理时会从 checkpoint 中恢复词表，因此预测应使用训练得到的模型文件。
- 预处理缓存默认会复用旧结果，修改数据或参数后请清理
  `datasets/processed/`。
- 如果使用 GPU，请先确认 `torch.cuda.is_available()` 为 `True`。
