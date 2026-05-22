from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def get_dir(dir: Path) -> Path:
    if not dir.exists():
        dir.mkdir(parents=True)
    return dir


DATASETS_DIR = get_dir(ROOT / "datasets")
OUTPUTS_DIR = get_dir(ROOT / "outputs")

RAW_DATASETS_DIR = get_dir(DATASETS_DIR / "raw")
PROCESSED_DATASETS_DIR = get_dir(DATASETS_DIR / "processed")

MODELS_DIR = get_dir(OUTPUTS_DIR / "models")
LOGS_DIR = get_dir(OUTPUTS_DIR / "logs")
CHECKPOINTS_DIR = get_dir(OUTPUTS_DIR / "checkpoints")
FIGURES_DIR = get_dir(OUTPUTS_DIR / "figures")

STOPWORDS_PATH = DATASETS_DIR / "stopwords.txt"
RAW_DATASETS_PATH = RAW_DATASETS_DIR / "online_shopping_10_cats.csv"
PROCESSED_DATASETS_PATH = PROCESSED_DATASETS_DIR / "processed_datasets.csv"

BEST_MODEL_PATH = MODELS_DIR / "best_model.pth"
LAST_MODEL_PATH = MODELS_DIR / "last_model.pth"

if __name__ == "__main__":
    print(ROOT)
