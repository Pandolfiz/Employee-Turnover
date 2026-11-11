"""Download turnover.csv from Kaggle into data/raw/ using kagglehub.

Public dataset: no Kaggle token is needed in most cases.

Usage:
    python scripts/download_data.py
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from turnover_survival import config  # noqa: E402


def main() -> None:
    if config.RAW_FILE.exists():
        print(f"Already present: {config.RAW_FILE}")
        return
    try:
        import kagglehub
    except ImportError:
        sys.exit("Install kagglehub first: pip install kagglehub")

    path = Path(kagglehub.dataset_download(config.KAGGLE_DATASET))
    src = next(path.rglob("turnover.csv"), None)
    if src is None:
        sys.exit(f"turnover.csv not found in {path}: {list(path.rglob('*'))}")

    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, config.RAW_FILE)
    print(f"Saved to {config.RAW_FILE}")


if __name__ == "__main__":
    main()
