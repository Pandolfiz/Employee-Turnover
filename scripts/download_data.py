"""Download turnover.csv from Kaggle into data/raw/.

Requires a Kaggle API token: https://www.kaggle.com/settings -> "Create New Token",
then save kaggle.json to ~/.kaggle/kaggle.json (Windows: C:\\Users\\<you>\\.kaggle\\kaggle.json).

Usage:
    python scripts/download_data.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from turnover_survival import config  # noqa: E402


def main() -> None:
    if config.RAW_FILE.exists():
        print(f"Already present: {config.RAW_FILE}")
        return
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        sys.exit("Install the Kaggle client first: pip install kaggle")

    api = KaggleApi()
    api.authenticate()
    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    api.dataset_download_files(config.KAGGLE_DATASET, path=str(config.DATA_RAW), unzip=True)

    if not config.RAW_FILE.exists():
        found = list(config.DATA_RAW.glob("*.csv"))
        sys.exit(f"Download finished but turnover.csv not found. CSVs in data/raw: {found}")
    print(f"Saved to {config.RAW_FILE}")


if __name__ == "__main__":
    main()
