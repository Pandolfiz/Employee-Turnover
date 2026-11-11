"""Project-wide constants: paths, column roles and modelling defaults."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

RAW_FILE = DATA_RAW / "turnover.csv"
KAGGLE_DATASET = "davinwijaya/employee-turnover"

# Survival target
DURATION_COL = "stag"  # tenure in months ("стаж")
EVENT_COL = "event"  # 1 = employee left, 0 = still employed (right-censored)

CATEGORICAL_COLS = [
    "gender",
    "industry",
    "profession",
    "traffic",  # recruitment channel
    "coach",  # had a coach/mentor during probation
    "head_gender",  # supervisor's gender
    "greywage",  # "white" (fully declared) vs "grey" (partly off-books) salary
    "way",  # commute mode
]

NUMERIC_COLS = [
    "age",
    "extraversion",  # Big Five-style scales (16PF test, 1-10)
    "independ",
    "selfcontrol",
    "anxiety",
    "novator",
]

# Categories with fewer than this many rows are lumped into "other"
MIN_CATEGORY_FREQ = 20

RANDOM_STATE = 42
TEST_SIZE = 0.25
