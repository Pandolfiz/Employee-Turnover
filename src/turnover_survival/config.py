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

# ~41% of rows share an identical 5-score personality profile with another row (groups of
# up to 5), so they are likely the same person / same test result. Rows sharing a profile get
# the same GROUP_COL id: kept on the same side of every split and used for cluster-robust SEs.
PERSONALITY_COLS = ["extraversion", "independ", "selfcontrol", "anxiety", "novator"]
GROUP_COL = "profile_id"

# Categories with fewer than this many rows are lumped into "other"
MIN_CATEGORY_FREQ = 20

RANDOM_STATE = 42
TEST_SIZE = 0.25
