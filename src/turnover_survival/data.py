"""Loading, cleaning and feature engineering for the turnover dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


def load_raw(path: str | Path = config.RAW_FILE) -> pd.DataFrame:
    """Read the raw CSV. The original file is not always UTF-8, so fall back to latin-1."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/download_data.py` or download "
            f"turnover.csv from https://www.kaggle.com/datasets/{config.KAGGLE_DATASET} "
            f"and place it in {config.DATA_RAW}."
        )
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        # The original file was produced in Russia (Windows-1251); e.g. "Finance" is
        # stored with a Cyrillic "с" (byte 0xF1).
        return pd.read_csv(path, encoding="cp1251")


# Cyrillic letters that look identical to Latin ones (typed by mistake in the source file)
_HOMOGLYPHS = str.maketrans("асеорхукмтнві", "aceopxykmthbi")


def clean(df: pd.DataFrame, drop_duplicates: bool = True) -> pd.DataFrame:
    """Normalise text columns, enforce dtypes and drop invalid/duplicate rows."""
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    missing = {config.DURATION_COL, config.EVENT_COL} - set(df.columns)
    if missing:
        raise ValueError(f"Missing survival columns: {sorted(missing)}")

    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype(str).str.strip().str.lower().str.translate(_HOMOGLYPHS)

    df[config.DURATION_COL] = pd.to_numeric(df[config.DURATION_COL], errors="coerce")
    df[config.EVENT_COL] = pd.to_numeric(df[config.EVENT_COL], errors="coerce")
    df = df.dropna(subset=[config.DURATION_COL, config.EVENT_COL])
    df = df[df[config.DURATION_COL] > 0]
    df[config.EVENT_COL] = df[config.EVENT_COL].astype(int)

    if drop_duplicates:
        df = df.drop_duplicates()

    return df.reset_index(drop=True)


def lump_rare_categories(
    df: pd.DataFrame,
    cols: list[str] | None = None,
    min_freq: int = config.MIN_CATEGORY_FREQ,
    other_label: str = "other",
) -> pd.DataFrame:
    """Replace categories with < `min_freq` rows by `other_label` (stabilises Cox estimates)."""
    df = df.copy()
    cols = cols if cols is not None else [c for c in config.CATEGORICAL_COLS if c in df]
    for col in cols:
        counts = df[col].value_counts()
        rare = counts[counts < min_freq].index
        if len(rare):
            df[col] = df[col].where(~df[col].isin(rare), other_label)
    return df


def make_design_matrix(
    df: pd.DataFrame,
    categorical: list[str] | None = None,
    numeric: list[str] | None = None,
    min_freq: int = config.MIN_CATEGORY_FREQ,
    drop_first: bool = True,
    reference_levels: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Build a model-ready frame: one-hot categoricals + numerics + duration/event.

    The most frequent level of each categorical is used as the reference (dropped) level
    unless `reference_levels` says otherwise, which makes hazard ratios easier to read.
    """
    categorical = categorical if categorical is not None else [
        c for c in config.CATEGORICAL_COLS if c in df
    ]
    numeric = numeric if numeric is not None else [c for c in config.NUMERIC_COLS if c in df]
    reference_levels = reference_levels or {}

    df = lump_rare_categories(df, categorical, min_freq=min_freq)

    parts = [df[numeric].astype(float)]
    for col in categorical:
        ref = reference_levels.get(col, df[col].value_counts().idxmax())
        levels = [ref] + sorted(set(df[col].unique()) - {ref})
        cat = pd.Categorical(df[col], categories=levels)
        dummies = pd.get_dummies(cat, prefix=col, drop_first=drop_first, dtype=float)
        parts.append(dummies)

    X = pd.concat(parts, axis=1)
    X[config.DURATION_COL] = df[config.DURATION_COL].to_numpy()
    X[config.EVENT_COL] = df[config.EVENT_COL].to_numpy()
    return X


def split(
    df: pd.DataFrame,
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train/test split stratified on the event indicator."""
    return train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df[config.EVENT_COL]
    )


def to_structured_y(df: pd.DataFrame) -> np.ndarray:
    """Convert duration/event columns to the structured array scikit-survival expects."""
    return np.array(
        list(zip(df[config.EVENT_COL].astype(bool), df[config.DURATION_COL].astype(float))),
        dtype=[("event", bool), ("time", float)],
    )


def load_clean(path: str | Path = config.RAW_FILE) -> pd.DataFrame:
    """Convenience: load + clean in one call."""
    return clean(load_raw(path))
