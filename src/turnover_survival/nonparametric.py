"""Non-parametric survival estimation: Kaplan-Meier, Nelson-Aalen and log-rank tests."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from lifelines import KaplanMeierFitter, NelsonAalenFitter
from lifelines.statistics import multivariate_logrank_test

from . import config

T, E = config.DURATION_COL, config.EVENT_COL


def fit_km(df: pd.DataFrame, label: str = "all employees") -> KaplanMeierFitter:
    return KaplanMeierFitter(label=label).fit(df[T], df[E])


def fit_na(df: pd.DataFrame, label: str = "all employees") -> NelsonAalenFitter:
    return NelsonAalenFitter(label=label).fit(df[T], df[E])


def survival_at(kmf: KaplanMeierFitter, times: list[float]) -> pd.DataFrame:
    """Survival probability (retention) and 95% CI at given tenures."""
    ci = kmf.confidence_interval_survival_function_.asof(times)  # step function: last value <= t
    ci.columns = ["ci_lower", "ci_upper"]
    ci.insert(0, "survival", kmf.survival_function_at_times(times).to_numpy())
    ci.index.name = "tenure_months"
    return ci


def logrank_by_group(df: pd.DataFrame, col: str) -> dict:
    """Multivariate log-rank test of H0: all groups of `col` share the same survival curve."""
    res = multivariate_logrank_test(df[T], df[col], df[E])
    return {
        "variable": col,
        "n_groups": df[col].nunique(),
        "test_statistic": res.test_statistic,
        "p_value": res.p_value,
    }


def logrank_table(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Log-rank test for each grouping variable, sorted by p-value."""
    return (
        pd.DataFrame([logrank_by_group(df, c) for c in cols])
        .sort_values("p_value")
        .reset_index(drop=True)
    )


def plot_km_by_group(
    df: pd.DataFrame,
    col: str,
    ax: plt.Axes | None = None,
    ci_show: bool = False,
    min_n: int = 1,
) -> plt.Axes:
    """Overlay Kaplan-Meier curves for each level of `col` (levels with < min_n rows skipped)."""
    ax = ax or plt.subplots(figsize=(8, 5))[1]
    for level, g in df.groupby(col):
        if len(g) < min_n:
            continue
        KaplanMeierFitter(label=f"{level} (n={len(g)})").fit(g[T], g[E]).plot_survival_function(
            ax=ax, ci_show=ci_show
        )
    p = logrank_by_group(df, col)["p_value"]
    ax.set_title(f"Kaplan-Meier by {col}  (log-rank p = {p:.3g})")
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Probability of staying")
    ax.set_ylim(0, 1.02)
    return ax
