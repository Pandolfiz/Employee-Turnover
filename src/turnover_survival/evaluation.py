"""Out-of-sample evaluation: concordance, time-dependent AUC and integrated Brier score."""

from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from sksurv.metrics import cumulative_dynamic_auc, integrated_brier_score

from . import config
from .data import to_structured_y
from .models import features

T, E = config.DURATION_COL, config.EVENT_COL


def evaluation_times(train: pd.DataFrame, test: pd.DataFrame, n: int = 20) -> np.ndarray:
    """Time grid strictly inside the follow-up range of both sets (required by sksurv metrics)."""
    lo = max(train[T].min(), test[T].min())
    hi = min(train[T].max(), test[T].max())
    q = np.quantile(test[T], [0.1, 0.8])
    return np.linspace(max(lo, q[0]), min(hi, q[1]) - 1e-6, n)


def _survival_matrix(model, X_test: pd.DataFrame, times: np.ndarray) -> np.ndarray:
    """Predicted S(t | x) for each test row (rows) at each time in `times` (columns)."""
    if hasattr(model, "predict_survival_function") and hasattr(model, "summary"):  # lifelines
        sf = model.predict_survival_function(features(X_test), times=times)
        return sf.T.to_numpy()
    fns = model.predict_survival_function(features(X_test))  # sksurv
    return np.vstack([fn(times) for fn in fns])


def _risk_score(model, X_test: pd.DataFrame) -> np.ndarray:
    """Higher = higher risk of leaving."""
    if hasattr(model, "predict_partial_hazard"):  # Cox
        return model.predict_partial_hazard(features(X_test)).to_numpy().ravel()
    if hasattr(model, "predict_median") and hasattr(model, "summary"):  # AFT
        return -model.predict_expectation(features(X_test)).to_numpy().ravel()
    return model.predict(features(X_test))  # sksurv: risk score


def evaluate(models: dict, train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Score several fitted models on a held-out set."""
    times = evaluation_times(train, test)
    y_train, y_test = to_structured_y(train), to_structured_y(test)
    rows = []
    for name, model in models.items():
        risk = _risk_score(model, test)
        surv = _survival_matrix(model, test, times)
        _, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk, times)
        rows.append(
            {
                "model": name,
                "c_index": concordance_index(test[T], -risk, test[E]),
                "mean_td_auc": mean_auc,
                "integrated_brier": integrated_brier_score(y_train, y_test, surv, times),
            }
        )
    return pd.DataFrame(rows).sort_values("c_index", ascending=False).reset_index(drop=True)


def km_baseline_ibs(train: pd.DataFrame, test: pd.DataFrame) -> float:
    """Integrated Brier score of the covariate-free Kaplan-Meier curve (the 'no-skill' benchmark)."""
    from lifelines import KaplanMeierFitter

    times = evaluation_times(train, test)
    kmf = KaplanMeierFitter().fit(train[T], train[E])
    s = kmf.survival_function_at_times(times).to_numpy()
    surv = np.tile(s, (len(test), 1))
    return integrated_brier_score(to_structured_y(train), to_structured_y(test), surv, times)
