"""Semi-parametric, parametric and machine-learning survival models."""

from __future__ import annotations

import warnings
from itertools import product

import numpy as np
import pandas as pd
from lifelines import (
    CoxPHFitter,
    LogLogisticAFTFitter,
    LogNormalAFTFitter,
    WeibullAFTFitter,
)
from lifelines.statistics import proportional_hazard_test
from lifelines.utils import concordance_index
from sksurv.ensemble import RandomSurvivalForest

from . import config
from .data import folds, to_structured_y

T, E, G = config.DURATION_COL, config.EVENT_COL, config.GROUP_COL


# --------------------------------------------------------------------------- helpers
def features(X: pd.DataFrame) -> pd.DataFrame:
    """Covariates only: drop duration, event and group-id columns."""
    return X.drop(columns=[T, E, G], errors="ignore")


def model_frame(X: pd.DataFrame) -> pd.DataFrame:
    """Covariates + duration/event (what lifelines fitters expect), without the group id."""
    return X.drop(columns=[G], errors="ignore")


def risk_score(model, X: pd.DataFrame) -> np.ndarray:
    """Higher = higher risk of leaving, for any fitted model in this module."""
    Xf = features(X)
    if hasattr(model, "predict_partial_hazard"):  # Cox
        return model.predict_partial_hazard(Xf).to_numpy().ravel()
    if hasattr(model, "predict_expectation") and hasattr(model, "summary"):  # lifelines AFT
        return -model.predict_expectation(Xf).to_numpy().ravel()
    return model.predict(Xf)  # sksurv


def cindex(model, X: pd.DataFrame) -> float:
    """Harrell's concordance index of `model` on `X`."""
    return concordance_index(X[T], -risk_score(model, X), X[E])


def _grid_search(X: pd.DataFrame, fit_fn, grid: list[dict], k: int, seed: int) -> pd.DataFrame:
    """Grouped k-fold CV of `fit_fn(train, **params)` for every params dict in `grid`."""
    splits = list(folds(X, n_splits=k, random_state=seed))
    rows = []
    for params in grid:
        scores = [
            cindex(fit_fn(X.iloc[tr], **params), X.iloc[te]) for tr, te in splits
        ]
        rows.append({**params, "cv_cindex_mean": np.mean(scores), "cv_cindex_std": np.std(scores)})
    return pd.DataFrame(rows).sort_values("cv_cindex_mean", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- Cox PH
def fit_cox(
    X: pd.DataFrame,
    penalizer: float = 0.01,
    l1_ratio: float = 0.0,
    strata: list[str] | None = None,
    robust: bool = True,
) -> CoxPHFitter:
    """Fit a Cox PH model on a design matrix containing T and E.

    If the group-id column is present and `robust=True`, standard errors are cluster-robust
    (sandwich estimator) so repeated rows of the same person don't overstate significance.

    For inference (hazard ratios, p-values) use `penalizer=0`: the sandwich estimator ignores the
    penalty term and underestimates uncertainty for penalised fits. For prediction, tune the
    penalizer (see `fit_cox_tuned`); robust SEs don't affect predictions.
    """
    if robust and G in X and penalizer > 0.01:
        warnings.warn(
            f"Cluster-robust SEs with penalizer={penalizer} are unreliable (too small); "
            "use penalizer=0 for inference.",
            stacklevel=2,
        )
    cph = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
    if robust and G in X:
        cph.fit(X, duration_col=T, event_col=E, strata=strata, cluster_col=G)
    else:
        cph.fit(model_frame(X), duration_col=T, event_col=E, strata=strata)
    return cph


def hazard_ratio_table(cph: CoxPHFitter, alpha: float = 0.05) -> pd.DataFrame:
    """Tidy hazard-ratio table with 95% CI, sorted by effect size."""
    s = cph.summary
    out = pd.DataFrame(
        {
            "HR": s["exp(coef)"],
            "HR_lower_95": s["exp(coef) lower 95%"],
            "HR_upper_95": s["exp(coef) upper 95%"],
            "p": s["p"],
        }
    )
    out["significant"] = out["p"] < alpha
    return out.sort_values("HR", ascending=False)


def ph_assumption_table(cph: CoxPHFitter, X: pd.DataFrame) -> pd.DataFrame:
    """Schoenfeld-residual test of the proportional-hazards assumption per covariate."""
    # pass the full frame: a cluster-robust fit needs its cluster column to compute residuals
    res = proportional_hazard_test(cph, X, time_transform="rank")
    return res.summary.sort_values("p")


def tune_cox_penalizer(
    X_train: pd.DataFrame,
    penalizers: tuple[float, ...] = (0.0, 0.001, 0.01, 0.05, 0.1, 0.5),
    l1_ratio: float = 0.0,
    k: int = 5,
    seed: int = config.RANDOM_STATE,
) -> pd.DataFrame:
    """Grouped k-fold CV of the Cox penalizer, scored by concordance index."""
    grid = [{"penalizer": p} for p in penalizers]
    fit = lambda tr, penalizer: fit_cox(tr, penalizer=penalizer, l1_ratio=l1_ratio, robust=False)  # noqa: E731
    return _grid_search(X_train, fit, grid, k, seed)


def fit_cox_tuned(X_train: pd.DataFrame, k: int = 3, **kwargs) -> CoxPHFitter:
    """Prediction model: tune the penalizer by inner grouped CV, then refit on all of X_train."""
    best = tune_cox_penalizer(X_train, k=k, **kwargs).iloc[0]["penalizer"]
    return fit_cox(X_train, penalizer=best, robust=False)


# --------------------------------------------------------------------------- Parametric AFT
AFT_MODELS = {
    "Weibull": WeibullAFTFitter,
    "LogNormal": LogNormalAFTFitter,
    "LogLogistic": LogLogisticAFTFitter,
}


def fit_aft(X: pd.DataFrame, name: str = "Weibull", penalizer: float = 0.01):
    """Fit one AFT model (the group id is not a covariate, so it is dropped)."""
    return AFT_MODELS[name](penalizer=penalizer).fit(model_frame(X), duration_col=T, event_col=E)


def fit_aft_models(X: pd.DataFrame, penalizer: float = 0.01) -> dict:
    """Fit all accelerated-failure-time models in AFT_MODELS."""
    return {name: fit_aft(X, name, penalizer) for name in AFT_MODELS}


def compare_aft(models: dict) -> pd.DataFrame:
    """Compare fitted AFT models by AIC / log-likelihood / concordance."""
    rows = [
        {
            "model": name,
            "AIC": m.AIC_,
            "log_likelihood": m.log_likelihood_,
            "concordance_train": m.concordance_index_,
        }
        for name, m in models.items()
    ]
    return pd.DataFrame(rows).sort_values("AIC").reset_index(drop=True)


# --------------------------------------------------------------------------- Random Survival Forest
RSF_GRID = {"min_samples_leaf": [5, 15, 30], "max_features": ["sqrt", 0.5]}


def fit_rsf(
    X: pd.DataFrame,
    n_estimators: int = 300,
    min_samples_leaf: int = 10,
    max_features: str | float = "sqrt",
    random_state: int = config.RANDOM_STATE,
) -> RandomSurvivalForest:
    """Fit a Random Survival Forest on a design matrix containing T and E."""
    rsf = RandomSurvivalForest(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        n_jobs=-1,
        random_state=random_state,
    )
    rsf.fit(features(X), to_structured_y(X))
    return rsf


def tune_rsf(
    X_train: pd.DataFrame,
    grid: dict = RSF_GRID,
    k: int = 3,
    n_estimators: int = 200,
    seed: int = config.RANDOM_STATE,
) -> pd.DataFrame:
    """Grouped k-fold CV over the RSF hyper-parameter grid, scored by concordance index."""
    combos = [dict(zip(grid, v)) for v in product(*grid.values())]
    fit = lambda tr, **p: fit_rsf(tr, n_estimators=n_estimators, **p)  # noqa: E731
    return _grid_search(X_train, fit, combos, k, seed)


def fit_rsf_tuned(X_train: pd.DataFrame, k: int = 3, n_estimators: int = 300) -> RandomSurvivalForest:
    """Tune RSF by inner grouped CV, then refit on all of X_train."""
    best = tune_rsf(X_train, k=k).iloc[0]
    return fit_rsf(
        X_train,
        n_estimators=n_estimators,
        min_samples_leaf=int(best["min_samples_leaf"]),
        max_features=best["max_features"],
    )
