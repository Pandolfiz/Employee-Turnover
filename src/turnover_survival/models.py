"""Semi-parametric, parametric and machine-learning survival models."""

from __future__ import annotations

import pandas as pd
from lifelines import (
    CoxPHFitter,
    LogLogisticAFTFitter,
    LogNormalAFTFitter,
    WeibullAFTFitter,
)
from lifelines.statistics import proportional_hazard_test
from sksurv.ensemble import RandomSurvivalForest

from . import config
from .data import to_structured_y

T, E = config.DURATION_COL, config.EVENT_COL


# --------------------------------------------------------------------------- Cox PH
def fit_cox(
    X: pd.DataFrame,
    penalizer: float = 0.01,
    l1_ratio: float = 0.0,
    strata: list[str] | None = None,
) -> CoxPHFitter:
    """Fit a Cox proportional hazards model on a design matrix containing T and E."""
    cph = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
    cph.fit(X, duration_col=T, event_col=E, strata=strata)
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
    res = proportional_hazard_test(cph, X, time_transform="rank")
    return res.summary.sort_values("p")


def tune_cox_penalizer(
    X_train: pd.DataFrame,
    penalizers: tuple[float, ...] = (0.0, 0.001, 0.01, 0.05, 0.1, 0.5),
    l1_ratio: float = 0.0,
    k: int = 5,
) -> pd.DataFrame:
    """K-fold CV of the Cox penalizer, scored by concordance index."""
    from lifelines.utils import k_fold_cross_validation

    rows = []
    for p in penalizers:
        scores = k_fold_cross_validation(
            CoxPHFitter(penalizer=p, l1_ratio=l1_ratio),
            X_train,
            duration_col=T,
            event_col=E,
            k=k,
            scoring_method="concordance_index",
            seed=config.RANDOM_STATE,
        )
        rows.append({"penalizer": p, "cv_cindex_mean": pd.Series(scores).mean(),
                     "cv_cindex_std": pd.Series(scores).std()})
    return pd.DataFrame(rows).sort_values("cv_cindex_mean", ascending=False)


# --------------------------------------------------------------------------- Parametric AFT
AFT_MODELS = {
    "Weibull": WeibullAFTFitter,
    "LogNormal": LogNormalAFTFitter,
    "LogLogistic": LogLogisticAFTFitter,
}


def fit_aft_models(X: pd.DataFrame, penalizer: float = 0.01) -> dict:
    """Fit all accelerated-failure-time models in AFT_MODELS."""
    return {
        name: cls(penalizer=penalizer).fit(X, duration_col=T, event_col=E)
        for name, cls in AFT_MODELS.items()
    }


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
def fit_rsf(
    X: pd.DataFrame,
    n_estimators: int = 500,
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


def features(X: pd.DataFrame) -> pd.DataFrame:
    """Drop the survival target columns from a design matrix."""
    return X.drop(columns=[T, E])
