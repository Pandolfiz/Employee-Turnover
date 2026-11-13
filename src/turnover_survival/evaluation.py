"""Leakage-safe evaluation: repeated grouped CV with fold-wise preprocessing, model selection,
a one-shot final hold-out test with bootstrap CIs, and calibration."""

from __future__ import annotations

from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index
from scipy import stats
from sksurv.metrics import cumulative_dynamic_auc, integrated_brier_score

from . import config
from .data import DesignMatrixBuilder, folds, is_raw, to_structured_y
from .models import features, risk_score

T, E, G = config.DURATION_COL, config.EVENT_COL, config.GROUP_COL

Fitter = Callable[[pd.DataFrame], object]


# --------------------------------------------------------------------------- building blocks
def evaluation_times(train: pd.DataFrame, test: pd.DataFrame, n: int = 20) -> np.ndarray:
    """Time grid strictly inside the follow-up range of both sets (required by sksurv metrics).

    Only the *grid* of evaluation times depends on the test durations; no model sees them.
    """
    lo = max(train[T].min(), test[T].min())
    hi = min(train[T].max(), test[T].max())
    q = np.quantile(test[T], [0.1, 0.8])
    return np.linspace(max(lo, q[0]), min(hi, q[1]) - 1e-6, n)


def survival_matrix(model, X: pd.DataFrame, times) -> np.ndarray:
    """Predicted S(t | x): one row per row of X, one column per time."""
    times = np.atleast_1d(times)
    if hasattr(model, "summary"):  # lifelines
        return model.predict_survival_function(features(X), times=times).T.to_numpy()
    return np.vstack([fn(times) for fn in model.predict_survival_function(features(X))])


def _within_follow_up(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    """Mask of test rows observed before the last training time.

    The censoring-weighted metrics (td-AUC, Brier) estimate the censoring distribution on the
    training set and cannot extrapolate past its last observed time.
    """
    return test[T] < train[T].max()


def score(model, train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """C-index (all test rows), mean time-dependent AUC and integrated Brier score."""
    c = concordance_index(test[T], -risk_score(model, test), test[E])
    test = test[_within_follow_up(train, test)]
    times = evaluation_times(train, test)
    y_train, y_test = to_structured_y(train), to_structured_y(test)
    _, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk_score(model, test), times)
    ibs = integrated_brier_score(y_train, y_test, survival_matrix(model, test, times), times)
    return {"c_index": c, "mean_td_auc": mean_auc, "integrated_brier": ibs}


def km_baseline_ibs(train: pd.DataFrame, test: pd.DataFrame) -> float:
    """Integrated Brier score of the covariate-free Kaplan-Meier curve (the 'no-skill' benchmark)."""
    test = test[_within_follow_up(train, test)]
    times = evaluation_times(train, test)
    s = KaplanMeierFitter().fit(train[T], train[E]).survival_function_at_times(times).to_numpy()
    surv = np.tile(s, (len(test), 1))
    return integrated_brier_score(to_structured_y(train), to_structured_y(test), surv, times)


def evaluate(models: dict, train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Score several already-fitted models on a single held-out set (design matrices)."""
    rows = [{"model": name, **score(m, train, test)} for name, m in models.items()]
    return pd.DataFrame(rows).sort_values("c_index", ascending=False).reset_index(drop=True)


def prepare_fold(
    train_raw: pd.DataFrame, test_raw: pd.DataFrame, builder_factory=DesignMatrixBuilder
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit preprocessing on train only and apply it to both sets.

    If the inputs are already design matrices (no raw categorical columns) they pass through.
    """
    if not is_raw(train_raw):
        return train_raw, test_raw
    builder = builder_factory().fit(train_raw)
    return builder.transform(train_raw), builder.transform(test_raw)


# --------------------------------------------------------------------------- repeated CV
def cross_validate(
    df: pd.DataFrame,
    fitters: dict[str, Fitter],
    n_splits: int = 5,
    n_repeats: int = 3,
    grouped: bool = True,
    horizons: tuple[float, ...] = (12, 24),
    seed: int = config.RANDOM_STATE,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Repeated (grouped) k-fold CV with fold-wise preprocessing.

    `df` should be the *cleaned raw* data: the design matrix is then built inside every fold,
    fitted on the training part only. `fitters` maps a model name to `train_matrix -> fitted
    model`; any hyper-parameter tuning must happen inside that function (nested CV).

    Returns
    -------
    scores : one row per (repeat, fold, model) with C-index, td-AUC, IBS (+ KM baseline IBS).
    oof    : out-of-fold predicted P(leave by h) per horizon from the first repeat
             (every row predicted exactly once), indexed like `df` — used for calibration.
    """
    score_rows, oof_rows = [], []
    for r in range(n_repeats):
        for f, (tr_idx, te_idx) in enumerate(
            folds(df, n_splits=n_splits, random_state=seed + r, grouped=grouped)
        ):
            train, test = prepare_fold(df.iloc[tr_idx], df.iloc[te_idx])
            km_ibs = km_baseline_ibs(train, test)
            for name, fit in fitters.items():
                model = fit(train)
                score_rows.append(
                    {"repeat": r, "fold": f, "model": name, "n_train": len(train),
                     "n_test": len(test), **score(model, train, test), "km_ibs": km_ibs}
                )
                if r == 0:
                    p_leave = 1 - survival_matrix(model, test, list(horizons))
                    oof_rows.append(
                        pd.DataFrame(p_leave, index=test.index,
                                     columns=[f"p_leave_{h:g}" for h in horizons]).assign(model=name)
                    )
            if verbose:
                print(f"repeat {r + 1}/{n_repeats} · fold {f + 1}/{n_splits} done")
    return pd.DataFrame(score_rows), pd.concat(oof_rows)


def shuffled_outcome_control(
    df: pd.DataFrame,
    fitters: dict[str, Fitter],
    n_permutations: int = 3,
    n_splits: int = 5,
    seed: int = config.RANDOM_STATE,
) -> pd.DataFrame:
    """Negative control: permute (duration, event) pairs across rows and re-run the full CV.

    With the outcome detached from the covariates there is nothing to learn, so a leak-free
    pipeline must give C-index ≈ 0.5. A mean clearly above 0.5 means information leaks.
    """
    rows = []
    for s in range(n_permutations):
        shuffled = df.copy()
        perm = np.random.default_rng(seed + s).permutation(len(df))
        shuffled[[T, E]] = df[[T, E]].to_numpy()[perm]
        sc, _ = cross_validate(shuffled, fitters, n_splits=n_splits, n_repeats=1,
                               seed=seed + 1000 + s, horizons=(12,), verbose=False)
        rows.append(sc.assign(permutation=s))
    return pd.concat(rows, ignore_index=True)


def summarize_control(ctrl: pd.DataFrame) -> pd.DataFrame:
    """Mean C-index under shuffled outcome and a one-sample t-test against 0.5."""
    out = []
    for name, g in ctrl.groupby("model"):
        t = stats.ttest_1samp(g["c_index"], 0.5)
        out.append({"model": name, "c_index_mean": g["c_index"].mean(),
                    "c_index_std": g["c_index"].std(), "n_folds": len(g),
                    "p_value_vs_0.5": t.pvalue,
                    "ibs_mean": g["integrated_brier"].mean(), "km_ibs_mean": g["km_ibs"].mean()})
    return pd.DataFrame(out)


def summarize_cv(scores: pd.DataFrame) -> pd.DataFrame:
    """Mean, std and 2.5-97.5% percentile range across folds for each model and metric."""
    metrics = ["c_index", "mean_td_auc", "integrated_brier"]
    g = scores.groupby("model")[metrics]
    out = pd.concat(
        {"mean": g.mean(), "std": g.std(), "p2.5": g.quantile(0.025), "p97.5": g.quantile(0.975)},
        axis=1,
    ).swaplevel(axis=1)[metrics]
    out[("integrated_brier", "km_baseline")] = scores.groupby("model")["km_ibs"].mean()
    return out.sort_values(("c_index", "mean"), ascending=False)


def corrected_paired_ttest(
    scores: pd.DataFrame, model_a: str, model_b: str, metric: str = "c_index"
) -> dict:
    """Nadeau & Bengio corrected resampled t-test of model_a − model_b on `metric`.

    The plain paired t-test across CV folds is over-confident because training sets overlap;
    the correction inflates the variance by (1/J + n_test/n_train).
    """
    piv = scores.pivot_table(index=["repeat", "fold"], columns="model", values=metric)
    d = (piv[model_a] - piv[model_b]).to_numpy()
    J = len(d)
    ratio = (scores["n_test"] / scores["n_train"]).mean()
    var = d.var(ddof=1) * (1 / J + ratio)
    t = d.mean() / np.sqrt(var)
    p = 2 * stats.t.sf(abs(t), df=J - 1)
    half = stats.t.ppf(0.975, df=J - 1) * np.sqrt(var)
    return {
        "comparison": f"{model_a} − {model_b}",
        "metric": metric,
        "mean_diff": d.mean(),
        "ci95_low": d.mean() - half,
        "ci95_high": d.mean() + half,
        "share_folds_a_better": (d > 0).mean() if metric != "integrated_brier" else (d < 0).mean(),
        "t": t,
        "p_value": p,
    }


def select_model(
    scores: pd.DataFrame,
    preferred: str,
    metric: str = "integrated_brier",
    alpha: float = 0.05,
) -> dict:
    """Pre-registered selection rule, applied to CV scores only (never to the hold-out).

    Take the model with the best mean `metric`. If it is not *significantly* better than the
    `preferred` (more interpretable) model under the corrected t-test, choose `preferred`.
    """
    means = scores.groupby("model")[metric].mean()
    best = means.idxmin() if metric == "integrated_brier" else means.idxmax()
    if best == preferred:
        return {"selected": preferred, "best_by_cv": best, "reason": "preferred model is best in CV"}
    t = corrected_paired_ttest(scores, best, preferred, metric)
    if t["p_value"] < alpha:
        return {"selected": best, "best_by_cv": best, "p_value": t["p_value"],
                "reason": f"significantly better than {preferred}"}
    return {"selected": preferred, "best_by_cv": best, "p_value": t["p_value"],
            "reason": f"{best} not significantly better (p={t['p_value']:.2f}); prefer {preferred}"}


# --------------------------------------------------------------------------- final hold-out
def holdout_evaluate(
    dev_raw: pd.DataFrame,
    holdout_raw: pd.DataFrame,
    fitters: dict[str, Fitter],
    n_boot: int = 500,
    seed: int = config.RANDOM_STATE,
) -> tuple[pd.DataFrame, dict]:
    """Fit preprocessing + every model on the development set, score once on the hold-out.

    95% CIs come from a *grouped* bootstrap of the hold-out (profiles resampled with
    replacement), so repeated rows of one person move together.

    Returns (table of point estimates and CIs, fitted models).
    """
    train, test = prepare_fold(dev_raw, holdout_raw)
    test = test[_within_follow_up(train, test)]
    times = evaluation_times(train, test)
    y_train = to_structured_y(train)
    km_s = KaplanMeierFitter().fit(train[T], train[E]).survival_function_at_times(times).to_numpy()

    groups = test[G].to_numpy() if G in test else np.arange(len(test))
    uniq = np.unique(groups)
    pos_by_group = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed)
    boot_idx = [
        np.concatenate([pos_by_group[g] for g in rng.choice(uniq, size=len(uniq), replace=True)])
        for _ in range(n_boot)
    ]

    def metrics(risk, surv, idx):
        sub = test.iloc[idx]
        keep = (times >= sub[T].min()) & (times < sub[T].max())
        c = concordance_index(sub[T], -risk[idx], sub[E])
        ibs = integrated_brier_score(y_train, to_structured_y(sub), surv[idx][:, keep], times[keep])
        return c, ibs

    fitted, rows = {}, []
    all_idx = np.arange(len(test))
    for name, fit in {**fitters, "Kaplan-Meier (no covariates)": None}.items():
        if fit is None:
            risk = np.zeros(len(test))
            surv = np.tile(km_s, (len(test), 1))
        else:
            fitted[name] = model = fit(train)
            risk = risk_score(model, test)
            surv = survival_matrix(model, test, times)
        c, ibs = metrics(risk, surv, all_idx)
        boots = np.array([metrics(risk, surv, b) for b in boot_idx])
        rows.append({
            "model": name, "c_index": c,
            "c_index_low": np.quantile(boots[:, 0], 0.025),
            "c_index_high": np.quantile(boots[:, 0], 0.975),
            "integrated_brier": ibs,
            "ibs_low": np.quantile(boots[:, 1], 0.025),
            "ibs_high": np.quantile(boots[:, 1], 0.975),
        })
    return pd.DataFrame(rows), fitted


# --------------------------------------------------------------------------- calibration
def calibration_table(
    oof: pd.DataFrame, df: pd.DataFrame, horizon: float, n_bins: int = 5
) -> pd.DataFrame:
    """Predicted vs observed P(leave by `horizon`) per risk quantile bin, for each model.

    Observed risk is 1 − Kaplan-Meier at `horizon` within the bin, so censoring is handled.
    """
    col = f"p_leave_{horizon:g}"
    rows = []
    for name, g in oof.groupby("model"):
        bins = pd.qcut(g[col], n_bins, labels=False, duplicates="drop")
        for b, gb in g.groupby(bins):
            sub = df.loc[gb.index]
            kmf = KaplanMeierFitter().fit(sub[T], sub[E])
            obs = 1 - kmf.survival_function_at_times(horizon).iloc[0]
            ci = kmf.confidence_interval_survival_function_.asof(horizon)
            rows.append({"model": name, "bin": int(b) + 1, "n": len(gb),
                         "predicted": gb[col].mean(), "observed": obs,
                         "observed_low": 1 - ci.iloc[1], "observed_high": 1 - ci.iloc[0]})
    tab = pd.DataFrame(rows)
    tab["abs_error"] = (tab["predicted"] - tab["observed"]).abs()
    return tab


def calibration_summary(tab: pd.DataFrame) -> pd.DataFrame:
    """Mean absolute calibration error per model (lower is better)."""
    return (tab.groupby("model")["abs_error"].mean().rename("mean_abs_calibration_error")
            .sort_values().to_frame())


def plot_calibration(tab: pd.DataFrame, horizon: float, ax: plt.Axes | None = None) -> plt.Axes:
    ax = ax or plt.subplots(figsize=(6, 6))[1]
    for name, g in tab.groupby("model"):
        ax.errorbar(g["predicted"], g["observed"],
                    yerr=[g["observed"] - g["observed_low"], g["observed_high"] - g["observed"]],
                    marker="o", capsize=3, label=name)
    lim = max(tab["predicted"].max(), tab["observed_high"].max()) * 1.05
    ax.plot([0, lim], [0, lim], ls="--", c="grey", lw=1)
    ax.set(xlim=(0, lim), ylim=(0, lim), xlabel="Predicted P(leave)",
           ylabel="Observed P(leave) (Kaplan-Meier)",
           title=f"Calibration at {horizon:g} months (out-of-fold)")
    ax.legend(fontsize=8)
    return ax
