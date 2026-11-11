"""Smoke tests for the full pipeline, run on synthetic data with the real schema."""

import numpy as np
import pandas as pd
import pytest

from turnover_survival import config, data, evaluation, models, nonparametric
from turnover_survival.synthetic import make_synthetic_turnover


@pytest.fixture(scope="module")
def df():
    raw = make_synthetic_turnover(n=500, seed=1)
    raw.loc[0, "industry"] = "  IT "  # whitespace / case noise like the real file
    return data.clean(raw)


@pytest.fixture(scope="module")
def X(df):
    return data.make_design_matrix(df)


def test_clean_normalises_and_validates(df):
    assert df["industry"].str.strip().eq(df["industry"]).all()
    assert df["industry"].str.lower().eq(df["industry"]).all()
    assert (df[config.DURATION_COL] > 0).all()
    assert set(df[config.EVENT_COL].unique()) <= {0, 1}


def test_clean_fixes_cyrillic_homoglyphs(tmp_path):
    raw = make_synthetic_turnover(n=5, seed=2)
    raw["profession"] = "Finanсe"  # Cyrillic "с", as in the real file
    path = tmp_path / "t.csv"
    raw.to_csv(path, index=False, encoding="cp1251")
    assert (data.clean(data.load_raw(path))["profession"] == "finance").all()


def test_clean_rejects_missing_target():
    with pytest.raises(ValueError):
        data.clean(pd.DataFrame({"stag": [1.0]}))


def test_lump_rare_categories():
    d = pd.DataFrame({"x": ["a"] * 30 + ["b"] * 5})
    out = data.lump_rare_categories(d, ["x"], min_freq=20)
    assert set(out["x"]) == {"a", "other"}


def test_design_matrix_is_numeric_and_complete(X, df):
    assert len(X) == len(df)
    assert not X.isna().any().any()
    assert all(np.issubdtype(t, np.number) for t in X.dtypes)
    # most frequent level is the reference -> not present as a dummy
    top = df["coach"].value_counts().idxmax()
    assert f"coach_{top}" not in X.columns


def test_km_and_logrank(df):
    kmf = nonparametric.fit_km(df)
    tbl = nonparametric.survival_at(kmf, [6, 12, 24])
    assert tbl["survival"].is_monotonic_decreasing
    assert (tbl["ci_lower"] <= tbl["survival"]).all()
    lr = nonparametric.logrank_table(df, ["greywage", "way"])
    assert lr["p_value"].between(0, 1).all()


@pytest.fixture(scope="module")
def X_grouped():
    """Synthetic data where ~40% of rows duplicate another row's personality profile."""
    raw = make_synthetic_turnover(n=400, seed=3)
    pers = config.PERSONALITY_COLS
    src = raw.sample(160, random_state=0).index
    dst = raw.drop(src).sample(160, random_state=1).index
    raw.loc[dst, pers] = raw.loc[src, pers].to_numpy()
    return data.make_design_matrix(data.clean(raw))


def test_profile_id_groups_identical_profiles(X_grouped):
    assert config.GROUP_COL in X_grouped
    assert X_grouped[config.GROUP_COL].nunique() < len(X_grouped)
    assert config.GROUP_COL not in models.features(X_grouped)


def test_grouped_split_has_no_profile_overlap(X_grouped):
    train, test = data.split(X_grouped)
    assert not set(train[config.GROUP_COL]) & set(test[config.GROUP_COL])
    for tr, te in data.folds(X_grouped, n_splits=4):
        g = X_grouped[config.GROUP_COL].to_numpy()
        assert not set(g[tr]) & set(g[te])


def test_cox_cluster_robust_and_aft_ignore_group(X_grouped):
    cph = models.fit_cox(X_grouped, penalizer=0.0)
    assert config.GROUP_COL not in cph.params_.index
    naive = models.fit_cox(X_grouped, penalizer=0.0, robust=False)
    with pytest.warns(UserWarning, match="unreliable"):
        models.fit_cox(X_grouped, penalizer=0.5)
    assert np.allclose(cph.params_, naive.params_)  # same point estimates
    assert not np.allclose(cph.standard_errors_, naive.standard_errors_)  # different SEs
    aft = models.fit_aft(X_grouped, "Weibull")
    assert not any(config.GROUP_COL in str(i) for i in aft.params_.index)


def test_nested_cv_ttest_and_calibration(X_grouped):
    fitters = {
        "cox": lambda d: models.fit_cox_tuned(d, k=2, penalizers=(0.01, 0.1)),
        "rsf": lambda d: models.fit_rsf(d, n_estimators=30),
    }
    scores, oof = evaluation.cross_validate(
        X_grouped, fitters, n_splits=3, n_repeats=2, horizons=(12,), verbose=False
    )
    assert len(scores) == 3 * 2 * 2
    assert oof.groupby("model").size().eq(len(X_grouped)).all()  # every row predicted once
    assert oof["p_leave_12"].between(0, 1).all()
    summary = evaluation.summarize_cv(scores)
    assert ("c_index", "mean") in summary.columns
    t = evaluation.corrected_paired_ttest(scores, "rsf", "cox")
    assert t["ci95_low"] <= t["mean_diff"] <= t["ci95_high"]
    cal = evaluation.calibration_table(oof, X_grouped, horizon=12, n_bins=3)
    assert cal["observed"].between(0, 1).all()


def test_models_and_evaluation(X):
    train, test = data.split(X)
    cph = models.fit_cox(train)
    assert {"HR", "p"} <= set(models.hazard_ratio_table(cph).columns)
    assert not models.ph_assumption_table(cph, train).empty

    aft = models.fit_aft_models(train)
    assert len(models.compare_aft(aft)) == 3

    rsf = models.fit_rsf(train, n_estimators=50)
    res = evaluation.evaluate({"cox": cph, "weibull": aft["Weibull"], "rsf": rsf}, train, test)
    assert res["c_index"].between(0, 1).all()
    assert res["integrated_brier"].between(0, 1).all()
    assert 0 < evaluation.km_baseline_ibs(train, test) < 1
