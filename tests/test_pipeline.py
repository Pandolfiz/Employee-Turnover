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
