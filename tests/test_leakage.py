"""Data-leakage guards for the evaluation pipeline.

Negative control: with the outcome shuffled across rows there is nothing to learn, so a
leak-free pipeline must give C-index ≈ 0.5. Positive control: on the unshuffled synthetic data
(which has a planted signal) the same pipeline must clearly beat 0.5, proving the control has power.
"""

import numpy as np
import pytest

from turnover_survival import config, data, evaluation, models
from turnover_survival.synthetic import make_synthetic_turnover


@pytest.fixture(scope="module")
def raw():
    df = make_synthetic_turnover(n=500, seed=11)
    # duplicate personality profiles, like the real data, so grouping matters
    pers = config.PERSONALITY_COLS
    src = df.sample(150, random_state=0).index
    dst = df.drop(src).sample(150, random_state=1).index
    df.loc[dst, pers] = df.loc[src, pers].to_numpy()
    return data.clean(df)


FITTERS = {
    "cox": lambda d: models.fit_cox_tuned(d, k=2, penalizers=(0.01, 0.1)),
    "rsf": lambda d: models.fit_rsf(d, n_estimators=40, min_samples_leaf=15),
}


def test_shuffled_outcome_gives_chance_level(raw):
    ctrl = evaluation.shuffled_outcome_control(raw, FITTERS, n_permutations=3, n_splits=4)
    summary = evaluation.summarize_control(ctrl).set_index("model")
    # 12 folds per model; fold SD ~0.04 -> SE of the mean ~0.012
    assert (summary["c_index_mean"].sub(0.5).abs() < 0.04).all(), summary


def test_unshuffled_signal_is_detected(raw):
    scores, _ = evaluation.cross_validate(raw, FITTERS, n_splits=4, n_repeats=1,
                                          horizons=(12,), verbose=False)
    assert (scores.groupby("model")["c_index"].mean() > 0.56).all()


def test_cross_validate_builds_design_matrix_per_fold(raw, monkeypatch):
    fitted_sizes = []
    original_fit = data.DesignMatrixBuilder.fit

    def spy(self, df):
        fitted_sizes.append(len(df))
        return original_fit(self, df)

    monkeypatch.setattr(data.DesignMatrixBuilder, "fit", spy)
    evaluation.cross_validate(raw, {"cox": FITTERS["cox"]}, n_splits=4, n_repeats=1,
                              horizons=(12,), verbose=False)
    assert len(fitted_sizes) == 4
    assert all(n < len(raw) for n in fitted_sizes)  # never fitted on the full data


def test_holdout_split_and_selection_rule(raw):
    dev, hold = data.split(raw, test_size=0.2)
    assert not set(dev[config.GROUP_COL]) & set(hold[config.GROUP_COL])

    scores, _ = evaluation.cross_validate(dev, FITTERS, n_splits=3, n_repeats=2,
                                          horizons=(12,), verbose=False)
    choice = evaluation.select_model(scores, preferred="cox")
    assert choice["selected"] in FITTERS

    table, fitted = evaluation.holdout_evaluate(dev, hold, FITTERS, n_boot=50)
    assert set(fitted) == set(FITTERS)
    row = table.set_index("model").loc["cox"]
    assert row["c_index_low"] <= row["c_index"] <= row["c_index_high"]
    km = table.set_index("model").loc["Kaplan-Meier (no covariates)"]
    assert np.isclose(km["c_index"], 0.5)
