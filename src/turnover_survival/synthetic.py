"""Synthetic data with the same schema as turnover.csv — used ONLY for tests and CI.

The values are random draws, not real employees; never use them for conclusions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_turnover(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    choice = lambda opts, p=None: rng.choice(opts, size=n, p=p)  # noqa: E731

    df = pd.DataFrame(
        {
            "gender": choice(["f", "m"], [0.75, 0.25]),
            "age": rng.normal(31, 7, n).clip(18, 58).round(),
            "industry": choice(["retail", "manufacture", "it", "banks", "etc", "consult"]),
            "profession": choice(["hr", "it", "sales", "commercial", "finance", "etc"]),
            "traffic": choice(["youjs", "empjs", "rabrecnerab", "friends", "referal", "kagency"]),
            "coach": choice(["no", "yes", "my head"], [0.6, 0.15, 0.25]),
            "head_gender": choice(["f", "m"]),
            "greywage": choice(["white", "grey"], [0.88, 0.12]),
            "way": choice(["bus", "car", "foot"], [0.6, 0.3, 0.1]),
            "extraversion": rng.uniform(1, 10, n).round(1),
            "independ": rng.uniform(1, 10, n).round(1),
            "selfcontrol": rng.uniform(1, 10, n).round(1),
            "anxiety": rng.uniform(1, 10, n).round(1),
            "novator": rng.uniform(1, 10, n).round(1),
        }
    )
    lin = (
        0.5 * (df["greywage"] == "grey")
        + 0.3 * (df["way"] == "bus")
        - 0.03 * (df["age"] - 31)
        + 0.08 * (df["anxiety"] - 5.5)
    )
    true_time = rng.weibull(1.2, n) * 40 * np.exp(-lin)
    censor = rng.uniform(1, 120, n)
    df.insert(0, "stag", np.minimum(true_time, censor).clip(0.1).round(2))
    df.insert(1, "event", (true_time <= censor).astype(int))
    return df
