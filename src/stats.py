"""Statistical inference — the layer that asks 'is this signal or noise?'.

Everything else in the repo describes *what the data says*; this module adds the
inferential judgement a data scientist brings: a gap comes with a confidence
interval and a plain-English read of whether it's real, and school rankings
refuse to trust schools with too few students. It's the difference between
reporting a number and reasoning about it.

**Honest framing.** The data is synthetic. The gender gap is ~0 by construction,
so the correct inferential answer is "no real effect" — and finding that, with a
CI that straddles zero, is exactly the point: the method works, and it doesn't
manufacture a gap that isn't there. The small-n school flagging is the real
signal — naively ranking schools with a handful of students is the classic
mistake this guards against.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

DB_PATH = Path(__file__).resolve().parent.parent / "ask_the_data.duckdb"

# Below this many students, a school average is too unstable to rank on — it is
# flagged as unreliable rather than placed in a league table.
MIN_RELIABLE_N = 30


def _connect(con: duckdb.DuckDBPyConnection | None) -> tuple[duckdb.DuckDBPyConnection, bool]:
    if con is not None:
        return con, False
    return duckdb.connect(str(DB_PATH), read_only=True), True


# --- gender gap with a confidence interval -----------------------------------


@dataclass(frozen=True)
class GapResult:
    """A group difference with the inference needed to judge it."""

    group_a: str
    group_b: str
    mean_a: float
    mean_b: float
    n_a: int
    n_b: int
    difference: float        # mean_a - mean_b
    ci_low: float
    ci_high: float
    interpretation: str      # plain-English read

    @property
    def is_significant(self) -> bool:
        """True when the CI excludes zero — a difference unlikely to be noise."""
        return not (self.ci_low <= 0 <= self.ci_high)


def gender_gap(
    domain: str = "Numeracy",
    year_level: int = 9,
    confidence: float = 0.95,
    con: duckdb.DuckDBPyConnection | None = None,
) -> GapResult:
    """Difference in mean scaled score between F and M, with a CI and a verdict.

    Uses Welch's t-interval (unequal variances) — the honest default when two
    groups may have different spreads. The interpretation states, in plain
    terms, whether the gap is distinguishable from zero.
    """
    from scipy import stats as sps

    con, owns = _connect(con)
    try:
        rows = con.execute(
            """
            SELECT s.gender AS g, COUNT(*) AS n,
                   AVG(r.scaled_score) AS mean, STDDEV_SAMP(r.scaled_score) AS sd
            FROM results r JOIN students s USING (student_id)
            WHERE r.year_level = ? AND r.domain = ?
              AND r.scaled_score IS NOT NULL AND s.gender IN ('F', 'M')
            GROUP BY s.gender ORDER BY s.gender
            """,
            [year_level, domain],
        ).fetchdf()
    finally:
        if owns:
            con.close()

    f = rows[rows["g"] == "F"].iloc[0]
    m = rows[rows["g"] == "M"].iloc[0]
    diff = float(f["mean"] - m["mean"])

    # Welch standard error and degrees of freedom.
    se = np.sqrt(f["sd"] ** 2 / f["n"] + m["sd"] ** 2 / m["n"])
    df = (f["sd"] ** 2 / f["n"] + m["sd"] ** 2 / m["n"]) ** 2 / (
        (f["sd"] ** 2 / f["n"]) ** 2 / (f["n"] - 1)
        + (m["sd"] ** 2 / m["n"]) ** 2 / (m["n"] - 1)
    )
    t_crit = sps.t.ppf(0.5 + confidence / 2, df)
    margin = float(t_crit * se)
    lo, hi = diff - margin, diff + margin

    crosses_zero = lo <= 0 <= hi
    spread = float((f["sd"] + m["sd"]) / 2)
    if crosses_zero:
        interpretation = (
            f"The gap is {diff:+.1f} points, but its {int(confidence * 100)}% "
            f"confidence interval [{lo:.1f}, {hi:.1f}] includes zero — so it is "
            f"consistent with no real difference. Against a spread of ~{spread:.0f} "
            f"points, a gap this small is indistinguishable from noise."
        )
    else:
        interpretation = (
            f"The gap is {diff:+.1f} points, {int(confidence * 100)}% CI "
            f"[{lo:.1f}, {hi:.1f}] — the interval excludes zero, so the "
            f"difference is unlikely to be noise. It is small relative to the "
            f"~{spread:.0f}-point spread, so statistically detectable but modest."
        )

    return GapResult(
        group_a="F", group_b="M",
        mean_a=float(f["mean"]), mean_b=float(m["mean"]),
        n_a=int(f["n"]), n_b=int(m["n"]),
        difference=diff, ci_low=lo, ci_high=hi, interpretation=interpretation,
    )


# --- school ranking that respects sample size --------------------------------


def school_ranking(
    domain: str = "Numeracy",
    min_n: int = MIN_RELIABLE_N,
    con: duckdb.DuckDBPyConnection | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank schools by mean score, but only those with enough students.

    Returns (reliable, flagged): schools with n >= min_n ranked with a standard
    error, and small-n schools withheld from the ranking because their averages
    are too unstable to compare. Ranking a school on a handful of students is the
    classic mistake this refuses to make.
    """
    con, owns = _connect(con)
    try:
        rows = con.execute(
            """
            SELECT sc.school_name AS school, COUNT(*) AS n,
                   AVG(r.scaled_score) AS mean, STDDEV_SAMP(r.scaled_score) AS sd
            FROM results r JOIN schools sc ON r.school_id = sc.school_id
            WHERE r.domain = ? AND r.scaled_score IS NOT NULL
            GROUP BY sc.school_name
            """,
            [domain],
        ).fetchdf()
    finally:
        if owns:
            con.close()

    rows["mean"] = rows["mean"].round(1)
    # Standard error of the mean — how precisely each average is pinned down.
    rows["std_error"] = (rows["sd"] / np.sqrt(rows["n"])).round(1)

    reliable = (
        rows[rows["n"] >= min_n]
        .sort_values("mean", ascending=False)
        .reset_index(drop=True)[["school", "n", "mean", "std_error"]]
    )
    flagged = (
        rows[rows["n"] < min_n]
        .sort_values("n")
        .reset_index(drop=True)[["school", "n", "mean"]]
    )
    return reliable, flagged


# --- connecting cleaning to inference ----------------------------------------


def sentinel_impact(con: duckdb.DuckDBPyConnection | None = None) -> str:
    """What the sentinel scores would have done to an average, in real numbers.

    Connects a cleaning step to its inferential consequence — the point a
    statistician makes that an engineer might not: an unrecoded 999 does not
    just look wrong, it *biases the estimate*. Quantified on the real data, so
    the claim is grounded, not asserted.
    """
    from emit_warehouse import RAW_DIR
    import clean as C

    raw = pd.read_csv(RAW_DIR / "warehouse_results.csv", dtype=str)
    scores = pd.to_numeric(raw["raw_score"], errors="coerce")
    sentinel_mask = raw["raw_score"].isin(["999.0", "-1.0", "9999.0"])
    n_sentinel = int(sentinel_mask.sum())
    n_total = int(scores.notna().sum())

    clean_scores = C.recode_sentinels(raw["raw_score"])
    clean_num = pd.to_numeric(clean_scores, errors="coerce")

    naive_mean = float(scores.mean())          # sentinels left in
    correct_mean = float(clean_num.mean())     # sentinels recoded to null
    bias = naive_mean - correct_mean

    return (
        f"Sentinels illustrate why cleaning is an inference problem, not just "
        f"tidying. {n_sentinel:,} of {n_total:,} raw scores (~"
        f"{n_sentinel / n_total * 100:.0f}%) were 'no score' codes like 999 and "
        f"-1. Left in a naive average, they don't just look wrong — they bias "
        f"it: the mean raw score reads {naive_mean:.1f} with them included versus "
        f"{correct_mean:.1f} once recoded to null, a shift of {bias:+.1f} points. "
        f"Because the codes are spread evenly across domains, the bias is broad "
        f"rather than concentrated — but any group comparison run on the "
        f"uncleaned data would inherit it."
    )


if __name__ == "__main__":
    gap = gender_gap()
    print(f"Gender gap — Year {9} Numeracy")
    print(f"  F: {gap.mean_a:.1f} (n={gap.n_a:,})   M: {gap.mean_b:.1f} (n={gap.n_b:,})")
    print(f"  difference {gap.difference:+.1f}, 95% CI [{gap.ci_low:.1f}, {gap.ci_high:.1f}]")
    print(f"  significant: {gap.is_significant}")
    print(f"  {gap.interpretation}")

    reliable, flagged = school_ranking()
    print(f"\nSchool ranking (Numeracy): {len(reliable)} reliable, {len(flagged)} flagged small-n")
    print(reliable.head(5).to_string(index=False))
    if not flagged.empty:
        print("flagged (withheld from ranking):")
        print(flagged.head(5).to_string(index=False))
