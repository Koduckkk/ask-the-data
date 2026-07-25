"""Tests for the statistical inference layer.

Uses the real database (built by a fixture), so the tests exercise the same
path the page does. scipy is required; importorskip keeps the core suite
independent of it.
"""

import sys
from pathlib import Path

import duckdb
import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

pytest.importorskip("scipy")

import load as L
import stats as S


@pytest.fixture(scope="module")
def con(tmp_path_factory):
    path = tmp_path_factory.mktemp("db") / "test.duckdb"
    L.build_database(path)
    connection = duckdb.connect(str(path), read_only=True)
    yield connection
    connection.close()


# --- gender gap + CI ---------------------------------------------------------


def test_gender_gap_returns_a_ci_and_verdict(con):
    gap = S.gender_gap("Numeracy", 9, con=con)
    assert gap.ci_low < gap.ci_high
    # The difference sits inside its own CI.
    assert gap.ci_low <= gap.difference <= gap.ci_high
    assert gap.interpretation  # a plain-English read, not empty


def test_significance_matches_whether_ci_excludes_zero(con):
    gap = S.gender_gap("Reading", 9, con=con)
    excludes_zero = not (gap.ci_low <= 0 <= gap.ci_high)
    assert gap.is_significant == excludes_zero


def test_interpretation_names_noise_when_ci_includes_zero(con):
    # For a domain with no built-in effect, the honest read mentions noise /
    # no real difference when the interval spans zero.
    gap = S.gender_gap("Writing", 9, con=con)
    if not gap.is_significant:
        assert "noise" in gap.interpretation.lower() or "no real" in gap.interpretation.lower()


# --- school ranking with small-n flagging ------------------------------------


def test_reliable_schools_meet_the_threshold(con):
    reliable, _flagged = S.school_ranking("Numeracy", con=con)
    assert (reliable["n"] >= S.MIN_RELIABLE_N).all()
    # Ranked descending by mean.
    assert reliable["mean"].is_monotonic_decreasing


def test_flagged_schools_are_below_threshold(con):
    _reliable, flagged = S.school_ranking("Numeracy", con=con)
    if not flagged.empty:
        assert (flagged["n"] < S.MIN_RELIABLE_N).all()


def test_small_schools_are_withheld_not_ranked(con):
    # A tiny school must never appear in the reliable ranking.
    reliable, flagged = S.school_ranking("Numeracy", con=con)
    overlap = set(reliable["school"]) & set(flagged["school"])
    assert not overlap


# --- cleaning -> inference impact --------------------------------------------


def test_sentinel_impact_is_grounded_and_nonempty():
    note = S.sentinel_impact()
    assert note
    # It states a real bias magnitude, not a placeholder.
    assert "points" in note and "%" in note


# --- school effects (shrinkage) ----------------------------------------------


def test_shrinkage_pulls_small_schools_more(con):
    df = S.school_effects("Numeracy", con=con)
    # Reliability rises with sample size — big schools trust their own mean.
    assert df["reliability"].max() > df["reliability"].min()
    # The least reliable (smallest) school is pulled further than the most reliable.
    least = df.loc[df["reliability"].idxmin()]
    most = df.loc[df["reliability"].idxmax()]
    assert abs(least["pulled_by"]) > abs(most["pulled_by"])


def test_shrunk_estimate_lies_between_raw_and_grand_mean(con):
    df = S.school_effects("Numeracy", con=con)
    grand = float((df["raw_mean"] * df["n"]).sum() / df["n"].sum())
    # Each shrunk estimate is between its raw mean and the grand mean.
    for _, r in df.iterrows():
        lo, hi = sorted([r["raw_mean"], grand])
        assert lo - 0.1 <= r["shrunk_estimate"] <= hi + 0.1


def test_reliability_weight_matches_the_correct_formula(con):
    # Quantitative check — the earlier tests only verified DIRECTION, which a
    # wrong tau2 / sigma2 estimator still satisfies. Recompute the variance
    # components the correct way (pooled within-variance; between-variance with
    # the sigma2*mean(1/n) sampling term) and confirm the reliability weights
    # match. A regression in either estimator breaks this.
    import numpy as np

    # Group by school NAME so it lines up with school_effects' output key, and
    # so per-school n/var are matched exactly (school_effects groups by name).
    rows = con.execute(
        """
        SELECT sc.school_name AS school, COUNT(*) AS n,
               AVG(r.scaled_score) AS mean, VAR_SAMP(r.scaled_score) AS var
        FROM results r JOIN schools sc ON r.school_id = sc.school_id
        WHERE r.domain = 'Numeracy' AND r.scaled_score IS NOT NULL
        GROUP BY sc.school_name
        """
    ).fetchdf()

    n = rows["n"].to_numpy()
    pooled = rows[rows["n"] > 1]
    dof = pooled["n"].to_numpy() - 1
    sigma2 = float((pooled["var"].to_numpy() * dof).sum() / dof.sum())
    observed = float(rows["mean"].var(ddof=1))
    tau2 = max(observed - sigma2 * np.mean(1.0 / n), 1.0)
    rows["expected"] = tau2 / (tau2 + sigma2 / n)

    df = S.school_effects("Numeracy", con=con)
    merged = df.merge(rows[["school", "expected"]], on="school")
    assert (merged["reliability"] - merged["expected"]).abs().max() < 0.01


# --- marker anomaly detection ------------------------------------------------


def test_marker_detection_flags_the_biased_markers(con):
    from emit_vendor import MARKER_SEVERITY

    result = S.marker_anomalies(con=con)
    flagged = set(result[result["flag"] != "ok"]["marker"])
    # The generator's deliberately biased markers must be caught.
    harsh = {m for m, s in MARKER_SEVERITY.items() if s < -0.5}
    lenient = {m for m, s in MARKER_SEVERITY.items() if s > 0.5}
    assert harsh <= flagged
    assert lenient <= flagged


def test_marker_detection_does_not_flag_fair_markers(con):
    from emit_vendor import MARKER_SEVERITY

    result = S.marker_anomalies(con=con)
    flagged = set(result[result["flag"] != "ok"]["marker"])
    fair = {m for m, s in MARKER_SEVERITY.items() if s == 0.0}
    # Fair markers must not be flagged — the point of peer-relative thresholding.
    assert not (fair & flagged)


def test_marker_harsh_and_lenient_labelled_correctly(con):
    result = S.marker_anomalies(con=con)
    harsh = result[result["flag"] == "harsh"]["mean_residual"]
    lenient = result[result["flag"] == "lenient"]["mean_residual"]
    # Harsh markers under-score (negative residual), lenient over-score.
    assert (harsh < 0).all()
    assert (lenient > 0).all()
