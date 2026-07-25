"""Schema-drift detection — suggest column mappings, never decide them.

The pipeline maps each source's own column names to canonical ones with a
hardcoded rule (``platform_student_id`` → ``student_id``). That rule is correct
and auditable, but it is *manual*: when a feed renames a column between years
(``platform_student_id`` one year, ``student_platform_id`` the next), a human
has to notice and edit the rule.

This module catches that drift. Given the expected mapping and a source's actual
columns, it scores every candidate column against the target by three signals —
name similarity, value overlap, and format fingerprint — and, when the expected
column is missing, **suggests** the most likely replacement for a human to
confirm. It proposes; it does not silently remap. Wrong auto-matches corrupt
every downstream join, so the authority stays with a person — the same
human-in-the-loop principle as the NL→SQL layer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

import pandas as pd


@dataclass(frozen=True)
class MatchScore:
    """How well a candidate column matches a target, by each signal."""

    candidate: str
    name_similarity: float    # 0..1 fuzzy name match
    value_overlap: float      # 0..1 fraction of candidate values seen in the reference
    format_match: float       # 0..1 do the value shapes agree
    combined: float           # weighted total


@dataclass(frozen=True)
class DriftReport:
    """The outcome of checking one source's columns against the expected name."""

    target: str               # canonical column name, e.g. "student_id"
    expected_source: str      # the column we expected to find
    present: bool             # was the expected source column there?
    suggestion: str | None    # best replacement when it was not
    scores: list[MatchScore]  # ranked candidates


def _name_similarity(a: str, b: str) -> float:
    """Fuzzy similarity of two column names, token-aware.

    Combines a character-level ratio with token overlap, so
    ``platform_student_id`` and ``student_platform_id`` score high despite the
    reordering.
    """
    char = SequenceMatcher(None, a.lower(), b.lower()).ratio()
    ta, tb = set(re.split(r"[_\s]+", a.lower())), set(re.split(r"[_\s]+", b.lower()))
    token = len(ta & tb) / len(ta | tb) if (ta | tb) else 0.0
    return round(0.5 * char + 0.5 * token, 3)


def _format_fingerprint(values: pd.Series, sample: int = 500) -> str:
    """A coarse shape of a column's values: letters→A, digits→9, else same char.

    ``R938098930K`` → ``A999999999A``. Two id columns with the same scheme share
    a fingerprint even when the actual ids differ.
    """
    vals = values.dropna().astype(str).head(sample)
    if vals.empty:
        return ""
    shapes = vals.map(lambda v: re.sub(r"[A-Za-z]", "A", re.sub(r"\d", "9", v)))
    return shapes.mode().iloc[0] if not shapes.mode().empty else ""


def _value_overlap(candidate: pd.Series, reference: pd.Series, sample: int = 5000) -> float:
    """Fraction of the candidate's values that also appear in the reference set.

    The strongest signal: if the *same students* appear in both columns, they are
    almost certainly the same key, whatever the name. Normalised (stripped,
    leading zeros removed) so format drift doesn't hide a true match.
    """
    def norm(s: pd.Series) -> set:
        return set(s.dropna().astype(str).str.strip().str.lstrip("0").head(sample))

    cand, ref = norm(candidate), norm(reference)
    if not cand:
        return 0.0
    return round(len(cand & ref) / len(cand), 3)


def score_candidate(
    name: str,
    values: pd.Series,
    target: str,
    reference: pd.Series,
) -> MatchScore:
    """Score one candidate column against a target concept."""
    name_sim = _name_similarity(name, target)
    overlap = _value_overlap(values, reference)
    fmt = 1.0 if _format_fingerprint(values) == _format_fingerprint(reference) else 0.0
    # Value overlap is the most trustworthy signal, so it dominates the blend.
    combined = round(0.25 * name_sim + 0.6 * overlap + 0.15 * fmt, 3)
    return MatchScore(name, name_sim, overlap, fmt, combined)


def detect_drift(
    source: pd.DataFrame,
    target: str,
    expected_source: str,
    reference: pd.Series,
    threshold: float = 0.5,
) -> DriftReport:
    """Check whether ``expected_source`` is present; if not, suggest a match.

    ``reference`` is a known-good set of the target's values (e.g. last year's
    ids) to measure value overlap against. Returns a report ranking every source
    column as a candidate for ``target``.
    """
    scores = sorted(
        (
            score_candidate(col, source[col], target, reference)
            for col in source.columns
        ),
        key=lambda s: s.combined,
        reverse=True,
    )
    present = expected_source in source.columns
    suggestion = None
    if not present and scores and scores[0].combined >= threshold:
        suggestion = scores[0].candidate
    return DriftReport(
        target=target,
        expected_source=expected_source,
        present=present,
        suggestion=suggestion,
        scores=scores,
    )


if __name__ == "__main__":
    # Demonstrate on a renamed id column: the reference is last year's ids; the
    # "source" renamed the column and reordered the tokens.
    reference = pd.Series([f"R{i:09d}K" for i in range(1000)])
    source = pd.DataFrame(
        {
            "student_platform_id": reference,           # renamed, same values
            "surname": ["Smith"] * len(reference),
            "ylevel": [9] * len(reference),
        }
    )
    report = detect_drift(source, "student_id", "platform_student_id", reference)
    print(f"expected '{report.expected_source}' present: {report.present}")
    print(f"suggested replacement: {report.suggestion}")
    print("\nranked candidates:")
    for s in report.scores:
        print(f"  {s.candidate:22s} combined={s.combined:.2f} "
              f"(name {s.name_similarity:.2f}, overlap {s.value_overlap:.2f}, fmt {s.format_match:.0f})")
