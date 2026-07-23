"""Cleaning rules — the counterpart to ``mess.py``.

Each rule is a small, pure function that takes a dirty column (or frame) and
returns a cleaned one, recording what it changed in a ``CleaningReport``. Most
reverse a specific injected defect; a few can only *flag* what cannot be
recovered, which is a deliberate and more honest choice than pretending to fix
it.

Three properties are the whole point of doing it this way, and they are what a
reviewer is meant to see:

* **Deterministic.** A dictionary lookup, a regex, a parse — the same input
  always yields the same output. No model, no randomness. Cleaning has one
  correct answer, so it is code, not a guess.
* **Auditable.** Every change is counted and attributed to a named rule, so the
  pipeline can tell you exactly what it did rather than silently transforming
  the data underneath you.
* **Testable.** Dirty in, clean out — each rule has a unit test that feeds it a
  known defect and asserts the repair.

This module covers text defects (§1 of docs/quirks.md). Later groups —
identifiers, values, and the structural vendor reshape — build on the same
report object.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class CleaningReport:
    """A running tally of what the pipeline changed, by rule.

    The report is the verification-by-design theme made concrete: nothing is
    cleaned silently. Each rule calls ``record`` with a count, and the summary
    is printed and saved so a human can see, per rule, how many values were
    touched.
    """

    rows: list[dict] = field(default_factory=list)

    def record(self, rule: str, column: str, changed: int, detail: str = "") -> None:
        """Note that ``rule`` changed ``changed`` values in ``column``."""
        self.rows.append(
            {"rule": rule, "column": column, "changed": int(changed), "detail": detail}
        )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=["rule", "column", "changed", "detail"])

    def summary(self) -> str:
        if not self.rows:
            return "No changes recorded."
        frame = self.to_frame()
        lines = ["Cleaning report", "=" * 52]
        for _, r in frame.iterrows():
            label = f"{r['rule']} ({r['column']})"
            tail = f"  — {r['detail']}" if r["detail"] else ""
            lines.append(f"  {label:<44} {r['changed']:>7,}{tail}")
        lines.append("-" * 52)
        lines.append(f"  {'total values changed':<44} {frame['changed'].sum():>7,}")
        return "\n".join(lines)


# --- §1 text defects ---------------------------------------------------------

# The recoverable half of the mojibake table: double-encoded sequences map back
# to the character they should have been. The lossy corruptions ("?", the
# replacement character) are deliberately absent — you cannot recover "ü" from
# "?", and a rule that guessed would be inventing data. Those rows are left for
# the name-source preference to handle (the vendor feed keeps clean names).
_MOJIBAKE_REPAIRS = {
    "Ã©": "é", "Ã«": "ë", "Ã¼": "ü", "Ã¶": "ö",
    "Ã±": "ñ", "Ã§": "ç", "Ã¸": "ø", "â€™": "'",
}

_NULL_PLACEHOLDERS = frozenset(
    {"n/a", "na", "null", "-", "", "unknown", ".", "nan", "none"}
)

# Characters that have no place in a name. Kept in sync with mess._JUNK_CHARS.
_JUNK_CHARS = set("[]&#~!@$%^*{}|\\<>?=+_/")


def repair_mojibake(
    values: pd.Series, report: CleaningReport | None = None, column: str = ""
) -> pd.Series:
    """Reverse recoverable encoding corruption (double-encoded sequences).

    Only the reversible corruptions are touched. Lossily corrupted values (a
    name reduced to "Jos?" or containing the replacement character) are left as
    they are — inventing the missing character would be worse than flagging it.
    """
    out = values.copy()
    mask = pd.Series(False, index=out.index)
    for bad, good in _MOJIBAKE_REPAIRS.items():
        contains = out.astype(str).str.contains(bad, regex=False, na=False)
        if contains.any():
            out = out.mask(contains, out.str.replace(bad, good, regex=False))
            mask |= contains
    if report is not None:
        report.record("repair_mojibake", column, mask.sum(), "double-encoded only")
    return out


def blank_placeholders(
    values: pd.Series, report: CleaningReport | None = None, column: str = ""
) -> pd.Series:
    """Turn placeholder text ("N/A", "-", "unknown", ...) into a true null.

    A placeholder is worse than a null: it survives a dropna and is counted as a
    real value. Normalising it first is what makes every later completeness
    check honest.
    """
    stripped = values.astype("string").str.strip()
    is_placeholder = stripped.str.lower().isin(_NULL_PLACEHOLDERS)
    # A genuine NaN is already null; only count values we actually blanked.
    is_placeholder = is_placeholder.fillna(False)
    out = values.mask(is_placeholder, pd.NA)
    if report is not None:
        report.record("blank_placeholders", column, is_placeholder.sum())
    return out


def strip_junk_characters(
    values: pd.Series, report: CleaningReport | None = None, column: str = ""
) -> pd.Series:
    """Remove characters that cannot legitimately appear in a name.

    Unlike a blanket "keep only letters" rule, this removes exactly the known
    junk set and preserves apostrophes, hyphens and accented letters — so
    O'Brien and Renée survive intact.
    """
    junk_pattern = "[" + "".join("\\" + c for c in sorted(_JUNK_CHARS)) + "]"
    as_str = values.astype("string")
    has_junk = as_str.str.contains(junk_pattern, regex=True, na=False)
    out = values.mask(has_junk, as_str.str.replace(junk_pattern, "", regex=True))
    if report is not None:
        report.record("strip_junk_characters", column, has_junk.sum())
    return out


def normalise_whitespace_case(
    values: pd.Series,
    report: CleaningReport | None = None,
    column: str = "",
    case: str | None = None,
) -> pd.Series:
    """Trim surrounding whitespace and optionally normalise case.

    ``case`` may be ``"upper"``, ``"lower"``, ``"title"`` or ``None`` (trim
    only). This is what makes ``" p "`` and ``"P"`` the same group key before an
    aggregate or a join splits them apart.
    """
    as_str = values.astype("string")
    trimmed = as_str.str.strip()
    if case == "upper":
        trimmed = trimmed.str.upper()
    elif case == "lower":
        trimmed = trimmed.str.lower()
    elif case == "title":
        trimmed = trimmed.str.title()
    changed = (trimmed != as_str).fillna(False)
    out = values.mask(changed, trimmed)
    if report is not None:
        report.record("normalise_whitespace_case", column, changed.sum(), case or "trim")
    return out


if __name__ == "__main__":
    from pathlib import Path

    raw = Path(__file__).resolve().parent.parent / "data" / "raw"
    students = pd.read_csv(raw / "warehouse_students.csv", dtype=str)
    report = CleaningReport()

    students["family_name"] = strip_junk_characters(
        repair_mojibake(students["family_name"], report, "family_name"),
        report,
        "family_name",
    )
    students["given_name"] = blank_placeholders(
        repair_mojibake(students["given_name"], report, "given_name"),
        report,
        "given_name",
    )
    students["gender_code"] = normalise_whitespace_case(
        students["gender_code"], report, "gender_code", case="upper"
    )

    print(report.summary())
