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

# Characters that have no place in a name. A subset of mess._JUNK_CHARS: '?' and
# the replacement character are deliberately EXCLUDED, because they double as the
# lossy-mojibake markers (é/ç/... reduced to '?' or the replacement char, which
# repair_mojibake cannot recover). Stripping them would erase the unrecoverable
# marker — turning "Ren?e" into a plausible-but-wrong "Rene" that mis-joins and
# is no longer flaggable — violating the flag-don't-guess contract. Both lossy
# markers are preserved for the vendor name to be the source of truth.
_JUNK_CHARS = set("[]&#~!@$%^*{}|\\<>=+_/")


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


# --- §2 identifier and row defects -------------------------------------------


def normalise_id(
    values: pd.Series, report: CleaningReport | None = None, column: str = ""
) -> pd.Series:
    """Reduce an identifier to a canonical join key.

    The warehouse stores ``local_id`` as a zero-padded string (``0173501``); the
    vendor stores the same value as an integer with the padding gone
    (``173501``). Neither is more correct — they simply disagree about type. The
    fix is to normalise *both* sides to the same form before joining, here by
    stripping leading zeros and surrounding whitespace so the two meet.

    A join on the raw values would silently lose every zero-padded row; this is
    what prevents that.
    """
    as_str = values.astype("string").str.strip()
    normalised = as_str.str.lstrip("0")
    # A value that was all zeros must not vanish to an empty string.
    normalised = normalised.mask((as_str.notna()) & (normalised == ""), "0")
    changed = (normalised != as_str).fillna(False)
    if report is not None:
        report.record("normalise_id", column, changed.sum())
    return normalised


def flag_unmatched_ids(
    frame: pd.DataFrame,
    key: str,
    reference_keys: set,
    report: CleaningReport | None = None,
) -> pd.DataFrame:
    """Mark rows whose id matches nothing in a reference set.

    These are the transposed-digit typos: still well-formed, so validation
    cannot catch them — they can only be found by failing to join. The rule does
    **not** try to repair them, because the original digits are unrecoverable.
    It adds an ``id_unmatched`` flag and counts them, so the report surfaces the
    rows rather than a later inner join dropping them without trace.
    """
    out = frame.copy()
    unmatched = ~out[key].isin(reference_keys)
    out["id_unmatched"] = unmatched
    if report is not None:
        report.record(
            "flag_unmatched_ids", key, unmatched.sum(), "flagged, not repaired"
        )
    return out


def drop_exact_duplicates(
    frame: pd.DataFrame, report: CleaningReport | None = None, name: str = ""
) -> pd.DataFrame:
    """Remove rows that are exact copies, counting how many fell out.

    The count matters: a load that silently doubled is a different problem from
    a source that genuinely repeats, and only the number tells them apart.
    """
    before = len(frame)
    out = frame.drop_duplicates().reset_index(drop=True)
    removed = before - len(out)
    if report is not None:
        report.record("drop_exact_duplicates", name, removed)
    return out


def resolve_conflicting_duplicates(
    frame: pd.DataFrame,
    key: str,
    prefer: str,
    ascending: bool = False,
    report: CleaningReport | None = None,
) -> pd.DataFrame:
    """Collapse rows that share a key but disagree, by a documented keep-rule.

    Exact duplicates are easy; the hard case is two rows with the same key and a
    different value in some column. Something must decide which wins, and that
    decision has to be explicit rather than left to whichever row the file
    happened to list first.

    The rule here: within a key, keep the row with the largest (or smallest)
    value of ``prefer``. It is a stand-in for a real priority — a real pipeline
    would prefer, say, the most recent record — but the point is that the choice
    is named and reproducible.
    """
    before = len(frame)
    ordered = frame.sort_values(prefer, ascending=ascending, kind="stable")
    out = ordered.drop_duplicates(subset=key, keep="first").reset_index(drop=True)
    collapsed = before - len(out)
    if report is not None:
        report.record(
            "resolve_conflicting_duplicates",
            key,
            collapsed,
            f"keep {'min' if ascending else 'max'} {prefer}",
        )
    return out


# --- §3 value defects --------------------------------------------------------

# The numeric sentinels the pipeline must treat as "no value". Kept in sync with
# mess.SCORE_SENTINELS. Both the int and float spelling, since a CSV round-trip
# turns 999 into "999.0".
SCORE_SENTINELS = frozenset({"999", "999.0", "-1", "-1.0", "9999", "9999.0"})

# Non-numeric strings that appear in the score columns.
_TEXT_NON_SCORES = frozenset({"absent", "n/a", "-", "exempt", "abs"})


def _canonical_map(vocabulary: str) -> dict[str, str]:
    """Invert mess.CODE_VARIANTS so every variant maps to its canonical value.

    Built from the injector's own table rather than hand-copied, so the cleaner
    can never fall out of step with what was injected. Matching is done on a
    stripped, lower-cased key so whitespace and case variants collapse too.
    """
    from mess import CODE_VARIANTS

    inverted = {}
    for canonical, variants in CODE_VARIANTS[vocabulary].items():
        for variant in variants:
            inverted[variant.strip().lower()] = canonical
    return inverted


def recode_sentinels(
    values: pd.Series, report: CleaningReport | None = None, column: str = ""
) -> pd.Series:
    """Replace numeric sentinels (999, -1, 9999) with a true null.

    A sentinel in a numeric column is the quiet killer: it aggregates. One 999
    left in a score column moves every mean and total that touches it, and
    nothing errors to warn you.
    """
    as_str = values.astype("string").str.strip()
    is_sentinel = as_str.isin(SCORE_SENTINELS).fillna(False)
    out = values.mask(is_sentinel, pd.NA)
    if report is not None:
        report.record("recode_sentinels", column, is_sentinel.sum())
    return out


def coerce_numeric(
    values: pd.Series, report: CleaningReport | None = None, column: str = ""
) -> pd.Series:
    """Force a column to numeric, turning non-numeric text into null.

    A single "absent" in a score column loads the whole column as text, after
    which every comparison is lexical ("9" > "10"). Coercing with errors→null
    restores a real numeric column and records how many values could not be
    parsed.
    """
    numeric = pd.to_numeric(values, errors="coerce")
    # Count values that were non-null before but became null — i.e. text that
    # could not be parsed, not values that were already missing.
    lost = (values.notna() & numeric.isna()).sum()
    if report is not None:
        report.record("coerce_numeric", column, lost, "non-numeric -> null")
    return numeric


def canonicalise_code(
    values: pd.Series,
    vocabulary: str,
    report: CleaningReport | None = None,
    column: str = "",
) -> pd.Series:
    """Map every spelling variant of a coded value to its canonical form.

    ``NUM``, ``Maths``, ``numeracy`` and ``Numeracy `` all become ``Numeracy``.
    This is the rule that stops a group-by silently splitting one category into
    five. Matching is whitespace- and case-insensitive; an unrecognised value is
    left unchanged and shows up in the report's residual count.
    """
    mapping = _canonical_map(vocabulary)
    key = values.astype("string").str.strip().str.lower()
    mapped = key.map(mapping)
    out = mapped.where(mapped.notna(), values)  # keep unknowns as-is
    changed = (out != values).fillna(False)
    unresolved = (values.notna() & mapped.isna()).sum()
    if report is not None:
        report.record(
            "canonicalise_code",
            column,
            changed.sum(),
            f"{vocabulary}; {unresolved} unresolved" if unresolved else vocabulary,
        )
    return out


def parse_year_level(
    values: pd.Series, report: CleaningReport | None = None, column: str = ""
) -> pd.Series:
    """Extract the integer year level from a mix of ``3`` and ``"Year 3"``.

    A column that mixes ``3`` and ``Year 3`` looks numeric but is not, so a join
    on it drops the string rows. Pulling the digits out gives one clean integer
    key.
    """
    extracted = values.astype("string").str.extract(r"(\d+)", expand=False)
    numeric = pd.to_numeric(extracted, errors="coerce").astype("Int64")
    changed = (numeric.astype("string") != values.astype("string")).fillna(False)
    if report is not None:
        report.record("parse_year_level", column, changed.sum())
    return numeric


def parse_dates(
    values: pd.Series, report: CleaningReport | None = None, column: str = ""
) -> pd.Series:
    """Parse mixed date formats to a single ISO date.

    The source mixes ``2016-03-14``, ``14/03/2016``, ``14-Mar-16`` and the
    ambiguous ``03/14/2016``. Values already in ISO ``YYYY-MM-DD`` form are
    parsed as-is; everything else is parsed day-first — the convention these
    feeds use — which resolves the ambiguous slash cases consistently. Applying
    day-first to an ISO date would silently swap its day and month, so the two
    are handled separately. Values that will not parse become null and counted.
    """
    as_str = values.astype("string")
    is_iso = as_str.str.match(r"^\d{4}-\d{2}-\d{2}$", na=False)

    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    if is_iso.any():
        parsed = parsed.mask(
            is_iso, pd.to_datetime(as_str.where(is_iso), errors="coerce")
        )
    if (~is_iso).any():
        parsed = parsed.mask(
            ~is_iso,
            pd.to_datetime(
                as_str.where(~is_iso), errors="coerce", dayfirst=True, format="mixed"
            ),
        )
    iso = parsed.dt.strftime("%Y-%m-%d")
    unparsed = (values.notna() & parsed.isna()).sum()
    if report is not None:
        report.record("parse_dates", column, iso.notna().sum(), f"{unparsed} unparseable")
    return iso


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
