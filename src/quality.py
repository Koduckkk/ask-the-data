"""Data-quality showcase — make the cleaning visible.

The generator, the cleaning rules, and the reconciliation proof are the most
distinctive work in this repo, and until now they lived in ``docs/`` and a
terminal report. This module surfaces them: for each defect, it pulls real
dirty values from the raw CSVs and shows exactly what the cleaning rule turns
them into — the pipeline's judgement, on real records, side by side.

Nothing here re-implements cleaning. Each before/after applies the *actual*
rule from ``clean.py`` to the raw column, so the "after" is precisely what the
pipeline produces, not a hand-written illustration.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

import clean as C
from emit_warehouse import RAW_DIR


@dataclass(frozen=True)
class BeforeAfter:
    """One defect's story: a few real dirty -> clean example rows.

    ``function`` names the actual ``clean.py`` rule that produced the "after"
    column, so a reviewer can trace each transformation to its source — the QA
    point of the page: every change attributable to a named, tested function.
    """

    defect: str          # human name of the defect
    explanation: str     # one line: what the rule does and why
    examples: pd.DataFrame  # columns: before, after
    function: str = ""   # e.g. "clean.repair_mojibake"


def _pairs(raw: pd.Series, cleaned: pd.Series, limit: int = 5) -> pd.DataFrame:
    """Rows where cleaning actually changed the value, de-duplicated.

    A value that was cleaned to null (a placeholder → NA) is a real change and
    must be shown, so nulls are rendered as an explicit "(null)" rather than
    dropped by a NaN comparison.
    """
    raw_str = raw.astype("string")
    clean_str = cleaned.astype("string")
    # A change is: the raw value existed and the cleaned value differs — where
    # "differs" includes becoming null. Compare on a null-safe filled form.
    changed = raw_str.notna() & (clean_str.fillna("\x00") != raw_str.fillna("\x00"))
    frame = pd.DataFrame(
        {
            "before": raw_str[changed],
            "after": clean_str[changed].fillna("(null)"),
        }
    )
    return frame.drop_duplicates().head(limit).reset_index(drop=True)


def _qualname(fn) -> str:
    """A short 'clean.<function>' reference for display, from the callable itself."""
    return f"clean.{fn.__name__}"


def before_after_examples(limit: int = 5) -> list[BeforeAfter]:
    """Real dirty -> clean pairs for each defect, drawn from the raw data.

    Each story names the exact ``clean.py`` function that produced its "after"
    column, derived from the callable so the reference can never drift from the
    code that actually ran.
    """
    students = pd.read_csv(RAW_DIR / "warehouse_students.csv", dtype=str)
    results = pd.read_csv(RAW_DIR / "warehouse_results.csv", dtype=str)

    def story(defect, explanation, raw, fn, *args):
        cleaned = fn(raw, *args)
        return BeforeAfter(
            defect, explanation, _pairs(raw, cleaned.astype("string"), limit), _qualname(fn)
        )

    stories = [
        story(
            "Encoding corruption (mojibake)",
            "Double-encoded characters are repaired via lookup; lossy ones "
            "(reduced to '?') are left for the clean vendor name, not guessed.",
            students["family_name"], C.repair_mojibake,
        ),
        story(
            "Placeholder text → null",
            "'N/A', '-', 'unknown' and friends become true nulls, so they stop "
            "being counted as real values.",
            students["given_name"], C.blank_placeholders,
        ),
        story(
            "Junk characters stripped",
            "Characters that can't appear in a name are removed; apostrophes, "
            "hyphens and accents survive.",
            students["family_name"], C.strip_junk_characters,
        ),
        story(
            "Gender codes canonicalised",
            "M/F/X, Male/Female, 1/2 and blanks all map to a single canonical "
            "code, so a group-by doesn't split one category into five.",
            students["gender_code"], C.canonicalise_code, "gender",
        ),
        story(
            "Domain names canonicalised",
            "NUM / Maths / numeracy / 'Numeracy ' all become 'Numeracy'.",
            results["test_domain"], C.canonicalise_code, "domain",
        ),
        story(
            "Score sentinels → null",
            "999 / -1 / 9999 are 'no score' codes, not marks — recoded to null "
            "before any average touches them.",
            results["raw_score"], C.recode_sentinels,
        ),
        story(
            "Mixed date formats → ISO",
            "14/03/2016, 14-Mar-16 and 03/14/2016 are parsed day-first to one "
            "ISO date; values already in ISO form pass through unchanged.",
            students["birth_date"], C.parse_dates,
        ),
        story(
            "Year level unified",
            "'Year 9' and 9 are the same value; the digits are extracted so the "
            "join key is one clean integer.",
            results["test_level"], C.parse_year_level,
        ),
        story(
            "Identifier normalised for joining",
            "The warehouse stores local_id zero-padded ('0173501'); the vendor "
            "strips it. Both are normalised so the join lines up.",
            students["local_id"], C.normalise_id,
        ),
    ]

    # Keep only stories that actually found examples.
    return [s for s in stories if not s.examples.empty]


# --- refused-but-attempted: the cross-table contradiction --------------------


def refused_but_attempted(limit: int = 5) -> pd.DataFrame:
    """The signature defect: refused students who nonetheless carry a score.

    This one is invisible in any single file — the score looks legitimate; only
    the disagreement with the participation table reveals it. The correct
    handling is to trust the participation code and zero the score.
    """
    results = pd.read_csv(RAW_DIR / "warehouse_results.csv", dtype=str)
    participation = pd.read_csv(RAW_DIR / "warehouse_participation.csv", dtype=str)

    # Canonicalise both sides so the join and the code comparison line up.
    results["student_id"] = C.normalise_id(results["STUDENT_KEY"])
    results["domain"] = C.canonicalise_code(results["test_domain"], "domain")
    participation["student_id"] = C.normalise_id(participation["platform_student_id"])
    participation["domain"] = C.canonicalise_code(participation["domain_name"], "domain")
    participation["code"] = C.canonicalise_code(
        participation["participation_code"], "participation"
    )

    refused = participation[participation["code"] == "R"][["student_id", "domain"]]
    joined = results.merge(refused, on=["student_id", "domain"])
    # A plausible score (not a sentinel) that a refusal should override to zero.
    scored = joined[
        joined["raw_score"].notna()
        & ~joined["raw_score"].isin(["999.0", "-1.0", "9999.0"])
    ]
    out = scored[["student_id", "domain", "raw_score"]].drop_duplicates().head(limit)
    out = out.rename(
        columns={"raw_score": "score_in_results"}
    ).reset_index(drop=True)
    out["participation_code"] = "R (refused)"
    out["corrected_score"] = "0"
    return out


if __name__ == "__main__":
    for story in before_after_examples():
        print(f"\n=== {story.defect}  [{story.function}()] ===")
        print(story.explanation)
        print(story.examples.to_string(index=False))

    print("\n=== Refused but attempted (cross-table) ===")
    print(refused_but_attempted().to_string(index=False))
