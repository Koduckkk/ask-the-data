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
    """One defect's story: a few real dirty -> clean example rows."""

    defect: str          # human name of the defect
    explanation: str     # one line: what the rule does and why
    examples: pd.DataFrame  # columns: before, after


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


def before_after_examples(limit: int = 5) -> list[BeforeAfter]:
    """Real dirty -> clean pairs for each defect, drawn from the raw data."""
    students = pd.read_csv(RAW_DIR / "warehouse_students.csv", dtype=str)
    results = pd.read_csv(RAW_DIR / "warehouse_results.csv", dtype=str)
    schools = pd.read_csv(RAW_DIR / "warehouse_schools.csv", dtype=str)

    stories: list[BeforeAfter] = []

    # 1. Encoding corruption in names.
    stories.append(
        BeforeAfter(
            "Encoding corruption (mojibake)",
            "Double-encoded characters are repaired via lookup; lossy ones "
            "(reduced to '?') are left for the clean vendor name, not guessed.",
            _pairs(students["family_name"], C.repair_mojibake(students["family_name"]), limit),
        )
    )

    # 2. Placeholder text standing in for null.
    stories.append(
        BeforeAfter(
            "Placeholder text → null",
            "'N/A', '-', 'unknown' and friends become true nulls, so they stop "
            "being counted as real values.",
            _pairs(students["given_name"], C.blank_placeholders(students["given_name"]), limit),
        )
    )

    # 3. Junk characters in names.
    stories.append(
        BeforeAfter(
            "Junk characters stripped",
            "Characters that can't appear in a name are removed; apostrophes, "
            "hyphens and accents survive.",
            _pairs(students["family_name"], C.strip_junk_characters(students["family_name"]), limit),
        )
    )

    # 4. Coded values with many spellings.
    stories.append(
        BeforeAfter(
            "Gender codes canonicalised",
            "M/F/X, Male/Female, 1/2 and blanks all map to a single canonical "
            "code, so a group-by doesn't split one category into five.",
            _pairs(
                students["gender_code"],
                C.canonicalise_code(students["gender_code"], "gender"),
                limit,
            ),
        )
    )

    # 5. Domain spelled many ways.
    stories.append(
        BeforeAfter(
            "Domain names canonicalised",
            "NUM / Maths / numeracy / 'Numeracy ' all become 'Numeracy'.",
            _pairs(
                results["test_domain"],
                C.canonicalise_code(results["test_domain"], "domain"),
                limit,
            ),
        )
    )

    # 6. Numeric sentinels in the score column.
    stories.append(
        BeforeAfter(
            "Score sentinels → null",
            "999 / -1 / 9999 are 'no score' codes, not marks — recoded to null "
            "before any average touches them.",
            _pairs(
                results["raw_score"],
                C.recode_sentinels(results["raw_score"]).astype("string"),
                limit,
            ),
        )
    )

    # 7. Mixed date formats.
    stories.append(
        BeforeAfter(
            "Mixed date formats → ISO",
            "14/03/2016, 14-Mar-16 and 03/14/2016 are parsed day-first to one "
            "ISO date.",
            _pairs(students["birth_date"], C.parse_dates(students["birth_date"]), limit),
        )
    )

    # 8. Numeric-looking year level.
    stories.append(
        BeforeAfter(
            "Year level unified",
            "'Year 9' and 9 are the same value; the digits are extracted so the "
            "join key is one clean integer.",
            _pairs(
                results["test_level"],
                C.parse_year_level(results["test_level"]).astype("string"),
                limit,
            ),
        )
    )

    # 9. Leading-zero id drift.
    stories.append(
        BeforeAfter(
            "Identifier normalised for joining",
            "The warehouse stores local_id zero-padded ('0173501'); the vendor "
            "strips it. Both are normalised so the join lines up.",
            _pairs(students["local_id"], C.normalise_id(students["local_id"]), limit),
        )
    )

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
        print(f"\n=== {story.defect} ===")
        print(story.explanation)
        print(story.examples.to_string(index=False))

    print("\n=== Refused but attempted (cross-table) ===")
    print(refused_but_attempted().to_string(index=False))
