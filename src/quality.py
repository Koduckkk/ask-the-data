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


# --- LLM narrative of the cleaning report ------------------------------------
#
# The LLM narrates; it never counts. The real per-rule figures come from the
# deterministic cleaning report and are passed in — the model only phrases them.
# This mirrors the NL->SQL design: the AI transforms, the facts stay verifiable,
# and the raw report is shown alongside. With a key the summary is generated
# live; without one, a pre-written summary of the *same real numbers* is shown,
# clearly labelled — the same demo-mode pattern the query page uses.

from nl_query import MODEL  # single source of truth for the model id

_SUMMARY_SYSTEM = """You write a short executive summary of a data-cleaning run \
for a non-technical reader. You are given the exact per-rule counts of what the \
pipeline changed. Rules:
- Use ONLY the numbers provided. Never invent, estimate, or total figures \
yourself beyond what is given.
- 3-4 sentences. Lead with the scale and the largest category of fixes.
- Explain in plain terms why a couple of the fixes matter (e.g. sentinels would \
skew averages; code variants would split a group-by).
- No preamble, no bullet points, no markdown."""


def _report_lines(report_frame: pd.DataFrame) -> str:
    """The per-rule counts as compact text for the prompt (and demo grounding)."""
    agg = report_frame.groupby("rule")["changed"].sum().sort_values(ascending=False)
    lines = [f"{rule}: {n:,}" for rule, n in agg.items() if n > 0]
    lines.append(f"total values changed: {int(report_frame['changed'].sum()):,}")
    return "\n".join(lines)


# An accurate one-clause description per rule, so the demo summary never pairs a
# fixed claim with a variable rule name. If a rule isn't here, a neutral fallback
# is used — never a fabricated description.
_RULE_DESCRIPTIONS = {
    "canonicalise_code": "standardised inconsistent coded values (spellings like "
    "'NUM', 'Maths' and 'numeracy' that would otherwise split one group into several)",
    "parse_dates": "parsed dates from mixed formats into one ISO form",
    "drop_exact_duplicates": "removed exact duplicate rows",
    "parse_year_level": "unified year level (turning 'Year 9' and 9 into one value)",
    "recode_sentinels": "recoded 'no score' sentinels (999, -1) to null",
    "strip_junk_characters": "stripped junk characters from names",
    "coerce_numeric": "coerced non-numeric text out of score columns",
    "normalise_id": "normalised identifiers so the sources join",
    "repair_mojibake": "repaired encoding-corrupted names",
    "blank_placeholders": "turned placeholder text into true nulls",
    "zero_refused_scores": "zeroed refused-but-scored records",
}


def _demo_summary(report_frame: pd.DataFrame) -> str:
    """A pre-written summary grounded on the real counts (keyless fallback).

    Not a fabricated stand-in: it states the same figures the deterministic
    report computed, and each rule is paired with its OWN accurate description
    (never a fixed claim on a variable rule), so it stays correct whatever the
    top rule happens to be.
    """
    agg = report_frame.groupby("rule")["changed"].sum().sort_values(ascending=False)
    total = int(report_frame["changed"].sum())
    top_rule, top_n = agg.index[0], int(agg.iloc[0])
    top_desc = _RULE_DESCRIPTIONS.get(top_rule, f"applied the {top_rule.replace('_', ' ')} rule")
    sentinels = int(agg.get("recode_sentinels", 0))
    refused = int(agg.get("zero_refused_scores", 0))
    return (
        f"The pipeline changed {total:,} values across the raw sources. The "
        f"largest single category ({top_n:,} values) {top_desc}. It also recoded "
        f"{sentinels:,} sentinel scores (999, -1) that would have skewed every "
        f"average that touched them, and overrode {refused:,} refused-but-scored "
        f"records to zero using the authoritative participation code. Every "
        f"change is attributable to a named, tested rule."
    )


def summarise_report(report_frame: pd.DataFrame, mode: str | None = None) -> tuple[str, str]:
    """A narrative summary of the cleaning report. Returns (summary, mode).

    ``mode`` is resolved the same way as the query layer: live LLM when a key is
    present, otherwise a pre-written demo summary of the same real figures.
    """
    import nl_query  # reuse the shared mode resolution

    mode = mode or nl_query.resolve_mode()
    if mode == "demo":
        return _demo_summary(report_frame), "demo"

    import anthropic

    client = anthropic.Anthropic()
    message = client.messages.create(
        model=MODEL,
        max_tokens=400,
        system=_SUMMARY_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": f"Cleaning report counts:\n\n{_report_lines(report_frame)}",
            }
        ],
    )
    text = "".join(b.text for b in message.content if b.type == "text").strip()
    return text, "llm"


def schema_drift_demo() -> tuple[str, str | None, pd.DataFrame]:
    """Demonstrate drift detection on a real column renamed as if next year.

    Takes the participation feed, renames its id column the way a real feed might
    between years, and shows the detector suggesting the remap by value overlap.
    Returns (renamed_column, suggestion, ranked-candidates frame).
    """
    import schema_match as SM

    results = pd.read_csv(RAW_DIR / "warehouse_results.csv", dtype=str)
    reference = results["STUDENT_KEY"]

    part = pd.read_csv(RAW_DIR / "warehouse_participation.csv", dtype=str)
    renamed = "student_platform_ref"  # a plausible next-year rename
    part = part.rename(columns={"platform_student_id": renamed})

    report = SM.detect_drift(part, "student_id", "platform_student_id", reference)
    scores = pd.DataFrame(
        {
            "column": [s.candidate for s in report.scores],
            "name similarity": [s.name_similarity for s in report.scores],
            "value overlap": [s.value_overlap for s in report.scores],
            "format match": [s.format_match for s in report.scores],
            "combined": [s.combined for s in report.scores],
        }
    )
    return renamed, report.suggestion, scores


if __name__ == "__main__":
    for story in before_after_examples():
        print(f"\n=== {story.defect}  [{story.function}()] ===")
        print(story.explanation)
        print(story.examples.to_string(index=False))

    print("\n=== Refused but attempted (cross-table) ===")
    print(refused_but_attempted().to_string(index=False))
