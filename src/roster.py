"""Build the canonical roster — the clean truth that every source table
derives from, and then corrupts independently.

The point of a single roster is that the *same* student can appear in the
warehouse and in the vendor feed with different identifiers, different name
spellings and different formatting, while still having one correct answer
behind them. Without a canonical truth there is nothing to reconcile against,
and no way to assert that a cleaning rule recovered the right value.

Nothing in this module injects mess. It produces the ideal, and
``mess.py`` degrades it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# --- configuration -----------------------------------------------------------

SEED = 20260722

YEAR_LEVELS = (3, 5, 7, 9)

# ~50k students total. Large enough that query performance and join strategy
# actually matter — a corpus of a few hundred rows hides both — while still
# generating in well under a second. Real cohorts run several times this; the
# generator handles 400k in a couple of seconds if you raise this.
STUDENTS_PER_YEAR_LEVEL = 12_500

TEST_YEAR = 2024

DOMAINS = ("Reading", "Numeracy", "Spelling", "Grammar and Punctuation", "Writing")

# Items per domain in the vendor feed. Literacy arrives as one "L" block that
# has to be split by question number: L01-L06 are Spelling, L26-L31 are Grammar
# and Punctuation. Nothing in the file says so — it is inferred from the number.
ITEMS_PER_DOMAIN = {
    "Reading": 8,
    "Numeracy": 8,
    "Spelling": 6,
    "Grammar and Punctuation": 6,
    "Writing": 10,
}
MAX_RAW_SCORE = {d: n for d, n in ITEMS_PER_DOMAIN.items()}

# Proficiency bands, lowest to highest.
PROFICIENCY_BANDS = (
    "Needs Additional Support",
    "Developing",
    "Approaching Expectations",
    "Strong",
    "Exceeding",
)

# Cut points are expressed as offsets from each year level's own scale floor,
# not as absolute scaled scores. Every year level sits on a different scale, so
# a fixed set of absolute cuts would grade Year 9 against a Year 3 yardstick and
# put nearly the whole cohort in the top band.
PROFICIENCY_CUT_OFFSETS = (60.0, 110.0, 165.0, 215.0)

# Each year level sits on its own scale, so the same raw score means something
# different in Year 3 and Year 9.
SCALE_SPAN = 260.0

# Per-domain profile: real assessment domains do not sit on one shared scale or
# have one shared difficulty. Each domain gets its own scale offset (so a scaled
# score means something different across domains) and its own mean difficulty (so
# "average score by domain" actually varies). Values are deliberate, not random,
# so the corpus stays reproducible.
#   scale_offset: added to the domain's scaled-score floor (points)
#   difficulty:   shift to the mean proportion-correct (+ easier, - harder)
DOMAIN_PROFILE = {
    "Reading":                 {"scale_offset":  15.0, "difficulty":  0.04},
    "Numeracy":                {"scale_offset":  35.0, "difficulty":  0.02},
    "Spelling":                {"scale_offset":   0.0, "difficulty":  0.08},
    "Grammar and Punctuation": {"scale_offset": -10.0, "difficulty": -0.02},
    "Writing":                 {"scale_offset": -30.0, "difficulty": -0.06},
}


def scale_floor(year_level: int, domain: str | None = None) -> float:
    """Bottom of the scaled-score range for a year level (and optional domain).

    Passing a domain shifts the floor by that domain's scale offset, so the same
    raw score maps to a different scaled score across domains — as it does in a
    real assessment, where domain scales are not interchangeable.
    """
    base = 250.0 + (year_level - 3) * 45.0
    if domain is None:
        return base
    return base + DOMAIN_PROFILE[domain]["scale_offset"]


def _logistic(p: float, steepness: float = 6.0) -> float:
    """S-curve on [0, 1], centred at 0.5."""
    return 1.0 / (1.0 + np.exp(-steepness * (p - 0.5)))

# Pseudo-schools: administrative placeholders that appear in the source data but
# are not real schools. Records carrying these must be excluded from reporting.
PSEUDO_SCHOOL_IDS = ("00000", "99999", "REGION-A", "REGION-B", "unassigned")

# Home schooling is a real cohort — students flow through with this code, but it
# is not a school and must not appear in school-level aggregates.
HOME_SCHOOL_ID = "88000"

_SCHOOL_PREFIXES = (
    "Riverside", "Mountain View", "Sunnybank", "Grange", "Holy Family",
    "St Mark's", "Parkfield", "Eastwood", "Northbridge", "Lakeview",
)
_SCHOOL_SUFFIXES = ("Primary School", "High School", "College", "Public School")

_FAMILY_NAMES = (
    "Smith", "Nguyen", "O'Brien", "Zhang", "Patel", "Lee", "Brown",
    "Kaur", "Ali", "Wilson", "Taylor", "Singh", "Chen", "Murphy", "Kelly",
    "Muller",
)
# Zoe, Renee, Jose, Francois and Bjorn are here on purpose: they have accented
# forms, so they are the names that encoding corruption can plausibly damage.
_GIVEN_NAMES = (
    "Ava", "Liam", "Mia", "Noah", "Zoe", "Aarav", "Ruby", "Kai",
    "Ivy", "Omar", "Ella", "Jack", "Aisha", "Leo", "Grace",
    "Renee", "Jose", "Francois", "Bjorn",
)
_MIDDLE_NAMES = ("", "", "", "James", "Rose", "Lee", "Grace", "May")


@dataclass(frozen=True)
class Roster:
    """The canonical truth, split into the entities the sources describe."""

    students: pd.DataFrame
    schools: pd.DataFrame
    results: pd.DataFrame
    score_map: pd.DataFrame


def build_score_map(year_levels=YEAR_LEVELS) -> pd.DataFrame:
    """Raw score -> scaled score lookup, one row per (year level, domain, raw).

    Scaled scores are not independent of raw scores: the same raw score on the
    same test form always maps to the same scaled score. Modelling this as a
    lookup rather than a second random draw is what makes the relationship
    *checkable* — a cleaning bug that shuffles rows or mismatches a join shows
    up as a raw/scaled pair that does not appear in this table.

    The curve is deliberately non-linear — shallow at both ends and steep in
    the middle, where an extra correct answer actually discriminates between
    students. A naive linear back-calculation therefore does not reproduce it,
    so the lookup has to genuinely be used.
    """
    rows = []
    for year_level in year_levels:
        span = SCALE_SPAN
        for domain, max_raw in MAX_RAW_SCORE.items():
            # Each domain sits on its own scale (via the domain-aware floor), so
            # the same raw score maps to a different scaled score per domain.
            floor = scale_floor(year_level, domain)
            # Logistic curve, rescaled so raw 0 maps to the floor and max raw
            # maps to floor + span exactly.
            lo, hi = _logistic(0.0), _logistic(1.0)
            for raw in range(max_raw + 1):
                p = raw / max_raw
                shaped = (_logistic(p) - lo) / (hi - lo)
                rows.append(
                    {
                        "year_level": year_level,
                        "domain": domain,
                        "raw_score": raw,
                        "scaled_score": round(floor + shaped * span, 1),
                    }
                )
    return pd.DataFrame(rows)


def assign_proficiency(scaled: pd.Series, year_level: pd.Series) -> pd.Series:
    """Derive the proficiency band from the scaled score and year level.

    Derived, never drawn independently — so a record whose band disagrees with
    its score is a genuine inconsistency for the pipeline to catch, rather than
    noise the generator invented.

    Banding is relative to the year level's own scale: a scaled score of 400 is
    a strong Year 3 result and a weak Year 9 one.
    """
    floor = year_level.map(scale_floor).astype(float)
    offset = scaled.astype(float) - floor

    band = pd.Series(PROFICIENCY_BANDS[0], index=scaled.index, dtype="object")
    for cut, label in zip(PROFICIENCY_CUT_OFFSETS, PROFICIENCY_BANDS[1:]):
        band = band.mask(offset >= cut, label)
    return band.mask(scaled.isna(), None)


def _make_student_ids(rng: np.random.Generator, n: int) -> list[str]:
    """Warehouse student identifier: letter + 9 digits + letter.

    Deliberately not a bare integer — the vendor feed keys on something else
    entirely, and an opaque string makes accidental numeric coercion visible.
    """
    digits = rng.integers(10**8, 10**9, size=n)
    head = rng.choice(list("ABCDEFGHJKLMNPRSTUVWXYZ"), size=n)
    tail = rng.choice(list("ABCDEFGHJKLMNPRSTUVWXYZ"), size=n)
    return [f"{h}{d:09d}{t}" for h, d, t in zip(head, digits, tail)]


def _make_enrolment_ids(rng: np.random.Generator, n: int) -> list[str]:
    """Local enrolment id, zero-padded to 7 characters.

    The leading zero is load-bearing: the vendor feed stores this same value as
    an integer, so ``0173501`` and ``173501`` are the same student. Any join
    that does not normalise loses these rows silently.

    Only some ids have a leading zero — drawing from a range that spans the
    width boundary keeps it an edge case. If every id started with a zero, a
    rule that merely stripped a constant prefix would pass by accident.
    """
    return [f"{v:07d}" for v in rng.integers(100_000, 9_999_999, size=n)]


def _build_schools(rng: np.random.Generator, n_schools: int = 40) -> pd.DataFrame:
    """Real schools, plus the pseudo- and home-school codes that must be excluded."""
    ids = [f"{v:05d}" for v in rng.choice(np.arange(10_000, 89_999), size=n_schools, replace=False)]
    names = [
        f"{rng.choice(_SCHOOL_PREFIXES)} {rng.choice(_SCHOOL_SUFFIXES)}"
        for _ in range(n_schools)
    ]
    schools = pd.DataFrame(
        {
            "school_id": ids,
            "school_name": names,
            "sector": rng.choice(["GOV", "CATH", "IND"], size=n_schools, p=[0.65, 0.2, 0.15]),
            # Some postcodes genuinely begin with a zero. They are the ones that
            # lose it to a numeric column somewhere upstream, so the corpus
            # needs them or that defect has nothing to act on.
            "postcode": [
                f"{v:04d}"
                for v in np.where(
                    rng.random(n_schools) < 0.25,
                    rng.integers(800, 999, size=n_schools),
                    rng.integers(2000, 2999, size=n_schools),
                )
            ],
            "is_real_school": True,
        }
    )

    placeholders = pd.DataFrame(
        {
            "school_id": [*PSEUDO_SCHOOL_IDS, HOME_SCHOOL_ID],
            "school_name": [
                "Unallocated", "Test School", "Region A Office",
                "Region B Office", "Unassigned", "Home Schooling",
            ],
            "sector": ["NA"] * len(PSEUDO_SCHOOL_IDS) + ["HOME"],
            "postcode": [""] * (len(PSEUDO_SCHOOL_IDS) + 1),
            "is_real_school": False,
        }
    )
    return pd.concat([schools, placeholders], ignore_index=True)


def _build_students(rng: np.random.Generator, schools: pd.DataFrame) -> pd.DataFrame:
    """One row per student, spread across year levels."""
    real_ids = schools.loc[schools["is_real_school"], "school_id"].to_numpy()

    frames = []
    for year_level in YEAR_LEVELS:
        n = STUDENTS_PER_YEAR_LEVEL
        # Date of birth consistent with the year level, so an implausible age is
        # a genuine defect rather than an artefact of the generator.
        birth_year = TEST_YEAR - year_level - 5
        frames.append(
            pd.DataFrame(
                {
                    "student_id": _make_student_ids(rng, n),
                    "enrolment_id": _make_enrolment_ids(rng, n),
                    "year_level": year_level,
                    "school_id": rng.choice(real_ids, size=n),
                    "family_name": rng.choice(_FAMILY_NAMES, size=n),
                    "given_name": rng.choice(_GIVEN_NAMES, size=n),
                    "middle_name": rng.choice(_MIDDLE_NAMES, size=n),
                    "gender": rng.choice(["M", "F", "X"], size=n, p=[0.49, 0.49, 0.02]),
                    "date_of_birth": [
                        f"{birth_year}-{m:02d}-{d:02d}"
                        for m, d in zip(
                            rng.integers(1, 13, size=n), rng.integers(1, 29, size=n)
                        )
                    ],
                }
            )
        )

    students = pd.concat(frames, ignore_index=True)
    students = students.drop_duplicates(subset="student_id").reset_index(drop=True)

    # A small cohort sits at pseudo-schools or is home schooled. These are not
    # errors in the source — they are real records that reporting must handle.
    n_pseudo, n_home = 12, 8
    picks = rng.choice(len(students), size=n_pseudo + n_home, replace=False)
    students.loc[picks[:n_pseudo], "school_id"] = rng.choice(
        list(PSEUDO_SCHOOL_IDS), size=n_pseudo
    )
    students.loc[picks[n_pseudo:], "school_id"] = HOME_SCHOOL_ID

    # A cohort registered but never sat — every result should be "did not attend".
    students["never_sat"] = False
    students.loc[rng.choice(len(students), size=10, replace=False), "never_sat"] = True

    return students


def _build_results(
    rng: np.random.Generator,
    students: pd.DataFrame,
    score_map: pd.DataFrame,
    test_year: int,
) -> pd.DataFrame:
    """One row per (student, domain) — the assessment results themselves.

    Raw score is the primitive; scaled score and proficiency are *derived* from
    it. That ordering matters: it means the three columns agree by construction
    here, so any disagreement in the emitted sources is mess that was injected
    deliberately, not an accident of the generator.
    """
    results = students[["student_id", "year_level", "school_id", "never_sat"]].merge(
        pd.DataFrame({"domain": list(DOMAINS)}), how="cross"
    )

    n = len(results)
    results["test_year"] = test_year

    # Participation: most sit, some are absent, exempt, withdrawn or refuse.
    results["participation"] = rng.choice(
        ["P", "A", "E", "W", "R"], size=n, p=[0.88, 0.06, 0.03, 0.02, 0.01]
    )
    results.loc[results["never_sat"], "participation"] = "X"

    # Ability is a per-student trait, so a student who reads well tends to also
    # spell well. Without this every domain is independent and student-level
    # aggregates are uninteresting.
    ability = pd.Series(
        rng.normal(0, 1, size=len(students)), index=students["student_id"]
    )
    max_raw = results["domain"].map(MAX_RAW_SCORE).to_numpy()
    z = ability.reindex(results["student_id"]).to_numpy() + rng.normal(0, 0.6, size=n)
    # Each domain has its own mean difficulty, so cohorts genuinely score
    # differently across domains rather than identically.
    difficulty = results["domain"].map(
        lambda d: DOMAIN_PROFILE[d]["difficulty"]
    ).to_numpy()
    # Squash the latent ability onto [0, 1], shifted by domain difficulty, and
    # scale to the domain's item count.
    proportion = np.clip(0.5 + z * 0.18 + difficulty, 0.02, 0.98)
    results["raw_score"] = np.rint(proportion * max_raw).astype(int)

    # Scaled score comes from the lookup — never recomputed independently.
    results = results.merge(
        score_map, on=["year_level", "domain", "raw_score"], how="left"
    )
    results["proficiency"] = assign_proficiency(
        results["scaled_score"], results["year_level"]
    )

    # Anyone who did not participate has no result at all. Note this is *not*
    # the refused-but-attempted case: a refusal (R) legitimately scores zero,
    # and stray item rows claiming otherwise are the artefact, not the truth.
    # That contradiction is injected later, in the emitted sources.
    absent = results["participation"] != "P"
    results.loc[absent, ["raw_score", "scaled_score"]] = np.nan
    results.loc[absent, "proficiency"] = None

    return results.drop(columns="never_sat").reset_index(drop=True)


def build_roster(seed: int = SEED, test_year: int = TEST_YEAR) -> Roster:
    """Build the canonical roster. Deterministic for a given (seed, year).

    ``test_year`` is threaded through rather than read from the module constant
    so later work can emit several years from one call site. Varying the seed
    with the year keeps each year's cohort distinct.
    """
    rng = np.random.default_rng(seed + test_year)
    schools = _build_schools(rng)
    students = _build_students(rng, schools)
    score_map = build_score_map()
    results = _build_results(rng, students, score_map, test_year)
    return Roster(
        students=students, schools=schools, results=results, score_map=score_map
    )


if __name__ == "__main__":
    roster = build_roster()
    print(f"schools:   {len(roster.schools):>6,} rows")
    print(f"students:  {len(roster.students):>6,} rows")
    print(f"results:   {len(roster.results):>6,} rows")
    print(f"score_map: {len(roster.score_map):>6,} rows")
    print()
    print(roster.results.head(6).to_string(index=False))
    print()
    print(roster.results["proficiency"].value_counts().to_string())
