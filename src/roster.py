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
STUDENTS_PER_YEAR_LEVEL = 150
TEST_YEAR = 2024

DOMAINS = ("Reading", "Numeracy", "Spelling", "Grammar and Punctuation", "Writing")

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
)
_GIVEN_NAMES = (
    "Ava", "Liam", "Mia", "Noah", "Zoe", "Aarav", "Ruby", "Kai",
    "Ivy", "Omar", "Ella", "Jack", "Aisha", "Leo", "Grace",
)
_MIDDLE_NAMES = ("", "", "", "James", "Rose", "Lee", "Grace", "May")


@dataclass(frozen=True)
class Roster:
    """The canonical truth, split into the entities the sources describe."""

    students: pd.DataFrame
    schools: pd.DataFrame
    results: pd.DataFrame


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
            "postcode": [f"{v:04d}" for v in rng.integers(2000, 2999, size=n_schools)],
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


def _build_results(rng: np.random.Generator, students: pd.DataFrame) -> pd.DataFrame:
    """One row per (student, domain) — the assessment results themselves."""
    results = students[["student_id", "year_level", "school_id", "never_sat"]].merge(
        pd.DataFrame({"domain": list(DOMAINS)}), how="cross"
    )

    n = len(results)
    # Scaled score drifts upward with year level, so aggregates are plausible and
    # a cleaning bug that mixes year levels is visible in the output.
    centre = 380 + (results["year_level"].to_numpy() - 3) * 25
    results["score"] = np.clip(rng.normal(centre, 70, size=n), 100, 800).round(1)
    results["raw_score"] = rng.integers(0, 41, size=n)
    results["test_year"] = TEST_YEAR

    # Participation: most sit, some are absent, exempt, withdrawn or refuse.
    results["participation"] = rng.choice(
        ["P", "A", "E", "W", "R"], size=n, p=[0.88, 0.06, 0.03, 0.02, 0.01]
    )
    results.loc[results["never_sat"], "participation"] = "X"

    # A score only exists where the student actually participated.
    absent = results["participation"] != "P"
    results.loc[absent, ["score", "raw_score"]] = np.nan

    return results.drop(columns="never_sat").reset_index(drop=True)


def build_roster(seed: int = SEED) -> Roster:
    """Build the canonical roster. Deterministic for a given seed."""
    rng = np.random.default_rng(seed)
    schools = _build_schools(rng)
    students = _build_students(rng, schools)
    results = _build_results(rng, students)
    return Roster(students=students, schools=schools, results=results)


if __name__ == "__main__":
    roster = build_roster()
    print(f"schools:  {len(roster.schools):>6,} rows")
    print(f"students: {len(roster.students):>6,} rows")
    print(f"results:  {len(roster.results):>6,} rows")
    print()
    print(roster.students.head(5).to_string(index=False))
