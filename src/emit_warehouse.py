"""Emit the warehouse tables — the "internal" source, dirtied per table.

Three tables land in ``data/raw/``: students, results and schools. Each is
derived from the same canonical roster and then damaged *independently*, which
is the point. A student's family name can be mojibaked in the student table and
clean in the vendor feed; a school id can be whitespace-padded in one place and
not the other. That disagreement between sources is what the cleaning pipeline
has to reconcile, and it cannot be reproduced by dirtying one shared frame and
copying it around.

Column names here deliberately differ from the vendor feed's names for the same
concepts — see ``docs/quirks.md``. The mapping is not guessable from the data,
which is why the cleaning layer needs an explicit alias table rather than a
clever heuristic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import mess as M
from roster import Roster, build_roster

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def emit_students(roster: Roster, rng: np.random.Generator) -> pd.DataFrame:
    """Student demographics, as the warehouse spells them."""
    students = roster.students.rename(
        columns={
            "student_id": "STUDENT_KEY",
            "enrolment_id": "local_id",
            "year_level": "most_recent_test_level",
            "school_id": "most_recent_school_id",
            "family_name": "family_name",
            "given_name": "given_name",
            "middle_name": "middle_name",
            "gender": "gender_code",
            "date_of_birth": "birth_date",
        }
    ).drop(columns=["never_sat"])

    # Names: encoding damage, placeholder text, and junk characters. Applied to
    # the warehouse copy only — the vendor feed keeps clean names, so the
    # pipeline has a reason to prefer one source over the other.
    students["family_name"] = M.inject_junk_characters(
        M.inject_mojibake(students["family_name"], rng), rng
    )
    students["given_name"] = M.inject_null_placeholders(
        M.inject_mojibake(students["given_name"], rng), rng
    )
    students["middle_name"] = M.inject_junk_characters(students["middle_name"], rng)

    # Dates arrive in whatever format the contributing system used.
    students["birth_date"] = M.inject_date_formats(students["birth_date"], rng)

    # Gender coded inconsistently: M/F/X alongside Male/Female, 1/2 and blanks.
    students["gender_code"] = M.inject_code_variants(students["gender_code"], rng, "gender")
    students["gender_code"] = M.inject_missing(students["gender_code"], rng, rate=0.02)

    # A student whose id is mistyped cannot be matched to anything. Left
    # deliberately unrepairable so the cleaning report has to surface it.
    students["local_id"] = M.inject_id_typos(students["local_id"], rng)

    students = M.inject_duplicate_rows(students, rng)
    # Same key, different school — a genuine conflict the dedup rule must
    # resolve rather than silently keeping whichever row came first.
    students = M.inject_conflicting_duplicates(
        students, rng, key="STUDENT_KEY", conflict_column="most_recent_school_id", n=40
    )
    return students


def emit_results(roster: Roster, rng: np.random.Generator) -> pd.DataFrame:
    """Assessment results, one row per student and domain."""
    results = roster.results.rename(
        columns={
            "student_id": "STUDENT_KEY",
            "year_level": "test_level",
            "school_id": "school_id",
            "domain": "test_domain",
            "raw_score": "raw_score",
            "scaled_score": "scale_score",
            "proficiency": "proficiency_band",
            "participation": "participation_code",
            "test_year": "cal_year",
        }
    )

    # Domain and participation both accumulate spelling and case variants as
    # they pass between systems.
    results["test_domain"] = M.inject_code_variants(results["test_domain"], rng, "domain")
    # Participation is NOT here — it lives in its own table, keyed on a
    # differently-spelled student id. See emit_participation().
    results = results.drop(columns=["participation_code"])

    # Refused students who nonetheless have a score here. The participation
    # code is authoritative: a refusal scores zero regardless of what this
    # table claims, so these rows are the artefact and must not be believed.
    #
    # This is the one defect that cannot be spotted from a single file. The
    # score is well-formed and in range; only the disagreement with the
    # participation table reveals it — which is precisely what makes it worth
    # demonstrating.
    stray = np.flatnonzero((roster.results["participation"] == "R").to_numpy())
    contradicted = np.array([], dtype=int)
    if len(stray):
        contradicted = rng.choice(stray, size=min(200, len(stray)), replace=False)
        results.loc[results.index[contradicted], "raw_score"] = rng.integers(
            5, 30, size=len(contradicted)
        )
        results.loc[results.index[contradicted], "scale_score"] = np.round(
            rng.uniform(300, 550, size=len(contradicted)), 1
        )

    # Sentinels and text in the score columns. Both force the column to object
    # dtype and both aggregate as if they were real marks.
    #
    # Held clear of the contradicted rows above: a refused row carrying 999 is
    # just a sentinel, and the interesting case is a refused row carrying a
    # score that looks entirely legitimate.
    keep = np.ones(len(results), dtype=bool)
    keep[contradicted] = False
    positions = np.flatnonzero(keep)

    # copy=True because pandas hands back a read-only view under copy-on-write.
    raw = results["raw_score"].astype("object").to_numpy(copy=True)
    raw[positions] = M.inject_score_sentinels(
        results["raw_score"].iloc[positions], rng
    ).to_numpy()
    results["raw_score"] = raw

    scaled = results["scale_score"].astype("object").to_numpy(copy=True)
    scaled[positions] = M.inject_text_in_numeric(
        results["scale_score"].iloc[positions], rng
    ).to_numpy()
    results["scale_score"] = scaled

    # Year level as "Year 9" as well as 9 — a join key that looks numeric but
    # is not, in a fraction of rows.
    level = results["test_level"].astype("object")
    idx = rng.choice(len(level), size=int(len(level) * 0.05), replace=False)
    level.iloc[idx] = "Year " + level.iloc[idx].astype(str)
    results["test_level"] = level

    return M.inject_duplicate_rows(results, rng)


def emit_participation(roster: Roster, rng: np.random.Generator) -> pd.DataFrame:
    """Participation codes, in their own table with their own spelling of the key.

    Separating this from results is what the real feeds do, and it changes the
    problem. Participation is no longer a column to tidy — it is a join, and the
    join key is spelled ``platform_student_id`` here against ``STUDENT_KEY``
    next door. Nothing in either file says they are the same concept.

    The tables also do not line up row for row. Some students appear here with
    no result, some results have no participation row at all. That mismatch is
    the point: an inner join silently drops both, and only a deliberate join
    choice plus a count of what fell out will show it.
    """
    participation = roster.results[
        ["student_id", "domain", "participation", "test_year"]
    ].rename(
        columns={
            "student_id": "platform_student_id",
            "domain": "domain_name",
            "participation": "participation_code",
            "test_year": "cal_year",
        }
    )

    # Whitespace first, then spelling variants. The other order compounds:
    # every spelling would gain every padding combination, exploding a
    # six-value vocabulary into hundreds of distinct strings.
    participation["participation_code"] = M.inject_whitespace_and_case(
        participation["participation_code"], rng, rate=0.04
    )
    participation["participation_code"] = M.inject_code_variants(
        participation["participation_code"], rng, "participation", rate=0.2
    )
    participation["domain_name"] = M.inject_code_variants(
        participation["domain_name"], rng, "domain", rate=0.15
    )

    # Orphans on the participation side: registered for a test that produced no
    # result row.
    extra = participation.sample(n=400, random_state=int(rng.integers(1e6))).copy()
    extra["platform_student_id"] = [
        f"Z{int(v):09d}Q" for v in rng.integers(1e8, 9.9e8, size=len(extra))
    ]
    participation = pd.concat([participation, extra], ignore_index=True)

    # Orphans on the results side: drop some participation rows so their
    # results have nothing to join to.
    drop = rng.choice(len(participation), size=600, replace=False)
    participation = participation.drop(participation.index[drop]).reset_index(drop=True)

    return M.inject_duplicate_rows(participation, rng, rate=0.03)


def emit_schools(roster: Roster, rng: np.random.Generator) -> pd.DataFrame:
    """School reference data."""
    schools = roster.schools.rename(
        columns={
            "school_id": "acara_id",
            "school_name": "school_name",
            "sector": "sector_code",
            "postcode": "postcode",
        }
    ).drop(columns=["is_real_school"])

    schools["school_name"] = M.inject_whitespace_and_case(schools["school_name"], rng)
    schools["school_name"] = M.inject_junk_characters(schools["school_name"], rng, rate=0.03)

    # Postcodes that lost a leading zero on their way through a numeric column.
    # Three digits where four are expected — and nothing in the value says
    # whether it was ever four, so the fix needs to know the valid range.
    # Targeted at the ones that actually start with a zero: stripping a value
    # that never had one is a no-op that inflates the apparent defect rate.
    postcodes = schools["postcode"].astype("object")
    eligible = np.flatnonzero(postcodes.astype(str).str.startswith("0").to_numpy())
    if len(eligible):
        hit = rng.choice(eligible, size=max(1, int(len(eligible) * 0.6)), replace=False)
        postcodes.iloc[hit] = postcodes.iloc[hit].astype(str).str.lstrip("0")
    schools["postcode"] = postcodes

    schools["sector_code"] = M.inject_null_placeholders(schools["sector_code"], rng, rate=0.05)
    return M.inject_duplicate_rows(schools, rng, rate=0.04)


def write_warehouse(roster: Roster, rng: np.random.Generator, out_dir: Path = RAW_DIR) -> dict[str, Path]:
    """Emit all three warehouse tables and return where they landed."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "students": emit_students(roster, rng),
        "results": emit_results(roster, rng),
        "participation": emit_participation(roster, rng),
        "schools": emit_schools(roster, rng),
    }
    written = {}
    for name, frame in tables.items():
        path = out_dir / f"warehouse_{name}.csv"
        frame.to_csv(path, index=False)
        written[name] = path
    return written


if __name__ == "__main__":
    roster = build_roster()
    rng = np.random.default_rng(11)
    for name, path in write_warehouse(roster, rng).items():
        size = path.stat().st_size / 1e6
        print(f"{name:10s} -> {path.name:28s} {size:6.1f} MB")
