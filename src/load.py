"""Assemble the cleaned tables, load them into DuckDB, and document the schema.

This is where the cleaning rules stop being a library and become a pipeline. It
reads the raw CSVs, applies the rules from ``clean.py`` and the reshape from
``reshape.py`` in order, and writes three clean tables to a DuckDB file:

    students   one row per student, deduplicated, names repaired
    results    one row per (student, domain), scores tidy and reconciled,
               refused scores overridden to zero, vendor reshape folded in
    schools    reference data, real schools only

Every step reports what it changed to a shared ``CleaningReport``, so running
this prints exactly what the pipeline did. It also writes ``schema.md`` — the
table and column documentation that both a human and (later) the NL→SQL prompt
read to understand the database.

    python src/load.py            # build ask_the_data.duckdb + docs/schema.md

The DuckDB file is gitignored and rebuilt from the raw CSVs, which are
themselves regenerated from the seed. Nothing in the data path is committed;
the code that produces it is.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

import clean as C
import reshape as R
from emit_warehouse import RAW_DIR

DB_PATH = Path(__file__).resolve().parent.parent / "ask_the_data.duckdb"
SCHEMA_DOC = Path(__file__).resolve().parent.parent / "docs" / "schema.md"

# Pseudo-schools and the home-school code — real records flow through with these
# ids, but they are not schools and must be excluded from school-level reporting.
# Mirrors the roster's PSEUDO_SCHOOL_IDS / HOME_SCHOOL_ID.
NON_SCHOOL_IDS = frozenset(
    {"00000", "99999", "REGION-A", "REGION-B", "unassigned", "88000"}
)


def clean_students(report: C.CleaningReport) -> pd.DataFrame:
    """Read and clean the warehouse student table."""
    df = pd.read_csv(RAW_DIR / "warehouse_students.csv", dtype=str)

    df["family_name"] = C.strip_junk_characters(
        C.repair_mojibake(df["family_name"], report, "family_name"),
        report, "family_name",
    )
    df["given_name"] = C.blank_placeholders(
        C.repair_mojibake(df["given_name"], report, "given_name"),
        report, "given_name",
    )
    df["middle_name"] = C.strip_junk_characters(df["middle_name"], report, "middle_name")
    df["gender_code"] = C.canonicalise_code(df["gender_code"], "gender", report, "gender_code")
    df["birth_date"] = C.parse_dates(df["birth_date"], report, "birth_date")
    df["year_level"] = C.parse_year_level(df["most_recent_test_level"], report, "year_level")
    df["student_id"] = C.normalise_id(df["STUDENT_KEY"], report, "student_id")
    df["enrolment_id"] = C.normalise_id(df["local_id"], report, "enrolment_id")

    tidy = df[[
        "student_id", "enrolment_id", "year_level", "most_recent_school_id",
        "family_name", "given_name", "middle_name", "gender_code", "birth_date",
    ]].rename(columns={"most_recent_school_id": "school_id", "gender_code": "gender"})

    # Same key, different school is a genuine conflict; keep one row per student
    # by a documented rule (the alphabetically-first school id, as a stand-in for
    # a real recency priority), after removing exact duplicates.
    tidy = C.drop_exact_duplicates(tidy, report, "students")
    tidy = C.resolve_conflicting_duplicates(
        tidy, key="student_id", prefer="school_id", ascending=True, report=report
    )
    return tidy


def clean_results(report: C.CleaningReport) -> pd.DataFrame:
    """Read and clean the warehouse results, fold in the vendor reshape."""
    df = pd.read_csv(RAW_DIR / "warehouse_results.csv", dtype=str)

    df["domain"] = C.canonicalise_code(df["test_domain"], "domain", report, "domain")
    df["year_level"] = C.parse_year_level(df["test_level"], report, "year_level")
    df["student_id"] = C.normalise_id(df["STUDENT_KEY"], report, "student_id")
    df["raw_score"] = C.coerce_numeric(
        C.recode_sentinels(df["raw_score"], report, "raw_score"), report, "raw_score"
    )
    df["scaled_score"] = C.coerce_numeric(
        C.recode_sentinels(df["scale_score"], report, "scaled_score"), report, "scaled_score"
    )

    df["test_year"] = C.coerce_numeric(df["cal_year"], report, "test_year")

    tidy = df[[
        "student_id", "year_level", "school_id", "domain",
        "raw_score", "scaled_score", "proficiency_band", "test_year",
    ]].rename(columns={"proficiency_band": "proficiency"})
    tidy["test_year"] = tidy["test_year"].astype("Int64")
    tidy = C.drop_exact_duplicates(tidy, report, "results")

    # Clean the participation table and apply the authoritative refused-override.
    participation = clean_participation(report)
    tidy = R.zero_refused_scores(tidy, participation, report=report)

    return tidy


def clean_participation(report: C.CleaningReport) -> pd.DataFrame:
    """Read and clean the separate participation table."""
    df = pd.read_csv(RAW_DIR / "warehouse_participation.csv", dtype=str)
    df["student_id"] = C.normalise_id(df["platform_student_id"], report, "part.student_id")
    df["domain"] = C.canonicalise_code(df["domain_name"], "domain", report, "part.domain")
    df["participation_code"] = C.canonicalise_code(
        df["participation_code"], "participation", report, "participation_code"
    )
    tidy = df[["student_id", "domain", "participation_code", "cal_year"]]
    return C.drop_exact_duplicates(tidy, report, "participation")


def clean_schools(report: C.CleaningReport) -> pd.DataFrame:
    """Read and clean the school reference table, dropping non-schools."""
    df = pd.read_csv(RAW_DIR / "warehouse_schools.csv", dtype=str)
    df["school_name"] = C.normalise_whitespace_case(
        C.strip_junk_characters(df["school_name"], report, "school_name"),
        report, "school_name", case="title",
    )
    df["sector_code"] = C.blank_placeholders(df["sector_code"], report, "sector_code")
    # Zero-pad postcodes back to four digits where a leading zero was lost.
    df["postcode"] = df["postcode"].astype("string").str.zfill(4)

    tidy = df.rename(columns={"acara_id": "school_id", "sector_code": "sector"})[
        ["school_id", "school_name", "sector", "postcode"]
    ]
    tidy = C.drop_exact_duplicates(tidy, report, "schools")

    # Drop pseudo- and home-school ids from the reference table — they are not
    # schools and must not appear in school-level reporting.
    before = len(tidy)
    tidy = tidy[~tidy["school_id"].isin(NON_SCHOOL_IDS)].reset_index(drop=True)
    report.record("drop_non_schools", "school_id", before - len(tidy), "pseudo/home")
    return tidy


def build_database(db_path: Path = DB_PATH) -> tuple[dict[str, pd.DataFrame], C.CleaningReport]:
    """Run the full cleaning pipeline and load the clean tables into DuckDB."""
    report = C.CleaningReport()
    tables = {
        "students": clean_students(report),
        "results": clean_results(report),
        "schools": clean_schools(report),
    }

    db_path.unlink(missing_ok=True)  # rebuild from scratch, never append
    con = duckdb.connect(str(db_path))
    for name, frame in tables.items():
        con.register(f"_{name}", frame)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM _{name}")
        con.unregister(f"_{name}")
    con.close()
    return tables, report


# --- schema documentation ----------------------------------------------------

# One-line description per column, so schema.md documents meaning, not just type.
_COLUMN_NOTES = {
    "student_id": "canonical student identifier (normalised across sources)",
    "enrolment_id": "local enrolment id (leading zeros normalised)",
    "year_level": "year level: 3, 5, 7 or 9",
    "school_id": "school identifier (ACARA id)",
    "family_name": "family name (encoding repaired, junk stripped)",
    "given_name": "given name",
    "middle_name": "middle name (may be empty)",
    "gender": "gender: M, F or X",
    "birth_date": "date of birth (ISO yyyy-mm-dd)",
    "domain": "assessment domain: Reading, Numeracy, Spelling, Grammar and Punctuation, Writing",
    "raw_score": "raw score (sum of item marks); null if not participated",
    "scaled_score": "scaled score; refused students overridden to 0",
    "proficiency": "proficiency band",
    "test_year": "calendar year of the test",
    "participation_code": "participation: P, A, E, W, R or X",
    "school_name": "school name",
    "sector": "school sector: GOV, CATH or IND",
    "postcode": "school postcode (four digits)",
}


def write_schema_doc(
    tables: dict[str, pd.DataFrame], db_path: Path = DB_PATH, out: Path = SCHEMA_DOC
) -> None:
    """Auto-generate schema.md from the loaded tables.

    This doc is both user documentation and, later, the grounding the NL→SQL
    prompt is given so the LLM writes queries against real column names. It is
    generated, not hand-maintained, so it can never drift from the database.
    """
    con = duckdb.connect(str(db_path), read_only=True)
    lines = [
        "# Database schema",
        "",
        "Auto-generated by `src/load.py` from the loaded DuckDB tables. Do not",
        "edit by hand — it is regenerated on every load and is the grounding the",
        "NL→SQL layer reads to write queries against real column names.",
        "",
    ]
    for name in tables:
        info = con.execute(f"PRAGMA table_info('{name}')").fetchdf()
        rows = con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
        lines += [f"## `{name}`  ({rows:,} rows)", "", "| column | type | description |", "|---|---|---|"]
        for _, col in info.iterrows():
            note = _COLUMN_NOTES.get(col["name"], "")
            lines.append(f"| `{col['name']}` | {col['type']} | {note} |")
        lines.append("")
    con.close()
    out.write_text("\n".join(lines))


if __name__ == "__main__":
    tables, report = build_database()
    write_schema_doc(tables)

    print(report.summary())
    print()
    print(f"Loaded {len(tables)} tables into {DB_PATH.name}:")
    for name, frame in tables.items():
        print(f"  {name:12s} {len(frame):>8,} rows")
    print(f"\nSchema written to {SCHEMA_DOC.relative_to(SCHEMA_DOC.parent.parent)}")
