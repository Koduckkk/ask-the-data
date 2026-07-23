# Ask the Data

> A prototype that lets non-technical staff query messy assessment data in plain English.
> Raw data flows through a documented, tested cleaning pipeline into a small database;
> users type a question ("average year 9 numeracy score by gender in 2024"); an LLM
> translates it to SQL against a documented schema; **the generated SQL is always
> displayed next to the results so a human can verify the work.** Read-only guardrails,
> offline demo mode, fully reproducible on synthetic data.

**Status: in progress.** Built incrementally — see the commit history.

## Why the mess is the point

Real assessment data does not arrive clean. It arrives as a warehouse export that
disagrees with a vendor feed about what a student's identifier is, with leading zeros
surviving in one system and stripped in the other, encoding corruption in name fields,
sentinel values standing in for "absent", and the same domain spelled three ways.

This repo ships a generator that reproduces those defects deliberately and by name, so
the cleaning pipeline has something real to earn its keep against — and so every
cleaning rule has a test with a known-dirty input.

The defect catalogue is in [docs/quirks.md](docs/quirks.md).

## Quickstart

```bash
pip install -r requirements.txt
python src/generate_data.py          # write the messy corpus to data/raw/
```

This writes ten CSVs — three warehouse tables plus a separate participation
table, four wide vendor files, and two writing files — and prints a summary of
what it wrote. The data is not committed; the generator is, and regenerating is
the intended way to get it.

```bash
python src/generate_data.py --students 400000   # production scale (~2.5s)
python src/generate_data.py --seed 7            # a different reproducible batch
```

Everything is deterministic: the same arguments always produce the same files.
The defect catalogue is in [docs/quirks.md](docs/quirks.md).

Then clean the data and load it into a small database:

```bash
python src/load.py                   # clean, reshape, load -> ask_the_data.duckdb
```

This applies every cleaning rule, folds in the wide vendor reshape, prints a
report of exactly what it changed, and writes three clean tables to DuckDB plus
an auto-generated [docs/schema.md](docs/schema.md). The cleaning is deterministic
and tested — `pytest` covers every rule, including a reconciliation check that
proves the vendor reshape reproduces the warehouse totals exactly.

```bash
pytest                               # 62 tests: dirty-in / clean-out per rule
```

Further steps (natural-language query layer) are added as the pipeline is built.

## Data

All data in this repo is **synthetic**, generated from a seeded RNG. Names, schools and
identifiers are invented. No real assessment data is included.

## Design notes

*(written as the pipeline is built)*

## Limitations & future work

*(written as the pipeline is built)*
