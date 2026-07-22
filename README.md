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
python src/generate_data.py     # write messy CSVs to data/raw/
```

Further steps (clean, load, query) are added as the pipeline is built.

## Data

All data in this repo is **synthetic**, generated from a seeded RNG. Names, schools and
identifiers are invented. No real assessment data is included.

## Design notes

*(written as the pipeline is built)*

## Limitations & future work

*(written as the pipeline is built)*
