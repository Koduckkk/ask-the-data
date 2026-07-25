# Ask the Data

> A prototype that lets non-technical staff query messy assessment data in plain English.
> Raw data flows through a documented, tested cleaning pipeline into a small database;
> users type a question ("average year 9 numeracy score by gender in 2024"); an LLM
> translates it to SQL against a documented schema; **the generated SQL is always
> displayed next to the results so a human can verify the work.** Read-only guardrails,
> offline demo mode, fully reproducible on synthetic data.

Built incrementally — see the commit history.

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
pytest                               # dirty-in / clean-out per rule + guardrails
```

Then ask questions in plain English:

```bash
streamlit run src/app.py             # text box, results, and the SQL beside them
# or, from the command line:
python src/nl_query.py "average writing score by year level"
```

The app has five pages, ordered so the sidebar reads as a narrative — the
data-science work first, then the interactive query, then the cleaning that
underpins it all:

- **Anomaly Detection** — the operational "what needs a second look": **school
  effects** with empirical-Bayes shrinkage (small schools pulled toward the mean
  by their unreliability — the principled version of a hard n-cutoff), and
  **marker-anomaly detection** that flags harsh/lenient markers after controlling
  for student ability via their other-domain scores, ranked as a review queue
  ("give me the top N markers to investigate").
- **IRT Analysis** — a methodology demonstration: real assessment scaled scores
  come from Item Response Theory, so this page fits a 2-parameter logistic (2PL)
  IRT model to the item-level responses and shows the estimated item difficulty
  and discrimination, plus each student's latent ability. Framed honestly — the
  data is synthetic, so it demonstrates the *workflow*, not real-world insight.
- **Statistical Insights** — the inference layer: is a gender gap real or noise?
  (a difference with a 95% confidence interval and a plain-English verdict), and a
  note connecting a cleaning step to its inferential consequence — skipping the
  sentinel recode biases the mean by ~125 points.
- **Ask the Data** — the interactive query page: ask a question in plain English,
  see the generated SQL beside the results.
- **Data Quality** — the cleaning made visible: raw dirty values next to their
  cleaned form for each defect (each labelled with the `clean.py` function that
  produced it), a count of every change, the cross-table "refused but attempted"
  contradiction that no single file reveals, and a **schema-drift detector** that
  catches a source renaming its id column between years and suggests the remap by
  value overlap (proposing, never silently re-joining on a guessed key).

With no API key this runs in **demo mode** — a set of canned questions — so the
whole pipeline is runnable without one. Set `ANTHROPIC_API_KEY` (copy
`.env.example` to `.env`) to ask free-form questions; the LLM translates them to
SQL against the schema, and either way the generated SQL is shown next to the
result. Results are charted automatically when the shape suits one (a trend over
year levels as a line, a breakdown by category as bars), with a humanised table
alongside — while the SQL panel keeps the literal column names you verify against.

## Example questions

Each of these runs in demo mode, no key required:

- *average year 9 numeracy score by gender*
- *average scaled score by domain*
- *proficiency band distribution for year 5 reading*
- *top 10 schools by average numeracy score*
- *how many students sat, by school sector*
- *average writing score by year level*

## Data

All data in this repo is **synthetic**, generated from a seeded RNG. Names, schools and
identifiers are invented. No real assessment data is included.

## Design notes

**Verification by design — the SQL is always shown.** The LLM appears in exactly
one place ([`nl_query.py`](src/nl_query.py)) and does exactly one thing: turn a
question into SQL. That step can be wrong, so its output is never trusted — it is
validated, run read-only, and **displayed next to the result** so a person can
check it. The tool treats the model as a translator whose work is shown for human
verification, not as an oracle.

**Cleaning is deterministic; only translation is not.** Everything before the
query — generate, clean, reshape, load — is plain Python with tests. Cleaning has
one correct answer, so it is code, not a model. The AI is confined to the one
task that genuinely has no single right answer (natural language → SQL), and even
there a human stays in the loop. This split is the whole architecture.

**Guardrails, not trust.** [`guardrails.py`](src/guardrails.py) is the boundary
between "the model wrote SQL" and "we ran it": comments are stripped, only a
single `SELECT`/`WITH` is allowed, every write/DDL/`ATTACH`/`PRAGMA` keyword is
rejected, and the result is row-limited. A rejected query is shown with its
reason, never executed. There is a test for a question that resolves to
`DROP TABLE` — it is caught before execution and the table survives.

**The cleaning report tells you what it did.** [`load.py`](src/load.py) prints a
per-rule tally of every value it changed — sentinels recoded, names repaired,
refused scores zeroed, duplicates dropped. The pipeline is auditable by
construction rather than a black box.

**Reconciliation as a correctness proof.** The generator builds the vendor feed
so its item scores sum to the warehouse totals exactly. After the reshape, that
equality is asserted across every (student, domain) pair — a cleaning bug shows
up as a failed reconciliation, not a silently wrong number. Most pipelines cannot
prove their reshape is correct; this one does.

**Why synthetic data.** The mess patterns are drawn from real experience with
assessment data, but every value here is invented from a seed. That keeps the
repo reproducible on any machine, free of confidentiality concerns, and honest —
the defect catalogue ([docs/quirks.md](docs/quirks.md)) says exactly what was
injected and where.

## Limitations & future work

This is a prototype, and it is scoped as one:

- **Demo mode answers a fixed set of questions.** Free-form querying needs an API
  key. The canned set covers the common patterns; it is not a full NL surface.
- **NL → SQL is single-shot.** There is no clarification loop for an ambiguous
  question, and no retry when a generated query fails — the tool shows the SQL
  and the error and asks the user to rephrase.
- **No evaluation harness** for NL→SQL accuracy. A next step would be a labelled
  set of question→SQL pairs scored automatically.
- **Cleaning rules are code, not config.** A production version would lift them
  into declarative, dbt-style rule definitions so non-engineers could review and
  extend them.
- **Continuously-updated data is out of scope.** The real feeds arrive
  incrementally (a value missing today may populate tomorrow); this repo models a
  single snapshot.

## Author

Built as a portfolio piece demonstrating a documented, tested cleaning pipeline
with an AI query layer kept honest by design.

## License

[MIT](LICENSE) — free to use, modify, and share with attribution.
