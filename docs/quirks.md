# Quirk catalogue

Every defect the generator injects, where it lives, how it presents, and how the
cleaning layer is expected to resolve it. This is the contract between the
generator and the pipeline: each row here should have a cleaning rule in
`clean.py` and a test that proves the rule reverses the defect.

Defects are drawn from patterns common to administrative data — identifier drift
between systems, encoding damage, placeholder text for nulls, sentinels in
numeric columns, and codes that pick up whitespace and case variants as they
pass through spreadsheets. The examples below are real values pulled from the
generated files, not illustrations.

All data is synthetic and reproducible from a seed. See the
[README](../README.md) for how it is generated.

---

## How to read this

* **Defect** — what is wrong.
* **Where** — the file and column it is injected into.
* **Injector** — the function in [`src/mess.py`](../src/mess.py) responsible.
* **Presents as** — a real example from the generated data.
* **Correct handling** — what the cleaning layer should do. "Flag, don't drop"
  means the row is kept with a reason recorded, not silently discarded.

---

## 1. Text and name defects

| # | Defect | Where | Injector | Presents as | Correct handling |
|---|--------|-------|----------|-------------|------------------|
| 1 | **Encoding corruption (mojibake)** | `warehouse_students` family/given name | `inject_mojibake` | `MÃ¼ller`, `O�Brien`, `Ren?e` | Prefer the vendor feed's clean name; repair known corruptions via lookup, never blanket-strip non-ASCII |
| 2 | **Placeholder text for null** | `warehouse_students` given name | `inject_null_placeholders` | `-`, `.`, `unknown`, `N/A`, `null` | Map the placeholder set to a true null before any completeness check counts it as a value |
| 3 | **Junk characters in names** | `warehouse_students` family/middle name | `inject_junk_characters` | `A^li`, `Kelly>`, `Zhang{` | Validate against a name pattern; flag failures, do not silently delete the character |
| 4 | **Whitespace and case chaos** | codes, school names, vendor surnames | `inject_whitespace_and_case` | `' p '`, `MOUNTAIN VIEW COLLEGE ` | `strip()` then normalise case before grouping or joining |

## 2. Identifier defects

| # | Defect | Where | Injector | Presents as | Correct handling |
|---|--------|-------|----------|-------------|------------------|
| 5 | **Leading-zero loss across systems** | vendor `LocalId` (int) vs warehouse `local_id` (`0173501`) | `strip_leading_zeros` | `173501` vs `0173501` | Normalise both sides to a common form before joining — treat the id as a string, not a number |
| 6 | **Transposed-digit id typos** | `warehouse_students` `local_id` | `inject_id_typos` | a well-formed id that matches nothing | Unrepairable by design — report the unjoinable rows, do not drop them silently |
| 7 | **Key spelled differently per table** | `STUDENT_KEY` / `platform_student_id` / `PlatformId` | (structural, in emitters) | same student, three column names | An explicit alias map, not a heuristic — nothing in the data connects them |

## 3. Row-level defects

| # | Defect | Where | Injector | Presents as | Correct handling |
|---|--------|-------|----------|-------------|------------------|
| 8 | **Exact duplicate rows** | every table | `inject_duplicate_rows` | identical rows repeated | `drop_duplicates`, but **count** them — a load that doubled is a different problem from a source that repeats |
| 9 | **Conflicting duplicates** | `warehouse_students` (same key, different school) | `inject_conflicting_duplicates` | one `STUDENT_KEY`, two `most_recent_school_id` | A documented keep-rule; the choice must be recorded, not left to row order |
| 10 | **True nulls in required fields** | `warehouse_students` gender | `inject_missing` | empty `gender_code` | Flag; decide whether the row survives downstream |
| 11 | **Orphan rows across tables** | `results` ↔ `participation` | (structural) | ~600 results with no participation, ~400 the reverse | A deliberate join type + a count of what fell out; an inner join hides ~1,000 rows |

## 4. Value defects

| # | Defect | Where | Injector | Presents as | Correct handling |
|---|--------|-------|----------|-------------|------------------|
| 12 | **Numeric sentinels** | `warehouse_results` `raw_score` | `inject_score_sentinels` | `999`, `-1`, `9999` | Recode to null before any aggregate; a single 999 moves a mean |
| 13 | **Text in a numeric column** | `warehouse_results` `scale_score` | `inject_text_in_numeric` | `absent`, `ABS`, `exempt`, `-` | Coerce with errors→null; do not let the column stay object and compare lexically |
| 14 | **Mixed date formats** | `warehouse_students` `birth_date` | `inject_date_formats` | `14/03/2016`, `03/14/2016`, `14-Mar-16` | Parse with an explicit day-first convention; the `mdy` cases are the trap |
| 15 | **Code spelling variants** | domain, gender, participation | `inject_code_variants` | `Numeracy`/`NUM`/`Maths`; `M`/`Male`/`1` | Map every variant to a canonical value via an explicit dictionary |
| 16 | **Numeric-looking year level** | `warehouse_results` `test_level` | (in `emit_results`) | `3` and `Year 3` in one column | Extract the number; a string/int mix breaks the join |
| 17 | **Lost-zero postcodes** | `warehouse_schools` `postcode` | (in `emit_schools`) | `899` where `0899` was meant | Zero-pad to width; indistinguishable from a genuine 3-digit value without the valid range |

## 5. Structural defects (the ones no single file reveals)

| # | Defect | Where | Presents as | Correct handling |
|---|--------|-------|-------------|------------------|
| 18 | **Domain encoded in the column name** | vendor `N3Q01` / `R3Q07` / `L3Q01` | wide, one row per student | Parse the header prefix to a domain, melt wide→long, sum per domain |
| 19 | **Literacy block splits by question number** | vendor `L##` columns | `L01–L06` Spelling, `L26–L31` Grammar | Split the `L` block by the question number — nothing in the file says so |
| 20 | **"Not attempted" sentinel in items** | vendor item cells | `inject`-set `9` (5,965 cells) | `9` is not a score of nine | Recode `9 → 0` before summing, or one item inflates the raw score by nine |
| 21 | **Writing in separate files, criterion-scored** | `vendor_writing_y3` / `_y579` | ten `wr_*` sub-scores per student | Assemble the domain from two files of different scope; sum the criteria |
| 22 | **Refused-but-attempted** | `results` (score) vs `participation` (`R`) | 199 refused students carrying plausible scores | The participation code wins — recode the score to zero; invisible in either file alone |

---

## Reconciliation invariants

These are the properties that make the corpus *checkable* — a correct pipeline
should reproduce them, and a bug shows up as a violation:

* **Vendor items sum to the warehouse raw score exactly**, once the `9` sentinel
  is recoded and the `L` block is split. Verified across 176,100 (student,
  domain) pairs. Before the reshape, a naive join reconciles ~88%.
* **Writing criterion sub-scores sum to the warehouse writing raw score
  exactly.** Verified across 44,097 students.
* **Scaled score and proficiency are derivable** from raw score and year level
  via the lookup in [`src/roster.py`](../src/roster.py). A raw/scaled pair that
  is not in the lookup, or a band that disagrees with its score, is a genuine
  defect rather than generator noise.

---

## Coverage checklist

Confirm the generated data actually exercises each defect:

- [x] 1 mojibake — `inject_mojibake` on warehouse names; vendor kept clean
- [x] 2 null placeholders — `inject_null_placeholders` on given name
- [x] 3 junk characters — `inject_junk_characters`
- [x] 4 whitespace/case — `inject_whitespace_and_case`
- [x] 5 leading-zero loss — `strip_leading_zeros` on vendor `LocalId`
- [x] 6 id typos — `inject_id_typos`, deliberately unrepairable
- [x] 7 key drift — three spellings across the emitters
- [x] 8 exact duplicates — `inject_duplicate_rows` everywhere
- [x] 9 conflicting duplicates — `inject_conflicting_duplicates`
- [x] 10 true nulls — `inject_missing` on gender
- [x] 11 orphans — 599 / 399 across results and participation
- [x] 12 score sentinels — `999`, `-1`, `9999`
- [x] 13 text in numeric — `absent`, `exempt`, ...
- [x] 14 mixed dates — four formats including the ambiguous `mdy`
- [x] 15 code variants — domain, gender, participation
- [x] 16 numeric-looking year level — `Year 3` alongside `3`
- [x] 17 lost-zero postcodes
- [x] 18 domain-in-header — vendor wide format
- [x] 19 `L`-block split by question number
- [x] 20 not-attempted `9` sentinel in vendor items
- [x] 21 writing as separate criterion-scored files
- [x] 22 refused-but-attempted cross-table contradiction
