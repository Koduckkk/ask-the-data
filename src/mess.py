"""Deliberate defect injection.

Every function here takes clean values and returns damaged ones. They are the
counterpart to the rules in ``clean.py``: each injector should have a cleaning
rule that undoes it, and a test that proves it.

Three properties matter throughout:

* **Named.** ``inject_mojibake`` says what it does. A generator built from
  anonymous one-off corruptions is impossible to reason about or document.
* **Partial.** Defects hit a fraction of rows, never all of them. A defect
  present in every row is a constant, and a cleaning rule can pass by
  accident — strip a prefix that is always there and you have proven nothing.
* **Pure.** No global state; the caller supplies the ``Generator``. Two calls
  with the same inputs give the same output, so the corpus is reproducible.

Defects are drawn from patterns common to administrative data generally:
identifier drift between systems, encoding damage, placeholder text standing
in for nulls, and codes that accumulate whitespace and case variants as they
pass through spreadsheets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Fraction of eligible rows a defect touches, unless the caller says otherwise.
DEFAULT_RATE = 0.06


def _pick(rng: np.random.Generator, n: int, rate: float) -> np.ndarray:
    """Row positions to damage — at least one, never more than all of them."""
    if n == 0:
        return np.array([], dtype=int)
    k = min(n, max(1, int(round(n * rate))))
    return rng.choice(n, size=k, replace=False)


# --- text defects -------------------------------------------------------------

# Text that has been decoded with the wrong codec. "José" read as latin-1 when
# it was written as UTF-8 becomes "JosÃ©"; a codec with no mapping at all
# yields a replacement character.
_MOJIBAKE = {
    "é": ["Ã©", "?", "�"],
    "ë": ["Ã«", "?", "�"],
    "ü": ["Ã¼", "?", "�"],
    "ö": ["Ã¶", "?", "�"],
    "ñ": ["Ã±", "?", "�"],
    "ç": ["Ã§", "?", "�"],
    "ø": ["Ã¸", "?", "�"],
    "'": ["?", "�", "â€™"],
}
_ACCENTED_FORMS = {
    "Zoe": "Zoë", "Renee": "Renée", "Jose": "José", "Muller": "Müller",
    "Francois": "François", "Bjorn": "Björn", "O'Brien": "O'Brien",
}

assert all(
    any(char in accented for char in _MOJIBAKE)
    for accented in _ACCENTED_FORMS.values()
), "every accented form needs a corruptible character, or injection is a no-op"

# Each accented character has this many corruption styles: double-encoded,
# lossy "?", and the replacement character.
_MOJIBAKE_VARIANTS = 3


def _corrupt_variants(accented: str) -> list[str]:
    """Every corrupted form of one accented name, one per corruption style."""
    forms = []
    for variant in range(_MOJIBAKE_VARIANTS):
        text = accented
        for char, corruptions in _MOJIBAKE.items():
            text = text.replace(char, corruptions[variant])
        forms.append(text)
    return forms


def inject_mojibake(
    values: pd.Series, rng: np.random.Generator, rate: float = DEFAULT_RATE
) -> pd.Series:
    """Corrupt accented characters the way a codec mismatch does.

    Names are where this shows up, because names are where non-ASCII characters
    live. The damage is not random noise: it is a specific, reversible-looking
    substitution, which is why a lookup of known corruptions can repair it while
    a general "strip non-ASCII" rule destroys the name instead.
    """
    out = values.copy()
    # Only values that have an accented form can plausibly be corrupted this
    # way, so target those rather than sampling blindly — otherwise the rate is
    # silently diluted by every plain-ASCII name that cannot be damaged.
    eligible = np.flatnonzero(values.astype(str).isin(_ACCENTED_FORMS.keys()).to_numpy())
    if len(eligible) == 0:
        return out

    k = min(len(eligible), max(1, int(round(len(out) * rate))))
    targets = rng.choice(eligible, size=k, replace=False)

    # The vocabulary of names is small and the corruptions are deterministic per
    # (name, variant), so enumerate every damaged form once and index into it.
    # Corrupting row by row costs seconds on a large corpus and buys nothing.
    corrupted = {name: _corrupt_variants(accented) for name, accented in _ACCENTED_FORMS.items()}
    picks = values.iloc[targets].astype(str)
    choice = rng.integers(0, _MOJIBAKE_VARIANTS, size=k)
    out.iloc[targets] = [corrupted[name][c] for name, c in zip(picks, choice)]
    return out


# Placeholder text typed into a field that should have been left empty. These
# are worse than a true null: they survive a "drop the nulls" filter and are
# counted as real values by anything that does not know to look for them.
_NULL_PLACEHOLDERS = ("N/A", "n/a", "NA", "null", "NULL", "-", "", "unknown", ".")


def inject_null_placeholders(
    values: pd.Series, rng: np.random.Generator, rate: float = DEFAULT_RATE
) -> pd.Series:
    """Replace values with text that means "missing" without being null."""
    out = values.copy()
    idx = _pick(rng, len(out), rate)
    out.iloc[idx] = [str(v) for v in rng.choice(list(_NULL_PLACEHOLDERS), size=len(idx))]
    return out


# Characters that have no business in a name, arriving via copy-paste, bad form
# validation, or a delimiter that leaked in from an upstream format.
_JUNK_CHARS = tuple("[]&#~!@$%^*{}|\\<>?=+_/")


def inject_junk_characters(
    values: pd.Series, rng: np.random.Generator, rate: float = DEFAULT_RATE
) -> pd.Series:
    """Append or embed characters that a name validation regex should reject."""
    out = values.copy()
    idx = _pick(rng, len(out), rate)
    if len(idx) == 0:
        return out

    # np.char needs a fixed-width string dtype; pandas hands back object.
    text = values.iloc[idx].astype(str).to_numpy(dtype=np.str_)
    junk = np.array(_JUNK_CHARS)[rng.integers(0, len(_JUNK_CHARS), size=len(idx))]
    lengths = np.char.str_len(text)

    # Sometimes trailing, sometimes mid-string — a rule that only strips the
    # ends should not get full marks.
    trailing = (rng.random(len(idx)) < 0.7) | (lengths < 2)
    cuts = np.maximum(1, (rng.random(len(idx)) * np.maximum(lengths - 1, 1)).astype(int))

    damaged = np.where(
        trailing,
        np.char.add(text, junk),
        [t[:c] + j + t[c:] for t, c, j in zip(text, cuts, junk)],
    )
    out.iloc[idx] = damaged
    return out


def inject_whitespace_and_case(
    values: pd.Series, rng: np.random.Generator, rate: float = DEFAULT_RATE
) -> pd.Series:
    """Add stray padding and randomise case.

    The classic spreadsheet defect. Harmless-looking, but ``" p "`` and ``"P"``
    are different group keys, so an aggregate silently splits in two.
    """
    out = values.copy()
    idx = _pick(rng, len(out), rate)
    if len(idx) == 0:
        return out

    text = values.iloc[idx].astype(str).to_numpy(dtype=np.str_)
    roll = rng.random(len(idx))
    text = np.where(roll < 0.35, np.char.lower(text), text)
    text = np.where((roll >= 0.35) & (roll < 0.7), np.char.upper(text), text)

    lead = np.array(["", " ", "  ", "\t"])[rng.integers(0, 4, size=len(idx))]
    trail = np.array(["", " ", "  "])[rng.integers(0, 3, size=len(idx))]
    out.iloc[idx] = np.char.add(np.char.add(lead, text), trail)
    return out


# --- identifier defects -------------------------------------------------------


def strip_leading_zeros(values: pd.Series) -> pd.Series:
    """Drop leading zeros from every value, as an integer round-trip does.

    Applied wholesale rather than to a fraction, because this models a *system*
    boundary: the vendor stores the field as an integer, so every value loses
    its padding. The defect is not that some rows are wrong — it is that the
    two systems disagree about the type, and the join has to reconcile them.
    """
    return values.astype(str).str.lstrip("0").replace("", "0")


def inject_id_typos(
    values: pd.Series, rng: np.random.Generator, rate: float = 0.02
) -> pd.Series:
    """Transpose adjacent digits in a small number of identifiers.

    Unrecoverable by design: a transposed id is still well-formed, so it cannot
    be detected by validation, only by failing to match anything. It exists so
    the pipeline has genuinely unjoinable rows to report rather than silently
    drop — the cleaning report should surface them, not fix them.
    """
    out = values.copy()
    idx = _pick(rng, len(out), rate)
    if len(idx) == 0:
        return out

    text = values.iloc[idx].astype(str).to_numpy()
    draw = rng.random(len(idx))
    swapped = []
    for value, d in zip(text, draw):
        # Only swap unequal neighbours — transposing "77" changes nothing and
        # would quietly report a typo that is not there.
        positions = [j for j in range(len(value) - 1) if value[j] != value[j + 1]]
        if not positions:
            swapped.append(value)
            continue
        j = positions[int(d * len(positions))]
        swapped.append(value[:j] + value[j + 1] + value[j] + value[j + 2 :])
    out.iloc[idx] = swapped
    return out


# --- row-level defects --------------------------------------------------------


def inject_duplicate_rows(
    frame: pd.DataFrame, rng: np.random.Generator, rate: float = DEFAULT_RATE
) -> pd.DataFrame:
    """Append exact copies of existing rows.

    Re-delivered files and re-run loads are the usual cause. Exact duplicates
    are the easy case — ``drop_duplicates`` handles them — but they still need
    counting, because a load that silently doubled is a different problem from
    a source that genuinely contains repeats.
    """
    if frame.empty:
        return frame
    idx = _pick(rng, len(frame), rate)
    return pd.concat([frame, frame.iloc[idx]], ignore_index=True)


def inject_conflicting_duplicates(
    frame: pd.DataFrame,
    rng: np.random.Generator,
    key: str,
    conflict_column: str,
    n: int = 8,
) -> pd.DataFrame:
    """Duplicate rows by key, but with a differing value in one column.

    The hard case, and the reason a bare ``drop_duplicates`` is not enough:
    two rows claim the same key and disagree. Something has to decide which
    wins, and that decision has to be written down rather than left to
    whichever row the file happened to list first.
    """
    if frame.empty or key not in frame or conflict_column not in frame:
        return frame
    idx = _pick(rng, len(frame), min(1.0, n / max(len(frame), 1)))
    clashing = frame.iloc[idx].copy()
    pool = frame[conflict_column].dropna().unique()
    if len(pool) < 2:
        return frame

    # Draw a *different* value for each clash. Sampling the pool blindly lets a
    # row draw its own value back, producing a duplicate that does not actually
    # disagree — which would understate how many conflicts the corpus contains.
    replacements = []
    for current in clashing[conflict_column]:
        alternatives = [v for v in pool if v != current]
        replacements.append(alternatives[int(rng.integers(len(alternatives)))])
    clashing[conflict_column] = replacements
    return pd.concat([frame, clashing], ignore_index=True)


def inject_missing(
    values: pd.Series, rng: np.random.Generator, rate: float = DEFAULT_RATE
) -> pd.Series:
    """Blank out values entirely — a true null, not a placeholder."""
    out = values.copy()
    out.iloc[_pick(rng, len(out), rate)] = None
    return out


# --- value defects ------------------------------------------------------------

# Sentinels: in-band values standing in for a state the column cannot express.
# Deadly in a numeric column, because they aggregate. A single 999 in a mean
# moves it; a "not attempted" 9 scored as 9 marks inflates a total silently.
SCORE_SENTINELS = (999, -1, 9999)


def inject_score_sentinels(
    values: pd.Series, rng: np.random.Generator, rate: float = 0.03
) -> pd.Series:
    """Replace scores with numeric codes meaning "no score"."""
    out = values.copy()
    idx = _pick(rng, len(out), rate)
    out.iloc[idx] = rng.choice(list(SCORE_SENTINELS), size=len(idx))
    return out


def inject_text_in_numeric(
    values: pd.Series, rng: np.random.Generator, rate: float = 0.03
) -> pd.Series:
    """Put words into a numeric column, forcing the whole column to text.

    One ``"absent"`` in a score column and the entire column loads as string.
    Every downstream comparison then either errors or, worse, compares
    lexically: ``"9" > "10"``.
    """
    out = values.astype("object").copy()
    idx = _pick(rng, len(out), rate)
    out.iloc[idx] = [
        str(v) for v in rng.choice(["absent", "n/a", "-", "exempt", "ABS"], size=len(idx))
    ]
    return out


def inject_date_formats(
    values: pd.Series, rng: np.random.Generator, rate: float = 0.25
) -> pd.Series:
    """Re-render ISO dates in a mix of regional formats.

    ``03/04/2016`` is ambiguous without knowing the convention, and the two
    readings are both valid dates — so the error is invisible. Applied at a
    higher rate than other defects because mixed formats in one column are the
    norm wherever data has been touched by more than one system.
    """
    out = values.astype("object").copy()
    idx = _pick(rng, len(out), rate)
    if len(idx) == 0:
        return out

    stamps = pd.to_datetime(values.iloc[idx], errors="coerce")
    styles = ["%d/%m/%Y", "%d-%m-%Y", "%m/%d/%Y", "%d-%b-%y"]
    pick = rng.integers(0, len(styles), size=len(idx))

    # Format per style in bulk rather than per row: .dt.strftime is vectorised,
    # so four passes over subsets beats one pass per value.
    rendered = pd.Series(index=stamps.index, dtype="object")
    for s, fmt in enumerate(styles):
        mask = pick == s
        if mask.any():
            rendered.iloc[mask] = stamps.iloc[mask].dt.strftime(fmt)

    # Unparseable values keep whatever they already had.
    out.iloc[idx] = rendered.where(stamps.notna(), values.iloc[idx]).to_numpy()
    return out


# Coded values that mean the same thing but were entered differently. Any
# aggregate that groups on the raw value splits one category into several.
CODE_VARIANTS = {
    "gender": {
        "M": ["M", "m", "Male", "MALE", "1"],
        "F": ["F", "f", "Female", "FEMALE", "2"],
        "X": ["X", "Other", "Unspecified", "9"],
    },
    "participation": {
        "P": ["P", "Present", "PARTICIPATED"],
        "A": ["A", "Absent", "ABS"],
        "E": ["E", "Exempt"],
        "W": ["W", "Withdrawn"],
        "R": ["R", "Refused"],
        "X": ["X", "Not attempted"],
    },
    "domain": {
        "Reading": ["Reading", "reading", "READ", "RD"],
        "Numeracy": ["Numeracy", "numeracy", "NUM", "Maths"],
        "Spelling": ["Spelling", "spelling", "SPELL", "SP"],
        "Grammar and Punctuation": [
            "Grammar and Punctuation", "grammar_and_punctuation",
            "Grammar & Punctuation", "GANDP", "G&P",
        ],
        "Writing": ["Writing", "writing", "WRIT", "WR"],
    },
}


def inject_code_variants(
    values: pd.Series,
    rng: np.random.Generator,
    vocabulary: str,
    rate: float = 0.3,
) -> pd.Series:
    """Re-spell coded values using recognised variants of the same meaning."""
    mapping = CODE_VARIANTS[vocabulary]
    out = values.astype("object").copy()
    idx = _pick(rng, len(out), rate)
    if len(idx) == 0:
        return out

    picks = values.iloc[idx].astype("object")
    # One uniform draw per row, scaled to each canonical value's own number of
    # variants — the lists are different lengths, so a shared index would skew
    # toward whichever spellings happen to come first.
    draw = rng.random(len(idx))
    replaced = [
        mapping[v][int(d * len(mapping[v]))] if v in mapping else v
        for v, d in zip(picks, draw)
    ]
    out.iloc[idx] = replaced
    return out
