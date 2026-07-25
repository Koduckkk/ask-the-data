"""Shared vendor item-column parsing — one source of truth for the L-block split.

The vendor feed encodes the domain in the column name (``N3Q01``, ``R3Q07``,
``L3Q26``). Three modules need to parse those headers — ``emit_vendor`` writes
them, ``reshape`` sums them, ``analysis`` fits IRT on them — so the parsing and,
critically, the literacy-block split (``L01-06`` Spelling, ``L26-31`` Grammar)
live here once and are derived from ``roster.ITEMS_PER_DOMAIN``. Hardcoding the
ranges separately in each module let them silently disagree if item counts ever
changed.
"""

from __future__ import annotations

import re

from roster import ITEMS_PER_DOMAIN

# A vendor item column: prefix (N/R/L), year level, question number.
ITEM_RE = re.compile(r"^([NRL])(\d+)Q(\d+)$")

# The combined literacy block covers two domains, told apart only by question
# number. Spelling starts at 1; Grammar starts at 26 (a gap the real feed uses).
GRAMMAR_START = 26
SPELLING_QUESTIONS = range(1, 1 + ITEMS_PER_DOMAIN["Spelling"])
GRAMMAR_QUESTIONS = range(
    GRAMMAR_START, GRAMMAR_START + ITEMS_PER_DOMAIN["Grammar and Punctuation"]
)
_SPELLING_Q = set(SPELLING_QUESTIONS)
_GRAMMAR_Q = set(GRAMMAR_QUESTIONS)


def domain_of(column: str) -> str | None:
    """Map an item column name to its domain, or None if it is not an item.

    ``N`` and ``R`` are one domain each; ``L`` splits by question number — the
    only thing in the file that distinguishes Spelling from Grammar.
    """
    match = ITEM_RE.match(column)
    if not match:
        return None
    prefix, _year, question = match.group(1), match.group(2), int(match.group(3))
    if prefix == "N":
        return "Numeracy"
    if prefix == "R":
        return "Reading"
    if question in _SPELLING_Q:
        return "Spelling"
    if question in _GRAMMAR_Q:
        return "Grammar and Punctuation"
    return None  # an L column outside both ranges — not expected, not summed
