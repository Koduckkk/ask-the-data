"""IRT Analysis — the psychometric method behind assessment scores.

A methodology demonstration: real assessment scaled scores come from Item
Response Theory, and this page fits a 2PL IRT model to the item-level responses
in the vendor feed, showing the estimated item parameters. Framed honestly — the
data is synthetic, so this demonstrates the *workflow*, not real-world insight.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import analysis as A
import display as D


st.title("📐 IRT Analysis")
st.markdown(
    "Real assessment scaled scores aren't averages of raw marks — they come from "
    "**Item Response Theory**. This page fits a **2-parameter logistic (2PL)** "
    "IRT model to the item-level responses, estimating each item's *difficulty* "
    "and *discrimination* by marginal maximum likelihood (via the `girth` library)."
)

with st.expander("How this works — and why 2PL, not 1PL"):
    st.markdown(
        "**In plain terms.** A raw score of 8/10 means something different on an "
        "easy test than on a hard one. IRT fixes that by putting *students* and "
        "*questions* on the **same skill scale**, so a student's estimated ability "
        "doesn't depend on which particular questions they happened to get. That's "
        "why real programs (including NAPLAN) report *scaled scores*, not raw marks."
    )
    st.markdown(
        "It describes every question with two numbers:\n"
        "- **Difficulty** — how much skill it takes to have an even chance of "
        "getting it right.\n"
        "- **Discrimination** — how sharply the question separates students just "
        "below that skill level from those just above. A high-discrimination item "
        "is a clean test of skill; a low one barely tells you who's stronger."
    )

    st.markdown("---")
    st.markdown(
        "**For the technical reader.** The 2PL gives each item a difficulty $b$ "
        "and a discrimination $a$; a person has latent ability $\\theta$. The "
        "probability of a correct response is a logistic curve:"
    )
    st.latex(r"P(\text{correct}\mid\theta) = \frac{1}{1 + e^{-a(\theta - b)}}")
    st.markdown(
        "**Why 2PL and not 1PL?** The 1PL (Rasch) model drops $a$ — it *forces "
        "every item to discriminate equally* and lets items differ only in "
        "difficulty:"
    )
    st.latex(r"P(\text{correct}\mid\theta) = \frac{1}{1 + e^{-(\theta - b)}}")
    st.markdown(
        "But equal discrimination is empirically false — real items separate "
        "students at very different sharpnesses — and **discrimination is itself a "
        "useful item-quality diagnostic** (a low-$a$ item is a candidate to "
        "revise or cut). For a *descriptive* demonstration like this one, we let "
        "the data reveal $a$ rather than assuming it away. If the goal were a "
        "locked measurement scale with invariance guarantees, Rasch would be the "
        "principled choice instead — it's a trade-off, not a strict upgrade. "
        "(3PL adds a guessing floor $c$; it's unstable to estimate and mostly "
        "matters for multiple-choice, so 2PL is the conventional stopping point.)"
    )
    st.caption(
        "Fitted here by marginal maximum likelihood via the `girth` library — "
        "see `src/analysis.py`."
    )

st.caption(
    "ℹ️ Because the items were generated with near-identical difficulty, the "
    "fit correctly recovers *clustered* item parameters — the workflow is the "
    "point; on real responses the parameters would spread."
)

# --- pick response set (year level + domain) ---------------------------------

st.subheader("Choose the response set")
fits = A.available_fits()
years = sorted({yl for yl, _ in fits})
domains = sorted({d for _, d in fits})

c1, c2 = st.columns(2)
year_level = c1.selectbox("Year level", years, index=0)
domain = c2.selectbox("Domain (the response — its 0/1 item columns)", domains, index=0)

st.caption(
    "The **response** is the set of 0/1 item scores for this domain "
    "(e.g. `N3Q01`…). A student's latent ability is the covariate the model "
    "estimates; each item gets a difficulty and a discrimination."
)


@st.cache_data(show_spinner="Fitting the 2PL model…")
def _fit(year_level: int, domain: str):
    r = A.fit_2pl(year_level, domain)
    return r.items, r.ability, r.student_ids, r.n_persons, r.aic, r.bic


if st.button("Fit the 2PL model", type="primary"):
    items, ability, student_ids, n_persons, aic, bic = _fit(year_level, domain)

    st.subheader(f"Item parameters — Year {year_level} {domain}")
    st.markdown(
        f"Fitted on **{n_persons:,} students**. Two numbers describe each question:"
    )
    dcol1, dcol2 = st.columns(2)
    dcol1.info(
        "**Difficulty** — how much skill a question demands. "
        "Higher = harder (fewer students get it right).",
        icon="🎯",
    )
    dcol2.info(
        "**Discrimination** — how cleanly a question separates stronger students "
        "from weaker ones. Higher = a sharper test of skill.",
        icon="🔪",
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Students", f"{n_persons:,}")
    m2.metric("AIC", f"{aic:,.0f}")
    m3.metric("BIC", f"{bic:,.0f}")

    left, right = st.columns([2, 3])
    with left:
        st.dataframe(
            items.rename(
                columns={
                    "item": "Item",
                    "difficulty": "Difficulty",
                    "discrimination": "Discrimination",
                }
            ),
            width='stretch',
            hide_index=True,
        )
    with right:
        # Difficulty per item, as a chart — sorted so it reads as a ranking.
        chart = items.set_index("item")[["difficulty"]].rename(
            columns={"difficulty": "Difficulty"}
        )
        st.bar_chart(chart.sort_values("Difficulty"), horizontal=True)

    st.caption(
        "Because every item in this synthetic domain has essentially the same "
        "generated difficulty, the estimates cluster — exactly what a correct "
        "fit should recover from this data. On real responses the parameters "
        "would spread, and this same workflow would reveal which items are hard "
        "and which discriminate well."
    )

    # --- the person side: estimated ability (theta) --------------------------

    st.subheader("Person ability")
    st.markdown(
        "The 2PL model returns **two things**: the item parameters above, and "
        "each student's **latent ability (θ)** — recovered purely from their "
        "answer pattern. With thousands of students, the distribution is what "
        "matters, not the individual list."
    )

    import numpy as np
    import pandas as pd

    a1, a2, a3 = st.columns(3)
    def _fmt(x: float) -> str:
        # A tiny negative mean rounds to "-0.00", which reads as a glitch. The
        # `+ 0.0` collapses negative zero to positive zero after rounding.
        return f"{round(float(x), 2) + 0.0:.2f}"

    a1.metric("Mean ability", _fmt(ability.mean()))
    a2.metric("Std. dev.", _fmt(ability.std()))
    a3.metric("Range", f"{_fmt(ability.min())} to {_fmt(ability.max())}")

    # Ability clusters around the k+1 possible raw-score levels (a k-item test
    # has k+1 raw totals). Binning to about that many bins lands one cluster per
    # bar with no empty gaps; asking for more bins just subdivides the empty
    # space between clusters and looks sparse. Honest, not smoothed.
    counts, edges = np.histogram(ability, bins=len(items) + 1)
    hist = pd.DataFrame(
        {"ability": np.round((edges[:-1] + edges[1:]) / 2, 2), "students": counts}
    ).set_index("ability")
    st.bar_chart(hist)
    st.caption(
        f"Ability is centred near 0 and standardised (θ = 0 is an average "
        f"student, ±1 ≈ one standard deviation of skill). With {len(items)} "
        f"items, a student's raw score can only take {len(items) + 1} values, "
        f"so ability resolves to about that many distinct levels — a longer "
        f"test would give a smoother, finer-grained distribution."
    )

    with st.expander("A few individual ability estimates, ranked"):
        # Sort all students by ability, then pick 8 evenly spanning the range —
        # highest down to lowest — so the sample reads as a ranking and shows
        # the full span, not just the top slice.
        order = np.argsort(ability)[::-1]  # descending
        n_show = min(8, len(order))
        picks = order[np.linspace(0, len(order) - 1, n_show).astype(int)]
        sample = pd.DataFrame(
            {
                "rank": [f"{p + 1:,} of {len(order):,}" for p in
                         np.linspace(0, len(order) - 1, n_show).astype(int)],
                "PSI (student id)": student_ids[picks],
                "ability (θ)": np.round(ability[picks], 3),
            }
        )
        st.dataframe(sample, width='stretch', hide_index=True)
        st.caption(
            "Real students by platform id (PSI) — the same id that threads "
            "through the whole pipeline — sampled evenly from highest to lowest "
            "ability to show the full range, not just the top."
        )
