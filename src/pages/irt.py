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
    st.caption(
        f"Fitted on {n_persons:,} students. **Difficulty**: where on the ability "
        "scale an item is hardest to discriminate (higher = harder). "
        "**Discrimination**: how sharply the item separates ability levels."
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
