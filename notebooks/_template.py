# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # NN · Title — one line on the business question
#
# Copy this file to `notebooks/NN_short_name.py` to start a new analysis. State the decision this
# analysis informs up front; end with **Takeaways**. Keep the narrative
# load → inspect → analyze → visualize, and delegate every non-trivial step to a named, tested
# function in `core` (promote anything reusable).

# %%
from core.config import ROOT
from core.prelude import *

RAW = ROOT / "data" / "raw"

# %% [markdown]
# ## 1. Load
# One typed loader per source — or register it once and `catalog.load("name")`. Scan lazily
# (`scan_parquet`) when the data is big.

# %%
# df = read_parquet(RAW / "<source>.parquet")

# %% [markdown]
# ## 2. Inspect
# Shape, dtypes, missingness, cardinality — know the data before trusting it.

# %%
# stats.summary(df)
# stats.missingness(df)

# %% [markdown]
# ## 3. Analyze

# %%


# %% [markdown]
# ## 4. Visualize
# Compose static charts with `base.grid(...)`; use `interactive.*` (Plotly) where hover/zoom helps.

# %%
# fig, axes = base.grid(2)

# %% [markdown]
# **Takeaways:** what we learned, and the decision it supports.
