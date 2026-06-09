# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Example analysis
#
# Open this `.py` in Jupyter (jupytext pairs it to an `.ipynb`). Work here interactively;
# promote anything reusable into the `core` package and import it back. Commit the `.py`.

# %%
import polars as pl

from core.config import get_settings

settings = get_settings()
settings.environment

# %% [markdown]
# ## A tiny synthetic example (runs with no database)

# %%
df = pl.DataFrame(
    {
        "segment": ["retail", "corporate", "retail", "wealth", "corporate"],
        "revenue": [120.0, 980.5, 200.0, 5400.0, 760.0],
    }
)
df.group_by("segment").agg(
    pl.col("revenue").sum().alias("revenue"),
    pl.len().alias("customers"),
).sort("revenue", descending=True)
