"""Generate small, reproducible sample datasets for the example notebooks.

Run:  python scripts/make_sample_data.py

Everything is seeded (``SEED``), so re-running produces byte-identical files — ``data/`` is a
rebuildable cache (gitignored) and this script is the source of truth. Three datasets in one
retail-bank commercial domain so they join up:

  - transactions.csv    ~5,000 order lines, deliberately *messy* (missing values, string dates,
                        inconsistent casing, duplicate rows, a few revenue outliers) — for the
                        cleaning / EDA / KPI notebooks.
  - customers.parquet    5,000 customers with a churn target that has real signal plus a small
                        treatment-group effect — for modeling, segmentation and the A/B notebook.
  - daily_sales.parquet ~6 years of daily revenue (trend + weekly & yearly seasonality + a
                        December bump) — for the forecasting notebook.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl

from core.config import ROOT
from core.io.writers import write_csv, write_parquet

SEED = 42  # matches config/config.yaml run.seed
RAW = ROOT / "data" / "raw"

SEGMENTS = ["retail", "sme", "corporate", "wealth"]
REGIONS = ["NA", "EU", "UK", "APAC"]
CATEGORIES = ["loans", "cards", "savings", "advisory", "insurance"]
CHANNELS = ["web", "mobile", "branch", "partner"]
PLANS = ["basic", "standard", "premium"]
N_CUSTOMERS = 5000
CUSTOMER_IDS = np.arange(1000, 1000 + N_CUSTOMERS)  # 1000..5999, shared with transactions


def make_transactions(rng: np.random.Generator) -> pl.DataFrame:
    """~5,000 order lines with controlled, realistic mess for the cleaning notebook."""
    n = 5000
    order_id = np.arange(1, n + 1)
    start = date(2023, 1, 1)
    order_date = [start + timedelta(days=int(d)) for d in rng.integers(0, 730, n)]
    customer_id = rng.choice(CUSTOMER_IDS, size=n)
    segment: list[str | None] = rng.choice(SEGMENTS, n, p=[0.55, 0.25, 0.15, 0.05]).tolist()
    region = rng.choice(REGIONS, n).tolist()
    product_category = rng.choice(CATEGORIES, n).tolist()
    channel = rng.choice(CHANNELS, n, p=[0.40, 0.30, 0.20, 0.10]).tolist()
    units = rng.integers(1, 10, n)
    unit_price: list[float | None] = np.round(rng.gamma(3.0, 55.0, n) + 8.0, 2).tolist()
    discount_raw = np.where(rng.random(n) < 0.35, rng.choice([0.05, 0.1, 0.15, 0.2], n), 0.0)
    discount: list[float | None] = np.round(discount_raw, 2).tolist()
    revenue: list[float | None] = [
        round(u * p * (1 - d), 2) for u, p, d in zip(units, unit_price, discount, strict=True)
    ]
    is_returned = (rng.random(n) < 0.08).astype(int)

    # --- inject deliberate mess (the cleaning notebook fixes each of these) ---
    for i in rng.choice(n, size=int(0.08 * n), replace=False):  # inconsistent casing/whitespace
        base = segment[i] or "retail"
        segment[i] = str(rng.choice([base.upper(), base.capitalize() + " ", " " + base]))
    for i in rng.choice(n, size=int(0.03 * n), replace=False):  # missing category
        segment[i] = None
    for i in rng.choice(n, size=int(0.02 * n), replace=False):  # missing numerics
        unit_price[i] = None
    for i in rng.choice(n, size=int(0.02 * n), replace=False):
        discount[i] = None
    for i in rng.choice(n, size=12, replace=False):  # a handful of revenue outliers
        if revenue[i] is not None:
            revenue[i] = round(revenue[i] * float(rng.uniform(15, 40)), 2)

    df = pl.DataFrame(
        {
            "order_id": order_id,
            "order_date": order_date,
            "customer_id": customer_id,
            "segment": segment,
            "region": region,
            "product_category": product_category,
            "channel": channel,
            "units": units,
            "unit_price": unit_price,
            "discount": discount,
            "revenue": revenue,
            "is_returned": is_returned,
        }
    )
    duplicates = df.sample(n=50, seed=SEED)  # exact duplicate rows to dedupe later
    return pl.concat([df, duplicates]).sample(fraction=1.0, shuffle=True, seed=SEED)


def make_customers(rng: np.random.Generator) -> pl.DataFrame:
    """5,000 customers with a churn target driven by tenure, spend, tickets, plan and group."""
    n = N_CUSTOMERS
    reference = date(2025, 1, 1)
    signup_days_ago = rng.integers(30, 5 * 365, n)
    signup_date = [reference - timedelta(days=int(d)) for d in signup_days_ago]
    tenure_months = np.round(signup_days_ago / 30.0).astype(int)
    age = rng.integers(18, 86, n)
    region = rng.choice(REGIONS, n)
    segment = rng.choice(SEGMENTS, n, p=[0.55, 0.25, 0.15, 0.05])
    plan = rng.choice(PLANS, n, p=[0.50, 0.35, 0.15])
    num_products = rng.integers(1, 7, n)
    sessions_30d = rng.poisson(8, n)
    support_tickets = rng.poisson(1.2, n)
    plan_mult = np.select(
        [plan == "basic", plan == "standard", plan == "premium"], [1.0, 2.2, 4.5], default=1.0
    )
    monthly_spend: list[float | None] = np.round(rng.gamma(2.0, 40.0, n) * plan_mult, 2).tolist()
    satisfaction: list[float | None] = (
        np.clip(np.round(rng.normal(7.0, 1.8, n)), 1, 10).astype(float).tolist()
    )
    group = rng.choice(["control", "treatment"], n)

    # churn data-generating process: genuine drivers + a small treatment effect (for the A/B test)
    logit = (
        -1.0
        - 0.020 * tenure_months
        + 0.30 * support_tickets
        - 0.05 * sessions_30d
        - 0.0010 * np.asarray([s if s is not None else 80.0 for s in monthly_spend])
        - 0.18 * (np.asarray([s if s is not None else 7.0 for s in satisfaction]) - 7.0)
        - 0.25 * (group == "treatment")
        + 0.30 * (plan == "basic")
        + rng.normal(0, 0.4, n)
    )
    churned = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)

    for i in rng.choice(n, size=int(0.02 * n), replace=False):  # a little missingness to impute
        monthly_spend[i] = None
    for i in rng.choice(n, size=int(0.02 * n), replace=False):
        satisfaction[i] = None

    return pl.DataFrame(
        {
            "customer_id": CUSTOMER_IDS,
            "signup_date": signup_date,
            "region": region,
            "segment": segment,
            "plan": plan,
            "age": age,
            "tenure_months": tenure_months,
            "num_products": num_products,
            "sessions_30d": sessions_30d,
            "support_tickets": support_tickets,
            "monthly_spend": monthly_spend,
            "satisfaction": satisfaction,
            "group": group,
            "is_active": 1 - churned,
            "churned": churned,
        }
    )


def make_daily_sales(rng: np.random.Generator) -> pl.DataFrame:
    """~6 years of daily revenue: linear trend + weekly & yearly seasonality + December bump."""
    start = date(2019, 1, 1)
    n_days = (date(2024, 12, 31) - start).days + 1
    dates = [start + timedelta(days=i) for i in range(n_days)]
    t = np.arange(n_days)
    weekday = np.array([d.weekday() for d in dates])
    month = np.array([d.month for d in dates])
    trend = 1000.0 + 0.45 * t
    weekly = np.where(weekday < 5, 120.0, -180.0)
    yearly = 220.0 * np.sin(2 * np.pi * t / 365.25)
    december = np.where(month == 12, 180.0, 0.0)
    noise = rng.normal(0, 55, n_days)
    revenue = np.round(np.clip(trend + weekly + yearly + december + noise, 50, None), 2)
    orders = np.maximum(1, np.round(revenue / rng.normal(85, 8, n_days))).astype(int)
    return pl.DataFrame({"date": dates, "revenue": revenue, "orders": orders})


def main() -> None:
    rng = np.random.default_rng(SEED)
    transactions = make_transactions(rng)
    customers = make_customers(rng)
    daily_sales = make_daily_sales(rng)

    write_csv(transactions, RAW / "transactions.csv")
    write_parquet(customers, RAW / "customers.parquet")
    write_parquet(daily_sales, RAW / "daily_sales.parquet")

    print(f"transactions: {transactions.shape}  -> {RAW / 'transactions.csv'}")
    print(f"customers:    {customers.shape}  -> {RAW / 'customers.parquet'}")
    print(f"daily_sales:  {daily_sales.shape}  -> {RAW / 'daily_sales.parquet'}")
    print(f"churn rate:   {customers['churned'].mean():.1%}")


if __name__ == "__main__":
    main()
