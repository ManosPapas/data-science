"""Financial KPIs — growth, margins, unit economics, SaaS recurring, marketplace, liquidity.

Aggregate your data first (sums/counts), then call these scalar helpers. ``nan`` on divide-by-zero.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else float("nan")


# --- Growth -------------------------------------------------------------------------------------


def growth_rate(current: float, previous: float) -> float:
    """Period-over-period growth: (current - previous) / previous."""
    return _safe_div(current - previous, previous)


def cagr(begin: float, end: float, periods: float) -> float:
    """Compound growth rate per period over ``periods``."""
    return (end / begin) ** (1 / periods) - 1 if begin > 0 and periods > 0 else float("nan")


# --- Margins ------------------------------------------------------------------------------------


def gross_margin(revenue: float, cogs: float) -> float:
    """(revenue - cost of goods) / revenue."""
    return _safe_div(revenue - cogs, revenue)


def contribution_margin(revenue: float, variable_costs: float) -> float:
    """(revenue - variable costs) / revenue."""
    return _safe_div(revenue - variable_costs, revenue)


def net_margin(net_income: float, revenue: float) -> float:
    """Net profit margin: net income / revenue."""
    return _safe_div(net_income, revenue)


def operating_margin(operating_income: float, revenue: float) -> float:
    """Operating margin: operating income / revenue."""
    return _safe_div(operating_income, revenue)


def ebitda_margin(ebitda: float, revenue: float) -> float:
    """EBITDA margin: EBITDA / revenue."""
    return _safe_div(ebitda, revenue)


def markup(price: float, cost: float) -> float:
    """Markup over cost: (price - cost) / cost."""
    return _safe_div(price - cost, cost)


# --- Unit economics ------------------------------------------------------------------------------


def average_order_value(revenue: float, orders: float) -> float:
    """Revenue per order."""
    return _safe_div(revenue, orders)


def arpu(revenue: float, users: float) -> float:
    """Average revenue per user."""
    return _safe_div(revenue, users)


def arppu(revenue: float, paying_users: float) -> float:
    """Average revenue per *paying* user."""
    return _safe_div(revenue, paying_users)


def cac(acquisition_spend: float, new_customers: float) -> float:
    """Customer acquisition cost: spend / new customers."""
    return _safe_div(acquisition_spend, new_customers)


def clv(aov: float, purchase_frequency: float, margin: float, churn: float) -> float:
    """Customer lifetime value ~ AOV * frequency * margin / churn."""
    return _safe_div(aov * purchase_frequency * margin, churn)


def ltv_cac_ratio(ltv: float, cac_value: float) -> float:
    """LTV : CAC ratio (>3 is healthy)."""
    return _safe_div(ltv, cac_value)


def cac_payback_months(cac_value: float, monthly_margin_per_customer: float) -> float:
    """Months to recover CAC from per-customer monthly margin."""
    return _safe_div(cac_value, monthly_margin_per_customer)


def roi(gain: float, cost: float) -> float:
    """Return on investment: (gain - cost) / cost."""
    return _safe_div(gain - cost, cost)


def roas(revenue: float, ad_spend: float) -> float:
    """Return on ad spend: revenue / ad spend."""
    return _safe_div(revenue, ad_spend)


def break_even_units(fixed_costs: float, price: float, variable_cost: float) -> float:
    """Units to break even: fixed costs / (price - variable cost)."""
    return _safe_div(fixed_costs, price - variable_cost)


# --- Recurring revenue (SaaS) -------------------------------------------------------------------


def churn_rate(lost: float, at_start: float) -> float:
    """Fraction of customers/revenue lost over a period."""
    return _safe_div(lost, at_start)


def retention_rate(retained: float, at_start: float) -> float:
    """Fraction retained over a period."""
    return _safe_div(retained, at_start)


def net_revenue_retention(
    starting: float, expansion: float, contraction: float, churned: float
) -> float:
    """NRR: (starting + expansion - contraction - churned) / starting."""
    return _safe_div(starting + expansion - contraction - churned, starting)


def gross_revenue_retention(starting: float, contraction: float, churned: float) -> float:
    """GRR: (starting - contraction - churned) / starting."""
    return _safe_div(starting - contraction - churned, starting)


def mrr(monthly_revenue_per_account: ArrayLike) -> float:
    """Monthly recurring revenue = sum of per-account monthly revenue."""
    return float(np.sum(np.asarray(monthly_revenue_per_account, dtype=float)))


def arr(mrr_value: float) -> float:
    """Annual recurring revenue = 12 * MRR."""
    return 12.0 * mrr_value


def run_rate(period_revenue: float, *, periods_per_year: float = 12.0) -> float:
    """Annualized run rate from one period's revenue."""
    return period_revenue * periods_per_year


# --- Marketplace & scale -------------------------------------------------------------------------


def gmv(order_values: ArrayLike) -> float:
    """Gross merchandise value = sum of order values."""
    return float(np.sum(np.asarray(order_values, dtype=float)))


def take_rate(revenue: float, gmv_value: float) -> float:
    """Platform take rate: revenue / GMV."""
    return _safe_div(revenue, gmv_value)


def revenue_per_employee(revenue: float, employees: float) -> float:
    """Revenue per employee."""
    return _safe_div(revenue, employees)


# --- Liquidity & cash ---------------------------------------------------------------------------


def burn_rate(start_cash: float, end_cash: float, months: float) -> float:
    """Average monthly cash burn over the period."""
    return _safe_div(start_cash - end_cash, months)


def runway(cash: float, monthly_burn: float) -> float:
    """Months of runway = cash / monthly burn."""
    return _safe_div(cash, monthly_burn)


def inventory_turnover(cogs: float, average_inventory: float) -> float:
    """How many times inventory sells through: COGS / average inventory."""
    return _safe_div(cogs, average_inventory)


def days_sales_outstanding(receivables: float, revenue: float, *, days: float = 365.0) -> float:
    """Average days to collect receivables: receivables / revenue * days."""
    return _safe_div(receivables, revenue) * days
