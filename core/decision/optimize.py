"""Optimization / OR — linear programs and optimal assignment (pricing, allocation, scheduling)."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


def linear_program(
    cost: ArrayLike,
    *,
    a_ub: ArrayLike | None = None,
    b_ub: ArrayLike | None = None,
    a_eq: ArrayLike | None = None,
    b_eq: ArrayLike | None = None,
    bounds: Any = None,
    maximize: bool = False,
) -> Any:
    """Solve a linear program (scipy.optimize.linprog). ``maximize=True`` flips the objective."""
    from scipy.optimize import linprog

    c = np.asarray(cost, dtype=float)
    return linprog(
        c=-c if maximize else c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bounds
    )


def assign(
    cost_matrix: ArrayLike, *, maximize: bool = False
) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    """Optimal one-to-one assignment (Hungarian algorithm). Returns (row_indices, col_indices)."""
    from scipy.optimize import linear_sum_assignment

    rows, cols = linear_sum_assignment(np.asarray(cost_matrix, dtype=float), maximize=maximize)
    return np.asarray(rows), np.asarray(cols)
