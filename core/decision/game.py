"""Game theory — equilibria for strategic interaction: the other side moves too.

Optimization assumes the environment holds still; competitors don't. Payoff-matrix tools find
pure/mixed Nash equilibria and strip dominated strategies; ``best_response_dynamics`` simulates
competitive reaction (e.g. price wars) to its resting point. An equilibrium is a *prediction of
where dynamics settle*, not a recommendation — knowing it tells you which moves get matched.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _payoffs(a: ArrayLike, b: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    pa = np.asarray(a, dtype=float)
    pb = np.asarray(b, dtype=float)
    if pa.shape != pb.shape or pa.ndim != 2:
        raise ValueError("payoff matrices must be 2-D and the same shape")
    return pa, pb


def pure_nash(payoff_row: ArrayLike, payoff_col: ArrayLike) -> list[tuple[int, int]]:
    """All pure-strategy Nash equilibria of a two-player game (may be none or several).

    Payoffs are (row player, column player) per cell; a cell is an equilibrium when neither side
    gains by deviating alone. Prisoner's-dilemma pricing in one check: "both discount" can be the
    unique equilibrium even though "both hold price" pays more — that's the trap, not a bug.
    """
    a, b = _payoffs(payoff_row, payoff_col)
    equilibria = []
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if a[i, j] >= a[:, j].max() and b[i, j] >= b[i, :].max():
                equilibria.append((i, j))
    return equilibria


def mixed_nash_2x2(
    payoff_row: ArrayLike, payoff_col: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The interior mixed equilibrium of a 2x2 game: (row probabilities, column probabilities).

    Each side mixes to make the *other* indifferent — your equilibrium mix comes from their
    payoffs, not yours. Exists when neither player has a dominant strategy (otherwise use
    :func:`pure_nash`); raises when the game degenerates.
    """
    a, b = _payoffs(payoff_row, payoff_col)
    if a.shape != (2, 2):
        raise ValueError("mixed_nash_2x2 needs 2x2 payoff matrices")
    denominator_p = b[0, 0] - b[1, 0] - b[0, 1] + b[1, 1]
    denominator_q = a[0, 0] - a[0, 1] - a[1, 0] + a[1, 1]
    if denominator_p == 0 or denominator_q == 0:
        raise ValueError("degenerate game — no unique interior mixed equilibrium")
    p = (b[1, 1] - b[1, 0]) / denominator_p  # P(row plays strategy 0)
    q = (a[1, 1] - a[0, 1]) / denominator_q  # P(column plays strategy 0)
    if not (0.0 <= p <= 1.0 and 0.0 <= q <= 1.0):
        raise ValueError("no interior mixed equilibrium — check pure_nash instead")
    return np.array([p, 1.0 - p]), np.array([q, 1.0 - q])


def iterated_dominance(payoff_row: ArrayLike, payoff_col: ArrayLike) -> tuple[list[int], list[int]]:
    """Surviving (row, column) strategies after iterated elimination of strictly dominated ones.

    A strategy a rational player never uses (another beats it against *everything*) can be
    deleted; deleting may expose new dominance. What survives is where analysis should focus —
    if one cell survives, the game is dominance-solvable and that's the prediction.
    """
    a, b = _payoffs(payoff_row, payoff_col)
    rows = list(range(a.shape[0]))
    cols = list(range(a.shape[1]))
    changed = True
    while (changed and len(rows) > 1) or (changed and len(cols) > 1):
        changed = False
        for i in list(rows):
            if len(rows) > 1 and any(all(a[k, j] > a[i, j] for j in cols) for k in rows if k != i):
                rows.remove(i)
                changed = True
        for j in list(cols):
            if len(cols) > 1 and any(all(b[i, k] > b[i, j] for i in rows) for k in cols if k != j):
                cols.remove(j)
                changed = True
    return rows, cols


@dataclass(frozen=True)
class BestResponseResult:
    """Fixed point of best-response iteration with its convergence trace."""

    point: NDArray[np.float64]
    converged: bool
    iterations: int
    history: NDArray[np.float64]  # (iterations+1, n_players)


def best_response_dynamics(
    responses: Sequence[Callable[[NDArray[np.float64]], float]],
    start: ArrayLike,
    *,
    max_iter: int = 500,
    tol: float = 1e-8,
    damping: float = 0.5,
) -> BestResponseResult:
    """Iterate everyone's best reply until actions stop moving — a competitive equilibrium.

    ``responses[i](actions)`` returns player i's best action given the current action vector
    (e.g. reaction functions from each player's profit model). The resting point is a Nash
    equilibrium in continuous strategies — where a price move no longer pays *after* the match.
    ``damping`` < 1 averages old and new actions to stop oscillation; simulate competitor
    response by seeding ``start`` with your contemplated move.
    """
    if not 0.0 < damping <= 1.0:
        raise ValueError("damping must be in (0, 1]")
    actions = np.asarray(start, dtype=float).copy()
    if actions.size != len(responses):
        raise ValueError("start must have one action per response function")
    trace = [actions.copy()]
    for iteration in range(1, max_iter + 1):
        replies = np.array([float(response(actions)) for response in responses])
        updated = (1.0 - damping) * actions + damping * replies
        trace.append(updated.copy())
        if float(np.max(np.abs(updated - actions))) < tol:
            return BestResponseResult(updated, True, iteration, np.vstack(trace))
        actions = updated
    return BestResponseResult(actions, False, max_iter, np.vstack(trace))
