"""Tests for the capability-audit additions: correlations, tests, distances, transforms, etc."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from core.analytics import curves, distance, regression, stats
from core.modeling import evaluate, preprocess

# --- Correlation suite --------------------------------------------------------------------------


def test_correlation_test_methods(rng: np.random.Generator) -> None:
    x = rng.normal(size=200)
    y = 2.0 * x + rng.normal(0, 0.5, 200)
    for method in ("pearson", "spearman", "kendall"):
        result = stats.correlation_test(x, y, method=method)
        assert result.statistic > 0.5
        assert result.p_value < 1e-3


def test_point_biserial(rng: np.random.Generator) -> None:
    group = rng.integers(0, 2, 300)
    value = group * 5.0 + rng.normal(0, 1, 300)  # higher mean for group 1
    result = stats.point_biserial(group, value)
    assert result.statistic > 0.8
    with pytest.raises(ValueError, match="0/1"):
        stats.point_biserial([0, 2, 1], [1.0, 2.0, 3.0])


def test_cramers_v_and_phi() -> None:
    # perfectly associated 2x2 → V and |phi| near 1
    a = [0, 0, 1, 1, 0, 1]
    b = [0, 0, 1, 1, 0, 1]
    assert stats.cramers_v(a, b, bias_correction=False) == pytest.approx(1.0, abs=1e-6)
    assert stats.phi_coefficient(a, b) == pytest.approx(1.0, abs=1e-6)
    # independent → near 0
    rng = np.random.default_rng(0)
    indep_a = rng.integers(0, 3, 2000)
    indep_b = rng.integers(0, 3, 2000)
    assert stats.cramers_v(indep_a, indep_b) < 0.1


def test_tetrachoric_recovers_latent_correlation(rng: np.random.Generator) -> None:
    cov = [[1.0, 0.6], [0.6, 1.0]]
    latent = rng.multivariate_normal([0, 0], cov, 20000)
    a = (latent[:, 0] > 0).astype(int)
    b = (latent[:, 1] > 0).astype(int)
    table = (
        pl.DataFrame({"a": a, "b": b})
        .pivot(on="b", index="a", values="a", aggregate_function="len")
        .drop("a")
        .to_numpy()
    )
    rho = stats.tetrachoric(table)
    assert abs(rho - 0.6) < 0.07  # recovers the latent correlation, phi would understate it


def test_partial_correlation_removes_confounder(rng: np.random.Generator) -> None:
    z = rng.normal(size=2000)  # confounder driving both
    x = z + rng.normal(0, 0.3, 2000)
    y = z + rng.normal(0, 0.3, 2000)
    raw = stats.correlation_test(x, y).statistic
    partial = stats.partial_correlation(
        pl.DataFrame({"x": x, "y": y, "z": z}), x="x", y="y", covariates=["z"]
    )
    assert raw > 0.7  # strong spurious correlation
    assert abs(partial.statistic) < 0.15  # vanishes once z is controlled for


def test_correlation_kind_chooser() -> None:
    assert stats.correlation_kind("continuous", "continuous").startswith("correlation_test")
    assert stats.correlation_kind("binary", "binary") == "phi_coefficient"
    assert stats.correlation_kind("binary", "continuous") == "point_biserial"
    assert stats.correlation_kind("nominal", "continuous") == "cramers_v"
    assert "spearman" in stats.correlation_kind("ordinal", "continuous")


# --- Hypothesis tests ---------------------------------------------------------------------------


def test_one_sample_t_test() -> None:
    rng = np.random.default_rng(1)
    sample = rng.normal(155.0, 10.0, 500)  # mean above the 150 SLA
    assert stats.one_sample_t_test(sample, 150.0).p_value < 0.01
    assert stats.one_sample_t_test(sample, 155.0).p_value > 0.05


def test_chi_square_gof() -> None:
    observed = [30, 30, 30, 30]  # uniform → matches default
    assert stats.chi_square_gof(observed).p_value > 0.9
    skewed = [60, 20, 20, 20]
    assert stats.chi_square_gof(skewed).p_value < 0.01


def test_fishers_exact() -> None:
    table = [[8, 2], [1, 9]]  # strong association, small n
    result = stats.fishers_exact(table)
    assert result.p_value < 0.05
    assert result.statistic > 1.0  # odds ratio


def test_friedman_and_wilcoxon(rng: np.random.Generator) -> None:
    base = rng.normal(0, 1, 100)
    cond_a, cond_b, cond_c = base, base + 0.5, base + 1.0  # consistent shifts per subject
    assert stats.friedman_test(cond_a, cond_b, cond_c).p_value < 0.01
    with pytest.raises(ValueError, match="at least 3"):
        stats.friedman_test(cond_a, cond_b)
    assert stats.wilcoxon_signed_rank(base, base + 0.5).p_value < 0.01


def test_distribution_fits_report_bic(rng: np.random.Generator) -> None:
    fit = stats.fit_distribution(rng.normal(0, 1, 500), "norm")
    assert "bic" in fit and fit["bic"] > fit["aic"]  # BIC penalty heavier at n=500
    discrete = stats.fit_discrete(rng.poisson(3.0, 500), "poisson")
    assert "bic" in discrete


# --- Distances ----------------------------------------------------------------------------------


def test_vector_distances() -> None:
    a = [0.0, 0.0]
    b = [3.0, 4.0]
    assert distance.vector_distance(a, b, metric="euclidean") == pytest.approx(5.0)
    assert distance.vector_distance(a, b, metric="manhattan") == pytest.approx(7.0)
    assert distance.cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0, abs=1e-9)
    assert distance.cosine_similarity([1.0, 1.0], [2.0, 2.0]) == pytest.approx(1.0, abs=1e-9)
    with pytest.raises(ValueError, match="unknown metric"):
        distance.vector_distance(a, b, metric="nope")


def test_pairwise_and_mahalanobis(rng: np.random.Generator) -> None:
    pts = np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]])
    matrix = distance.pairwise(pts, metric="euclidean")
    assert matrix.shape == (3, 3)
    assert matrix[0, 1] == pytest.approx(5.0)
    # mahalanobis: a point far along the low-variance axis is "far" despite small raw distance
    data = rng.multivariate_normal([0, 0], [[10.0, 0.0], [0.0, 0.1]], 2000)
    along_tight_axis = distance.mahalanobis([0.0, 2.0], data)
    along_wide_axis = distance.mahalanobis([2.0, 0.0], data)
    assert along_tight_axis > along_wide_axis


# --- Specificity / power transform / integration / endogeneity ----------------------------------


def test_classification_metrics_specificity() -> None:
    y_true = [0, 0, 0, 0, 1, 1, 1, 1]
    y_pred = [0, 0, 0, 1, 1, 1, 1, 0]  # 1 FP, 1 FN
    m = evaluate.classification_metrics(y_true, y_pred)
    assert m["specificity"] == pytest.approx(3 / 4)  # TN / (TN + FP)
    assert m["sensitivity"] == pytest.approx(3 / 4)  # TP / (TP + FN)
    assert "npv" in m


def test_power_transformer_normalizes(rng: np.random.Generator) -> None:
    skewed = rng.lognormal(0, 1, 500).reshape(-1, 1)
    transformer = preprocess.make_power_transformer(method="yeo-johnson")
    out = transformer.fit_transform(skewed)
    assert abs(float(stats.describe_distribution(out.ravel()).get("skew", 0.0))) < 0.3
    with pytest.raises(ValueError, match="yeo-johnson"):
        preprocess.make_power_transformer(method="nope")


def test_curve_integration() -> None:
    x = np.linspace(0.0, 10.0, 1001)
    assert curves.integrate(x, x) == pytest.approx(50.0, abs=0.01)  # ∫x dx 0..10 = 50
    assert curves.integrate(x, np.ones_like(x), method="simpson") == pytest.approx(10.0, abs=1e-6)


def test_durbin_wu_hausman_detects_endogeneity(rng: np.random.Generator) -> None:
    n = 3000
    instrument = rng.normal(size=n)
    confounder = rng.normal(size=n)
    endog = instrument + confounder + rng.normal(0, 0.3, n)  # correlated with the error via u
    y = 1.5 * endog + confounder + rng.normal(0, 0.3, n)  # confounder is the omitted error term
    df = pl.DataFrame({"y": y, "endog": endog, "z": instrument})
    result = regression.durbin_wu_hausman(
        df, y="y", endogenous="endog", exogenous=[], instruments=["z"]
    )
    assert result.p_value < 0.01  # endogeneity detected → OLS biased, use IV
