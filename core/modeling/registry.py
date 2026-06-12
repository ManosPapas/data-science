"""Model registry — ``make_model(name, task, **params)`` returns a configured estimator.

Estimators are returned raw (sklearn-compatible) so they drop straight into ``Pipeline``s, search,
``cross_validate``, and the ``viz.model`` charts. ``**params`` passes through to the estimator, so
every hyper-parameter is yours. Add a model with :func:`register`; list them with
:func:`available_models`. xgboost / lightgbm / pygam are imported lazily so the registry loads even
when those (optional) libraries aren't installed.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    BaggingClassifier,
    BaggingRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    BayesianRidge,
    ElasticNet,
    GammaRegressor,
    HuberRegressor,
    Lasso,
    LinearRegression,
    LogisticRegression,
    PoissonRegressor,
    QuantileRegressor,
    Ridge,
    RidgeClassifier,
    SGDClassifier,
    TweedieRegressor,
)
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

Factory = Callable[..., Any]

_REGISTRY: dict[tuple[str, str], Factory] = {}


def register(name: str, task: str, factory: Factory) -> None:
    """Register ``factory(**params) -> estimator`` under (task, name)."""
    _REGISTRY[(task, name)] = factory


def available_models(task: str | None = None) -> list[str]:
    """Sorted model names, optionally filtered to a task ('regression' or 'classification')."""
    return sorted(
        name for (registered_task, name) in _REGISTRY if task is None or registered_task == task
    )


def make_model(name: str, *, task: str = "classification", **params: Any) -> Any:
    """Construct a registered model by name with ``**params`` passed to the estimator."""
    key = (task, name)
    if key not in _REGISTRY:
        choices = ", ".join(available_models(task)) or "(none)"
        raise KeyError(f"unknown {task} model {name!r}. available: {choices}")
    return _REGISTRY[key](**params)


_REGRESSORS: dict[str, Factory] = {
    "linear": LinearRegression,
    "ridge": Ridge,
    "lasso": Lasso,
    "elasticnet": ElasticNet,
    "bayesian_ridge": BayesianRidge,
    "huber": HuberRegressor,
    "quantile": QuantileRegressor,
    "poisson": PoissonRegressor,
    "gamma": GammaRegressor,
    "tweedie": TweedieRegressor,
    "knn": KNeighborsRegressor,
    "svr": SVR,
    "linear_svr": LinearSVR,
    "tree": DecisionTreeRegressor,
    "random_forest": RandomForestRegressor,
    "extra_trees": ExtraTreesRegressor,
    "gradient_boosting": GradientBoostingRegressor,
    "hist_gradient_boosting": HistGradientBoostingRegressor,
    "adaboost": AdaBoostRegressor,
    "bagging": BaggingRegressor,
    "mlp": MLPRegressor,
}

_CLASSIFIERS: dict[str, Factory] = {
    "logistic": LogisticRegression,
    "ridge": RidgeClassifier,
    "sgd": SGDClassifier,
    "knn": KNeighborsClassifier,
    "svc": SVC,
    "linear_svc": LinearSVC,
    "tree": DecisionTreeClassifier,
    "random_forest": RandomForestClassifier,
    "extra_trees": ExtraTreesClassifier,
    "gradient_boosting": GradientBoostingClassifier,
    "hist_gradient_boosting": HistGradientBoostingClassifier,
    "adaboost": AdaBoostClassifier,
    "bagging": BaggingClassifier,
    "mlp": MLPClassifier,
    "gaussian_nb": GaussianNB,
    "bernoulli_nb": BernoulliNB,
}

for _name, _factory in _REGRESSORS.items():
    register(_name, "regression", _factory)
for _name, _factory in _CLASSIFIERS.items():
    register(_name, "classification", _factory)


def _xgboost(task: str, **params: Any) -> Any:
    import xgboost

    estimator = xgboost.XGBClassifier if task == "classification" else xgboost.XGBRegressor
    return estimator(**params)


def _lightgbm(task: str, **params: Any) -> Any:
    import lightgbm

    estimator = lightgbm.LGBMClassifier if task == "classification" else lightgbm.LGBMRegressor
    return estimator(**params)


def _gam(task: str, **params: Any) -> Any:
    import pygam

    estimator = pygam.LogisticGAM if task == "classification" else pygam.LinearGAM
    return estimator(**params)


for _task in ("regression", "classification"):
    register("xgboost", _task, lambda _t=_task, **params: _xgboost(_t, **params))
    register("lightgbm", _task, lambda _t=_task, **params: _lightgbm(_t, **params))
    register("gam", _task, lambda _t=_task, **params: _gam(_t, **params))
