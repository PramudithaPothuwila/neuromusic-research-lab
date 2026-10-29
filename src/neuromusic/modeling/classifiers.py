"""Classifier registry. Every model is a Pipeline so imputation and scaling are fitted inside
each cross-validation training fold only."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from neuromusic.modeling.baselines import dummy_classifier


@dataclass(frozen=True)
class ModelSpec:
    name: str
    factory: Callable[[int], Any]
    param_grid: dict[str, list[Any]]
    description: str


def _pipeline(estimator: Any, scale: bool = True) -> Pipeline:
    steps: list[tuple[str, Any]] = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)


def _xgboost(seed: int) -> Pipeline:
    from xgboost import XGBClassifier

    return _pipeline(XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1,
                                   subsample=0.8, random_state=seed, n_jobs=1,
                                   eval_metric="mlogloss"), scale=False)


CLASSIFIERS: dict[str, ModelSpec] = {
    "dummy": ModelSpec("dummy", lambda s: _pipeline(dummy_classifier(s), scale=False), {},
                       "Most-frequent-class baseline"),
    "logistic_regression": ModelSpec(
        "logistic_regression",
        lambda s: _pipeline(LogisticRegression(C=1.0, max_iter=5000, random_state=s)),
        {"model__C": [0.01, 0.1, 1.0, 10.0]},
        "Multinomial logistic regression (L2)"),
    "svm": ModelSpec(
        "svm",
        lambda s: _pipeline(SVC(kernel="rbf", C=1.0, gamma="scale", random_state=s)),
        {"model__C": [0.1, 1.0, 10.0], "model__kernel": ["linear", "rbf"]},
        "Support vector machine"),
    "random_forest": ModelSpec(
        "random_forest",
        lambda s: _pipeline(RandomForestClassifier(n_estimators=300, random_state=s, n_jobs=1),
                            scale=False),
        {"model__max_depth": [None, 3, 6], "model__min_samples_leaf": [1, 3]},
        "Random forest"),
    "mlp": ModelSpec(
        "mlp",
        lambda s: _pipeline(MLPClassifier(hidden_layer_sizes=(32,), alpha=1e-2, max_iter=2000,
                                          random_state=s)),
        {"model__alpha": [1e-3, 1e-2, 1e-1], "model__hidden_layer_sizes": [(16,), (32,)]},
        "Small feed-forward neural network trained by backpropagation (optional; high "
        "variance with few participants)"),
    "xgboost": ModelSpec("xgboost", _xgboost, {"model__max_depth": [2, 3, 4]},
                         "Gradient-boosted trees (optional dependency)"),
}


def available_classifiers() -> list[str]:
    names = [n for n in CLASSIFIERS if n != "xgboost"]
    try:
        import xgboost  # noqa: F401
    except ImportError:
        return names
    return [*names, "xgboost"]
