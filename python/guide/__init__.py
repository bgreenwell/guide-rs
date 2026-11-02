"""GUIDE classification and regression trees with scikit-learn estimators."""

from importlib.metadata import version

from ._core import ALGORITHM_VERSION
from ._estimators import (
    GuideBoostingClassifier,
    GuideBoostingRegressor,
    GuideClassifier,
    GuideForestClassifier,
    GuideForestRegressor,
    GuideRegressor,
    importance,
)
from ._plot import plot_tree

__version__ = version("guide-rs")

__all__ = [
    "ALGORITHM_VERSION",
    "GuideBoostingClassifier",
    "GuideBoostingRegressor",
    "GuideClassifier",
    "GuideForestClassifier",
    "GuideForestRegressor",
    "GuideRegressor",
    "__version__",
    "importance",
    "plot_tree",
]
