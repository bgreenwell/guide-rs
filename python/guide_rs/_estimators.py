"""scikit-learn estimators backed by the guide-rs core."""

import json
import numbers
import warnings

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import (
    check_array,
    check_consistent_length,
    check_is_fitted,
    column_or_1d,
    validate_data,
)

from ._core import Model

_PRUNING = ("cv", "cost_complexity", "max_leaves", "none")


def _is_frame(X):
    return hasattr(X, "columns") and hasattr(X, "dtypes") and hasattr(X, "iloc")


def _is_missing(v):
    if v is None:
        return True
    if isinstance(v, (float, np.floating)):
        return bool(np.isnan(v))
    # pandas.NA and pandas.NaT, without importing pandas.
    return type(v).__name__ in ("NAType", "NaTType")


def _labels(values):
    return [None if _is_missing(v) else str(v) for v in values]


def _seed(random_state):
    if isinstance(random_state, numbers.Integral):
        return int(random_state)
    return int(check_random_state(random_state).randint(np.iinfo(np.int32).max))


class _GuideTree(BaseEstimator):
    """Shared input handling and options for single GUIDE trees."""

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags

    def _categorical_mask(self, X):
        names = getattr(self, "feature_names_in_", None)
        spec = self.categorical_features
        if spec is None:
            if _is_frame(X):
                mask = []
                for name, dtype in zip(X.columns, X.dtypes):
                    kind = getattr(dtype, "kind", "O")
                    if kind not in "iufOSUb":
                        raise TypeError(f"column {name!r} has unsupported dtype {dtype}")
                    mask.append(kind in "OSUb")
                return mask
            return [False] * self.n_features_in_
        spec = list(spec)
        if spec and all(isinstance(s, (bool, np.bool_)) for s in spec):
            if len(spec) != self.n_features_in_:
                raise ValueError("categorical_features mask must have one entry per feature")
            return [bool(s) for s in spec]
        mask = [False] * self.n_features_in_
        for s in spec:
            if isinstance(s, str):
                if names is None or s not in names:
                    raise ValueError(f"categorical feature {s!r} is not a column name")
                mask[list(names).index(s)] = True
            elif isinstance(s, numbers.Integral) and 0 <= s < self.n_features_in_:
                mask[int(s)] = True
            else:
                raise ValueError(f"invalid categorical feature {s!r}")
        return mask

    def _columns(self, X, y=None, *, reset):
        """Validates X (and y) and returns per-feature columns for the core."""
        # A fit needs two samples; check_array words the error as sklearn expects.
        minimum = 2 if reset else 1
        if _is_frame(X):
            validate_data(self, X, skip_check_array=True, reset=reset)
            if reset:
                if y is None:
                    raise ValueError(
                        f"{type(self).__name__} requires y to be passed, but the target y is None."
                    )
                y = column_or_1d(y, warn=True)
                check_consistent_length(X, y)
            if X.shape[0] < minimum:
                raise ValueError(
                    f"Found array with {X.shape[0]} sample(s) while a minimum of {minimum} "
                    "is required."
                )
            get = X.iloc.__getitem__
        else:
            object_input = self.categorical_features is not None and len(
                list(self.categorical_features)
            )
            params = {
                "reset": reset,
                "dtype": None if object_input else np.float64,
                "ensure_all_finite": False if object_input else "allow-nan",
                "ensure_min_samples": minimum,
            }
            if reset:
                X, y = validate_data(self, X, y, **params)
            else:
                X = validate_data(self, X, **params)

            def get(index):
                return X[index]

        if reset:
            mask = self._categorical_mask(X)
            names = getattr(self, "feature_names_in_", None)
            names = [str(n) for n in names] if names is not None else [
                f"x{j}" for j in range(self.n_features_in_)
            ]
            if len(set(names)) != len(names):
                raise ValueError("feature names must be unique")
            role = "both" if self._fits_features() else "split"
            self._schema = [
                {
                    "name": name,
                    "feature_type": {"kind": "categorical" if cat else "numeric"},
                    "role": "split" if cat else role,
                }
                for name, cat in zip(names, mask)
            ]
        columns = []
        for j, feature in enumerate(self._schema):
            column = get((slice(None), j))
            if feature["feature_type"]["kind"] == "categorical":
                values = column.tolist() if hasattr(column, "tolist") else list(column)
                columns.append(_labels(values))
            elif _is_frame(X):
                columns.append(column.to_numpy(dtype=float, na_value=np.nan).tolist())
            else:
                columns.append(np.asarray(column, dtype=float).tolist())
        return columns, y

    def _fits_features(self):
        return False

    def _pruning(self):
        if self.pruning == "cv":
            return {
                "kind": "cross_validation",
                "folds": self.cv_folds,
                "standard_errors": self.cv_standard_errors,
            }
        if self.pruning == "cost_complexity":
            return {"kind": "cost_complexity", "alpha": self.ccp_alpha}
        if self.pruning == "max_leaves":
            if self.max_leaves is None:
                raise ValueError("pruning='max_leaves' requires max_leaves")
            return {"kind": "fixed_leaves", "max_leaves": self.max_leaves}
        if self.pruning == "none":
            return {"kind": "none"}
        raise ValueError(f"pruning must be one of {_PRUNING}, got {self.pruning!r}")

    def _options(self, task, node_model):
        return {
            "task": task,
            "node_model": node_model,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "min_gain": self.min_gain,
            "interactions": self.interactions,
            "significance_level": self.significance_level,
            "max_split_candidates": self.max_split_candidates,
            "pruning": self._pruning(),
            "seed": _seed(self.random_state),
        }

    def _fit_model(self, columns, target, options, sample_weight, exposure=None):
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=float)
            if sample_weight.shape != (len(target),):
                raise ValueError("sample_weight must have one entry per sample")
            if not np.any(sample_weight > 0):
                raise ValueError("sample_weight must contain a positive weight, not only zero")
            sample_weight = sample_weight.tolist()
        self._model = Model.fit(
            json.dumps(self._schema),
            columns,
            target,
            json.dumps(options),
            weights=sample_weight,
            exposure=exposure,
        )
        self.fit_report_ = json.loads(self._model.report)
        for message in self.fit_report_["warnings"]:
            warnings.warn(message, stacklevel=3)
        return self

    def get_n_leaves(self):
        """Number of terminal nodes in the fitted tree."""
        check_is_fitted(self)
        return self._model.shapes[0][0]

    def get_depth(self):
        """Depth of the fitted tree; a single leaf has depth 0."""
        check_is_fitted(self)
        return self._model.shapes[0][1]

    def apply(self, X):
        """Terminal node id of each sample."""
        check_is_fitted(self)
        columns, _ = self._columns(X, reset=False)
        return np.asarray(self._model.apply(columns)[0], dtype=np.intp)

    def to_text(self):
        """The fitted tree as indented text."""
        check_is_fitted(self)
        return self._model.to_text()

    def to_dot(self):
        """The fitted tree in Graphviz DOT format."""
        check_is_fitted(self)
        return self._model.to_dot()

    def to_json(self):
        """The fitted model in the format saved by the guide-rs CLI."""
        check_is_fitted(self)
        return self._model.to_json()

    def __sklearn_is_fitted__(self):
        return hasattr(self, "_model")

    def __getstate__(self):
        state = super().__getstate__()
        if "_model" in state:
            state = {**state, "_model": state["_model"].to_json()}
        return state

    def __setstate__(self, state):
        if isinstance(state.get("_model"), str):
            state = {**state, "_model": Model.from_json(state["_model"])}
        super().__setstate__(state)


class GuideRegressor(RegressorMixin, _GuideTree):
    """GUIDE regression tree.

    Parameters
    ----------
    loss : {"squared_error", "quantile", "poisson"}, default="squared_error"
        Node loss. Poisson accepts an ``exposure`` in ``fit`` and ``predict``.
    quantile : float, default=0.5
        Quantile probability for ``loss="quantile"``.
    node_model : {"constant", "linear", "polynomial"}, default="constant"
        Leaf model. Linear and polynomial leaves regress on every numeric
        feature; categorical features are used only for splitting.
    degree : int, default=2
        Polynomial degree for ``node_model="polynomial"``.
    max_depth : int, default=10
    min_samples_leaf : int, default=2
    min_gain : float, default=1e-12
        Minimum loss reduction for a split.
    interactions : bool, default=True
        Run pairwise interaction tests during variable selection.
    significance_level : float, default=0.1
    max_split_candidates : int, default=100
        Cutpoints evaluated per numeric split; 0 evaluates all.
    pruning : {"cv", "cost_complexity", "max_leaves", "none"}, default="cv"
    cv_folds : int, default=10
    cv_standard_errors : float, default=0.25
        Standard-error rule for cross-validation pruning.
    ccp_alpha : float, default=0.0
        Complexity parameter for ``pruning="cost_complexity"``.
    max_leaves : int, default=None
        Leaf count for ``pruning="max_leaves"``.
    categorical_features : array-like of str, int or bool, default=None
        Categorical columns. By default, object, string, categorical and
        boolean DataFrame columns are categorical and arrays are numeric.
    random_state : int, RandomState instance or None, default=0
        Seed for cross-validation folds. Fits are deterministic by default.
    """

    def __init__(
        self,
        *,
        loss="squared_error",
        quantile=0.5,
        node_model="constant",
        degree=2,
        max_depth=10,
        min_samples_leaf=2,
        min_gain=1e-12,
        interactions=True,
        significance_level=0.1,
        max_split_candidates=100,
        pruning="cv",
        cv_folds=10,
        cv_standard_errors=0.25,
        ccp_alpha=0.0,
        max_leaves=None,
        categorical_features=None,
        random_state=0,
    ):
        self.loss = loss
        self.quantile = quantile
        self.node_model = node_model
        self.degree = degree
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_gain = min_gain
        self.interactions = interactions
        self.significance_level = significance_level
        self.max_split_candidates = max_split_candidates
        self.pruning = pruning
        self.cv_folds = cv_folds
        self.cv_standard_errors = cv_standard_errors
        self.ccp_alpha = ccp_alpha
        self.max_leaves = max_leaves
        self.categorical_features = categorical_features
        self.random_state = random_state

    def _fits_features(self):
        return self.node_model != "constant"

    def _loss(self):
        if self.loss == "squared_error":
            return {"kind": "squared_error"}
        if self.loss == "quantile":
            return {"kind": "quantile", "probability": self.quantile}
        if self.loss == "poisson":
            return {"kind": "poisson"}
        raise ValueError(
            f"loss must be 'squared_error', 'quantile' or 'poisson', got {self.loss!r}"
        )

    def _node_model(self):
        if self.node_model in ("constant", "linear"):
            return {"kind": self.node_model}
        if self.node_model == "polynomial":
            return {"kind": "polynomial", "degree": self.degree}
        raise ValueError(
            f"node_model must be 'constant', 'linear' or 'polynomial', got {self.node_model!r}"
        )

    def fit(self, X, y, sample_weight=None, exposure=None):
        """Fit the tree. ``exposure`` is the Poisson exposure (not its log)."""
        columns, y = self._columns(X, y, reset=True)
        y = check_array(y, ensure_2d=False, dtype=np.float64)
        options = self._options({"kind": "regression", "loss": self._loss()}, self._node_model())
        if exposure is not None:
            exposure = np.asarray(exposure, dtype=float).tolist()
        return self._fit_model(columns, y.tolist(), options, sample_weight, exposure)

    def predict(self, X, exposure=None):
        """Predicted response; Poisson models fitted with exposure need it here too."""
        check_is_fitted(self)
        columns, _ = self._columns(X, reset=False)
        if exposure is not None:
            exposure = np.asarray(exposure, dtype=float).tolist()
        return np.asarray(self._model.predict_values(columns, exposure), dtype=float)


class GuideClassifier(ClassifierMixin, _GuideTree):
    """GUIDE classification tree.

    Parameters
    ----------
    class_priors : array-like of shape (n_classes,), default=None
        Prior probabilities in the order of ``classes_``. By default they are
        estimated from the training data.
    class_costs : array-like of shape (n_classes, n_classes), default=None
        ``class_costs[i][j]`` is the cost of predicting class ``j`` for a
        sample of class ``i``, in the order of ``classes_``.

    The remaining parameters are those of :class:`GuideRegressor` that apply
    to classification. Sample weights are treated as 0 (excluded) or 1,
    following GUIDE.
    """

    def __init__(
        self,
        *,
        max_depth=10,
        min_samples_leaf=2,
        min_gain=1e-12,
        interactions=True,
        linear_splits=False,
        significance_level=0.1,
        max_split_candidates=100,
        pruning="cv",
        cv_folds=10,
        cv_standard_errors=0.25,
        ccp_alpha=0.0,
        max_leaves=None,
        class_priors=None,
        class_costs=None,
        categorical_features=None,
        random_state=0,
    ):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_gain = min_gain
        self.interactions = interactions
        self.linear_splits = linear_splits
        self.significance_level = significance_level
        self.max_split_candidates = max_split_candidates
        self.pruning = pruning
        self.cv_folds = cv_folds
        self.cv_standard_errors = cv_standard_errors
        self.ccp_alpha = ccp_alpha
        self.max_leaves = max_leaves
        self.class_priors = class_priors
        self.class_costs = class_costs
        self.categorical_features = categorical_features
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        """Fit the tree."""
        columns, y = self._columns(X, y, reset=True)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        names = [str(c) for c in self.classes_]
        if len(set(names)) != len(names):
            raise ValueError("class labels must have distinct string forms")
        options = self._options({"kind": "classification"}, {"kind": "constant"})
        options["linear_splits"] = self.linear_splits
        # The core orders classes by label string and drops classes with no
        # positive-weight sample; priors and costs follow that order.
        weights = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight)
        present = sorted({str(v) for v, w in zip(y, weights) if w > 0})
        index = [names.index(c) for c in present]
        if self.class_priors is not None:
            priors = np.asarray(self.class_priors, dtype=float)
            if priors.shape != (len(names),):
                raise ValueError("class_priors must have one entry per class")
            options["class_priors"] = priors[index].tolist()
        if self.class_costs is not None:
            costs = np.asarray(self.class_costs, dtype=float)
            if costs.shape != (len(names), len(names)):
                raise ValueError("class_costs must be a square matrix over classes")
            options["class_costs"] = costs[np.ix_(index, index)].tolist()
        return self._fit_model(columns, [str(v) for v in y], options, sample_weight)

    def predict_proba(self, X):
        """Class probabilities in the order of ``classes_``."""
        return self._predict(X)[1]

    def predict(self, X):
        """Predicted class labels (cost-minimizing when costs are given)."""
        return self._predict(X)[0]

    def _predict(self, X):
        check_is_fitted(self)
        columns, _ = self._columns(X, reset=False)
        labels, probabilities = self._model.predict_classes(columns)
        names = [str(c) for c in self.classes_]
        index = [names.index(c) for c in self._model.classes]
        proba = np.zeros((len(labels), len(names)))
        if labels:
            proba[:, index] = probabilities
        lookup = {name: i for i, name in enumerate(names)}
        return self.classes_[[lookup[label] for label in labels]], proba
