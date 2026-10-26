use crate::{Result, invalid};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Task {
    Classification,
    Regression {
        #[serde(default)]
        loss: Loss,
    },
}
impl Default for Task {
    fn default() -> Self {
        Self::Regression {
            loss: Loss::SquaredError,
        }
    }
}

/// Prediction limits for multiple-linear least-squares leaves. The default
/// preserves earlier Rust behavior; GUIDE's native default is global_range.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum PredictionTruncation {
    /// Preserve the unbounded fitted prediction.
    #[default]
    None,
    /// Observed response range among the leaf's active training rows.
    NodeRange,
    /// Leaf range extended by 10% of its width at each endpoint.
    NodeRangeExpanded,
    /// Full active training response range, fixed across CV folds.
    GlobalRange,
}
impl PredictionTruncation {
    pub fn is_none(&self) -> bool {
        *self == Self::None
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Loss {
    #[default]
    SquaredError,
    Quantile {
        probability: f64,
    },
    DualQuantile {
        lower: f64,
        upper: f64,
    },
    Poisson,
    Logistic,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum NodeModel {
    #[default]
    Constant,
    /// Multiple linear regression. Numeric squared-error designs use GUIDE's
    /// nominal-column/global-missing-flag sample thresholds before QR fitting.
    Linear,
    Polynomial {
        degree: usize,
    },
    /// Experimental, opt-in grouped forward selection with optional backward elimination.
    /// Native GUIDE selection/fallback parity is incomplete; further parity work is deferred.
    Stepwise {
        alpha: f64,
        /// Zero means no user-imposed variable limit.
        max_variables: usize,
        backward: bool,
    },
}

/// Held-out loss of classification trees: it chooses the cross-validation or
/// validation subtree and measures forests' out-of-bag loss. GUIDE uses
/// misclassification cost; the proper scoring rules are guide-rs extensions.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum PruningLoss {
    #[default]
    Misclassification,
    /// Negative log probability of the observed class.
    LogLoss,
    /// Squared error of the class probability vector, summed over classes.
    Brier,
}
impl PruningLoss {
    pub fn is_default(&self) -> bool {
        *self == Self::Misclassification
    }
    /// The loss of predicting `probabilities` for an observation of class
    /// `y`, given the misclassification `cost` of the predicted class.
    pub fn loss(&self, probabilities: &[f64], y: usize, cost: f64) -> f64 {
        match self {
            Self::Misclassification => cost,
            Self::LogLoss => -probabilities[y].ln(),
            Self::Brier => probabilities
                .iter()
                .enumerate()
                .map(|(k, p)| (p - f64::from(u8::from(k == y))).powi(2))
                .sum(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Pruning {
    None,
    CrossValidation { folds: usize, standard_errors: f64 },
    Validation,
    FixedLeaves { max_leaves: usize },
    CostComplexity { alpha: f64 },
}
impl Default for Pruning {
    fn default() -> Self {
        Self::CrossValidation {
            folds: 10,
            standard_errors: 0.25,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FitOptions {
    pub task: Task,
    pub node_model: NodeModel,
    #[serde(default, skip_serializing_if = "PredictionTruncation::is_none")]
    pub prediction_truncation: PredictionTruncation,
    pub max_depth: usize,
    /// Minimum child row count. Numerical threshold candidates additionally use
    /// observed split-value ranks: left >= this count, right > this count.
    pub min_samples_leaf: usize,
    pub min_gain: f64,
    pub interactions: bool,
    /// Test bivariate discriminant projections after main and interaction tests.
    pub linear_splits: bool,
    pub significance_level: f64,
    /// Zero evaluates all candidate cutpoints; otherwise at least two.
    pub max_split_candidates: usize,
    pub pruning: Pruning,
    /// Held-out loss for classification pruning and out-of-bag loss.
    #[serde(default, skip_serializing_if = "PruningLoss::is_default")]
    pub pruning_loss: PruningLoss,
    pub seed: u64,
    /// Probabilities in lexicographic class-label order. None estimates priors.
    pub class_priors: Option<Vec<f64>>,
    /// `costs[true_class][predicted_class]`, in lexicographic label order.
    pub class_costs: Option<Vec<Vec<f64>>>,
    /// Features sampled at each node. None considers every splitting feature.
    pub mtry: Option<usize>,
}
impl Default for FitOptions {
    fn default() -> Self {
        Self {
            task: Task::default(),
            node_model: NodeModel::Constant,
            prediction_truncation: PredictionTruncation::None,
            max_depth: 10,
            min_samples_leaf: 2,
            min_gain: 1e-12,
            interactions: true,
            linear_splits: false,
            significance_level: 0.10,
            max_split_candidates: 100,
            pruning: Pruning::default(),
            pruning_loss: PruningLoss::Misclassification,
            seed: 0,
            class_priors: None,
            class_costs: None,
            mtry: None,
        }
    }
}
impl FitOptions {
    pub fn validate(&self) -> Result<()> {
        if !self.prediction_truncation.is_none()
            && (self.node_model != NodeModel::Linear
                || self.task
                    != Task::Regression {
                        loss: Loss::SquaredError,
                    })
        {
            return Err(invalid(
                "prediction_truncation currently requires multiple-linear squared-error leaves",
            ));
        }
        if let Some(p) = &self.class_priors
            && (p.is_empty()
                || p.iter().any(|v| !v.is_finite() || *v <= 0.0)
                || (p.iter().sum::<f64>() - 1.0).abs() > 1e-8)
        {
            return Err(invalid(
                "class priors must be finite, positive, and sum to one",
            ));
        }
        if let Some(costs) = &self.class_costs
            && (costs.is_empty()
                || costs.iter().enumerate().any(|(i, r)| {
                    r.len() != costs.len()
                        || r.iter().any(|x| !x.is_finite() || *x < 0.0)
                        || r[i] != 0.0
                }))
        {
            return Err(invalid(
                "class costs must be finite, nonnegative, square, and have zero diagonal",
            ));
        }
        if !self.pruning_loss.is_default() {
            if self.task != Task::Classification {
                return Err(invalid("pruning_loss applies only to classification"));
            }
            if self.class_costs.is_some() {
                return Err(invalid(
                    "pruning_loss log_loss and brier ignore class costs; use misclassification",
                ));
            }
        }
        if self.max_depth > 64 {
            return Err(invalid("max_depth must be at most 64"));
        }
        if self.min_samples_leaf == 0 || self.max_split_candidates == 1 {
            return Err(invalid(
                "min_samples_leaf must be positive; max_split_candidates must be 0 (exhaustive) or at least 2",
            ));
        }
        if !self.min_gain.is_finite()
            || self.min_gain < 0.0
            || !self.significance_level.is_finite()
            || !(0.0..1.0).contains(&self.significance_level)
            || self.significance_level == 0.0
        {
            return Err(invalid(
                "min_gain must be finite and nonnegative; significance_level must be in (0, 1)",
            ));
        }
        if self.mtry == Some(0) {
            return Err(invalid("mtry must be positive"));
        }
        match &self.pruning {
            Pruning::CrossValidation {
                folds,
                standard_errors,
            } if *folds < 2 || !standard_errors.is_finite() || *standard_errors < 0.0 => {
                return Err(invalid(
                    "CV requires at least 2 folds and finite nonnegative standard_errors",
                ));
            }
            Pruning::FixedLeaves { max_leaves: 0 } => {
                return Err(invalid("max_leaves must be positive"));
            }
            Pruning::CostComplexity { alpha } if !alpha.is_finite() || *alpha < 0.0 => {
                return Err(invalid("alpha must be finite and nonnegative"));
            }
            _ => {}
        }
        if let Task::Regression { loss } = &self.task {
            if self.class_priors.is_some() || self.class_costs.is_some() {
                return Err(invalid("class priors and costs require classification"));
            }
            if let Loss::Quantile { probability } = loss
                && (!probability.is_finite() || *probability <= 0.0 || *probability >= 1.0)
            {
                return Err(invalid("quantile probability must be in (0, 1)"));
            }
            if let Loss::DualQuantile { lower, upper } = loss {
                if !lower.is_finite()
                    || !upper.is_finite()
                    || *lower <= 0.0
                    || *upper >= 1.0
                    || lower >= upper
                {
                    return Err(invalid("dual quantiles require 0 < lower < upper < 1"));
                }
                if self.node_model != NodeModel::Constant {
                    return Err(invalid(
                        "dual quantiles currently require constant node models",
                    ));
                }
            }
        } else if self.node_model != NodeModel::Constant {
            return Err(invalid(
                "classification currently supports constant node models only",
            ));
        }
        if let NodeModel::Polynomial { degree } = self.node_model
            && !(1..=6).contains(&degree)
        {
            return Err(invalid("polynomial degree must be between 1 and 6"));
        }
        if let NodeModel::Stepwise { alpha, .. } = self.node_model {
            if !alpha.is_finite() || alpha <= 0.0 || alpha >= 1.0 {
                return Err(invalid("stepwise alpha must be in (0, 1)"));
            }
            if !matches!(
                self.task,
                Task::Regression {
                    loss: Loss::SquaredError
                }
            ) {
                return Err(invalid(
                    "stepwise node fitting currently requires squared-error regression",
                ));
            }
        }
        Ok(())
    }
}
