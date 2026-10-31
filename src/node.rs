use crate::{
    Dataset, FeatureType, FitOptions, Loss, NodeModel, PredictionTruncation, Result, Task, Value,
    invalid, stats,
};
use serde::{Deserialize, Serialize};

pub(crate) fn sorted_category_levels(
    levels: impl Iterator<Item = Option<String>>,
) -> Vec<Option<String>> {
    let mut levels: Vec<_> = levels.collect();
    levels.sort_by(|a, b| {
        a.as_deref()
            .unwrap_or("NA")
            .cmp(b.as_deref().unwrap_or("NA"))
            .then_with(|| a.cmp(b))
    });
    levels.dedup();
    levels
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Prediction {
    Classification {
        label: String,
        probabilities: Vec<(String, f64)>,
    },
    Regression {
        value: f64,
    },
    Quantiles {
        lower: f64,
        upper: f64,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Term {
    pub feature: usize,
    pub power: usize,
    pub category: Option<String>,
    pub center: f64,
    pub scale: f64,
    pub impute: f64,
    pub missing_indicator: bool,
}
impl Term {
    pub(crate) fn value(&self, row: &[Value]) -> f64 {
        if self.missing_indicator {
            return if row[self.feature].is_missing() {
                1.0
            } else {
                0.0
            };
        }
        if let Some(cat) = &self.category {
            return if row[self.feature]
                .label_str()
                .is_some_and(|label| label == cat.as_str())
            {
                1.0
            } else {
                0.0
            };
        }
        let z = (row[self.feature].number().unwrap_or(self.impute) - self.center) / self.scale;
        // powi(1) returns its argument exactly; skip the library call.
        if self.power == 1 {
            z
        } else {
            z.powi(self.power as i32)
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PredictionBounds {
    pub lower: f64,
    pub upper: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LeafModel {
    Classification {
        probabilities: Vec<f64>,
        predicted_class: usize,
    },
    Constant {
        value: f64,
    },
    Quantiles {
        lower: f64,
        upper: f64,
    },
    Linear {
        intercept: f64,
        coefficients: Vec<f64>,
        terms: Vec<Term>,
        loss: Loss,
        converged: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        prediction_bounds: Option<PredictionBounds>,
    },
    /// A boosted tree's additive contribution on the link scale.
    Score {
        value: f64,
    },
    /// A linear discriminant (LDA) classification leaf.
    Discriminant(crate::Discriminant),
}
impl LeafModel {
    pub(crate) fn scalar(&self, row: &[Value], offset: f64, loss: &Loss) -> f64 {
        self.scalar_impl(row, offset, loss, true)
    }
    pub(crate) fn fitting_scalar(&self, row: &[Value], offset: f64, loss: &Loss) -> f64 {
        self.scalar_impl(row, offset, loss, false)
    }
    fn scalar_impl(&self, row: &[Value], offset: f64, loss: &Loss, truncate: bool) -> f64 {
        match self {
            Self::Quantiles { lower, upper } => lower / 2.0 + upper / 2.0,
            Self::Score { value } => *value,
            Self::Classification {
                predicted_class, ..
            } => *predicted_class as f64,
            Self::Discriminant(d) => d.predict(row) as f64,
            Self::Constant { value } => {
                if matches!(loss, Loss::Poisson) {
                    value * offset
                } else {
                    *value
                }
            }
            Self::Linear {
                intercept,
                coefficients,
                terms,
                loss,
                prediction_bounds,
                ..
            } => {
                let eta = intercept
                    + coefficients
                        .iter()
                        .zip(terms)
                        .map(|(b, t)| b * t.value(row))
                        .sum::<f64>();
                match loss {
                    Loss::Logistic => stats::sigmoid(eta),
                    Loss::Poisson => (eta + offset.ln()).exp(),
                    _ => {
                        if truncate {
                            prediction_bounds
                                .as_ref()
                                .map_or(eta, |b| eta.clamp(b.lower, b.upper))
                        } else {
                            eta
                        }
                    }
                }
            }
        }
    }
    /// The class probabilities of `row` under a classification leaf.
    pub(crate) fn class_probabilities(&self, row: &[Value]) -> Option<std::borrow::Cow<'_, [f64]>> {
        match self {
            Self::Classification { probabilities, .. } => Some(probabilities.as_slice().into()),
            Self::Discriminant(d) => Some(d.posteriors(row).into()),
            _ => None,
        }
    }
    /// The class a classification leaf predicts for `row`.
    pub(crate) fn predicted_class(&self, row: &[Value]) -> Option<usize> {
        match self {
            Self::Classification {
                predicted_class, ..
            } => Some(*predicted_class),
            Self::Discriminant(d) => Some(d.predict(row)),
            _ => None,
        }
    }
    pub(crate) fn regression_loss(&self, row: &[Value], offset: f64, y: f64, loss: &Loss) -> f64 {
        if let (
            Self::Quantiles { lower, upper },
            Loss::DualQuantile {
                lower: lo,
                upper: hi,
            },
        ) = (self, loss)
        {
            quantile_pair_loss(y, *lower, *upper, *lo, *hi)
        } else {
            observation_loss(loss, y, self.scalar(row, offset, loss))
        }
    }
}
pub(crate) fn quantile_pair_loss(y: f64, lower: f64, upper: f64, lo: f64, hi: f64) -> f64 {
    observation_loss(&Loss::Quantile { probability: lo }, y, lower)
        + observation_loss(&Loss::Quantile { probability: hi }, y, upper)
}

pub(crate) struct Context<'a> {
    pub data: &'a Dataset,
    pub options: &'a FitOptions,
    pub classes: Vec<String>,
    pub y: Vec<f64>,
    pub weights: Vec<f64>,
    pub costs: Vec<Vec<f64>>,
    pub loss: Loss,
    pub root_weight: f64,
    pub class_priors: Vec<f64>,
    pub root_samples: usize,
    /// Missing-value flag column of each feature, if any.
    pub flags: Vec<Option<usize>>,
    /// Candidate columns of LDA node models, fixed by the training rows.
    pub discriminant_columns: Vec<crate::discriminant::Column>,
    split_cost_weights: Vec<f64>,
    class_frequencies: Vec<f64>,
    /// Fewest rows for a linear node or leaf model.
    linear_min_samples: Option<usize>,
    /// Fewest rows for a linear model when scoring a candidate child.
    linear_split_min_samples: Option<usize>,
    linear_category_levels: Vec<Vec<Option<String>>>,
    global_prediction_bounds: Option<PredictionBounds>,
}
impl<'a> Context<'a> {
    pub fn new(
        data: &'a Dataset,
        options: &'a FitOptions,
        ids: &[usize],
        universe: Option<&[String]>,
    ) -> Result<Self> {
        let classifying = matches!(options.task, Task::Classification);
        let mut classes = universe.map(<[String]>::to_vec).unwrap_or_default();
        if classifying && classes.is_empty() {
            classes = ids
                .iter()
                .filter_map(|&i| data.targets[i].label())
                .collect();
            classes.sort();
            classes.dedup();
        }
        let y: Vec<f64> = data
            .targets
            .iter()
            .map(|v| {
                if classifying {
                    v.label()
                        .and_then(|s| classes.binary_search(&s).ok())
                        .unwrap_or(0) as f64
                } else {
                    v.number().unwrap_or(0.0)
                }
            })
            .collect();
        let mut weights: Vec<f64> = (0..data.rows.len())
            .map(|i| {
                if classifying {
                    if data.weight(i) > 0.0 { 1.0 } else { 0.0 }
                } else {
                    data.weight(i)
                }
            })
            .collect();
        let mut costs = vec![];
        let mut class_priors = vec![];
        let mut class_frequencies = vec![];
        if classifying {
            let k = classes.len();
            if k == 0 {
                return Err(invalid("no observed class labels"));
            }
            if universe.is_none() && k < 2 {
                return Err(invalid(
                    "classification requires at least two observed classes",
                ));
            }
            let mut counts = vec![0.0; k];
            for &i in ids {
                counts[y[i] as usize] += 1.0;
            }
            class_frequencies = counts.clone();
            class_priors = options
                .class_priors
                .clone()
                .unwrap_or_else(|| counts.iter().map(|n| n / ids.len() as f64).collect());
            if let Some(priors) = &options.class_priors {
                if priors.len() != k
                    || priors.iter().any(|x| !x.is_finite() || *x <= 0.0)
                    || (priors.iter().sum::<f64>() - 1.0).abs() > 1e-8
                {
                    return Err(invalid(
                        "class_priors must be positive, sum to 1, and match sorted class labels",
                    ));
                }
                for (i, w) in weights.iter_mut().enumerate() {
                    let j = y[i] as usize;
                    *w *= if counts[j] > 0.0 {
                        priors[j] * ids.len() as f64 / counts[j]
                    } else {
                        0.0
                    };
                }
            }
            costs = options.class_costs.clone().unwrap_or_else(|| {
                (0..k)
                    .map(|i| (0..k).map(|j| if i == j { 0.0 } else { 1.0 }).collect())
                    .collect()
            });
            if costs.len() != k
                || costs.iter().enumerate().any(|(i, r)| {
                    r.len() != k || r.iter().any(|x| !x.is_finite() || *x < 0.0) || r[i] != 0.0
                })
            {
                return Err(invalid(
                    "class_costs must be a finite nonnegative square matrix with zero diagonal",
                ));
            }
        }
        let root_weight: f64 = ids.iter().map(|&i| weights[i]).sum();
        if !root_weight.is_finite() || root_weight <= 0.0 {
            return Err(invalid("total training weight must be positive and finite"));
        }
        let loss = if let Task::Regression { loss } = &options.task {
            loss.clone()
        } else {
            Loss::SquaredError
        };
        let fitting: Vec<_> = data
            .schema
            .iter()
            .enumerate()
            .filter(|(_, f)| f.role.fits())
            .collect();
        // Native LS designs use all input rows, including excluded targets and
        // zero weights. Missing categorical values sort as the DSC token "NA".
        let linear_category_levels: Vec<_> = data
            .schema
            .iter()
            .enumerate()
            .map(|(j, f)| {
                if f.role.fits() && matches!(f.feature_type, FeatureType::Categorical) {
                    sorted_category_levels(data.rows.iter().map(|row| row[j].label()))
                } else {
                    vec![]
                }
            })
            .collect();
        // Thresholds count declared numeric columns and global categorical
        // dummies, not effective rank. Only numeric missing flags change the
        // threshold formula; categorical missingness is an ordinary level.
        let quantile = matches!(loss, Loss::Quantile { .. });
        let (linear_min_samples, linear_split_min_samples) = if options.node_model
            == NodeModel::Linear
            && (loss == Loss::SquaredError || quantile)
            && fitting.iter().all(|(_, f)| {
                matches!(
                    f.feature_type,
                    FeatureType::Numeric | FeatureType::Categorical
                )
            }) {
            let p: usize = fitting
                .iter()
                .map(|(j, f)| {
                    if matches!(f.feature_type, FeatureType::Categorical) {
                        linear_category_levels[*j].len().saturating_sub(1)
                    } else {
                        1
                    }
                })
                .sum();
            let flags = fitting
                .iter()
                .filter(|(j, f)| {
                    matches!(f.feature_type, FeatureType::Numeric)
                        && data.rows.iter().any(|r| r[*j].is_missing())
                })
                .count();
            let numeric = fitting
                .iter()
                .all(|(_, f)| matches!(f.feature_type, FeatureType::Numeric));
            if !quantile && flags == 0 && numeric {
                // Native least-squares trees with complete numeric regressors
                // fit constants throughout when the training sample has fewer
                // than p + 7 rows. Otherwise nodes and leaves are linear from
                // p + 3 rows, while split search scores a child as linear only
                // from p + 5 rows.
                if ids.len() < p.saturating_add(7) {
                    (Some(usize::MAX), Some(usize::MAX))
                } else {
                    (Some(p.saturating_add(3)), Some(p.saturating_add(5)))
                }
            } else if !quantile {
                let minimum = Some(if flags == 0 {
                    p.saturating_add(7)
                } else {
                    p.saturating_add(flags).saturating_add(3)
                });
                (minimum, minimum)
            } else if flags == 0 && numeric {
                // Native quantile leaves fit a linear model from p + 4 rows,
                // probed with complete numeric regressors only.
                let minimum = Some(p.saturating_add(4));
                (minimum, minimum)
            } else {
                (None, None)
            }
        } else {
            (None, None)
        };
        let global_prediction_bounds = (!options.prediction_truncation.is_none()).then(|| {
            // GUIDE fixes global bounds from the full active training dataset,
            // including during CV. Node bounds are instead local to each fold.
            let bound_ids = if options.prediction_truncation == PredictionTruncation::GlobalRange {
                data.active_rows()
            } else {
                ids.to_vec()
            };
            PredictionBounds {
                lower: bound_ids
                    .iter()
                    .map(|&i| y[i])
                    .fold(f64::INFINITY, f64::min),
                upper: bound_ids
                    .iter()
                    .map(|&i| y[i])
                    .fold(f64::NEG_INFINITY, f64::max),
            }
        });
        let cost_scale = costs.iter().flatten().copied().fold(0.0, f64::max);
        let mut split_cost_weights: Vec<f64> = costs
            .iter()
            .map(|row| {
                if cost_scale == 0.0 {
                    0.0
                } else {
                    row.iter().map(|c| c / cost_scale).sum()
                }
            })
            .collect();
        let max_cost_sum = split_cost_weights.iter().copied().fold(0.0, f64::max);
        if max_cost_sum > 0.0 {
            for w in &mut split_cost_weights {
                *w /= max_cost_sum;
            }
        }
        Ok(Self {
            data,
            options,
            classes,
            y,
            weights,
            costs,
            loss,
            root_weight,
            class_priors,
            root_samples: ids.len(),
            flags: crate::flag_columns(&data.schema),
            discriminant_columns: if options.node_model.discriminant_splits() {
                crate::discriminant::candidate_columns(data, ids)
            } else {
                vec![]
            },
            split_cost_weights,
            class_frequencies,
            linear_min_samples,
            linear_split_min_samples,
            linear_category_levels,
            global_prediction_bounds,
        })
    }
    pub fn classifying(&self) -> bool {
        !self.classes.is_empty()
    }
    /// The node or leaf model of rows `ids`.
    pub fn fit_leaf(&self, ids: &[usize]) -> LeafModel {
        self.fit_model(ids, self.linear_min_samples, false)
    }
    /// The model of a candidate child when scoring a split.
    pub fn fit_child(&self, ids: &[usize]) -> LeafModel {
        self.fit_model(ids, self.linear_split_min_samples, true)
    }
    fn fit_model(
        &self,
        ids: &[usize],
        linear_min_samples: Option<usize>,
        child: bool,
    ) -> LeafModel {
        if self.classifying() {
            let mut counts = vec![0.0; self.classes.len()];
            for &i in ids {
                counts[self.y[i] as usize] += 1.0;
            }
            return self.classification_leaf(counts);
        }
        if let Loss::DualQuantile { lower, upper } = self.loss {
            let values: Vec<(f64, f64)> =
                ids.iter().map(|&i| (self.y[i], self.weights[i])).collect();
            return LeafModel::Quantiles {
                lower: stats::weighted_quantile(&values, lower),
                upper: stats::weighted_quantile(&values, upper),
            };
        }
        let total: f64 = ids.iter().map(|&i| self.weights[i]).sum();
        let mean = ids
            .iter()
            .map(|&i| self.y[i] * self.weights[i])
            .sum::<f64>()
            / total;
        let value = match self.loss {
            Loss::Quantile { probability } => stats::weighted_quantile(
                &ids.iter()
                    .map(|&i| (self.y[i], self.weights[i]))
                    .collect::<Vec<_>>(),
                probability,
            ),
            Loss::Poisson => {
                let exposure_scale = ids.iter().map(|&i| self.data.offset(i)).fold(0.0, f64::max);
                let exposure_mean = ids
                    .iter()
                    .map(|&i| (self.data.offset(i) / exposure_scale) * (self.weights[i] / total))
                    .sum::<f64>();
                (mean / exposure_mean) / exposure_scale
            }
            _ => mean,
        };
        let constant = LeafModel::Constant { value };
        match self.options.node_model {
            // LDA leaves are classification models, fitted during growth.
            NodeModel::Constant | NodeModel::Lda | NodeModel::LdaSplits => constant,
            NodeModel::Linear => {
                if linear_min_samples.is_some_and(|minimum| ids.len() < minimum) {
                    constant
                } else {
                    self.linear(ids, self.terms(ids, None, 1), &constant)
                        .unwrap_or(constant)
                }
            }
            NodeModel::Stepwise {
                alpha,
                max_variables,
                backward,
            } => self.stepwise(ids, alpha, max_variables, backward, constant),
            // GUIDE scores candidate children with full-degree fits and drops
            // insignificant powers only in the leaves it keeps.
            NodeModel::Polynomial { degree, alpha }
                if matches!(self.loss, Loss::SquaredError) && alpha < 1.0 =>
            {
                self.significant_polynomial(ids, degree, if child { 1.0 } else { alpha }, constant)
            }
            NodeModel::Polynomial { degree, .. } => {
                let mut best = constant;
                let mut score = self.risk(&best, ids);
                for j in 0..self.data.schema.len() {
                    if !self.data.schema[j].role.fits()
                        || !matches!(self.data.schema[j].feature_type, FeatureType::Numeric)
                    {
                        continue;
                    }
                    for d in 1..=degree {
                        if let Some(model) = self.linear(ids, self.terms(ids, Some(j), d), &best) {
                            let risk = self.risk(&model, ids);
                            if stats::strictly_less(risk, score) {
                                best = model;
                                score = risk;
                            }
                        }
                    }
                }
                best
            }
        }
    }
    /// The classification leaf for the given (unweighted) class counts.
    /// GUIDE's count K of split variables, constant ones included, which
    /// sets its interaction and stopping thresholds.
    pub(crate) fn split_variable_count(&self) -> usize {
        self.data.schema.iter().filter(|s| s.role.splits()).count()
    }
    /// Whether interaction and linear-split tests run. GUIDE skips them when
    /// their Bonferroni level `alpha / (K (K - 1) / 2)` is at most 2e-6, the
    /// smallest probability its chi-square quantile routine accepts.
    pub(crate) fn pair_tests(&self) -> bool {
        let k = self.split_variable_count() as f64;
        k < 1.5 || self.options.significance_level / (k * (k - 1.0) / 2.0) > 2e-6
    }
    pub fn classification_leaf(&self, mut p: Vec<f64>) -> LeafModel {
        self.class_probabilities(&mut p);
        let predicted_class = self.least_cost_class(&p);
        LeafModel::Classification {
            probabilities: p,
            predicted_class,
        }
    }
    /// The class with the least expected misclassification cost under
    /// probabilities `p`, the first on ties.
    fn least_cost_class(&self, p: &[f64]) -> usize {
        // Classification contexts always hold at least one class.
        (0..p.len())
            .min_by(|&a, &b| {
                let cost = |j| {
                    p.iter()
                        .enumerate()
                        .map(|(i, v)| v * self.costs[i][j])
                        .sum::<f64>()
                };
                cost(a).total_cmp(&cost(b)).then(a.cmp(&b))
            })
            .unwrap()
    }
    /// Turns unweighted class counts into prior-adjusted probabilities.
    pub fn class_probabilities(&self, p: &mut [f64]) {
        for ((p, prior), frequency) in p
            .iter_mut()
            .zip(&self.class_priors)
            .zip(&self.class_frequencies)
        {
            *p = if *frequency > 0.0 {
                // GUIDE's evaluation order; its rounding decides near ties.
                *prior * *p / frequency
            } else {
                0.0
            };
        }
        let total: f64 = p.iter().sum();
        for value in p {
            *value /= total;
        }
    }
    fn stepwise(
        &self,
        ids: &[usize],
        alpha: f64,
        max_variables: usize,
        backward: bool,
        constant: LeafModel,
    ) -> LeafModel {
        let terms = self.terms(ids, None, 1);
        let groups: Vec<usize> = terms
            .iter()
            .map(|t| t.feature)
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        let limit = if max_variables == 0 {
            groups.len()
        } else {
            max_variables
        };
        let fit = |selected: &[usize]| -> Option<(LeafModel, f64, usize)> {
            let chosen: Vec<Term> = terms
                .iter()
                .filter(|t| selected.contains(&t.feature))
                .cloned()
                .collect();
            if chosen.is_empty() {
                return Some((constant.clone(), self.risk(&constant, ids), 1));
            }
            let x: Vec<Vec<f64>> = ids
                .iter()
                .map(|&i| {
                    std::iter::once(1.0)
                        .chain(chosen.iter().map(|t| t.value(&self.data.rows[i])))
                        .collect()
                })
                .collect();
            let y: Vec<f64> = ids.iter().map(|&i| self.y[i]).collect();
            let w: Vec<f64> = ids.iter().map(|&i| self.weights[i]).collect();
            let (beta, rank) = stats::least_squares_ranked(&x, &y, &w)?;
            if rank >= ids.len() {
                return None;
            }
            let model = LeafModel::Linear {
                intercept: beta[0],
                coefficients: beta[1..].to_vec(),
                terms: chosen,
                loss: Loss::SquaredError,
                converged: true,
                prediction_bounds: None,
            };
            let risk = self.risk(&model, ids);
            Some((model, risk, rank))
        };
        let test = |reduced: f64, full: f64, added: usize, rank: usize| {
            if added == 0 || !stats::strictly_less(full, reduced) {
                return 0.0;
            }
            if full == 0.0 {
                return f64::NEG_INFINITY;
            }
            // Compare log tails even when both ordinary probabilities round to
            // zero. Form log(F) directly to avoid overflow of the risk ratio.
            stats::f_log_survival_from_log(
                (reduced - full).ln() - (added as f64).ln() - full.ln()
                    + ((ids.len() - rank) as f64).ln(),
                added,
                ids.len() - rank,
            )
        };
        let mut selected = vec![];
        // With no selected terms `fit` returns the constant model, never None.
        let mut current = fit(&selected).unwrap();
        let mut visited = std::collections::BTreeSet::new();
        for _ in 0..(4 * groups.len() + 1) {
            if !visited.insert(selected.clone()) || selected.len() >= limit {
                break;
            }
            let remaining: Vec<usize> = groups
                .iter()
                .copied()
                .filter(|j| !selected.contains(j))
                .collect();
            let mut best = None;
            let mut pbest = 0.0;
            for &j in &remaining {
                let mut proposed = selected.clone();
                proposed.push(j);
                proposed.sort_unstable();
                if let Some(candidate) = fit(&proposed) {
                    let p = test(
                        current.1,
                        candidate.1,
                        candidate.2.saturating_sub(current.2),
                        candidate.2,
                    );
                    if p < pbest {
                        pbest = p;
                        best = Some((proposed, candidate));
                    }
                }
            }
            if pbest >= alpha.ln() - (remaining.len().max(1) as f64).ln() {
                break;
            }
            let Some((proposed, candidate)) = best else {
                break;
            };
            selected = proposed;
            current = candidate;
            if backward {
                loop {
                    let mut worst = None;
                    let mut pmax = alpha.ln();
                    for &j in &selected {
                        let reduced: Vec<usize> =
                            selected.iter().copied().filter(|k| *k != j).collect();
                        if let Some(candidate) = fit(&reduced) {
                            let p = test(
                                candidate.1,
                                current.1,
                                current.2.saturating_sub(candidate.2),
                                current.2,
                            );
                            if p > pmax {
                                pmax = p;
                                worst = Some((reduced, candidate));
                            }
                        }
                    }
                    if let Some((reduced, candidate)) = worst {
                        selected = reduced;
                        current = candidate;
                    } else {
                        break;
                    }
                }
            }
        }
        current.0
    }
    fn terms(&self, ids: &[usize], only: Option<usize>, degree: usize) -> Vec<Term> {
        let mut terms = vec![];
        for (j, f) in self.data.schema.iter().enumerate() {
            if !f.role.fits() || only.is_some_and(|i| i != j) {
                continue;
            }
            if matches!(f.feature_type, FeatureType::Categorical) {
                if self.options.node_model == NodeModel::Linear && self.loss == Loss::SquaredError {
                    let local =
                        sorted_category_levels(ids.iter().map(|&i| self.data.rows[i][j].label()));
                    let global = self.linear_category_levels[j].first();
                    // With the global reference absent, the last local dummy
                    // aliases the intercept in GUIDE's sequential rank check.
                    // Rebase explicitly so pivoted QR preserves its prediction
                    // convention for unseen levels without retaining aliases.
                    let reference = global
                        .filter(|level| local.contains(level))
                        .or_else(|| local.last());
                    for level in &local {
                        if Some(level) != reference {
                            terms.push(Term {
                                feature: j,
                                power: 1,
                                category: level.clone(),
                                center: 0.0,
                                scale: 1.0,
                                impute: 0.0,
                                missing_indicator: level.is_none(),
                            });
                        }
                    }
                    continue;
                }
                let mut cats: Vec<String> = ids
                    .iter()
                    .filter_map(|&i| self.data.rows[i][j].label())
                    .collect();
                cats.sort();
                cats.dedup();
                for category in cats.into_iter().skip(1) {
                    terms.push(Term {
                        feature: j,
                        power: 1,
                        category: Some(category),
                        center: 0.0,
                        scale: 1.0,
                        impute: 0.0,
                        missing_indicator: false,
                    });
                }
            } else {
                let values: Vec<(f64, f64)> = ids
                    .iter()
                    .filter_map(|&i| self.data.rows[i][j].number().map(|x| (x, self.weights[i])))
                    .collect();
                let w: f64 = values.iter().map(|x| x.1).sum();
                if w == 0.0 {
                    continue;
                }
                let center = values.iter().map(|(x, w)| x * w).sum::<f64>() / w;
                let scale = (values
                    .iter()
                    .map(|(x, w)| w * (x - center).powi(2))
                    .sum::<f64>()
                    / w)
                    .sqrt();
                if scale > 0.0 {
                    for power in 1..=degree {
                        terms.push(Term {
                            feature: j,
                            power,
                            category: None,
                            center,
                            scale,
                            impute: center,
                            missing_indicator: false,
                        });
                    }
                }
            }
            if ids.iter().any(|&i| self.data.rows[i][j].is_missing())
                && ids.iter().any(|&i| !self.data.rows[i][j].is_missing())
            {
                terms.push(Term {
                    feature: j,
                    power: 1,
                    category: None,
                    center: 0.0,
                    scale: 1.0,
                    impute: 0.0,
                    missing_indicator: true,
                });
            }
        }
        terms
    }
    /// GUIDE's best simple polynomial with insignificant powers dropped: the
    /// numeric fit feature whose degree-`degree` fit has the least error, then
    /// lower degrees while the highest power's two-sided t-test has p >
    /// `alpha`, down to a constant.
    fn significant_polynomial(
        &self,
        ids: &[usize],
        degree: usize,
        alpha: f64,
        constant: LeafModel,
    ) -> LeafModel {
        if ids.len() < degree + 3 {
            return constant;
        }
        let mut best: Option<(usize, LeafModel, f64)> = None;
        for j in 0..self.data.schema.len() {
            if !self.data.schema[j].role.fits()
                || !matches!(self.data.schema[j].feature_type, FeatureType::Numeric)
            {
                continue;
            }
            if let Some(model) = self.linear(ids, self.terms(ids, Some(j), degree), &constant) {
                let risk = self.risk(&model, ids);
                if best
                    .as_ref()
                    .is_none_or(|b| stats::strictly_less(risk, b.2))
                {
                    best = Some((j, model, risk));
                }
            }
        }
        let Some((j, mut model, _)) = best else {
            return constant;
        };
        for d in (1..=degree).rev() {
            let terms = self.terms(ids, Some(j), d);
            if self.highest_power_log_p(ids, &terms, d) <= alpha.ln() {
                return model;
            }
            if d > 1 {
                model = match self.linear(ids, self.terms(ids, Some(j), d - 1), &constant) {
                    Some(m) => m,
                    None => return constant,
                };
            }
        }
        constant
    }
    /// The log two-sided t-test p-value of the power-`d` term in the weighted
    /// least-squares fit of an intercept and `terms`; 0 (p = 1) when the fit
    /// leaves no residual degrees of freedom or is singular.
    fn highest_power_log_p(&self, ids: &[usize], terms: &[Term], d: usize) -> f64 {
        let p = terms.len() + 1;
        let Some(k) = terms
            .iter()
            .position(|t| t.power == d && !t.missing_indicator)
        else {
            return 0.0;
        };
        if ids.len() <= p {
            return 0.0;
        }
        let rows: Vec<Vec<f64>> = ids
            .iter()
            .map(|&i| {
                std::iter::once(1.0)
                    .chain(terms.iter().map(|t| t.value(&self.data.rows[i])))
                    .collect()
            })
            .collect();
        let mut xtx = vec![vec![0.0; p]; p];
        let mut xty = vec![0.0; p];
        for (r, &i) in rows.iter().zip(ids) {
            let w = self.weights[i];
            for a in 0..p {
                xty[a] += w * r[a] * self.y[i];
                for b in 0..p {
                    xtx[a][b] += w * r[a] * r[b];
                }
            }
        }
        let Some(inverse) = stats::invert(&xtx) else {
            return 0.0;
        };
        let beta: Vec<f64> = (0..p)
            .map(|a| (0..p).map(|b| inverse[a][b] * xty[b]).sum())
            .collect();
        let sse: f64 = rows
            .iter()
            .zip(ids)
            .map(|(r, &i)| {
                let fit: f64 = r.iter().zip(&beta).map(|(x, b)| x * b).sum();
                self.weights[i] * (self.y[i] - fit).powi(2)
            })
            .sum();
        let df = ids.len() - p;
        let variance = sse / df as f64 * inverse[k + 1][k + 1];
        if variance.is_nan() || variance <= 0.0 {
            return if beta[k + 1] == 0.0 {
                0.0
            } else {
                f64::NEG_INFINITY
            };
        }
        stats::f_log_survival(beta[k + 1] * beta[k + 1] / variance, 1, df)
    }
    fn linear(&self, ids: &[usize], terms: Vec<Term>, constant: &LeafModel) -> Option<LeafModel> {
        if terms.is_empty() || terms.len() + 1 > ids.len() {
            return None;
        }
        let x: Vec<Vec<f64>> = ids
            .iter()
            .map(|&i| {
                std::iter::once(1.0)
                    .chain(terms.iter().map(|t| t.value(&self.data.rows[i])))
                    .collect()
            })
            .collect();
        let y: Vec<f64> = ids.iter().map(|&i| self.y[i]).collect();
        let w: Vec<f64> = ids.iter().map(|&i| self.weights[i]).collect();
        let (beta, converged) = if matches!(self.loss, Loss::SquaredError) {
            (stats::least_squares(&x, &y, &w)?, true)
        } else if let Loss::Quantile { probability } = self.loss {
            if let Some(fit) = stats::quantile_regression(&x, &y, &w, probability) {
                (fit.coefficients, fit.converged)
            } else {
                // Retain a usable constant prediction but expose the numerical
                // failure through the node status and fit-report warning.
                let mut beta = vec![0.0; terms.len() + 1];
                beta[0] = constant.scalar(&self.data.rows[ids[0]], 1.0, &self.loss);
                (beta, false)
            }
        } else {
            let mut beta = vec![0.0; terms.len() + 1];
            let mu = constant.scalar(&self.data.rows[ids[0]], 1.0, &self.loss);
            // Boundary optima have no finite log/logit coefficient; retain the
            // exact constant instead of reporting a spurious converged GLM.
            if mu == 0.0 || (matches!(self.loss, Loss::Logistic) && mu == 1.0) {
                return Some(constant.clone());
            }
            beta[0] = if matches!(self.loss, Loss::Logistic) {
                (mu.clamp(1e-8, 1.0 - 1e-8) / (1.0 - mu.clamp(1e-8, 1.0 - 1e-8))).ln()
            } else {
                mu.ln()
            };
            let objective = |b: &[f64]| {
                ids.iter()
                    .enumerate()
                    .map(|(k, &i)| {
                        let eta = x[k].iter().zip(b).map(|(x, b)| x * b).sum::<f64>();
                        let mu = if matches!(self.loss, Loss::Logistic) {
                            stats::sigmoid(eta)
                        } else {
                            (eta + self.data.offset(i).ln()).exp()
                        };
                        self.weights[i] * observation_loss(&self.loss, self.y[i], mu)
                    })
                    .sum::<f64>()
            };
            let mut old = objective(&beta);
            let mut converged = false;
            for _ in 0..100 {
                let mut z = vec![];
                let mut iw = vec![];
                for (k, &i) in ids.iter().enumerate() {
                    let eta = x[k].iter().zip(&beta).map(|(x, b)| x * b).sum::<f64>();
                    let (mu, var) = if matches!(self.loss, Loss::Logistic) {
                        let p = stats::sigmoid(eta).clamp(1e-8, 1.0 - 1e-8);
                        (p, p * (1.0 - p))
                    } else {
                        let mu = (eta + self.data.offset(i).ln()).exp();
                        (mu, mu.max(1e-8))
                    };
                    z.push(eta + (self.y[i] - mu) / var);
                    iw.push(self.weights[i] * var);
                }
                let candidate = stats::least_squares(&x, &z, &iw)?;
                let mut step = 1.0;
                let mut accepted = None;
                for _ in 0..30 {
                    let b: Vec<f64> = beta
                        .iter()
                        .zip(&candidate)
                        .map(|(a, b)| a + step * (b - a))
                        .collect();
                    let obj = objective(&b);
                    if obj.is_finite() && stats::less_or_close(obj, old) {
                        accepted = Some((b, obj));
                        break;
                    }
                    step *= 0.5;
                }
                let Some((b, obj)) = accepted else {
                    break;
                };
                let change = beta
                    .iter()
                    .zip(&b)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f64::max);
                beta = b;
                old = obj;
                if change < 1e-7 {
                    converged = true;
                    break;
                }
            }
            (beta, converged)
        };
        Some(LeafModel::Linear {
            intercept: beta[0],
            coefficients: beta[1..].to_vec(),
            terms,
            loss: self.loss.clone(),
            converged,
            prediction_bounds: self.prediction_bounds(ids),
        })
    }
    fn prediction_bounds(&self, ids: &[usize]) -> Option<PredictionBounds> {
        match self.options.prediction_truncation {
            PredictionTruncation::None => None,
            PredictionTruncation::GlobalRange => self.global_prediction_bounds.clone(),
            mode => {
                let mut lower = ids.iter().map(|&i| self.y[i]).fold(f64::INFINITY, f64::min);
                let mut upper = ids
                    .iter()
                    .map(|&i| self.y[i])
                    .fold(f64::NEG_INFINITY, f64::max);
                if mode == PredictionTruncation::NodeRangeExpanded {
                    let padding = 0.1 * upper - 0.1 * lower;
                    lower -= padding;
                    upper += padding;
                }
                Some(PredictionBounds { lower, upper })
            }
        }
    }
    pub fn risk(&self, model: &LeafModel, ids: &[usize]) -> f64 {
        // A constant prediction without exposure does not depend on the row,
        // so skip the per-row model evaluation; the losses are the same.
        if let LeafModel::Constant { value } = model
            && self.loss != Loss::Poisson
        {
            return ids
                .iter()
                .map(|&i| self.weights[i] * observation_loss(&self.loss, self.y[i], *value))
                .sum();
        }
        ids.iter()
            .map(|&i| self.weights[i] * self.row_loss(model, i))
            .sum()
    }
    pub fn row_loss(&self, model: &LeafModel, i: usize) -> f64 {
        if let Some(predicted) = model.predicted_class(&self.data.rows[i]) {
            self.costs[self.y[i] as usize][predicted]
        } else if matches!(
            model,
            LeafModel::Linear {
                loss: Loss::SquaredError,
                ..
            }
        ) {
            // Fit/search/weakest-link risks use the ordinary least-squares fit;
            // prediction and held-out errors use the selected truncation rule.
            observation_loss(
                &self.loss,
                self.y[i],
                model.fitting_scalar(&self.data.rows[i], self.data.offset(i), &self.loss),
            )
        } else {
            model.regression_loss(
                &self.data.rows[i],
                self.data.offset(i),
                self.y[i],
                &self.loss,
            )
        }
    }
    pub fn impurity(&self, model: &LeafModel, ids: &[usize]) -> f64 {
        if let LeafModel::Classification { probabilities, .. } = model {
            self.gini(probabilities, ids.len())
        } else {
            // Including LDA leaves, whose impurity is their misclassification cost.
            self.risk(model, ids)
        }
    }
    /// The impurity of a classification leaf with `n` rows.
    pub fn gini(&self, probabilities: &[f64], n: usize) -> f64 {
        // GUIDE weights child Gini values by observation counts, not
        // prior-adjusted probability mass (Loh 2009, equation 2.2).
        // Unequal costs alter the splitting priors by class cost sums;
        // reported posteriors and predicted classes retain the original priors.
        let p = || {
            probabilities
                .iter()
                .zip(&self.split_cost_weights)
                .map(|(p, cost)| p * cost)
        };
        let total = p().sum::<f64>();
        if total <= 0.0 {
            return 0.0;
        }
        n as f64 * (1.0 - p().map(|v| (v / total).powi(2)).sum::<f64>()).max(0.0)
    }

    /// Apply GUIDE's zero-posterior correction after growth and pruning.
    /// Keeping raw probabilities during search preserves pure-node stopping.
    pub fn finalize_probabilities(&self, node: &mut crate::Node) {
        if let LeafModel::Classification {
            probabilities,
            predicted_class,
        } = &mut node.model
        {
            let raw = probabilities.clone();
            self.correct_zero_posteriors(probabilities);
            // GUIDE keeps the decision of the raw probabilities but breaks
            // ties by the corrected ones. Prior-adjusted probabilities of equal
            // counts can differ in the last bits, so ties are relative.
            let cost = |p: &[f64], j: usize| {
                p.iter()
                    .enumerate()
                    .map(|(i, v)| v * self.costs[i][j])
                    .sum::<f64>()
            };
            let best = cost(&raw, *predicted_class);
            let tied: Vec<usize> = (0..raw.len())
                .filter(|&j| cost(&raw, j) - best <= 1e-12 * best.abs())
                .collect();
            if tied.len() > 1 {
                *predicted_class = tied
                    .iter()
                    .copied()
                    .min_by(|&a, &b| {
                        cost(probabilities, a)
                            .total_cmp(&cost(probabilities, b))
                            .then(a.cmp(&b))
                    })
                    .unwrap_or(*predicted_class);
            }
        }
        if let Some(left) = &mut node.left {
            self.finalize_probabilities(left);
        }
        if let Some(right) = &mut node.right {
            self.finalize_probabilities(right);
        }
    }
    /// GUIDE's zero-posterior correction: if any class has probability 0,
    /// shrink all probabilities toward the priors by one pseudo-observation.
    pub fn correct_zero_posteriors(&self, probabilities: &mut [f64]) {
        if probabilities.contains(&0.0) {
            let n = self.root_samples as f64;
            for (p, prior) in probabilities.iter_mut().zip(&self.class_priors) {
                *p = (n * *p + prior) / (n + 1.0);
            }
        }
    }
}
pub(crate) fn observation_loss(loss: &Loss, y: f64, p: f64) -> f64 {
    match loss {
        // The scalar version denotes the degenerate interval [p,p]. Actual
        // two-quantile nodes are evaluated by regression_loss above.
        Loss::DualQuantile { lower, upper } => quantile_pair_loss(y, p, p, *lower, *upper),
        Loss::SquaredError => (y - p).powi(2),
        Loss::Quantile { probability } => {
            let r = y - p;
            if r >= 0.0 {
                probability * r
            } else {
                (probability - 1.0) * r
            }
        }
        Loss::Poisson => {
            if !p.is_finite() || p < 0.0 {
                return f64::INFINITY;
            }
            if y == 0.0 {
                return 2.0 * p;
            }
            let p = p.max(1e-300);
            let delta = (y - p) / p;
            let half = if delta.abs() < 1e-4 {
                // (1+d)log(1+d)-d, evaluated without subtracting first-order
                // terms. Direct deviance loses all accuracy near y == p.
                p * delta
                    * delta
                    * (0.5
                        + delta
                            * (-1.0 / 6.0
                                + delta * (1.0 / 12.0 + delta * (-1.0 / 20.0 + delta / 30.0))))
            } else if delta.is_finite() && delta > -1.0 {
                y * delta.ln_1p() - (y - p)
            } else {
                y * (y.ln() - p.ln()) - y + p
            };
            2.0 * half.max(0.0)
        }
        Loss::Logistic => {
            let p = p.clamp(1e-15, 1.0 - 1e-15);
            -2.0 * (y * p.ln() + (1.0 - y) * (-p).ln_1p())
        }
    }
}
