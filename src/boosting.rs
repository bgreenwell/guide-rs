//! Gradient boosting with GUIDE trees. In each node, chi-square tests of the
//! gradients against each predictor's GUIDE bins (and bin pairs) choose the
//! split variable, the regularized second-order gain chooses its split, and
//! leaves hold Newton steps (Chen and Guestrin 2016 for the gain; Friedman
//! 2001 for boosting). For squared error the tests are GUIDE's, on the signs
//! of the negative gradients. For the other losses the sign is mostly fixed
//! by the response (it is the class label for classification), so the tests
//! are score tests for bin effects on the current fit instead. The tests rank
//! variables; after the first round they are not significance tests.
use crate::{
    node::Context,
    selection::{self, SelectionMode},
    tree::{PARALLEL_MIN_ROWS, candidates, ordered_map, partition, validate_supported},
    *,
};
use rand::{SeedableRng, seq::SliceRandom};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct BoostingOptions {
    /// Maximum number of boosting rounds.
    pub rounds: usize,
    /// Shrinkage applied to every leaf value.
    pub learning_rate: f64,
    /// Depth of each tree; boosted trees are not pruned.
    pub max_depth: usize,
    /// Minimum sum of Hessians in a child.
    pub min_child_weight: f64,
    /// L2 penalty on leaf values.
    pub lambda: f64,
    /// Minimum gain to split.
    pub gamma: f64,
    /// Fraction of training rows sampled, without replacement, each round.
    pub subsample: f64,
    /// Stop when the validation loss has not improved for this many rounds;
    /// requires a validation dataset.
    pub early_stopping_rounds: Option<usize>,
}
impl Default for BoostingOptions {
    fn default() -> Self {
        Self {
            rounds: 100,
            learning_rate: 0.1,
            max_depth: 3,
            min_child_weight: 1.0,
            lambda: 1.0,
            gamma: 0.0,
            subsample: 1.0,
            early_stopping_rounds: None,
        }
    }
}
impl BoostingOptions {
    pub fn validate(&self) -> Result<()> {
        if self.rounds == 0
            || !self.learning_rate.is_finite()
            || self.learning_rate <= 0.0
            || self.max_depth == 0
            || self.max_depth > 64
            || !self.min_child_weight.is_finite()
            || self.min_child_weight < 0.0
            || !self.lambda.is_finite()
            || self.lambda < 0.0
            || !self.gamma.is_finite()
            || self.gamma < 0.0
            || !self.subsample.is_finite()
            || self.subsample <= 0.0
            || self.subsample > 1.0
            || self.early_stopping_rounds == Some(0)
        {
            return Err(invalid(
                "boosting needs positive rounds, learning_rate and early_stopping_rounds, \
                 max_depth in 1..=64, nonnegative min_child_weight, lambda and gamma, \
                 and subsample in (0, 1]",
            ));
        }
        Ok(())
    }
}

/// What a boosted model adds to its trees.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BoostedModel {
    pub options: BoostingOptions,
    /// Initial score of each output on the link scale: one for regression and
    /// binary classification (the log odds of the second class), one per
    /// class otherwise.
    pub base_scores: Vec<f64>,
    /// Rounds kept; with early stopping, the best validation round.
    pub rounds: usize,
    /// Validation loss after each round, when early stopping was used.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub validation_loss: Vec<f64>,
}
impl BoostedModel {
    /// Trees per round: one per class for three or more classes, else one.
    pub fn outputs(&self) -> usize {
        self.base_scores.len()
    }
}

/// The supported losses, with their gradients and inverse links.
#[derive(Clone, Copy, PartialEq)]
enum Objective {
    SquaredError,
    Poisson,
    /// Regression on a 0/1 response.
    Logistic,
    /// Classification; `classes` is the number of classes.
    Softmax {
        classes: usize,
    },
}
impl Objective {
    fn of(options: &FitOptions, classes: usize) -> Result<Self> {
        Ok(match &options.task {
            Task::Classification => Self::Softmax { classes },
            Task::Regression { loss } => match loss {
                Loss::SquaredError => Self::SquaredError,
                Loss::Poisson => Self::Poisson,
                Loss::Logistic => Self::Logistic,
                _ => {
                    return Err(invalid(
                        "boosting supports squared-error, Poisson and logistic regression \
                         and classification",
                    ));
                }
            },
        })
    }
    fn outputs(&self) -> usize {
        match self {
            Self::Softmax { classes } if *classes > 2 => *classes,
            _ => 1,
        }
    }
}

/// Score test for bin effects on the current fit: with `G` and `H` the
/// gradient and Hessian sums of the rows and `G_b`, `H_b` those of bin `b`,
/// `sum_b (G_b - H_b G / H)^2 / H_b`, on (nonempty bins - 1) degrees of
/// freedom. The Hessian is the variance of the gradient when the loss is a
/// log likelihood, as for the Poisson, logistic and softmax losses.
fn score_test(ids: &[usize], g: &[f64], h: &[f64], bins: &[usize]) -> (f64, usize, f64) {
    let width = bins.iter().max().map_or(0, |m| m + 1);
    let (mut gb, mut hb) = (vec![0.0; width], vec![0.0; width]);
    for (&i, &b) in ids.iter().zip(bins) {
        gb[b] += g[i];
        hb[b] += h[i];
    }
    let (gs, hs): (f64, f64) = (gb.iter().sum(), hb.iter().sum());
    let mut stat = 0.0;
    let mut active = 0_usize;
    for (&gb, &hb) in gb.iter().zip(&hb) {
        if hb > 0.0 {
            stat += (gb - hb * gs / hs).powi(2) / hb;
            active += 1;
        }
    }
    let df = active.saturating_sub(1);
    (stat, df, stats::chi_square_log_survival(stat, df))
}

/// Probabilities of every class from per-output scores.
fn class_probabilities(scores: &[f64], classes: usize) -> Vec<f64> {
    if classes == 2 {
        let p = stats::sigmoid(scores[0]);
        return vec![1.0 - p, p];
    }
    let top = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = scores.iter().map(|s| (s - top).exp()).collect();
    let total: f64 = e.iter().sum();
    e.iter().map(|v| v / total).collect()
}

impl FittedModel {
    /// Fits a boosted ensemble of GUIDE trees. `validation`, required with
    /// `early_stopping_rounds`, scores each round.
    pub fn fit_boosting(
        data: &Dataset,
        options: &FitOptions,
        boosting: &BoostingOptions,
        validation: Option<&Dataset>,
    ) -> Result<Self> {
        options.validate()?;
        boosting.validate()?;
        data.validate(&options.task)?;
        validate_supported(data, options)?;
        if options.pruning != Pruning::None {
            return Err(invalid("boosted trees are not pruned; set pruning to none"));
        }
        if options.node_model != NodeModel::Constant
            || options.linear_splits
            || options.class_priors.is_some()
            || options.class_costs.is_some()
            || !options.pruning_loss.is_default()
            || !options.prediction_truncation.is_none()
        {
            return Err(invalid(
                "boosting uses constant leaves without linear splits, priors, costs, \
                 pruning losses or prediction truncation",
            ));
        }
        if boosting.early_stopping_rounds.is_some() != validation.is_some() {
            return Err(invalid(
                "early stopping and a validation dataset must be used together",
            ));
        }
        let ids = data.active_rows();
        let global = Context::new(data, options, &ids, None)?;
        let objective = Objective::of(options, global.classes.len())?;
        let outputs = objective.outputs();
        if let Some(v) = validation {
            v.validate(&options.task)?;
            if v.schema != data.schema {
                return Err(invalid("validation feature schema differs from training"));
            }
            if data.offsets.is_empty() != v.offsets.is_empty() {
                return Err(invalid(
                    "validation data must supply exposure exactly when training does",
                ));
            }
        }

        // Selection and candidate generation run on a squared-error context
        // whose response is the negative gradient of the current output.
        let mut work_options = options.clone();
        work_options.task = Task::Regression {
            loss: Loss::SquaredError,
        };
        work_options.max_depth = boosting.max_depth;
        let mut work_data = data.clone();
        work_data.targets = data
            .targets
            .iter()
            .map(|t| {
                if t.is_missing() {
                    Value::Missing
                } else {
                    Value::Number(0.0)
                }
            })
            .collect();
        work_data.offsets.clear();
        let mut ctx = Context::new(&work_data, &work_options, &ids, None)?;
        ctx.weights.clone_from(&global.weights);

        let base_scores = base_scores(&global, &ids, objective)?;
        let n = data.rows.len();
        let mut scores = vec![base_scores.clone(); n];
        let validation_ctx = validation
            .map(|v| Context::new(v, options, &v.active_rows(), Some(&global.classes)))
            .transpose()?;
        let validation_ids = validation.map(|v| v.active_rows()).unwrap_or_default();
        let mut validation_scores =
            vec![base_scores.clone(); validation.map_or(0, |v| v.rows.len())];

        let mut rng = ChaCha8Rng::seed_from_u64(options.seed);
        let mut trees = vec![];
        let mut validation_loss = vec![];
        let mut best = (f64::INFINITY, 0);
        let mut gradients = vec![0.0; n];
        let mut hessians = vec![0.0; n];
        for round in 0..boosting.rounds {
            let mut sample = ids.clone();
            if boosting.subsample < 1.0 {
                sample.shuffle(&mut rng);
                sample.truncate(((ids.len() as f64 * boosting.subsample).round() as usize).max(1));
                sample.sort_unstable();
            }
            // Gradients for every output use the scores before this round.
            let round_scores = scores.clone();
            for k in 0..outputs {
                for &i in &ids {
                    let (g, h) = gradient(&global, data, objective, &round_scores[i], i, k);
                    gradients[i] = g * global.weights[i];
                    hessians[i] = h * global.weights[i];
                }
                for &i in &ids {
                    ctx.y[i] = -gradients[i];
                }
                let mut next = 0;
                let root = grow(
                    &ctx,
                    &sample,
                    &gradients,
                    &hessians,
                    objective == Objective::SquaredError,
                    0,
                    &mut next,
                    &mut rng,
                    boosting,
                );
                for (i, row) in data.rows.iter().enumerate() {
                    scores[i][k] += root.leaf(row).model.scalar(row, 1.0, &Loss::SquaredError);
                }
                if let Some(v) = validation {
                    for (i, row) in v.rows.iter().enumerate() {
                        validation_scores[i][k] +=
                            root.leaf(row).model.scalar(row, 1.0, &Loss::SquaredError);
                    }
                }
                trees.push(Tree { root });
            }
            if let (Some(v), Some(vctx), Some(patience)) =
                (validation, &validation_ctx, boosting.early_stopping_rounds)
            {
                let loss = mean_loss(vctx, v, objective, &validation_scores, &validation_ids);
                if !loss.is_finite() {
                    return Err(invalid("nonfinite validation loss during boosting"));
                }
                validation_loss.push(loss);
                if stats::strictly_less(loss, best.0) {
                    best = (loss, round + 1);
                } else if round + 1 - best.1 >= patience {
                    break;
                }
            }
        }
        let rounds = if validation.is_some() {
            best.1.max(1)
        } else {
            trees.len() / outputs
        };
        trees.truncate(rounds * outputs);
        let mut report = FitReport {
            training_rows: ids.len(),
            excluded_rows: data.rows.len() - ids.len(),
            ..FitReport::default()
        };
        report.unpruned_leaves = trees.iter().map(|t| t.root.leaves()).sum();
        report.fitted_leaves = report.unpruned_leaves;
        let mut fit_options = options.clone();
        fit_options.max_depth = boosting.max_depth;
        let model = Self {
            format_version: 1,
            algorithm_version: ALGORITHM_VERSION.into(),
            schema: data.schema.clone(),
            classes: global.classes.clone(),
            options: fit_options,
            input: None,
            requires_exposure: !data.offsets.is_empty(),
            ensemble: None,
            boosting: Some(BoostedModel {
                options: boosting.clone(),
                base_scores,
                rounds,
                validation_loss,
            }),
            trees,
            report,
        };
        model.validate()?;
        Ok(model)
    }

    /// Per-output scores of a boosted model on the link scale.
    pub(crate) fn boosted_scores(&self, boosted: &BoostedModel, row: &[Value]) -> Vec<f64> {
        let mut scores = boosted.base_scores.clone();
        for (t, tree) in self.trees.iter().enumerate() {
            scores[t % boosted.outputs()] +=
                tree.root
                    .leaf(row)
                    .model
                    .scalar(row, 1.0, &Loss::SquaredError);
        }
        scores
    }

    /// The prediction of a boosted model for one row.
    pub(crate) fn boosted_prediction(
        &self,
        boosted: &BoostedModel,
        row: &[Value],
        offset: f64,
    ) -> Prediction {
        let scores = self.boosted_scores(boosted, row);
        if self.classes.is_empty() {
            let value = match &self.options.task {
                Task::Regression {
                    loss: Loss::Poisson,
                } => (scores[0] + offset.ln()).exp(),
                Task::Regression {
                    loss: Loss::Logistic,
                } => stats::sigmoid(scores[0]),
                _ => scores[0],
            };
            return Prediction::Regression { value };
        }
        let p = class_probabilities(&scores, self.classes.len());
        // A classifier has at least two classes, so a most probable one exists.
        let best = (0..p.len())
            .max_by(|&a, &b| p[a].total_cmp(&p[b]).then(b.cmp(&a)))
            .unwrap();
        Prediction::Classification {
            label: self.classes[best].clone(),
            probabilities: self.classes.iter().cloned().zip(p).collect(),
        }
    }
}

/// The best constant score of each output.
fn base_scores(global: &Context<'_>, ids: &[usize], objective: Objective) -> Result<Vec<f64>> {
    let total: f64 = ids.iter().map(|&i| global.weights[i]).sum();
    let mean = |f: &dyn Fn(usize) -> f64| {
        ids.iter().map(|&i| global.weights[i] * f(i)).sum::<f64>() / total
    };
    let logit = |p: f64| {
        let p = p.clamp(1e-12, 1.0 - 1e-12);
        (p / (1.0 - p)).ln()
    };
    Ok(match objective {
        Objective::SquaredError => vec![mean(&|i| global.y[i])],
        Objective::Poisson => {
            let counts: f64 = ids.iter().map(|&i| global.weights[i] * global.y[i]).sum();
            let exposure: f64 = ids
                .iter()
                .map(|&i| global.weights[i] * global.data.offset(i))
                .sum();
            if counts <= 0.0 {
                return Err(invalid("Poisson boosting needs a positive count"));
            }
            vec![(counts / exposure).ln()]
        }
        Objective::Logistic => vec![logit(mean(&|i| global.y[i]))],
        Objective::Softmax { classes: 2 } => {
            vec![logit(mean(&|i| f64::from(u8::from(global.y[i] == 1.0))))]
        }
        Objective::Softmax { classes } => (0..classes)
            .map(|k| {
                mean(&|i| f64::from(u8::from(global.y[i] as usize == k)))
                    .max(1e-12)
                    .ln()
            })
            .collect(),
    })
}

/// Gradient and Hessian of row `i`'s loss for output `k`, unweighted.
fn gradient(
    global: &Context<'_>,
    data: &Dataset,
    objective: Objective,
    scores: &[f64],
    i: usize,
    k: usize,
) -> (f64, f64) {
    let y = global.y[i];
    match objective {
        Objective::SquaredError => (scores[0] - y, 1.0),
        Objective::Poisson => {
            let mu = (scores[0] + data.offset(i).ln()).exp();
            (mu - y, mu.max(1e-16))
        }
        Objective::Logistic => {
            let p = stats::sigmoid(scores[0]);
            (p - y, (p * (1.0 - p)).max(1e-16))
        }
        Objective::Softmax { classes } => {
            let p = class_probabilities(scores, classes);
            let (pk, target) = if classes == 2 {
                (p[1], y == 1.0)
            } else {
                (p[k], y as usize == k)
            };
            (
                pk - f64::from(u8::from(target)),
                (pk * (1.0 - pk)).max(1e-16),
            )
        }
    }
}

/// Weighted mean loss of the validation rows.
fn mean_loss(
    ctx: &Context<'_>,
    data: &Dataset,
    objective: Objective,
    scores: &[Vec<f64>],
    ids: &[usize],
) -> f64 {
    let total: f64 = ids.iter().map(|&i| ctx.weights[i]).sum();
    ids.iter()
        .map(|&i| {
            let y = ctx.y[i];
            let loss = match objective {
                Objective::SquaredError => (y - scores[i][0]).powi(2),
                Objective::Poisson => crate::node::observation_loss(
                    &Loss::Poisson,
                    y,
                    (scores[i][0] + data.offset(i).ln()).exp(),
                ),
                Objective::Logistic => {
                    let p = stats::sigmoid(scores[i][0]).clamp(1e-15, 1.0 - 1e-15);
                    -(y * p.ln() + (1.0 - y) * (1.0 - p).ln())
                }
                Objective::Softmax { classes } => -class_probabilities(&scores[i], classes)
                    [y as usize]
                    .max(1e-15)
                    .ln(),
            };
            ctx.weights[i] * loss
        })
        .sum::<f64>()
        / total
}

/// Grows one boosted tree on gradients `g` and Hessians `h` (weighted),
/// selecting split variables by gradient signs if `signs`, else by score
/// tests.
#[allow(clippy::too_many_arguments)]
fn grow(
    ctx: &Context<'_>,
    ids: &[usize],
    g: &[f64],
    h: &[f64],
    signs: bool,
    depth: usize,
    next_id: &mut usize,
    rng: &mut ChaCha8Rng,
    b: &BoostingOptions,
) -> Node {
    let sum = |ids: &[usize], v: &[f64]| ids.iter().map(|&i| v[i]).sum::<f64>();
    let (gs, hs) = (sum(ids, g), sum(ids, h));
    let score = |g: f64, h: f64| g * g / (h + b.lambda);
    let mut node = Node {
        id: *next_id,
        sample_count: ids.len(),
        weight_sum: ids.iter().map(|&i| ctx.weights[i]).sum(),
        risk: 0.0,
        impurity: score(gs, hs),
        model: LeafModel::Score {
            value: -b.learning_rate * gs / (hs + b.lambda),
        },
        diagnostics: vec![],
        split: None,
        left: None,
        right: None,
    };
    *next_id += 1;
    if depth >= b.max_depth
        || ids.len() < 2 * ctx.options.min_samples_leaf
        || hs < 2.0 * b.min_child_weight
    {
        return node;
    }
    let mut features: Vec<usize> = ctx
        .data
        .schema
        .iter()
        .enumerate()
        .filter(|(_, s)| s.role.splits())
        .map(|(j, _)| j)
        .collect();
    if let Some(m) = ctx.options.mtry {
        features.shuffle(rng);
        features.truncate(m);
        features.sort_unstable();
    }
    let selection = if signs {
        let labels: Vec<usize> = ids.iter().map(|&i| usize::from(-g[i] > 0.0)).collect();
        selection::select_with_labels(ctx, ids, &labels, &features, SelectionMode::Fit)
    } else {
        selection::select_with_test(ctx, ids, &features, SelectionMode::Fit, 2, None, |bins| {
            score_test(ids, g, h, bins)
        })
    };
    node.diagnostics = selection.diagnostics;
    let splits: Vec<Split> = selection
        .features
        .iter()
        .flat_map(|&j| candidates(ctx, ids, j))
        .collect();
    let gains = ordered_map(
        &splits,
        ids.len() >= PARALLEL_MIN_ROWS,
        || (vec![], vec![]),
        |(left, right), split| {
            partition(ids, split, ctx.data, left, right);
            let usable = |side: &[usize]| {
                side.len() >= ctx.options.min_samples_leaf
                    && side.iter().any(|&i| ctx.weights[i] > 0.0)
                    && sum(side, h) >= b.min_child_weight
            };
            (usable(left) && usable(right)).then(|| {
                0.5 * (score(sum(left, g), sum(left, h)) + score(sum(right, g), sum(right, h))
                    - score(gs, hs))
                    - b.gamma
            })
        },
    );
    // Highest gain; ties keep the first candidate.
    let best = splits
        .iter()
        .zip(gains)
        .filter_map(|(s, gain)| gain.filter(|v| v.is_finite()).map(|v| (s, v)))
        .fold(None::<(&Split, f64)>, |best, (s, v)| match best {
            Some((_, w)) if !stats::strictly_less(w, v) => best,
            _ => Some((s, v)),
        });
    if let Some((split, gain)) = best
        && gain > 0.0
    {
        let (mut left, mut right) = (vec![], vec![]);
        partition(ids, split, ctx.data, &mut left, &mut right);
        node.split = Some(split.clone());
        node.left = Some(Box::new(grow(
            ctx,
            &left,
            g,
            h,
            signs,
            depth + 1,
            next_id,
            rng,
            b,
        )));
        node.right = Some(Box::new(grow(
            ctx,
            &right,
            g,
            h,
            signs,
            depth + 1,
            next_id,
            rng,
            b,
        )));
    }
    node
}
