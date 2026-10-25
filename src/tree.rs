use crate::{
    node::{Context, sorted_category_levels},
    selection::{self, SelectionMode},
    *,
};
use rand::{SeedableRng, seq::SliceRandom};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Missing values routed by their flag code (GUIDE M variables): a missing
/// value goes left exactly when its code is in `left`. Missing values with
/// no code, or with a code unseen in training, go right.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct MissingCodes {
    pub flag: usize,
    pub left: Vec<String>,
}
impl MissingCodes {
    fn goes_left(&self, row: &[Value]) -> bool {
        missing_code(row, Some(self.flag)).is_some_and(|code| self.left.iter().any(|l| l == code))
    }
}
fn route_missing(codes: &Option<MissingCodes>, missing_left: bool, row: &[Value]) -> bool {
    codes.as_ref().map_or(missing_left, |c| c.goes_left(row))
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Split {
    Linear {
        features: [usize; 2],
        coefficients: [f64; 2],
        threshold: f64,
        missing_left: bool,
    },
    Numeric {
        feature: usize,
        threshold: f64,
        missing_left: bool,
        /// Overrides `missing_left` for features with a missing-value flag.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        missing_codes: Option<MissingCodes>,
    },
    Categorical {
        feature: usize,
        left_categories: Vec<String>,
        known_categories: Vec<String>,
        missing_left: bool,
        unknown_left: bool,
        /// Overrides `missing_left` for features with a missing-value flag.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        missing_codes: Option<MissingCodes>,
    },
    /// Missing values go left; with `missing_codes`, only those codes do.
    Missing {
        feature: usize,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        missing_codes: Option<MissingCodes>,
    },
    Periodic {
        feature: usize,
        start: f64,
        end: f64,
        period: f64,
        missing_left: bool,
    },
}
impl Split {
    pub fn feature(&self) -> usize {
        match self {
            Self::Linear { features, .. } => features[0],
            Self::Numeric { feature, .. }
            | Self::Categorical { feature, .. }
            | Self::Missing { feature, .. }
            | Self::Periodic { feature, .. } => *feature,
        }
    }
    pub fn goes_left(&self, row: &[Value]) -> bool {
        let v = &row[self.feature()];
        match self {
            Self::Linear {
                features,
                coefficients,
                threshold,
                missing_left,
            } => row[features[0]]
                .number()
                .zip(row[features[1]].number())
                .map(|(a, b)| a * coefficients[0] + b * coefficients[1] <= *threshold)
                .unwrap_or(*missing_left),
            Self::Missing { missing_codes, .. } => {
                v.is_missing() && route_missing(missing_codes, true, row)
            }
            Self::Numeric {
                threshold,
                missing_left,
                missing_codes,
                ..
            } => v
                .number()
                .map(|x| x <= *threshold)
                .unwrap_or_else(|| route_missing(missing_codes, *missing_left, row)),
            Self::Categorical {
                left_categories,
                known_categories,
                missing_left,
                unknown_left,
                missing_codes,
                ..
            } => {
                if let Some(s) = v.label() {
                    if !known_categories.contains(&s) {
                        *unknown_left
                    } else {
                        left_categories.contains(&s)
                    }
                } else {
                    route_missing(missing_codes, *missing_left, row)
                }
            }
            Self::Periodic {
                start,
                end,
                period,
                missing_left,
                ..
            } => v
                .number()
                .map(|x| {
                    let x = x.rem_euclid(*period);
                    if start <= end {
                        x > *start && x <= *end
                    } else {
                        x > *start || x <= *end
                    }
                })
                .unwrap_or(*missing_left),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SelectionDiagnostic {
    pub feature: usize,
    pub chi_square: f64,
    pub degrees_of_freedom: usize,
    pub p_value: f64,
    pub importance_p_value: f64,
    /// Natural log of p_value. Absent in models saved before reference4.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub log_p_value: Option<f64>,
    /// Natural log of the main/interaction probability used for importance.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub importance_log_p_value: Option<f64>,
    pub interaction_with: Option<usize>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Node {
    pub id: usize,
    pub sample_count: usize,
    pub weight_sum: f64,
    /// Training loss divided by total training weight, if this node is collapsed.
    pub risk: f64,
    pub impurity: f64,
    pub model: LeafModel,
    pub diagnostics: Vec<SelectionDiagnostic>,
    pub split: Option<Split>,
    pub left: Option<Box<Node>>,
    pub right: Option<Box<Node>>,
}
impl Node {
    pub fn leaves(&self) -> usize {
        match (&self.left, &self.right) {
            (Some(l), Some(r)) => l.leaves() + r.leaves(),
            _ => 1,
        }
    }
    pub fn depth(&self) -> usize {
        match (&self.left, &self.right) {
            (Some(l), Some(r)) => 1 + l.depth().max(r.depth()),
            _ => 0,
        }
    }
    pub fn leaf<'a>(&'a self, row: &[Value]) -> &'a Node {
        match (&self.split, &self.left, &self.right) {
            (Some(s), Some(l), Some(r)) => {
                if s.goes_left(row) {
                    l.leaf(row)
                } else {
                    r.leaf(row)
                }
            }
            _ => self,
        }
    }
    pub(crate) fn collapse(&mut self) {
        self.left = None;
        self.right = None;
        self.split = None;
    }
    pub(crate) fn leaf_risk(&self) -> f64 {
        match (&self.left, &self.right) {
            (Some(l), Some(r)) => l.leaf_risk() + r.leaf_risk(),
            _ => self.risk,
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Tree {
    pub root: Node,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct FitReport {
    pub training_rows: usize,
    pub excluded_rows: usize,
    pub unpruned_leaves: usize,
    pub fitted_leaves: usize,
    pub selected_alpha: f64,
    pub pruning_scores: Vec<crate::pruning::PruningScore>,
    pub out_of_bag_loss: Option<f64>,
    pub out_of_bag_rows: usize,
    pub warnings: Vec<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FittedModel {
    pub format_version: u32,
    pub algorithm_version: String,
    pub schema: Vec<FeatureSchema>,
    pub classes: Vec<String>,
    pub options: FitOptions,
    pub input: Option<CsvOptions>,
    pub requires_exposure: bool,
    pub ensemble: Option<EnsembleOptions>,
    pub trees: Vec<Tree>,
    pub report: FitReport,
}
impl FittedModel {
    pub fn fit(data: &Dataset, options: &FitOptions) -> Result<Self> {
        Self::fit_with_validation(data, options, None)
    }
    pub fn fit_with_validation(
        data: &Dataset,
        options: &FitOptions,
        validation: Option<&Dataset>,
    ) -> Result<Self> {
        options.validate()?;
        data.validate(&options.task)?;
        if validation.is_some() && options.pruning != Pruning::Validation {
            return Err(invalid("a validation dataset requires validation pruning"));
        }
        validate_supported(data, options)?;
        let ids = data.active_rows();
        let ctx = Context::new(data, options, &ids, None)?;
        let mut rng = ChaCha8Rng::seed_from_u64(options.seed);
        let mut next_id = 0;
        let tree = Tree {
            root: grow(&ctx, &ids, 0, &mut next_id, &mut rng, SelectionMode::Fit),
        };
        validate_node(&tree.root, &data.schema, ctx.classes.len(), options, 0)?;
        let mut report = FitReport {
            training_rows: ids.len(),
            excluded_rows: data.rows.len() - ids.len(),
            unpruned_leaves: tree.root.leaves(),
            ..FitReport::default()
        };
        if ctx.classifying() && ids.iter().any(|&i| data.weight(i) != 1.0) {
            report.warnings.push("Positive classification weights are treated as 1, following GUIDE; zero-weight rows are excluded.".into());
        }
        let mut tree = crate::pruning::prune(tree, &ctx, validation, &mut report)?;
        ctx.finalize_probabilities(&mut tree.root);
        report.fitted_leaves = tree.root.leaves();
        collect_warnings(&tree.root, &mut report.warnings);
        let model = Self {
            format_version: 1,
            algorithm_version: ALGORITHM_VERSION.into(),
            schema: data.schema.clone(),
            classes: ctx.classes.clone(),
            options: options.clone(),
            input: None,
            requires_exposure: !data.offsets.is_empty(),
            ensemble: None,
            trees: vec![tree],
            report,
        };
        model.validate()?;
        Ok(model)
    }
    pub fn predict(&self, rows: &[Vec<Value>]) -> Result<Vec<Prediction>> {
        self.predict_with_offsets(rows, None)
    }
    pub fn predict_with_offsets(
        &self,
        rows: &[Vec<Value>],
        offsets: Option<&[f64]>,
    ) -> Result<Vec<Prediction>> {
        self.validate()?;
        if self.requires_exposure && offsets.is_none() {
            return Err(invalid(
                "this model was fitted with exposure; provide prediction offsets",
            ));
        }
        if offsets.is_some_and(|v| v.len() != rows.len()) {
            return Err(invalid("one exposure is required per prediction row"));
        }
        if offsets.is_some()
            && !matches!(
                self.options.task,
                Task::Regression {
                    loss: Loss::Poisson
                }
            )
        {
            return Err(invalid("prediction offsets require Poisson regression"));
        }
        let loss = if let Task::Regression { loss } = &self.options.task {
            loss.clone()
        } else {
            Loss::SquaredError
        };
        rows.iter()
            .enumerate()
            .map(|(i, row)| {
                crate::data::validate_row(&self.schema, row)?;
                let offset = offsets.map(|o| o[i]).unwrap_or(1.0);
                if !offset.is_finite() || offset <= 0.0 {
                    return Err(invalid("prediction exposure must be positive and finite"));
                }
                let p = self.predict_one(row, offset, &loss);
                if let Prediction::Regression { value } = &p
                    && !value.is_finite()
                {
                    return Err(invalid(
                        "prediction overflow; check feature ranges and exposure",
                    ));
                }
                Ok(p)
            })
            .collect()
    }
    pub(crate) fn predict_one(&self, row: &[Value], offset: f64, loss: &Loss) -> Prediction {
        if matches!(loss, Loss::DualQuantile { .. }) {
            let mut lower = 0.0;
            let mut upper = 0.0;
            for t in &self.trees {
                if let LeafModel::Quantiles {
                    lower: lo,
                    upper: hi,
                } = &t.root.leaf(row).model
                {
                    lower += lo / self.trees.len() as f64;
                    upper += hi / self.trees.len() as f64;
                }
            }
            Prediction::Quantiles { lower, upper }
        } else if self.classes.is_empty() {
            let value = self
                .trees
                .iter()
                .map(|t| t.root.leaf(row).model.scalar(row, offset, loss) / self.trees.len() as f64)
                .sum();
            Prediction::Regression { value }
        } else {
            let mut p = vec![0.0; self.classes.len()];
            for t in &self.trees {
                if let LeafModel::Classification { probabilities, .. } = &t.root.leaf(row).model {
                    for (a, b) in p.iter_mut().zip(probabilities) {
                        *a += b / self.trees.len() as f64;
                    }
                }
            }
            let c = if self.ensemble.is_none() {
                let LeafModel::Classification {
                    predicted_class, ..
                } = &self.trees[0].root.leaf(row).model
                else {
                    unreachable!()
                };
                *predicted_class
            } else {
                // Validated classification models have at least one class.
                (0..p.len())
                    .min_by(|&a, &b| {
                        let cost = |j| {
                            p.iter()
                                .enumerate()
                                .map(|(i, v)| {
                                    v * self
                                        .options
                                        .class_costs
                                        .as_ref()
                                        .map(|c| c[i][j])
                                        .unwrap_or(if i == j { 0.0 } else { 1.0 })
                                })
                                .sum::<f64>()
                        };
                        cost(a).total_cmp(&cost(b)).then(a.cmp(&b))
                    })
                    .unwrap()
            };
            Prediction::Classification {
                label: self.classes[c].clone(),
                probabilities: self.classes.iter().cloned().zip(p).collect(),
            }
        }
    }
    pub fn to_json(&self) -> Result<String> {
        self.validate()?;
        Ok(serde_json::to_string_pretty(self)?)
    }
    pub fn from_json(input: &str) -> Result<Self> {
        let model: Self = serde_json::from_str(input)?;
        model.validate()?;
        Ok(model)
    }
    pub fn validate(&self) -> Result<()> {
        if self.format_version != 1 {
            return Err(invalid("unsupported model format_version"));
        }
        self.options.validate()?;
        crate::data::validate_schema(&self.schema)?;
        if self.trees.is_empty() {
            return Err(invalid("model has no trees"));
        }
        if let Some(e) = &self.ensemble {
            if e.trees != self.trees.len()
                || !e.sample_fraction.is_finite()
                || e.sample_fraction <= 0.0
                || e.sample_fraction > 1.0
            {
                return Err(invalid("saved ensemble configuration is inconsistent"));
            }
        } else if self.trees.len() != 1 {
            return Err(invalid("multiple trees require ensemble metadata"));
        }
        if self.requires_exposure
            && !matches!(
                self.options.task,
                Task::Regression {
                    loss: Loss::Poisson
                }
            )
        {
            return Err(invalid("exposure metadata requires Poisson regression"));
        }
        if matches!(self.options.task, Task::Classification) == self.classes.is_empty() {
            return Err(invalid("model classes do not match task"));
        }
        if self.classes.windows(2).any(|w| w[0] >= w[1]) {
            return Err(invalid("class labels must be unique and sorted"));
        }
        if self
            .options
            .class_priors
            .as_ref()
            .is_some_and(|p| p.len() != self.classes.len())
        {
            return Err(invalid("saved class priors do not match class labels"));
        }
        if let Some(c) = &self.options.class_costs {
            let k = self.classes.len();
            if c.len() != k
                || c.iter().enumerate().any(|(i, r)| {
                    r.len() != k || r.iter().any(|v| !v.is_finite() || *v < 0.0) || r[i] != 0.0
                })
            {
                return Err(invalid("invalid saved classification costs"));
            }
        }
        for tree in &self.trees {
            let mut pending = vec![&tree.root];
            let mut ids = std::collections::BTreeSet::new();
            while let Some(n) = pending.pop() {
                if !ids.insert(n.id) {
                    return Err(invalid("node IDs must be unique within a tree"));
                }
                if let Some(l) = &n.left {
                    pending.push(l);
                }
                if let Some(r) = &n.right {
                    pending.push(r);
                }
            }
            validate_node(
                &tree.root,
                &self.schema,
                self.classes.len(),
                &self.options,
                0,
            )?;
        }
        Ok(())
    }
}
pub(crate) fn collect_warnings(n: &Node, w: &mut Vec<String>) {
    if matches!(
        n.model,
        LeafModel::Linear {
            converged: false,
            ..
        }
    ) && !w.iter().any(|s| s.contains("convergence tolerance"))
    {
        w.push("At least one iterative node fit did not meet convergence tolerance; inspect before inference.".into());
    }
    if let Some(l) = &n.left {
        collect_warnings(l, w);
    }
    if let Some(r) = &n.right {
        collect_warnings(r, w);
    }
}
/// GUIDE converts flags to indicator variables for linear models; that
/// conversion is not implemented, and code subsets are enumerated exhaustively.
const MAX_MISSING_CODES: usize = 10;
pub(crate) fn validate_supported(data: &Dataset, options: &FitOptions) -> Result<()> {
    for (j, flag) in flag_columns(&data.schema).into_iter().enumerate() {
        let Some(flag) = flag else { continue };
        if options.node_model != NodeModel::Constant {
            return Err(invalid(
                "missing-value flags require constant node models; GUIDE's flag-to-indicator conversion for linear models is not implemented",
            ));
        }
        let codes: std::collections::BTreeSet<_> = data
            .rows
            .iter()
            .filter(|row| row[j].is_missing())
            .filter_map(|row| missing_code(row, Some(flag)))
            .collect();
        if codes.len() > MAX_MISSING_CODES {
            return Err(invalid(format!(
                "flag {} has {} missing-value codes; at most {MAX_MISSING_CODES} are supported",
                data.schema[flag].name,
                codes.len()
            )));
        }
    }
    if options.node_model != NodeModel::Constant && !data.schema.iter().any(|s| s.role.fits()) {
        return Err(invalid(
            "nonconstant node models require at least one feature with role fit or both",
        ));
    }
    let classes: std::collections::BTreeSet<_> = data
        .active_rows()
        .iter()
        .filter_map(|&i| data.targets[i].label())
        .collect();
    if matches!(options.task, Task::Classification) && classes.len() > 2 {
        for (j, f) in data.schema.iter().enumerate() {
            if f.role.splits() && matches!(f.feature_type, FeatureType::Categorical) {
                let categories: std::collections::BTreeSet<_> = data
                    .active_rows()
                    .iter()
                    .filter_map(|&i| data.rows[i][j].label())
                    .collect();
                if categories.len() > 11 {
                    return Err(invalid(format!(
                        "multiclass categorical feature {} has more than 11 levels; GUIDE's LDA/grouped-category search is not implemented",
                        f.name
                    )));
                }
            }
        }
    }
    Ok(())
}
pub(crate) fn validate_node(
    n: &Node,
    schema: &[FeatureSchema],
    classes: usize,
    options: &FitOptions,
    depth: usize,
) -> Result<()> {
    if depth > 64 {
        return Err(invalid("model exceeds depth limit"));
    }
    if !n.risk.is_finite()
        || !n.impurity.is_finite()
        || !n.weight_sum.is_finite()
        || n.sample_count == 0
        || n.weight_sum <= 0.0
    {
        return Err(invalid("invalid node statistics"));
    }
    match &n.model {
        LeafModel::Quantiles { lower, upper } => {
            if classes != 0
                || !lower.is_finite()
                || !upper.is_finite()
                || lower > upper
                || !matches!(
                    options.task,
                    Task::Regression {
                        loss: Loss::DualQuantile { .. }
                    }
                )
            {
                return Err(invalid("invalid dual-quantile node"));
            }
        }
        LeafModel::Classification {
            probabilities,
            predicted_class,
        } => {
            if probabilities.len() != classes
                || *predicted_class >= classes
                || probabilities.iter().any(|p| !p.is_finite() || *p < 0.0)
                || (probabilities.iter().sum::<f64>() - 1.0).abs() > 1e-8
            {
                return Err(invalid("invalid class probabilities"));
            }
        }
        LeafModel::Constant { value } => {
            if classes != 0
                || !value.is_finite()
                || matches!(options.task, Task::Regression {loss: Loss::Poisson} if *value<0.0)
                || matches!(options.task, Task::Regression {loss: Loss::Logistic} if !(0.0..=1.0).contains(value))
                || matches!(
                    options.task,
                    Task::Regression {
                        loss: Loss::DualQuantile { .. }
                    }
                )
            {
                return Err(invalid("invalid regression leaf"));
            }
        }
        LeafModel::Linear {
            intercept,
            coefficients,
            terms,
            loss,
            prediction_bounds,
            ..
        } => {
            if classes != 0
                || options.node_model == NodeModel::Constant
                || !matches!(&options.task,Task::Regression {loss:expected} if loss==expected)
                || !intercept.is_finite()
                || coefficients.len() != terms.len()
                || coefficients.iter().any(|b| !b.is_finite())
                || prediction_bounds.is_none() != options.prediction_truncation.is_none()
                || prediction_bounds.as_ref().is_some_and(|b| {
                    !b.lower.is_finite() || !b.upper.is_finite() || b.lower > b.upper
                })
                || terms.iter().any(|t| {
                    t.feature >= schema.len()
                        || t.power == 0
                        || t.power > 6
                        || !t.center.is_finite()
                        || !t.scale.is_finite()
                        || t.scale <= 0.0
                        || !t.impute.is_finite()
                        || !schema[t.feature].role.fits()
                        || (!t.missing_indicator
                            && t.category.is_some()
                                != matches!(
                                    schema[t.feature].feature_type,
                                    FeatureType::Categorical
                                ))
                })
            {
                return Err(invalid("invalid linear node model"));
            }
        }
    }
    if n.diagnostics.iter().any(|d| {
        d.feature >= schema.len()
            || d.interaction_with.is_some_and(|j| j >= schema.len())
            || !d.p_value.is_finite()
            || !(0.0..=1.0).contains(&d.p_value)
            || !d.importance_p_value.is_finite()
            || !(0.0..=1.0).contains(&d.importance_p_value)
            || d.log_p_value.is_some_and(|p| !p.is_finite() || p > 0.0)
            || d.importance_log_p_value
                .is_some_and(|p| !p.is_finite() || p > 0.0)
            || !d.chi_square.is_finite()
    }) {
        return Err(invalid("invalid selection diagnostics"));
    }
    match (&n.split, &n.left, &n.right) {
        (None, None, None) => {}
        (Some(s), Some(l), Some(r)) => {
            if s.feature() >= schema.len() || !schema[s.feature()].role.splits() {
                return Err(invalid("split feature outside schema"));
            }
            match s {
                Split::Linear {
                    features,
                    coefficients,
                    threshold,
                    ..
                } if features.iter().any(|&j| {
                    j >= schema.len()
                        || !schema[j].role.splits()
                        || !matches!(schema[j].feature_type, FeatureType::Numeric)
                }) || coefficients.iter().any(|x| !x.is_finite())
                    || !threshold.is_finite() =>
                {
                    return Err(invalid("invalid linear split"));
                }
                Split::Numeric { threshold, .. } if !threshold.is_finite() => {
                    return Err(invalid("nonfinite threshold"));
                }
                Split::Numeric { feature, .. }
                    if !matches!(schema[*feature].feature_type, FeatureType::Numeric) =>
                {
                    return Err(invalid("numeric split requires numeric feature"));
                }
                Split::Categorical {
                    feature,
                    left_categories,
                    known_categories,
                    ..
                } if !matches!(schema[*feature].feature_type, FeatureType::Categorical)
                    || left_categories
                        .iter()
                        .any(|v| !known_categories.contains(v)) =>
                {
                    return Err(invalid("invalid categorical split metadata"));
                }
                Split::Numeric {
                    feature,
                    missing_codes: Some(c),
                    ..
                }
                | Split::Categorical {
                    feature,
                    missing_codes: Some(c),
                    ..
                }
                | Split::Missing {
                    feature,
                    missing_codes: Some(c),
                } if c.flag >= schema.len()
                    || schema[c.flag].role != Role::Flag
                    || schema[c.flag].flags.as_ref() != Some(&schema[*feature].name) =>
                {
                    return Err(invalid("split missing codes must use the feature's flag"));
                }
                Split::Periodic {
                    start, end, period, ..
                } if !start.is_finite()
                    || !end.is_finite()
                    || !period.is_finite()
                    || *period <= 0.0 =>
                {
                    return Err(invalid("invalid periodic split"));
                }
                _ => {}
            }
            validate_node(l, schema, classes, options, depth + 1)?;
            validate_node(r, schema, classes, options, depth + 1)?;
        }
        _ => return Err(invalid("split and child nodes must appear together")),
    }
    Ok(())
}

pub(crate) fn grow(
    ctx: &Context<'_>,
    ids: &[usize],
    depth: usize,
    next_id: &mut usize,
    rng: &mut ChaCha8Rng,
    mode: SelectionMode,
) -> Node {
    let model = ctx.fit_leaf(ids);
    let risk = ctx.risk(&model, ids) / ctx.root_weight;
    let impurity = ctx.impurity(&model, ids);
    let mut node = Node {
        id: *next_id,
        sample_count: ids.len(),
        weight_sum: ids.iter().map(|&i| ctx.weights[i]).sum(),
        risk,
        impurity,
        model,
        diagnostics: vec![],
        split: None,
        left: None,
        right: None,
    };
    *next_id += 1;
    if depth >= ctx.options.max_depth
        || ids.len() < 2 * ctx.options.min_samples_leaf
        || impurity / ctx.root_weight <= ctx.options.min_gain
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
    let selection = selection::select(ctx, ids, &node.model, &features, mode);
    node.diagnostics = selection.diagnostics;
    let pair = selection.pair && mode == SelectionMode::Fit && depth + 1 < ctx.options.max_depth;
    let mut best: Option<(Split, Vec<usize>, Vec<usize>, f64)> = None;
    let split_candidates = if let Some(p) = &selection.linear {
        let values: Vec<Option<f64>> = ids
            .iter()
            .map(|&i| {
                ctx.data.rows[i][p.features[0]]
                    .number()
                    .zip(ctx.data.rows[i][p.features[1]].number())
                    .map(|(a, b)| a * p.coefficients[0] + b * p.coefficients[1])
            })
            .collect();
        let mut observed: Vec<f64> = values.iter().flatten().copied().collect();
        observed.sort_by(f64::total_cmp);
        observed.dedup();
        let cuts = reduce(
            observed
                .windows(2)
                .map(|x| {
                    let mid = x[0] / 2.0 + x[1] / 2.0;
                    if mid >= x[1] { x[0] } else { mid }
                })
                .collect(),
            ctx.options.max_split_candidates,
        );
        let directions = if values.iter().any(Option::is_none) {
            vec![false, true]
        } else {
            vec![false]
        };
        cuts.into_iter()
            .flat_map(|threshold| {
                directions.iter().map(move |&missing_left| Split::Linear {
                    features: p.features,
                    coefficients: p.coefficients,
                    threshold,
                    missing_left,
                })
            })
            .collect::<Vec<_>>()
    } else {
        selection
            .features
            .iter()
            .flat_map(|&j| candidates(ctx, ids, j))
            .collect()
    };
    for split in split_candidates {
        let j = split.feature();
        let (left, right) = partition(ids, &split, ctx.data);
        if !admissible(ctx, &left, &right) {
            continue;
        }
        let lm = ctx.fit_leaf(&left);
        let rm = ctx.fit_leaf(&right);
        let mut score = ctx.impurity(&lm, &left) + ctx.impurity(&rm, &right);
        if pair {
            // Pair selections hold two distinct features (a < b in selection).
            let other = selection
                .features
                .iter()
                .copied()
                .find(|k| *k != j)
                .unwrap();
            score = best_child_score(ctx, &left, &lm, other)
                + best_child_score(ctx, &right, &rm, other);
        }
        if score.is_finite()
            && best.as_ref().is_none_or(|b| {
                stats::strictly_less(score, b.3)
                    || (pair && j > b.0.feature() && stats::less_or_close(score, b.3))
                    || (ctx.classifying()
                        && j == b.0.feature()
                        && matches!(split, Split::Numeric { .. })
                        && stats::less_or_close(score, b.3)
                        && left.len().abs_diff(right.len()) < b.1.len().abs_diff(b.2.len()))
            })
        {
            best = Some((split, left, right, score));
        }
    }
    // GUIDE splits on the selected variable even when the best admissible
    // split leaves the loss unchanged, so `min_gain = 0` accepts zero gain
    // (up to roundoff). A positive `min_gain` remains a strict threshold.
    if let Some((mut split, left, right, score)) = best
        && (impurity - score > ctx.options.min_gain * ctx.root_weight
            || ((ctx.options.min_gain == 0.0 || mode == SelectionMode::Importance)
                && stats::less_or_close(score, impurity)))
    {
        // Numeric cuts default to the right when missingness was not observed,
        // as confirmed by GUIDE 46.2/46.3 reference fixtures. Categorical cuts
        // also default right. Linear and periodic splits remain unverified.
        if !ids.iter().any(|&i| match &split {
            Split::Linear { features, .. } => {
                features.iter().any(|&j| ctx.data.rows[i][j].is_missing())
            }
            _ => ctx.data.rows[i][split.feature()].is_missing(),
        }) {
            match &mut split {
                Split::Linear { missing_left, .. } | Split::Periodic { missing_left, .. } => {
                    *missing_left = left.len() >= right.len()
                }
                Split::Numeric { .. } | Split::Categorical { .. } | Split::Missing { .. } => {}
            }
        }
        let l = grow(ctx, &left, depth + 1, next_id, rng, mode);
        let r = grow(ctx, &right, depth + 1, next_id, rng, mode);
        // GUIDE 46.2/46.3 merge sibling leaves that predict the same class,
        // bottom-up, so a split survives only if its subtree changes a
        // prediction. Predicted classes already reflect priors and costs.
        // Importance growth keeps every split: its scores sum node statistics.
        if mode != SelectionMode::Fit || !same_class_leaves(&l, &r) {
            node.split = Some(split);
            node.left = Some(Box::new(l));
            node.right = Some(Box::new(r));
        }
    }
    node
}
fn same_class_leaves(left: &Node, right: &Node) -> bool {
    match (&left.model, &right.model) {
        (
            LeafModel::Classification {
                predicted_class: a, ..
            },
            LeafModel::Classification {
                predicted_class: b, ..
            },
        ) => left.split.is_none() && right.split.is_none() && a == b,
        _ => false,
    }
}
fn best_child_score(ctx: &Context<'_>, ids: &[usize], model: &LeafModel, j: usize) -> f64 {
    let mut best = ctx.impurity(model, ids);
    if ids.len() < 2 * ctx.options.min_samples_leaf {
        return best;
    }
    for s in candidates(ctx, ids, j) {
        let (l, r) = partition(ids, &s, ctx.data);
        if admissible(ctx, &l, &r) {
            best =
                best.min(ctx.impurity(&ctx.fit_leaf(&l), &l) + ctx.impurity(&ctx.fit_leaf(&r), &r));
        }
    }
    best
}
fn admissible(ctx: &Context<'_>, l: &[usize], r: &[usize]) -> bool {
    l.len() >= ctx.options.min_samples_leaf
        && r.len() >= ctx.options.min_samples_leaf
        && l.iter().any(|&i| ctx.weights[i] > 0.0)
        && r.iter().any(|&i| ctx.weights[i] > 0.0)
}
fn partition(ids: &[usize], split: &Split, data: &Dataset) -> (Vec<usize>, Vec<usize>) {
    ids.iter()
        .copied()
        .partition(|&i| split.goes_left(&data.rows[i]))
}
fn reduce<T: Clone>(v: Vec<T>, max: usize) -> Vec<T> {
    if max == 0 || v.len() <= max {
        v
    } else {
        (0..max)
            .map(|k| v[k * (v.len() - 1) / (max - 1)].clone())
            .collect()
    }
}
/// Missing-value routes for a split on feature `j`: all missing right, all
/// missing left and, for a flagged feature, each nonempty subset of the
/// node's missing codes going left. Unlike GUIDE 46.3, which never sends all
/// flagged missing values right, the empty subset stays a candidate.
fn missing_routes(ctx: &Context<'_>, ids: &[usize], j: usize) -> Vec<(bool, Option<MissingCodes>)> {
    let missing: Vec<usize> = ids
        .iter()
        .copied()
        .filter(|&i| ctx.data.rows[i][j].is_missing())
        .collect();
    if missing.is_empty() {
        return vec![(false, None)];
    }
    let Some(flag) = ctx.flags[j] else {
        return vec![(false, None), (true, None)];
    };
    let mut routes = vec![(false, None)];
    {
        let codes: std::collections::BTreeSet<&str> = missing
            .iter()
            .filter_map(|&i| missing_code(&ctx.data.rows[i], Some(flag)))
            .collect();
        let codes: Vec<&str> = codes.into_iter().collect();
        // GUIDE writes "or NA" only when the node's codes are every code the
        // data file records for this feature's missing values; otherwise it
        // names the codes, and unseen or uncoded values go right.
        let mut universe = std::collections::BTreeSet::new();
        let mut uncoded = false;
        for row in ctx.data.rows.iter().filter(|r| r[j].is_missing()) {
            match missing_code(row, Some(flag)) {
                Some(code) => {
                    universe.insert(code);
                }
                None => uncoded = true,
            }
        }
        let complete = !uncoded && universe.len() == codes.len();
        let all = (1usize << codes.len()) - 1;
        for mask in 1..=all {
            if mask == all && complete {
                continue; // identical to sending every missing value left
            }
            let left = codes
                .iter()
                .enumerate()
                .filter(|(b, _)| mask & (1 << b) != 0)
                .map(|(_, c)| (*c).to_owned())
                .collect();
            routes.push((false, Some(MissingCodes { flag, left })));
        }
    }
    // Last, so that cut splits prefer named codes on equal loss, as in GUIDE.
    routes.push((true, None));
    routes
}
/// GUIDE 46.2/46.3 put the side with fewer categories on the left, counting
/// missing values as one category, or each flag code and uncoded missing
/// values separately; ties keep the given orientation. Unseen levels and
/// uncoded missing values always go right, so a split needing them on the
/// left keeps its orientation.
fn orient_fewer_left(split: Split, node_missing: &[&[Value]]) -> Split {
    let Split::Categorical {
        feature,
        left_categories,
        known_categories,
        missing_left,
        unknown_left,
        missing_codes,
    } = split
    else {
        return split;
    };
    let flag = missing_codes.as_ref().map(|c| c.flag);
    let codes: std::collections::BTreeSet<&str> = node_missing
        .iter()
        .filter_map(|row| missing_code(row, flag))
        .collect();
    let uncoded = node_missing
        .iter()
        .any(|row| missing_code(row, flag).is_none());
    let (left_missing, right_missing) = match &missing_codes {
        Some(c) => (
            c.left.len(),
            codes.len() - c.left.len() + usize::from(uncoded),
        ),
        None if node_missing.is_empty() => (0, 0),
        None => (usize::from(missing_left), usize::from(!missing_left)),
    };
    let left = left_categories.len() + left_missing;
    let right = known_categories.len() - left_categories.len() + right_missing;
    if left <= right || (missing_codes.is_some() && uncoded) {
        return Split::Categorical {
            feature,
            left_categories,
            known_categories,
            missing_left,
            unknown_left,
            missing_codes,
        };
    }
    let flipped_codes = missing_codes.and_then(|c| {
        let left: Vec<String> = codes
            .iter()
            .filter(|code| !c.left.iter().any(|l| l == *code))
            .map(|code| (*code).to_owned())
            .collect();
        (!left.is_empty()).then_some(MissingCodes { flag: c.flag, left })
    });
    Split::Categorical {
        feature,
        left_categories: known_categories
            .iter()
            .filter(|c| !left_categories.contains(c))
            .cloned()
            .collect(),
        known_categories,
        // Without missing values in the node the route is a default, kept right.
        missing_left: !node_missing.is_empty() && flag.is_none() && !missing_left,
        unknown_left,
        missing_codes: flipped_codes,
    }
}
fn candidates(ctx: &Context<'_>, ids: &[usize], j: usize) -> Vec<Split> {
    let mut result = vec![];
    let routes = missing_routes(ctx, ids, j);
    if routes.len() > 1 && ids.iter().any(|&i| !ctx.data.rows[i][j].is_missing()) {
        // Missing-only splits try "all missing left" before named codes,
        // so equal-loss ties choose it, as in GUIDE.
        result.push(Split::Missing {
            feature: j,
            missing_codes: None,
        });
        for (_, missing_codes) in routes.iter().filter(|(_, c)| c.is_some()) {
            result.push(Split::Missing {
                feature: j,
                missing_codes: missing_codes.clone(),
            });
        }
    }
    let directions: Vec<bool> = routes
        .iter()
        .filter(|(_, codes)| codes.is_none())
        .map(|(left, _)| *left)
        .collect();
    match ctx.data.schema[j].feature_type {
        FeatureType::Numeric => {
            let mut values: Vec<f64> = ids
                .iter()
                .filter_map(|&i| ctx.data.rows[i][j].number())
                .collect();
            values.sort_by(f64::total_cmp);
            // Native numerical rank bounds apply before routing missing rows:
            // left >= minimum observed values, right > minimum. The exclusive
            // upper endpoint is confirmed by forward/reversed boundary probes.
            // Keep missingness-only splits as a separate candidate above.
            let minimum = ctx.options.min_samples_leaf;
            if values.len() <= minimum || values.len() - minimum <= minimum {
                return result;
            }
            let lower = values[minimum - 1];
            let upper = values[values.len() - minimum - 1];
            values.dedup();
            let cuts: Vec<f64> = values
                .windows(2)
                .map(|p| {
                    let mid = p[0] / 2.0 + p[1] / 2.0;
                    if mid >= p[1] { p[0] } else { mid }
                })
                .collect();
            for threshold in reduce(cuts, ctx.options.max_split_candidates) {
                if threshold < lower || threshold >= upper {
                    continue;
                }
                for (missing_left, missing_codes) in &routes {
                    result.push(Split::Numeric {
                        feature: j,
                        threshold,
                        missing_left: *missing_left,
                        missing_codes: missing_codes.clone(),
                    });
                }
            }
        }
        FeatureType::Periodic { period } => {
            let mut values: Vec<f64> = ids
                .iter()
                .filter_map(|&i| ctx.data.rows[i][j].number())
                .map(|x| x.rem_euclid(period))
                .collect();
            values.sort_by(f64::total_cmp);
            values.dedup();
            if values.len() > 1 {
                let mut cuts: Vec<f64> = values.windows(2).map(|p| (p[0] + p[1]) / 2.0).collect();
                cuts.push(
                    ((values[0] + values[values.len() - 1] + period) / 2.0).rem_euclid(period),
                );
                cuts.sort_by(f64::total_cmp);
                let cuts = reduce(cuts, ctx.options.max_split_candidates);
                for a in 0..cuts.len() {
                    for b in a + 1..cuts.len() {
                        for &missing_left in &directions {
                            result.push(Split::Periodic {
                                feature: j,
                                start: cuts[a],
                                end: cuts[b],
                                period,
                                missing_left,
                            });
                        }
                    }
                }
            }
        }
        FeatureType::Categorical => {
            let mut groups: BTreeMap<String, (f64, f64)> = BTreeMap::new();
            let exposure_scale = if matches!(ctx.loss, Loss::Poisson) {
                ids.iter().map(|&i| ctx.data.offset(i)).fold(0.0, f64::max)
            } else {
                1.0
            };
            for &i in ids {
                if let Some(s) = ctx.data.rows[i][j].label() {
                    let e = groups.entry(s).or_default();
                    e.0 += ctx.y[i] * ctx.weights[i];
                    e.1 += ctx.weights[i]
                        * if matches!(ctx.loss, Loss::Poisson) {
                            ctx.data.offset(i) / exposure_scale
                        } else {
                            1.0
                        };
                }
            }
            let known: Vec<String> = groups.keys().cloned().collect();
            let k = known.len();
            // Mean ordering is sufficient for constant SSE leaves, but can
            // omit the optimum when child regression slopes differ. Bound the
            // exhaustive model-based search at the existing 11-level limit.
            let exhaustive_linear = ctx.options.node_model == NodeModel::Linear
                && ctx.loss == Loss::SquaredError
                && k <= 11;
            if exhaustive_linear {
                // Native orientation fixes the first locally observed level
                // on the left. Include missingness as the lexical token NA;
                // excluded global levels cannot anchor a local split.
                let levels =
                    sorted_category_levels(ids.iter().map(|&i| ctx.data.rows[i][j].label()));
                if levels.len() < 2 {
                    return result;
                }
                // Replace the generic missing-only candidate as well, so its
                // orientation and unknown-level route use the same convention.
                result.clear();
                for mask in 0..(1usize << (levels.len() - 1)) - 1 {
                    let left: Vec<_> = levels
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| *i == 0 || mask & (1 << (i - 1)) != 0)
                        .map(|(_, level)| level)
                        .collect();
                    result.push(Split::Categorical {
                        feature: j,
                        left_categories: left.iter().filter_map(|level| (*level).clone()).collect(),
                        known_categories: known.clone(),
                        missing_left: left.iter().any(|level| level.is_none()),
                        unknown_left: false,
                        missing_codes: None,
                    });
                }
                return result;
            }
            if k < 2 {
                return result;
            }
            let sets: Vec<Vec<String>> = if ctx.classes.len() > 2 {
                (1..(1usize << (k - 1)))
                    .map(|mask| {
                        known
                            .iter()
                            .take(k - 1)
                            .enumerate()
                            .filter(|(i, _)| mask & (1 << i) != 0)
                            .map(|(_, s)| s.clone())
                            .collect()
                    })
                    .collect()
            } else {
                let mut order = known.clone();
                order.sort_by(|a, b| {
                    let x = groups[a];
                    let y = groups[b];
                    let order = (x.0 / x.1).total_cmp(&(y.0 / y.1));
                    // Binary classification puts the higher class proportion
                    // on the left; regression orders by increasing response.
                    (if ctx.classifying() {
                        order.reverse()
                    } else {
                        order
                    })
                    .then(a.cmp(b))
                });
                reduce(
                    (1..k).map(|n| order[..n].to_vec()).collect(),
                    ctx.options.max_split_candidates,
                )
            };
            // Prefixes of the real-level order crossed with every missing
            // route contain the optimum over levels and codes jointly.
            let node_missing: Vec<&[Value]> = ids
                .iter()
                .map(|&i| ctx.data.rows[i].as_slice())
                .filter(|row| row[j].is_missing())
                .collect();
            // Uncoded missing values always go right, so a partition placing
            // them with a level set needs that set's complement on the left.
            let mut sets = sets;
            if let Some(flag) = ctx.flags[j]
                && node_missing
                    .iter()
                    .any(|row| missing_code(row, Some(flag)).is_none())
            {
                let complements: Vec<Vec<String>> = sets
                    .iter()
                    .map(|set| known.iter().filter(|c| !set.contains(c)).cloned().collect())
                    .collect();
                sets.extend(complements);
            }
            for left_categories in sets {
                for (missing_left, missing_codes) in &routes {
                    result.push(orient_fewer_left(
                        Split::Categorical {
                            feature: j,
                            left_categories: left_categories.clone(),
                            known_categories: known.clone(),
                            missing_left: *missing_left,
                            unknown_left: false,
                            missing_codes: missing_codes.clone(),
                        },
                        &node_missing,
                    ));
                }
            }
        }
    }
    result
}
