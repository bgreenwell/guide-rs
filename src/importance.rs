use crate::{
    node::Context,
    selection::SelectionMode,
    tree::{grow, validate_supported},
    *,
};
use rand::{SeedableRng, seq::SliceRandom};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ImportanceOptions {
    pub permutations: usize,
    pub max_depth: usize,
    pub seed: u64,
}
impl Default for ImportanceOptions {
    fn default() -> Self {
        Self {
            permutations: 300,
            max_depth: 4,
            seed: 0,
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ImportanceScore {
    pub feature: String,
    pub raw_score: f64,
    pub null_mean: f64,
    /// Raw/null mean; zero is a compatibility placeholder when null_mean is zero.
    /// Positive raw scores with zero null mean have indeterminate groups and warnings.
    pub adjusted_score: f64,
    pub normalized_score: Option<f64>,
    /// highly_important, likely_important, unimportant, or indeterminate.
    pub group: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ImportanceThreshold {
    pub confidence: f64,
    pub raw_maximum_threshold: f64,
    /// Rank-based target count. Ties can leave individual memberships indeterminate.
    pub important_count: usize,
    /// Midpoint between the last selected and first unselected adjusted score.
    /// None if all or none qualify, or the boundary is tied.
    pub adjusted_threshold: Option<f64>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ImportanceResult {
    /// Numerical behavior revision; absent in older serialized importance reports.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub algorithm_version: Option<String>,
    pub scores: Vec<ImportanceScore>,
    pub thresholds: Vec<ImportanceThreshold>,
    pub permutations: usize,
    pub seed: u64,
    pub warnings: Vec<String>,
}

// If a rank boundary bisects equal adjusted scores, selecting particular names
// inside that block would invent evidence that the scores do not contain.
fn rank_membership(scores: &[ImportanceScore], index: usize, count: usize) -> Option<bool> {
    if count == 0 {
        return Some(false);
    }
    if count == scores.len() {
        return Some(true);
    }
    let upper = scores[count - 1].adjusted_score;
    let lower = scores[count].adjusted_score;
    let score = scores[index].adjusted_score;
    if upper == lower && score == upper {
        None
    } else {
        Some(index < count)
    }
}

fn score_group(
    scores: &[ImportanceScore],
    index: usize,
    at_95: usize,
    at_99: usize,
) -> &'static str {
    match rank_membership(scores, index, at_99) {
        Some(true) => "highly_important",
        None => "indeterminate",
        Some(false) => match rank_membership(scores, index, at_95) {
            Some(true) => "likely_important",
            Some(false) => "unimportant",
            None => "indeterminate",
        },
    }
}

/// Four-level GUIDE-style scoring and permutation bias correction (Loh & Zhou, 2021).
/// Thresholds use the paper's rank/midpoint approximation; boundary cases are explicit.
pub fn importance(
    data: &Dataset,
    options: &FitOptions,
    importance: &ImportanceOptions,
) -> Result<ImportanceResult> {
    options.validate()?;
    data.validate(&options.task)?;
    validate_supported(data, options)?;
    if !matches!(
        options.task,
        Task::Classification
            | Task::Regression {
                loss: Loss::SquaredError
            }
    ) || !(options.node_model == NodeModel::Constant || options.node_model.discriminant_splits())
    {
        return Err(invalid(
            "importance currently supports constant classification and squared-error regression trees",
        ));
    }
    if options.class_priors.is_some() || options.class_costs.is_some() {
        return Err(invalid("importance requires default priors and costs"));
    }
    if importance.permutations < 2 || importance.max_depth == 0 || importance.max_depth > 16 {
        return Err(invalid(
            "importance requires at least 2 permutations and depth 1..16",
        ));
    }
    // Scores come from constant-node trees whatever the fitted node model, so
    // an LDA classifier is scored like a constant one.
    let mut opt = options.clone();
    opt.node_model = NodeModel::Constant;
    opt.max_depth = importance.max_depth;
    opt.pruning = Pruning::None;
    opt.mtry = None;
    let ids = data.active_rows();
    let raw = raw_scores(data, &opt, &ids, importance.seed)?;
    let mut null_means = vec![0.0; data.schema.len()];
    let mut null_maxima = vec![];
    // Permutations are drawn in order, a batch at a time, and each batch is
    // scored in parallel; null means accumulate in permutation order, so the
    // result does not depend on the number of threads.
    let mut rng = ChaCha8Rng::seed_from_u64(importance.seed);
    let batch = 4 * rayon::current_num_threads();
    let mut b = 0;
    while b < importance.permutations {
        let orders: Vec<(usize, Vec<usize>)> = (b..importance.permutations.min(b + batch))
            .map(|k| {
                let mut order = ids.clone();
                order.shuffle(&mut rng);
                (k, order)
            })
            .collect();
        let batch_scores: Vec<Result<Vec<f64>>> = orders
            .par_iter()
            .map_init(
                || data.clone(),
                |shuffled, (k, order)| {
                    for (&i, &j) in ids.iter().zip(order) {
                        shuffled.targets[i] = data.targets[j].clone();
                    }
                    raw_scores(
                        shuffled,
                        &opt,
                        &ids,
                        importance.seed.wrapping_add(*k as u64 + 1),
                    )
                },
            )
            .collect();
        for scores in batch_scores {
            let scores = scores?;
            for (m, s) in null_means.iter_mut().zip(&scores) {
                *m += s / importance.permutations as f64;
            }
            null_maxima.push(scores.iter().copied().fold(0.0, f64::max));
        }
        b += orders.len();
    }
    let mut scores: Vec<ImportanceScore> = data
        .schema
        .iter()
        .enumerate()
        .filter(|(_, f)| f.role.splits())
        .map(|(j, f)| ImportanceScore {
            feature: f.name.clone(),
            raw_score: raw[j],
            null_mean: null_means[j],
            adjusted_score: if null_means[j] > 0.0 {
                raw[j] / null_means[j]
            } else {
                0.0
            },
            normalized_score: None,
            group: "unimportant".into(),
        })
        .collect();
    scores.sort_by(|a, b| {
        b.adjusted_score
            .total_cmp(&a.adjusted_score)
            .then(a.feature.cmp(&b.feature))
    });
    let mut thresholds = vec![];
    for confidence in [0.80, 0.90, 0.95, 0.99] {
        let cutoff = stats::quantile(&null_maxima, confidence);
        let k = scores.iter().filter(|s| s.raw_score > cutoff).count();
        let midpoint =
            if k > 0 && k < scores.len() && scores[k - 1].adjusted_score > scores[k].adjusted_score
            {
                Some(scores[k - 1].adjusted_score / 2.0 + scores[k].adjusted_score / 2.0)
            } else {
                None
            };
        thresholds.push(ImportanceThreshold {
            confidence,
            raw_maximum_threshold: cutoff,
            important_count: k,
            adjusted_threshold: midpoint,
        });
    }
    let groups: Vec<_> = (0..scores.len())
        .map(|j| {
            score_group(
                &scores,
                j,
                thresholds[2].important_count,
                thresholds[3].important_count,
            )
        })
        .collect();
    for (s, group) in scores.iter_mut().zip(groups) {
        let undefined_adjustment = s.null_mean == 0.0 && s.raw_score > 0.0;
        s.normalized_score = thresholds[2]
            .adjusted_threshold
            .filter(|x| *x > 0.0 && !undefined_adjustment)
            .map(|x| s.adjusted_score / x);
        s.group = if undefined_adjustment {
            "indeterminate"
        } else {
            group
        }
        .into();
    }
    let mut warnings = vec![];
    if importance.permutations < 300 {
        warnings.push(
            "Fewer than 300 permutations: tail thresholds have limited Monte Carlo resolution."
                .into(),
        );
    }
    if thresholds.iter().any(|t| t.adjusted_threshold.is_none()) {
        warnings.push("A threshold selects all/no features or cuts through tied scores; a numerical normalized cutoff is unavailable.".into());
    }
    if scores
        .iter()
        .any(|s| s.null_mean == 0.0 && s.raw_score > 0.0)
    {
        warnings.push("A feature has positive raw score but zero estimated null mean; increase permutations before interpreting its adjusted score.".into());
    }
    if scores.iter().any(|s| s.group == "indeterminate") {
        warnings.push("An importance group is indeterminate because a rank boundary crosses tied adjusted scores or the null adjustment is undefined.".into());
    }
    Ok(ImportanceResult {
        algorithm_version: Some(ALGORITHM_VERSION.into()),
        scores,
        thresholds,
        permutations: importance.permutations,
        seed: importance.seed,
        warnings,
    })
}

fn raw_scores(data: &Dataset, options: &FitOptions, ids: &[usize], seed: u64) -> Result<Vec<f64>> {
    let ctx = Context::new(data, options, ids, None)?;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut next = 0;
    let root = grow(&ctx, ids, 0, &mut next, &mut rng, SelectionMode::Importance);
    crate::tree::validate_node(&root, &data.schema, ctx.classes.len(), options, 0)?;
    let mut scores = vec![0.0; data.schema.len()];
    fn collect(n: &Node, s: &mut [f64]) {
        if n.split.is_none() {
            return;
        }
        for d in &n.diagnostics {
            s[d.feature] += (n.sample_count as f64).sqrt()
                * stats::chi_square_log_upper_quantile(
                    d.importance_log_p_value
                        .unwrap_or_else(|| d.importance_p_value.ln()),
                    1,
                );
        }
        if let Some(l) = &n.left {
            collect(l, s);
        }
        if let Some(r) = &n.right {
            collect(r, s);
        }
    }
    collect(&root, &mut scores);
    Ok(scores)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tied_adjusted_boundary_does_not_select_arbitrary_feature_names() {
        // A raw cutoff of 15 selects one variable, but both bias-adjusted scores
        // equal 2. Their names supply no basis for assigning different groups.
        let scores: Vec<_> = [
            ("first", 20.0, 10.0),
            ("second", 10.0, 5.0),
            ("lower", 1.0, 1.0),
        ]
        .into_iter()
        .map(|(feature, raw_score, null_mean)| ImportanceScore {
            feature: feature.into(),
            raw_score,
            null_mean,
            adjusted_score: raw_score / null_mean,
            normalized_score: None,
            group: String::new(),
        })
        .collect();
        for index in [0, 1] {
            assert_eq!(score_group(&scores, index, 1, 0), "indeterminate");
            assert_eq!(score_group(&scores, index, 2, 1), "indeterminate");
            assert_eq!(score_group(&scores, index, 2, 0), "likely_important");
            assert_eq!(score_group(&scores, index, 2, 2), "highly_important");
        }
        assert_eq!(score_group(&scores, 2, 1, 0), "unimportant");
    }
}
