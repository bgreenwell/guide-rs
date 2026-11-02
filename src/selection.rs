use crate::node::{Context, LeafModel};
use crate::tree::{PARALLEL_MIN_ROWS, ordered_map};
use crate::{FeatureType, SelectionDiagnostic, Value, missing_code, stats};
use std::collections::BTreeMap;

#[derive(Clone, Copy, PartialEq)]
pub(crate) enum SelectionMode {
    Fit,
    Importance,
}
pub(crate) struct Selection {
    pub features: Vec<usize>,
    pub diagnostics: Vec<SelectionDiagnostic>,
    pub pair: bool,
    pub linear: Option<Projection>,
}
pub(crate) struct Projection {
    pub features: [usize; 2],
    pub coefficients: [f64; 2],
}

fn discretize(
    ctx: &Context<'_>,
    ids: &[usize],
    j: usize,
    interaction: bool,
    mode: SelectionMode,
    classes: usize,
) -> Vec<usize> {
    let x: Vec<f64> = ids
        .iter()
        .filter_map(|&i| ctx.data.rows[i][j].number())
        .collect();
    let missing = x.len() != ids.len();
    // A flagged feature's missing values form one group per flag code.
    let code = |i: usize| missing_code(&ctx.data.rows[i], ctx.flags[j]).unwrap_or("");
    let categorical = matches!(ctx.data.schema[j].feature_type, FeatureType::Categorical);
    // Quantile cuts and the importance-mode level count need sorted values.
    let quantile_cuts = mode == SelectionMode::Importance || !ctx.classifying();
    let mut sorted = vec![];
    if !categorical && quantile_cuts {
        sorted.clone_from(&x);
        sorted.sort_by(f64::total_cmp);
    }
    let few_values = || {
        let mut distinct = sorted.clone();
        distinct.dedup();
        distinct.len() <= 4
    };
    let key = |i: usize| match &ctx.data.rows[i][j] {
        Value::Missing => (0, code(i).to_owned()),
        // label() is None only for Missing, handled above.
        v => (1, v.label().unwrap()),
    };
    // Regression and importance main-effect tests group categorical
    // variables, and numeric ones with at most four distinct values, by
    // value. GUIDE pools the values seen once, and a pool of one row joins
    // the largest group. A numeric variable with a single observed value is
    // not tested.
    let by_value = quantile_cuts && !interaction;
    if by_value && !categorical && sorted.first() == sorted.last() {
        return vec![0; ids.len()];
    }
    if by_value && (categorical || few_values()) {
        let mut counts: BTreeMap<(u8, String), usize> = BTreeMap::new();
        for &i in ids {
            *counts.entry(key(i)).or_default() += 1;
        }
        let single = |k: &(u8, String)| k.0 == 1 && counts[k] == 1;
        let pooled = counts.keys().filter(|k| single(k)).count();
        let mut group = BTreeMap::new();
        for k in counts.keys().filter(|k| !single(k)) {
            let next = group.len();
            group.insert(k, next);
        }
        // The largest group, the first among equals, absorbs a lone value.
        let largest = counts
            .iter()
            .filter(|(k, _)| !single(k))
            .max_by(|a, b| a.1.cmp(b.1).then(b.0.cmp(a.0)))
            .map(|(k, _)| group[k]);
        let pool = match largest {
            Some(g) if pooled == 1 => g,
            _ => group.len(),
        };
        return ids
            .iter()
            .map(|&i| group.get(&key(i)).copied().unwrap_or(pool))
            .collect();
    }
    if categorical || (mode == SelectionMode::Importance && few_values()) {
        let mut levels = BTreeMap::new();
        return ids
            .iter()
            .map(|&i| {
                let next = levels.len();
                *levels.entry(key(i)).or_insert(next)
            })
            .collect();
    }
    if x.is_empty() {
        return vec![0; ids.len()];
    }
    let codes: std::collections::BTreeSet<&str> = ids
        .iter()
        .filter(|&&i| ctx.data.rows[i][j].is_missing())
        .map(|&i| code(i))
        .collect();
    let codes: Vec<&str> = codes.into_iter().collect();
    let cuts: Vec<f64> = if interaction && mode == SelectionMode::Importance {
        importance_pair_cut(&sorted, ids.len())
    } else if quantile_cuts {
        quartile_cuts(&sorted, ids.len(), interaction, missing)
    } else {
        let m = stats::mean(&x);
        let s = stats::variance(&x).sqrt();
        if interaction {
            if ids.len() < 45 * classes {
                vec![m]
            } else {
                vec![m - s / 3.0_f64.sqrt(), m + s / 3.0_f64.sqrt()]
            }
        } else if ids.len() >= 20 * classes {
            vec![
                m - s * 3.0_f64.sqrt() / 2.0,
                m,
                m + s * 3.0_f64.sqrt() / 2.0,
            ]
        } else {
            vec![m - s / 3.0_f64.sqrt(), m + s / 3.0_f64.sqrt()]
        }
    };
    let mut bins: Vec<usize> = ids
        .iter()
        .map(|&i| {
            ctx.data.rows[i][j]
                .number()
                .map(|x| cuts.partition_point(|c| x > *c))
                .unwrap_or_else(|| cuts.len() + 1 + codes.binary_search(&code(i)).unwrap_or(0))
        })
        .collect();
    if quantile_cuts && !interaction {
        pool_single_rows(&mut bins, cuts.len() + 2 + codes.len());
    }
    bins
}
/// Index of the strongest of `tests` (statistic, df, log tail), the first
/// among equals. Fitting ranks them on GUIDE's 1-df scale, whose
/// Wilson-Hilferty tails can reorder strong tests with different df;
/// importance scoring inverts tail probabilities exactly, as GUIDE does there.
fn strongest(mode: SelectionMode, tests: impl Iterator<Item = (f64, usize, f64)>) -> Option<usize> {
    if mode == SelectionMode::Fit {
        return stats::guide_strongest(tests);
    }
    tests
        .enumerate()
        .min_by(|a, b| a.1.2.total_cmp(&b.1.2).then(a.0.cmp(&b.0)))
        .map(|(k, _)| k)
}
/// Quartile cuts of sorted values, three groups below 60 rows, one group
/// fewer with missing values, kept up to the first repeated cut: quartiles
/// 0, 0 and 41 give the single cut 0, not the two cuts 0 and 41.
fn quartile_cuts(sorted: &[f64], rows: usize, interaction: bool, missing: bool) -> Vec<f64> {
    let groups = if interaction || rows < 60 { 3 } else { 4 };
    let groups = if missing { groups - 1 } else { groups };
    let mut cuts: Vec<f64> = (1..groups)
        .map(|k| stats::sorted_quantile(sorted, k as f64 / groups as f64))
        .collect();
    if let Some(k) = (1..cuts.len()).find(|&k| cuts[k] == cuts[k - 1]) {
        cuts.truncate(k);
    }
    cuts
}
/// GUIDE's importance interaction tests cut a numeric variable once, at the
/// median of all `rows` with its missing values sorted first, so missing
/// values pull the cut below the median of the observed values `sorted`.
fn importance_pair_cut(sorted: &[f64], rows: usize) -> Vec<f64> {
    let at = (rows - 1) as f64 / 2.0 - (rows - sorted.len()) as f64;
    if at < 0.0 {
        return vec![];
    }
    let (k, f) = (at.floor() as usize, at.fract());
    let next = sorted[(k + 1).min(sorted.len() - 1)];
    vec![sorted[k] + f * (next - sorted[k])]
}
/// GUIDE 46.3 treats a quantile group holding one row, observed or missing,
/// as it treats a value seen once: such groups are pooled, and a pool of one
/// row joins the largest group, the first among equals.
fn pool_single_rows(bins: &mut [usize], groups: usize) {
    let mut counts = vec![0; groups];
    for &b in bins.iter() {
        counts[b] += 1;
    }
    let single: Vec<usize> = (0..groups).filter(|&b| counts[b] == 1).collect();
    if let [lone] = single[..] {
        let largest = (0..groups).rev().max_by_key(|&b| counts[b]).unwrap_or(0);
        bins.iter_mut()
            .filter(|b| **b == lone)
            .for_each(|b| *b = largest);
    } else if let [first, ..] = single[..] {
        bins.iter_mut()
            .filter(|b| single.contains(b))
            .for_each(|b| *b = first);
    }
}
/// Numbers the joint cells of two binnings in order of first appearance,
/// which fixes the column order of the contingency table. Up to `dense_limit`
/// cells are indexed in a flat table; more, from many-level categorical
/// pairs, use a hash map.
fn joint_cells(
    a: &[usize],
    b: &[usize],
    dense_limit: usize,
    index: &mut Vec<usize>,
    joint: &mut Vec<usize>,
) {
    let width = b.iter().max().map_or(0, |m| m + 1);
    let cells = a.iter().max().map_or(0, |m| m + 1) * width;
    joint.clear();
    if cells <= dense_limit {
        index.clear();
        index.resize(cells, usize::MAX);
        let mut next = 0;
        for (&x, &y) in a.iter().zip(b) {
            let cell = &mut index[x * width + y];
            if *cell == usize::MAX {
                *cell = next;
                next += 1;
            }
            joint.push(*cell);
        }
    } else {
        let mut map = std::collections::HashMap::new();
        for (&x, &y) in a.iter().zip(b) {
            let next = map.len();
            joint.push(*map.entry((x, y)).or_insert(next));
        }
    }
}
/// Pearson statistic, active degrees of freedom, and log upper-tail probability.
fn test(labels: &[usize], groups: &[usize]) -> (f64, usize, f64) {
    let nr = labels.iter().max().map(|n| n + 1).unwrap_or(0);
    let nc = groups.iter().max().map(|n| n + 1).unwrap_or(0);
    let mut table = vec![vec![0.0; nc]; nr];
    for (&r, &c) in labels.iter().zip(groups) {
        table[r][c] += 1.0;
    }
    let (stat, df) = stats::pearson(&table);
    (stat, df, stats::chi_square_log_survival(stat, df))
}
pub(crate) fn select(
    ctx: &Context<'_>,
    ids: &[usize],
    model: &LeafModel,
    features: &[usize],
    mode: SelectionMode,
) -> Selection {
    let labels: Vec<usize> = ids
        .iter()
        .map(|&i| {
            if let LeafModel::Quantiles { lower, upper } = model {
                if ctx.y[i] < *lower {
                    0
                } else if ctx.y[i] > *upper {
                    2
                } else {
                    1
                }
            } else if ctx.classifying() {
                ctx.y[i] as usize
            } else {
                usize::from(
                    ctx.y[i]
                        > model.fitting_scalar(&ctx.data.rows[i], ctx.data.offset(i), &ctx.loss),
                )
            }
        })
        .collect();
    let labels = match (&ctx.loss, model) {
        (crate::Loss::Poisson, _) => anscombe_groups(ctx, ids, model),
        (crate::Loss::Quantile { .. }, LeafModel::Quantiles { .. }) => labels,
        (crate::Loss::Quantile { .. }, _) => zero_residuals_to_smaller(ctx, ids, model, labels),
        _ => labels,
    };
    let mut selection = select_with_labels(ctx, ids, &labels, features, mode);
    // Simple polynomial trees stop when the node's polynomial fit leaves no
    // candidate with any residual-sign association (native probes: such a
    // node with a fitted slope is terminal, one fitted by a constant splits).
    if mode == SelectionMode::Fit
        && matches!(ctx.options.node_model, crate::NodeModel::Polynomial { .. })
        && ctx.loss == crate::Loss::SquaredError
        && !matches!(model, LeafModel::Constant { .. })
        && selection.diagnostics.iter().all(|d| d.chi_square == 0.0)
    {
        selection.features.clear();
        selection.pair = false;
    }
    selection
}
/// GUIDE 34.2 groups Poisson rows by whether their Anscombe residual,
/// `1.5 (y^(2/3) - mu^(2/3)) / mu^(1/6)`, lies above the mean residual.
fn anscombe_groups(ctx: &Context<'_>, ids: &[usize], model: &LeafModel) -> Vec<usize> {
    let r: Vec<f64> = ids
        .iter()
        .map(|&i| {
            let mu = model
                .fitting_scalar(&ctx.data.rows[i], ctx.data.offset(i), &ctx.loss)
                .max(f64::MIN_POSITIVE);
            1.5 * (ctx.y[i].powf(2.0 / 3.0) - mu.powf(2.0 / 3.0)) / mu.powf(1.0 / 6.0)
        })
        .collect();
    let mean = r.iter().sum::<f64>() / r.len().max(1) as f64;
    r.iter().map(|&v| usize::from(v > mean)).collect()
}
/// GUIDE 28.1 puts quantile rows with zero residuals in the smaller of the
/// positive and negative residual groups; probes show the positive group
/// takes them when the two are equal.
fn zero_residuals_to_smaller(
    ctx: &Context<'_>,
    ids: &[usize],
    model: &LeafModel,
    mut labels: Vec<usize>,
) -> Vec<usize> {
    let fitted: Vec<f64> = ids
        .iter()
        .map(|&i| model.fitting_scalar(&ctx.data.rows[i], ctx.data.offset(i), &ctx.loss))
        .collect();
    let zero: Vec<bool> = ids
        .iter()
        .zip(&fitted)
        .map(|(&i, f)| ctx.y[i] == *f)
        .collect();
    let positive = labels.iter().filter(|&&l| l == 1).count();
    let negative = labels.len() - positive - zero.iter().filter(|&&z| z).count();
    let group = usize::from(positive <= negative);
    for (label, z) in labels.iter_mut().zip(zero) {
        if z {
            *label = group;
        }
    }
    labels
}
/// Variable selection from precomputed response groups, one per row of
/// `ids`: class indices, residual signs or, for boosting, gradient signs.
pub(crate) fn select_with_labels(
    ctx: &Context<'_>,
    ids: &[usize],
    labels: &[usize],
    features: &[usize],
    mode: SelectionMode,
) -> Selection {
    let classes = labels
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    select_with_test(ctx, ids, features, mode, classes, Some(labels), |bins| {
        test(labels, bins)
    })
}
/// Variable selection with a custom association test of the response with a
/// predictor's bins, one bin per row of `ids`, returning the statistic,
/// degrees of freedom and log upper-tail probability. `classes` sets the bin
/// count for classification contexts; linear-combination splits need the
/// response groups `labels`.
pub(crate) fn select_with_test(
    ctx: &Context<'_>,
    ids: &[usize],
    features: &[usize],
    mode: SelectionMode,
    classes: usize,
    labels: Option<&[usize]>,
    test: impl Fn(&[usize]) -> (f64, usize, f64) + Sync,
) -> Selection {
    let parallel = ids.len() >= PARALLEL_MIN_ROWS;
    let tests = ordered_map(
        features,
        parallel,
        || (),
        |_, &j| {
            let bins = discretize(ctx, ids, j, false, mode, classes);
            (!bins.iter().all(|x| *x == bins[0])).then(|| test(&bins))
        },
    );
    let mut diagnostics = vec![];
    let mut eligible = vec![];
    for (&j, result) in features.iter().zip(tests) {
        let Some((stat, df, log_p)) = result else {
            continue;
        };
        diagnostics.push(SelectionDiagnostic {
            feature: j,
            chi_square: stat,
            degrees_of_freedom: df,
            p_value: log_p.exp(),
            importance_p_value: log_p.exp(),
            log_p_value: Some(log_p),
            importance_log_p_value: Some(log_p),
            interaction_with: None,
        });
        eligible.push(j);
    }
    let mut selection = Selection {
        features: vec![],
        diagnostics,
        pair: false,
        linear: None,
    };
    if eligible.is_empty() {
        return selection;
    }
    // Every diagnostic constructed above contains a log tail, and eligibility
    // guarantees one exists; features are in index order, so ties go to the
    // lower index.
    let best = &selection.diagnostics[strongest(
        mode,
        selection
            .diagnostics
            .iter()
            .map(|d| (d.chi_square, d.degrees_of_freedom, d.log_p_value.unwrap())),
    )
    .unwrap()];
    selection.features = vec![best.feature];
    let best_main_log_p = best.log_p_value.unwrap();
    // Pinned multiple-linear SSE controls stop when the sole nonconstant
    // candidate has exactly zero residual-sign association. This is not a
    // significance cutoff: positive scores with p > .99 still reach search.
    // Keep all diagnostics, and leave multi-predictor interaction/projection
    // selection intact; the singleton rule cannot be generalized to it.
    if mode == SelectionMode::Fit
        && ctx.options.node_model == crate::NodeModel::Linear
        && ctx.loss == crate::Loss::SquaredError
        && eligible.len() == 1
        && best.chi_square == 0.0
    {
        selection.features.clear();
        return selection;
    }
    let log_alpha = ctx.options.significance_level.ln();
    let main_log_threshold = log_alpha - (eligible.len() as f64).ln();
    if ctx.options.interactions
        && ctx.pair_tests()
        && eligible.len() > 1
        && best_main_log_p >= main_log_threshold
    {
        let bins: Vec<Vec<usize>> = ordered_map(
            &eligible,
            parallel,
            || (),
            |_, &j| discretize(ctx, ids, j, true, mode, classes),
        );
        let pairs: Vec<(usize, usize)> = (0..eligible.len())
            .flat_map(|a| (a + 1..eligible.len()).map(move |b| (a, b)))
            .collect();
        let pair_tests = ordered_map(
            &pairs,
            parallel,
            || (vec![], vec![]),
            |(index, joint), &(a, b)| {
                joint_cells(&bins[a], &bins[b], 16 * ids.len() + 1024, index, joint);
                test(joint)
            },
        );
        let best = strongest(mode, pair_tests.iter().copied());
        let best_pair = best.map(|k| (eligible[pairs[k].0], eligible[pairs[k].1]));
        let best_pair_log_p = best.map_or(0.0, |k| pair_tests[k].2);
        let pair_log_threshold = log_alpha + std::f64::consts::LN_2
            - (eligible.len() as f64).ln()
            - ((eligible.len() - 1) as f64).ln();
        if best_pair_log_p < pair_log_threshold
            && let Some((a, b)) = best_pair
        {
            selection.features = vec![a, b];
            selection.pair = true;
            for d in &mut selection.diagnostics {
                // The 42.5 history explicitly extends the winning interaction
                // score to identical predictor copies, not just the selected pair.
                let identical = |j: usize| {
                    ctx.data.schema[d.feature].feature_type == ctx.data.schema[j].feature_type
                        && ids
                            .iter()
                            .all(|&i| ctx.data.rows[i][d.feature] == ctx.data.rows[i][j])
                };
                let other = if d.feature == a {
                    Some(b)
                } else if d.feature == b {
                    Some(a)
                } else if identical(a) {
                    Some(b)
                } else if identical(b) {
                    Some(a)
                } else {
                    None
                };
                if let Some(other) = other {
                    d.importance_p_value = best_pair_log_p.exp();
                    d.importance_log_p_value = Some(best_pair_log_p);
                    d.interaction_with = Some(other);
                }
            }
        }
    }
    if let Some(labels) = labels
        && mode == SelectionMode::Fit
        && ctx.options.linear_splits
        && ctx.pair_tests()
        && !selection.pair
        && best_main_log_p >= main_log_threshold
    {
        let numeric: Vec<usize> = eligible
            .iter()
            .copied()
            .filter(|&j| matches!(ctx.data.schema[j].feature_type, FeatureType::Numeric))
            .collect();
        if numeric.len() > 1 {
            let threshold = log_alpha + std::f64::consts::LN_2
                - (numeric.len() as f64).ln()
                - ((numeric.len() - 1) as f64).ln();
            let mut best_p = threshold;
            for a in 0..numeric.len() {
                for b in a + 1..numeric.len() {
                    if let Some((projection, p)) =
                        discriminant(ctx, ids, labels, [numeric[a], numeric[b]])
                        && p < best_p
                    {
                        best_p = p;
                        selection.linear = Some(projection);
                    }
                }
            }
        }
    }
    selection
}

/// Trim class-wise at two standard deviations, then solve the 2x2
/// generalized between/within scatter eigenproblem using Cholesky whitening.
fn discriminant(
    ctx: &Context<'_>,
    ids: &[usize],
    labels: &[usize],
    features: [usize; 2],
) -> Option<(Projection, f64)> {
    let mut samples = vec![];
    for (&i, &label) in ids.iter().zip(labels) {
        if let (Some(a), Some(b)) = (
            ctx.data.rows[i][features[0]].number(),
            ctx.data.rows[i][features[1]].number(),
        ) {
            samples.push(([a, b], label));
        }
    }
    if samples.len() < 6 {
        return None;
    }
    let center: [f64; 2] =
        std::array::from_fn(|j| samples.iter().map(|s| s.0[j]).sum::<f64>() / samples.len() as f64);
    let scale: [f64; 2] = std::array::from_fn(|j| {
        (samples
            .iter()
            .map(|s| (s.0[j] - center[j]).powi(2))
            .sum::<f64>()
            / (samples.len() - 1) as f64)
            .sqrt()
    });
    if scale.iter().any(|s| *s <= 0.0 || !s.is_finite()) {
        return None;
    }
    for (x, _) in &mut samples {
        for j in 0..2 {
            x[j] = (x[j] - center[j]) / scale[j];
        }
    }
    let classes = labels
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    if classes.len() < 2 {
        return None;
    }
    let mut trimmed = vec![];
    for class in classes {
        let subset: Vec<[f64; 2]> = samples
            .iter()
            .filter(|s| s.1 == class)
            .map(|s| s.0)
            .collect();
        if subset.len() < 2 {
            continue;
        }
        let mean: [f64; 2] =
            std::array::from_fn(|j| subset.iter().map(|x| x[j]).sum::<f64>() / subset.len() as f64);
        let sd: [f64; 2] = std::array::from_fn(|j| {
            (subset.iter().map(|x| (x[j] - mean[j]).powi(2)).sum::<f64>()
                / (subset.len() - 1) as f64)
                .sqrt()
        });
        for x in subset {
            if (0..2).all(|j| (x[j] - mean[j]).abs() <= 2.0 * sd[j] + 1e-12) {
                trimmed.push((x, class));
            }
        }
    }
    if trimmed.len() < 4 {
        return None;
    }
    let mean: [f64; 2] =
        std::array::from_fn(|j| trimmed.iter().map(|s| s.0[j]).sum::<f64>() / trimmed.len() as f64);
    let mut groups: BTreeMap<usize, Vec<[f64; 2]>> = BTreeMap::new();
    for (x, c) in trimmed {
        groups.entry(c).or_default().push(x);
    }
    if groups.len() < 2 {
        return None;
    }
    let mut within = [[0.0; 2]; 2];
    let mut between = [[0.0; 2]; 2];
    for g in groups.values() {
        let cm: [f64; 2] =
            std::array::from_fn(|j| g.iter().map(|x| x[j]).sum::<f64>() / g.len() as f64);
        for a in 0..2 {
            for b in 0..2 {
                between[a][b] += g.len() as f64 * (cm[a] - mean[a]) * (cm[b] - mean[b]);
                within[a][b] += g
                    .iter()
                    .map(|x| (x[a] - cm[a]) * (x[b] - cm[b]))
                    .sum::<f64>();
            }
        }
    }
    let ridge = (within[0][0] + within[1][1]).max(1.0) * 1e-10;
    within[0][0] += ridge;
    within[1][1] += ridge;
    let l00 = within[0][0].sqrt();
    let l10 = within[1][0] / l00;
    let l11 = (within[1][1] - l10 * l10).sqrt();
    if !l11.is_finite() || l11 == 0.0 {
        return None;
    }
    let inv = [[1.0 / l00, 0.0], [-l10 / (l00 * l11), 1.0 / l11]];
    let whitened: [[f64; 2]; 2] = std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            (0..2)
                .flat_map(|a| (0..2).map(move |b| inv[i][a] * between[a][b] * inv[j][b]))
                .sum()
        })
    });
    let angle = 0.5 * (2.0 * whitened[0][1]).atan2(whitened[0][0] - whitened[1][1]);
    let q = [angle.cos(), angle.sin()];
    let mut coef = [
        (inv[0][0] * q[0] + inv[1][0] * q[1]) / scale[0],
        inv[1][1] * q[1] / scale[1],
    ];
    let norm = coef[0].hypot(coef[1]);
    if !norm.is_finite() || norm == 0.0 {
        return None;
    }
    let sign = if coef[0] < 0.0 { -1.0 } else { 1.0 };
    for c in &mut coef {
        *c *= sign / norm;
    }
    let values: Vec<Option<f64>> = ids
        .iter()
        .map(|&i| {
            ctx.data.rows[i][features[0]]
                .number()
                .zip(ctx.data.rows[i][features[1]].number())
                .map(|(a, b)| a * coef[0] + b * coef[1])
        })
        .collect();
    let observed: Vec<f64> = values.iter().flatten().copied().collect();
    let m = stats::mean(&observed);
    let s = stats::variance(&observed).sqrt();
    let cuts = if ids.len() >= 20 * groups.len() {
        vec![
            m - s * 3.0_f64.sqrt() / 2.0,
            m,
            m + s * 3.0_f64.sqrt() / 2.0,
        ]
    } else {
        vec![m - s / 3.0_f64.sqrt(), m + s / 3.0_f64.sqrt()]
    };
    let bins: Vec<usize> = values
        .iter()
        .map(|v| {
            v.map(|x| cuts.partition_point(|c| x > *c))
                .unwrap_or(cuts.len() + 1)
        })
        .collect();
    let (_, _, p) = test(labels, &bins);
    Some((
        Projection {
            features,
            coefficients: coef,
        },
        p,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quartile_cuts_stop_at_the_first_repeated_cut() {
        let mut x: Vec<f64> = vec![0.0; 108];
        x.extend((0..92).map(|i| [9.0, 26.0, 37.0, 41.0, 44.0, 55.0][i % 6]));
        x.sort_by(f64::total_cmp);
        assert_eq!(quartile_cuts(&x, 200, false, false), vec![0.0]);
        let y: Vec<f64> = (0..200)
            .map(|i| {
                if (50..170).contains(&i) {
                    50.0
                } else {
                    i as f64
                }
            })
            .collect();
        assert_eq!(quartile_cuts(&y, 200, false, false), vec![49.75, 50.0]);
    }

    #[test]
    fn a_one_row_quantile_group_joins_the_largest_group() {
        // GUIDE 46.3 at an RHC importance node: edu's third group holds one row.
        let mut bins: Vec<usize> = [vec![0; 19], vec![1; 32], vec![2], vec![3; 17]].concat();
        pool_single_rows(&mut bins, 5);
        assert_eq!(bins.iter().filter(|&&b| b == 1).count(), 33);
        // Two one-row groups form one group.
        let mut bins = vec![0, 0, 0, 1, 2, 2, 2, 3];
        pool_single_rows(&mut bins, 4);
        assert_eq!(bins, vec![0, 0, 0, 1, 2, 2, 2, 1]);
    }

    #[test]
    fn importance_pair_cuts_sort_missing_values_first() {
        // A GUIDE 46.3 probe: 300 rows, 91 missing; native cuts at 14, the
        // 60th observed value, not at the observed median.
        let sorted: Vec<f64> = (0..209).map(|i| (i / 4) as f64).collect();
        assert_eq!(importance_pair_cut(&sorted, 300), vec![sorted[59]]);
        let full: Vec<f64> = (0..10).map(f64::from).collect();
        assert_eq!(importance_pair_cut(&full, 10), vec![4.5]);
        assert!(importance_pair_cut(&full[..2], 10).is_empty());
    }

    #[test]
    fn joint_cells_number_by_first_appearance() {
        let a: Vec<usize> = (0..200).map(|i| (i * 7) % 5).collect();
        let b: Vec<usize> = (0..200).map(|i| (i * 3 + i / 11) % 9).collect();
        let (mut index, mut dense, mut sparse) = (vec![], vec![], vec![]);
        joint_cells(&a, &b, usize::MAX, &mut index, &mut dense);
        joint_cells(&a, &b, 0, &mut index, &mut sparse);
        assert_eq!(dense, sparse);
        let mut seen = BTreeMap::new();
        let expected: Vec<usize> = a
            .iter()
            .zip(&b)
            .map(|cell| {
                let next = seen.len();
                *seen.entry(cell).or_insert(next)
            })
            .collect();
        assert_eq!(dense, expected);
    }
}
