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
    if categorical || (mode == SelectionMode::Importance && few_values()) {
        let mut levels = BTreeMap::new();
        return ids
            .iter()
            .map(|&i| {
                let key = match &ctx.data.rows[i][j] {
                    Value::Missing => (0, code(i).to_owned()),
                    // label() is None only for Missing, handled above.
                    v => (1, v.label().unwrap()),
                };
                let next = levels.len();
                *levels.entry(key).or_insert(next)
            })
            .collect();
    }
    if x.is_empty() {
        return vec![0; ids.len()];
    }
    let cuts: Vec<f64> = if quantile_cuts {
        let groups = if interaction || ids.len() < 60 { 3 } else { 4 };
        let groups = if missing { groups - 1 } else { groups };
        (1..groups)
            .map(|k| stats::sorted_quantile(&sorted, k as f64 / groups as f64))
            .collect()
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
    let codes: std::collections::BTreeSet<&str> = ids
        .iter()
        .filter(|&&i| ctx.data.rows[i][j].is_missing())
        .map(|&i| code(i))
        .collect();
    let codes: Vec<&str> = codes.into_iter().collect();
    ids.iter()
        .map(|&i| {
            ctx.data.rows[i][j]
                .number()
                .map(|x| cuts.partition_point(|c| x > *c))
                .unwrap_or_else(|| cuts.len() + 1 + codes.binary_search(&code(i)).unwrap_or(0))
        })
        .collect()
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
    let classes = labels
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    let parallel = ids.len() >= PARALLEL_MIN_ROWS;
    let tests = ordered_map(
        features,
        parallel,
        || (),
        |_, &j| {
            let bins = discretize(ctx, ids, j, false, mode, classes);
            (!bins.iter().all(|x| *x == bins[0])).then(|| test(&labels, &bins))
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
    // Every diagnostic constructed above contains a log tail; eligibility also
    // guarantees this minimum is nonempty.
    let best = selection
        .diagnostics
        .iter()
        .min_by(|a, b| {
            a.log_p_value
                .unwrap()
                .total_cmp(&b.log_p_value.unwrap())
                .then(a.feature.cmp(&b.feature))
        })
        .unwrap();
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
    if ctx.options.interactions && eligible.len() > 1 && best_main_log_p >= main_log_threshold {
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
                test(&labels, joint).2
            },
        );
        let mut best_pair = None;
        let mut best_pair_log_p = 0.0;
        for (&(a, b), log_p) in pairs.iter().zip(pair_tests) {
            if log_p < best_pair_log_p {
                best_pair_log_p = log_p;
                best_pair = Some((eligible[a], eligible[b]));
            }
        }
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
    if mode == SelectionMode::Fit
        && ctx.options.linear_splits
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
                        discriminant(ctx, ids, &labels, [numeric[a], numeric[b]])
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
