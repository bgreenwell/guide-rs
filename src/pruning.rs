use crate::{node::Context, selection::SelectionMode, tree::grow, *};
use rand::{SeedableRng, seq::SliceRandom};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PruningScore {
    pub alpha: f64,
    pub leaves: usize,
    pub mean_loss: f64,
    pub standard_error: f64,
}
pub(crate) fn weakest_link_path(tree: &Tree) -> Vec<(f64, Tree)> {
    let mut current = tree.clone();
    let mut path = vec![(0.0, current.clone())];
    let mut last = 0.0_f64;
    while current.root.leaves() > 1 {
        let mut weakest = None;
        minimum_alpha(&current.root, &mut weakest);
        let (mut alpha, id) = weakest.expect("an internal node exists");
        alpha = alpha.max(last).max(0.0);
        prune_node(&mut current.root, id);
        path.push((alpha, current.clone()));
        last = alpha;
    }
    path
}
fn minimum_alpha(n: &Node, best: &mut Option<(f64, usize)>) {
    if n.split.is_none() {
        return;
    }
    let a = (n.risk - n.leaf_risk()) / (n.leaves() - 1) as f64;
    let a = a.max(0.0);
    // Native GUIDE retains separate stages for tied branches. Visit parents
    // before children, left before right, and prune only one weakest link.
    if best.is_none_or(|(alpha, _)| stats::strictly_less(a, alpha)) {
        *best = Some((a, n.id));
    }
    if let Some(l) = &n.left {
        minimum_alpha(l, best);
    }
    if let Some(r) = &n.right {
        minimum_alpha(r, best);
    }
}
fn prune_node(n: &mut Node, id: usize) {
    if n.id == id {
        n.collapse();
        return;
    }
    if let Some(l) = &mut n.left {
        prune_node(l, id);
    }
    if let Some(r) = &mut n.right {
        prune_node(r, id);
    }
}
fn at_alpha(path: &[(f64, Tree)], alpha: f64) -> &Tree {
    path.iter()
        .rev()
        .find(|(a, _)| *a <= alpha)
        .map(|(_, t)| t)
        .unwrap_or(&path[0].1)
}
pub(crate) fn prune(
    tree: Tree,
    ctx: &Context<'_>,
    validation: Option<&Dataset>,
    report: &mut FitReport,
) -> Result<Tree> {
    if matches!(ctx.options.pruning, Pruning::None) {
        return Ok(tree);
    }
    let path = weakest_link_path(&tree);
    match ctx.options.pruning {
        Pruning::None => Ok(tree),
        Pruning::CostComplexity { alpha } => {
            let t = at_alpha(&path, alpha).clone();
            report.selected_alpha = alpha;
            Ok(t)
        }
        Pruning::FixedLeaves { max_leaves } => {
            let (alpha, t) = path
                .iter()
                .find(|(_, t)| t.root.leaves() <= max_leaves)
                .unwrap();
            report.selected_alpha = *alpha;
            Ok(t.clone())
        }
        Pruning::Validation => {
            let data = validation
                .ok_or_else(|| invalid("validation pruning requires a validation dataset"))?;
            data.validate(&ctx.options.task)?;
            if data.schema != ctx.data.schema {
                return Err(invalid("validation feature schema differs from training"));
            }
            if !ctx.data.offsets.is_empty() && data.offsets.is_empty() {
                return Err(invalid(
                    "validation data must supply the exposure used during training",
                ));
            }
            let ids = data.active_rows();
            let mut best = 0;
            let mut best_loss = f64::INFINITY;
            for (j, (alpha, t)) in path.iter().enumerate() {
                let losses = external_losses(ctx, t, data, &ids)?;
                let (m, se) = weighted_summary(&losses);
                report.pruning_scores.push(PruningScore {
                    alpha: *alpha,
                    leaves: t.root.leaves(),
                    mean_loss: m,
                    standard_error: se,
                });
                if stats::less_or_close(m, best_loss) {
                    best = j;
                    best_loss = m;
                }
            }
            report.selected_alpha = path[best].0;
            Ok(path[best].1.clone())
        }
        Pruning::CrossValidation {
            folds,
            standard_errors,
        } => {
            let ids = ctx.data.active_rows();
            if ids.len() < folds {
                return Err(invalid(format!(
                    "{} active rows are insufficient for {folds} CV folds; lower folds or disable pruning",
                    ids.len()
                )));
            }
            let mut rng = ChaCha8Rng::seed_from_u64(ctx.options.seed);
            let mut parts = vec![vec![]; folds];
            let mut groups = std::collections::BTreeMap::<usize, Vec<usize>>::new();
            for &i in &ids {
                groups
                    .entry(if ctx.classifying() {
                        ctx.y[i] as usize
                    } else {
                        0
                    })
                    .or_default()
                    .push(i);
            }
            let mut shift = 0;
            for g in groups.values_mut() {
                g.shuffle(&mut rng);
                for (k, &i) in g.iter().enumerate() {
                    parts[(shift + k) % folds].push(i);
                }
                shift = (shift + g.len()) % folds;
            }
            let alphas: Vec<f64> = path
                .iter()
                .enumerate()
                .map(|(i, (a, _))| {
                    if i + 1 == path.len() {
                        f64::INFINITY
                    } else if *a == 0.0 {
                        0.0
                    } else {
                        a.sqrt() * path[i + 1].0.sqrt()
                    }
                })
                .collect();
            let mut results = vec![vec![]; path.len()];
            let mut fold_options = ctx.options.clone();
            if ctx.classifying() {
                // Estimated priors belong to the full training sample. Native
                // GUIDE holds them fixed while fitting each CV training fold.
                fold_options.class_priors = Some(ctx.class_priors.clone());
            }
            let mut fold_of = vec![usize::MAX; ctx.data.rows.len()];
            for (fold, held) in parts.iter().enumerate() {
                for &i in held {
                    fold_of[i] = fold;
                }
            }
            for (f, held) in parts.iter().enumerate() {
                let train: Vec<usize> = ids.iter().copied().filter(|&i| fold_of[i] != f).collect();
                let fold_ctx = Context::new(ctx.data, &fold_options, &train, Some(&ctx.classes))?;
                let mut fold_rng =
                    ChaCha8Rng::seed_from_u64(ctx.options.seed.wrapping_add(f as u64 + 1));
                let mut next = 0;
                let full = Tree {
                    root: grow(
                        &fold_ctx,
                        &train,
                        0,
                        &mut next,
                        &mut fold_rng,
                        SelectionMode::Fit,
                    ),
                };
                crate::tree::validate_node(
                    &full.root,
                    &ctx.data.schema,
                    ctx.classes.len(),
                    ctx.options,
                    0,
                )?;
                let fp = weakest_link_path(&full);
                for (j, &alpha) in alphas.iter().enumerate() {
                    let subtree = if j == 0 && path.len() > 1 {
                        &fp[0].1
                    } else {
                        at_alpha(&fp, alpha)
                    };
                    results[j].extend(external_losses(ctx, subtree, ctx.data, held)?);
                }
            }
            for (j, (alpha, t)) in path.iter().enumerate() {
                let (m, se) = weighted_summary(&results[j]);
                report.pruning_scores.push(PruningScore {
                    alpha: *alpha,
                    leaves: t.root.leaves(),
                    mean_loss: m,
                    standard_error: se,
                });
            }
            let best = (0..path.len())
                .min_by(|&a, &b| {
                    report.pruning_scores[a]
                        .mean_loss
                        .total_cmp(&report.pruning_scores[b].mean_loss)
                })
                .unwrap();
            let cutoff = report.pruning_scores[best].mean_loss
                + standard_errors * report.pruning_scores[best].standard_error;
            let chosen = (0..path.len())
                .rev()
                .find(|&j| stats::less_or_close(report.pruning_scores[j].mean_loss, cutoff))
                .unwrap();
            report.selected_alpha = path[chosen].0;
            Ok(path[chosen].1.clone())
        }
    }
}
fn external_losses(
    ctx: &Context<'_>,
    tree: &Tree,
    data: &Dataset,
    ids: &[usize],
) -> Result<Vec<(f64, f64)>> {
    // Priors affect the fitted classifier, not the held-out population. Native
    // GUIDE averages the observed misclassification costs over validation rows.
    ids.iter()
        .map(|&i| {
            let leaf = &tree.root.leaf(&data.rows[i]).model;
            let loss = if let LeafModel::Classification {
                predicted_class, ..
            } = leaf
            {
                let label = data.targets[i].label().unwrap();
                let y = ctx
                    .classes
                    .binary_search(&label)
                    .map_err(|_| invalid(format!("unknown validation class {label}")))?;
                ctx.costs[y][*predicted_class]
            } else {
                leaf.regression_loss(
                    &data.rows[i],
                    data.offset(i),
                    data.targets[i].number().unwrap(),
                    &ctx.loss,
                )
            };
            if !loss.is_finite() {
                return Err(invalid(
                    "nonfinite validation loss; rescale the response or inspect the fitted model",
                ));
            }
            Ok((
                loss,
                if ctx.classifying() {
                    1.0
                } else {
                    data.weight(i)
                },
            ))
        })
        .collect()
}
fn weighted_summary(values: &[(f64, f64)]) -> (f64, f64) {
    let scale = values.iter().map(|v| v.1).fold(0.0, f64::max);
    let w: f64 = values.iter().map(|v| v.1 / scale).sum();
    let loss_scale = values.iter().map(|v| v.0.abs()).fold(0.0, f64::max);
    let loss_scale = if loss_scale == 0.0 { 1.0 } else { loss_scale };
    let mean_scaled = values
        .iter()
        .map(|(x, weight)| (x / loss_scale) * (weight / scale))
        .sum::<f64>()
        / w;
    let n = values.len() as f64;
    let se = if n > 1.0 {
        (values
            .iter()
            .map(|(x, wi)| ((x / loss_scale) * (wi / scale) * n / w - mean_scaled).powi(2))
            .sum::<f64>()
            / n
            / n)
            .sqrt()
    } else {
        0.0
    };
    (mean_scaled * loss_scale, se * loss_scale)
}
