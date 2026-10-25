use crate::{node::Context, selection::SelectionMode, tree::grow, *};
use rand::{SeedableRng, seq::SliceRandom};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PruningScore {
    pub alpha: f64,
    pub leaves: usize,
    pub mean_loss: f64,
    pub standard_error: f64,
}
/// The weakest-link pruning sequence of a tree. Stage 0 is the full tree and
/// each later stage collapses one more internal node; subtrees are built only
/// on request. Cached leaf counts and risks are updated with the same
/// left-plus-right sums as a recursive recount, so the alphas are unchanged.
pub(crate) struct PrunePath<'a> {
    tree: &'a Tree,
    /// Complexity parameter of each stage.
    alphas: Vec<f64>,
    /// Leaf count of each stage.
    leaves: Vec<usize>,
    /// The stage at which each node id becomes a leaf; `usize::MAX` if never.
    collapsed_at: Vec<usize>,
}
fn internal(n: &Node) -> Option<(&Node, &Node)> {
    match (&n.split, &n.left, &n.right) {
        (Some(_), Some(l), Some(r)) => Some((l, r)),
        _ => None,
    }
}
impl<'a> PrunePath<'a> {
    pub(crate) fn new(tree: &'a Tree) -> Self {
        fn max_id(n: &Node) -> usize {
            internal(n).map_or(n.id, |(l, r)| n.id.max(max_id(l)).max(max_id(r)))
        }
        let size = max_id(&tree.root) + 1;
        let mut leaves = vec![1; size];
        let mut leaf_risk = vec![0.0; size];
        let mut parent = vec![usize::MAX; size];
        let mut children = vec![None; size];
        let mut risk = vec![0.0; size];
        fn index(
            n: &Node,
            leaves: &mut [usize],
            leaf_risk: &mut [f64],
            parent: &mut [usize],
            children: &mut [Option<(usize, usize)>],
            risk: &mut [f64],
        ) {
            risk[n.id] = n.risk;
            if let Some((l, r)) = internal(n) {
                index(l, leaves, leaf_risk, parent, children, risk);
                index(r, leaves, leaf_risk, parent, children, risk);
                parent[l.id] = n.id;
                parent[r.id] = n.id;
                children[n.id] = Some((l.id, r.id));
                leaves[n.id] = leaves[l.id] + leaves[r.id];
                leaf_risk[n.id] = leaf_risk[l.id] + leaf_risk[r.id];
            } else {
                leaf_risk[n.id] = n.risk;
            }
        }
        index(
            &tree.root,
            &mut leaves,
            &mut leaf_risk,
            &mut parent,
            &mut children,
            &mut risk,
        );
        let root = tree.root.id;
        let mut path = Self {
            tree,
            alphas: vec![0.0],
            leaves: vec![leaves[root]],
            collapsed_at: vec![usize::MAX; size],
        };
        let mut last = 0.0_f64;
        while leaves[root] > 1 {
            let mut weakest = None;
            path.minimum_alpha(&tree.root, &leaves, &leaf_risk, &mut weakest);
            let (mut alpha, id) = weakest.expect("an internal node exists");
            alpha = alpha.max(last).max(0.0);
            path.collapsed_at[id] = path.alphas.len();
            leaves[id] = 1;
            leaf_risk[id] = risk[id];
            let mut a = parent[id];
            while a != usize::MAX {
                // Every parent has two children.
                let (l, r) = children[a].expect("parent of a node");
                leaves[a] = leaves[l] + leaves[r];
                leaf_risk[a] = leaf_risk[l] + leaf_risk[r];
                a = parent[a];
            }
            path.alphas.push(alpha);
            path.leaves.push(leaves[root]);
            last = alpha;
        }
        path
    }
    fn minimum_alpha(
        &self,
        n: &Node,
        leaves: &[usize],
        leaf_risk: &[f64],
        best: &mut Option<(f64, usize)>,
    ) {
        let Some((l, r)) = internal(n) else {
            return;
        };
        if self.collapsed_at[n.id] != usize::MAX {
            return;
        }
        let a = (n.risk - leaf_risk[n.id]) / (leaves[n.id] - 1) as f64;
        let a = a.max(0.0);
        // Native GUIDE retains separate stages for tied branches. Visit parents
        // before children, left before right, and prune only one weakest link.
        if best.is_none_or(|(alpha, _)| stats::strictly_less(a, alpha)) {
            *best = Some((a, n.id));
        }
        self.minimum_alpha(l, leaves, leaf_risk, best);
        self.minimum_alpha(r, leaves, leaf_risk, best);
    }
    fn len(&self) -> usize {
        self.alphas.len()
    }
    /// The last stage whose alpha does not exceed `alpha`, or the full tree.
    fn stage_at(&self, alpha: f64) -> usize {
        (0..self.len())
            .rev()
            .find(|&k| self.alphas[k] <= alpha)
            .unwrap_or(0)
    }
    fn collapsed(&self, n: &Node, stage: usize) -> bool {
        self.collapsed_at[n.id] <= stage
    }
    fn subtree(&self, stage: usize) -> Tree {
        fn copy(path: &PrunePath<'_>, n: &Node, stage: usize) -> Node {
            let mut node = Node {
                id: n.id,
                sample_count: n.sample_count,
                weight_sum: n.weight_sum,
                risk: n.risk,
                impurity: n.impurity,
                model: n.model.clone(),
                diagnostics: n.diagnostics.clone(),
                split: None,
                left: None,
                right: None,
            };
            if let Some((l, r)) = internal(n)
                && !path.collapsed(n, stage)
            {
                node.split = n.split.clone();
                node.left = Some(Box::new(copy(path, l, stage)));
                node.right = Some(Box::new(copy(path, r, stage)));
            }
            node
        }
        Tree {
            root: copy(self, &self.tree.root, stage),
        }
    }
    /// Held-out `(loss, weight)` of each row in `ids` at each of `stages`.
    /// A row's leaf at any stage is the first collapsed node on its path
    /// through the full tree, so each row is routed once.
    fn losses(
        &self,
        ctx: &Context<'_>,
        data: &Dataset,
        ids: &[usize],
        stages: &[usize],
    ) -> Result<Vec<Vec<(f64, f64)>>> {
        let mut result = vec![Vec::with_capacity(ids.len()); stages.len()];
        let mut nodes = vec![];
        let mut cache = vec![];
        for &i in ids {
            nodes.clear();
            let mut n = &self.tree.root;
            nodes.push(n);
            while let Some((l, r)) = internal(n) {
                // `internal` returned children, so the split exists.
                n = if n.split.as_ref().expect("split").goes_left(&data.rows[i]) {
                    l
                } else {
                    r
                };
                nodes.push(n);
            }
            cache.clear();
            cache.resize(nodes.len(), None);
            let weight = if ctx.classifying() {
                1.0
            } else {
                data.weight(i)
            };
            for (out, &stage) in result.iter_mut().zip(stages) {
                let d = nodes
                    .iter()
                    .position(|n| self.collapsed(n, stage))
                    .unwrap_or(nodes.len() - 1);
                let loss = match cache[d] {
                    Some(loss) => loss,
                    None => {
                        let loss = row_loss(ctx, &nodes[d].model, data, i)?;
                        cache[d] = Some(loss);
                        loss
                    }
                };
                out.push((loss, weight));
            }
        }
        Ok(result)
    }
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
    let path = PrunePath::new(&tree);
    match ctx.options.pruning {
        Pruning::None => Ok(tree.clone()),
        Pruning::CostComplexity { alpha } => {
            report.selected_alpha = alpha;
            Ok(path.subtree(path.stage_at(alpha)))
        }
        Pruning::FixedLeaves { max_leaves } => {
            // The path ends at the root-only tree and max_leaves is validated
            // positive, so some subtree always qualifies.
            let stage = (0..path.len())
                .find(|&k| path.leaves[k] <= max_leaves)
                .unwrap();
            report.selected_alpha = path.alphas[stage];
            Ok(path.subtree(stage))
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
            let stages: Vec<usize> = (0..path.len()).collect();
            for (j, losses) in path.losses(ctx, data, &ids, &stages)?.iter().enumerate() {
                let (m, se) = weighted_summary(losses);
                report.pruning_scores.push(PruningScore {
                    alpha: path.alphas[j],
                    leaves: path.leaves[j],
                    mean_loss: m,
                    standard_error: se,
                });
                if stats::less_or_close(m, best_loss) {
                    best = j;
                    best_loss = m;
                }
            }
            report.selected_alpha = path.alphas[best];
            Ok(path.subtree(best))
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
                .alphas
                .iter()
                .enumerate()
                .map(|(i, a)| {
                    if i + 1 == path.len() {
                        f64::INFINITY
                    } else if *a == 0.0 {
                        0.0
                    } else {
                        a.sqrt() * path.alphas[i + 1].sqrt()
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
            // Folds are fitted in parallel and their losses combined in fold
            // order, so results and the reported error do not depend on threads.
            let fold_losses: Vec<Result<Vec<_>>> = parts
                .par_iter()
                .enumerate()
                .map(|(f, held)| {
                    let train: Vec<usize> =
                        ids.iter().copied().filter(|&i| fold_of[i] != f).collect();
                    let fold_ctx =
                        Context::new(ctx.data, &fold_options, &train, Some(&ctx.classes))?;
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
                    let fp = PrunePath::new(&full);
                    let stages: Vec<usize> = alphas
                        .iter()
                        .enumerate()
                        .map(|(j, &alpha)| {
                            if j == 0 && path.len() > 1 {
                                0
                            } else {
                                fp.stage_at(alpha)
                            }
                        })
                        .collect();
                    fp.losses(ctx, ctx.data, held, &stages)
                })
                .collect();
            for losses in fold_losses {
                for (j, l) in losses?.into_iter().enumerate() {
                    results[j].extend(l);
                }
            }
            for (j, losses) in results.iter().enumerate() {
                let (m, se) = weighted_summary(losses);
                report.pruning_scores.push(PruningScore {
                    alpha: path.alphas[j],
                    leaves: path.leaves[j],
                    mean_loss: m,
                    standard_error: se,
                });
            }
            // The pruning path always contains at least the full tree.
            let best = (0..path.len())
                .min_by(|&a, &b| {
                    report.pruning_scores[a]
                        .mean_loss
                        .total_cmp(&report.pruning_scores[b].mean_loss)
                })
                .unwrap();
            let cutoff = report.pruning_scores[best].mean_loss
                + standard_errors * report.pruning_scores[best].standard_error;
            // `best` itself satisfies the cutoff because standard errors are
            // nonnegative, so the search always succeeds.
            let chosen = (0..path.len())
                .rev()
                .find(|&j| stats::less_or_close(report.pruning_scores[j].mean_loss, cutoff))
                .unwrap();
            report.selected_alpha = path.alphas[chosen];
            Ok(path.subtree(chosen))
        }
    }
}
/// The held-out loss of row `i` under a leaf model.
fn row_loss(ctx: &Context<'_>, leaf: &LeafModel, data: &Dataset, i: usize) -> Result<f64> {
    // Priors affect the fitted classifier, not the held-out population. Native
    // GUIDE averages the observed misclassification costs over validation rows.
    // Callers pass only active rows, whose responses are observed and typed
    // by the validated task, so the response accessors below cannot fail.
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
    Ok(loss)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indexed_stages_match_materialized_subtrees() {
        let x = |i: usize| ((i * 37) % 101) as f64;
        let data = Dataset {
            schema: vec![FeatureSchema {
                name: "x".into(),
                feature_type: FeatureType::Numeric,
                role: Role::Split,
                flags: None,
            }],
            rows: (0..120).map(|i| vec![Value::Number(x(i))]).collect(),
            targets: (0..120)
                .map(|i| Value::Number((x(i) / 10.0).floor() + (i % 7) as f64 * 0.3))
                .collect(),
            weights: vec![],
            offsets: vec![],
        };
        let options = FitOptions {
            pruning: Pruning::None,
            min_samples_leaf: 3,
            ..Default::default()
        };
        let ids: Vec<usize> = (0..120).collect();
        let ctx = Context::new(&data, &options, &ids, None).unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let tree = Tree {
            root: grow(&ctx, &ids, 0, &mut 0, &mut rng, SelectionMode::Fit),
        };
        let path = PrunePath::new(&tree);
        assert!(path.len() > 5);
        assert_eq!(path.leaves[0], tree.root.leaves());
        assert_eq!(path.leaves[path.len() - 1], 1);
        let stages: Vec<usize> = (0..path.len()).rev().collect();
        let losses = path.losses(&ctx, &data, &ids, &stages).unwrap();
        for (losses, &stage) in losses.iter().zip(&stages) {
            let subtree = path.subtree(stage);
            assert_eq!(subtree.root.leaves(), path.leaves[stage]);
            let expected: Vec<(f64, f64)> = ids
                .iter()
                .map(|&i| {
                    let leaf = &subtree.root.leaf(&data.rows[i]).model;
                    (row_loss(&ctx, leaf, &data, i).unwrap(), 1.0)
                })
                .collect();
            assert_eq!(losses, &expected);
        }
    }
}
