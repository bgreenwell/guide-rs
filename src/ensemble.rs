use crate::{
    node::{Context, observation_loss},
    selection::SelectionMode,
    tree::{grow, validate_supported},
    *,
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EnsembleKind {
    #[default]
    Bagging,
    Forest,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EnsembleOptions {
    pub kind: EnsembleKind,
    pub trees: usize,
    pub sample_fraction: f64,
}
impl Default for EnsembleOptions {
    fn default() -> Self {
        Self {
            kind: EnsembleKind::Bagging,
            trees: 100,
            sample_fraction: 1.0,
        }
    }
}
impl FittedModel {
    pub fn fit_ensemble(
        data: &Dataset,
        options: &FitOptions,
        ensemble: &EnsembleOptions,
    ) -> Result<Self> {
        options.validate()?;
        data.validate(&options.task)?;
        validate_supported(data, options)?;
        if matches!(ensemble.kind, EnsembleKind::Bagging) && options.mtry.is_some() {
            return Err(invalid(
                "per-node feature sampling requires ensemble kind forest",
            ));
        }
        if ensemble.trees == 0
            || !ensemble.sample_fraction.is_finite()
            || ensemble.sample_fraction <= 0.0
            || ensemble.sample_fraction > 1.0
        {
            return Err(invalid(
                "ensemble trees must be positive and sample_fraction in (0, 1]",
            ));
        }
        if matches!(options.pruning, Pruning::Validation) {
            return Err(invalid("ensemble validation pruning is not implemented"));
        }
        let ids = data.active_rows();
        let global = Context::new(data, options, &ids, None)?;
        let size = ((ids.len() as f64 * ensemble.sample_fraction).round() as usize).max(1);
        let mut opts = options.clone();
        if matches!(ensemble.kind, EnsembleKind::Forest) {
            if opts.pruning != Pruning::None {
                return Err(invalid("forest requires pruning kind none"));
            }
            let p = data.schema.iter().filter(|s| s.role.splits()).count();
            if opts.mtry.is_none() {
                opts.mtry = Some(
                    if global.classifying() {
                        (p as f64).sqrt() as usize
                    } else {
                        p / 3
                    }
                    .max(1),
                );
            }
        }
        let mut rng = ChaCha8Rng::seed_from_u64(options.seed);
        let mut trees = vec![];
        let mut oob: Vec<Vec<usize>> = vec![vec![]; data.rows.len()];
        let mut report = FitReport {
            training_rows: ids.len(),
            excluded_rows: data.rows.len() - ids.len(),
            ..FitReport::default()
        };
        for t in 0..ensemble.trees {
            let sampled: Vec<usize> = (0..size)
                .map(|_| ids[rng.gen_range(0..ids.len())])
                .collect();
            let mut in_bag = vec![false; data.rows.len()];
            for &i in &sampled {
                in_bag[i] = true;
            }
            let bootstrap = data.subset(&sampled);
            let bi: Vec<usize> = (0..size).collect();
            let mut tree_options = opts.clone();
            tree_options.seed = rng.r#gen();
            let ctx = Context::new(&bootstrap, &tree_options, &bi, Some(&global.classes))?;
            let mut next = 0;
            let mut tree_rng = ChaCha8Rng::seed_from_u64(tree_options.seed);
            let full = Tree {
                root: grow(&ctx, &bi, 0, &mut next, &mut tree_rng, SelectionMode::Fit),
            };
            crate::tree::validate_node(
                &full.root,
                &data.schema,
                global.classes.len(),
                &tree_options,
                0,
            )?;
            report.unpruned_leaves += full.root.leaves();
            let mut tree_report = FitReport::default();
            let mut tree = crate::pruning::prune(full, &ctx, None, &mut tree_report)?;
            ctx.finalize_probabilities(&mut tree.root);
            report.fitted_leaves += tree.root.leaves();
            crate::tree::collect_warnings(&tree.root, &mut report.warnings);
            trees.push(tree);
            for &i in &ids {
                if !in_bag[i] {
                    oob[i].push(t);
                }
            }
        }
        let mut total_loss = 0.0;
        let mut total_weight = 0.0;
        for &i in &ids {
            if oob[i].is_empty() {
                continue;
            }
            report.out_of_bag_rows += 1;
            let prediction = if global.classifying() {
                let mut p = vec![0.0; global.classes.len()];
                for &t in &oob[i] {
                    if let LeafModel::Classification { probabilities, .. } =
                        &trees[t].root.leaf(&data.rows[i]).model
                    {
                        for (p, q) in p.iter_mut().zip(probabilities) {
                            *p += q / oob[i].len() as f64;
                        }
                    }
                }
                let best = (0..p.len())
                    .min_by(|&a, &b| {
                        let cost = |j| {
                            p.iter()
                                .enumerate()
                                .map(|(k, v)| v * global.costs[k][j])
                                .sum::<f64>()
                        };
                        cost(a).total_cmp(&cost(b)).then(a.cmp(&b))
                    })
                    .unwrap();
                global.costs[global.y[i] as usize][best]
            } else if let Loss::DualQuantile { lower, upper } = global.loss {
                let mut lo = 0.0;
                let mut hi = 0.0;
                for &t in &oob[i] {
                    if let LeafModel::Quantiles { lower, upper } =
                        &trees[t].root.leaf(&data.rows[i]).model
                    {
                        lo += lower / oob[i].len() as f64;
                        hi += upper / oob[i].len() as f64;
                    }
                }
                crate::node::quantile_pair_loss(global.y[i], lo, hi, lower, upper)
            } else {
                let p = oob[i]
                    .iter()
                    .map(|&t| {
                        trees[t].root.leaf(&data.rows[i]).model.scalar(
                            &data.rows[i],
                            data.offset(i),
                            &global.loss,
                        ) / oob[i].len() as f64
                    })
                    .sum();
                observation_loss(&global.loss, global.y[i], p)
            };
            let w = if global.classifying() {
                global.weights[i]
            } else {
                data.weight(i)
            };
            total_loss += w * prediction;
            total_weight += w;
        }
        if total_weight > 0.0 {
            report.out_of_bag_loss = Some(total_loss / total_weight);
        }
        if report.out_of_bag_rows < ids.len() {
            report.warnings.push("Some training rows have no out-of-bag predictions; they are excluded from OOB loss.".into());
        }
        let model = Self {
            format_version: 1,
            algorithm_version: ALGORITHM_VERSION.into(),
            schema: data.schema.clone(),
            classes: global.classes,
            options: opts,
            input: None,
            requires_exposure: !data.offsets.is_empty(),
            ensemble: Some(ensemble.clone()),
            trees,
            report,
        };
        model.validate()?;
        Ok(model)
    }
}
