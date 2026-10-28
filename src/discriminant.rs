//! GUIDE's linear discriminant (LDA) node models for classification.
//!
//! A node selects its own columns, projects them on its leading discriminant
//! coordinates and classifies with class-specific Gaussian densities on those
//! coordinates. The rules follow black-box probes of GUIDE 46.3; see
//! `docs/discriminant-notes.md`.
use crate::node::{Context, Term};
use crate::{Dataset, FeatureType, stats};
use crate::{LeafModel, Node, Split};
use serde::{Deserialize, Serialize};

/// Relative residual sum of squares below which a selected column is treated
/// as a linear combination of columns selected before it and dropped.
const COLLINEAR_TOLERANCE: f64 = 1e-10;

/// A candidate column of the node design.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Column {
    /// A numeric feature, with missing values imputed by the node mean.
    Numeric(usize),
    /// The 0/1 missing-value indicator of a numeric feature.
    Missing(usize),
}

/// The candidate columns of a training set: numeric fit features, then an
/// indicator for each of them that has missing values. Columns constant over
/// the training rows are left out, as GUIDE excludes constant variables.
pub(crate) fn candidate_columns(data: &Dataset, ids: &[usize]) -> Vec<Column> {
    let numeric: Vec<usize> = (0..data.schema.len())
        .filter(|&j| {
            matches!(data.schema[j].feature_type, FeatureType::Numeric)
                && data.schema[j].role.fits()
        })
        .collect();
    let mut columns = vec![];
    for &j in &numeric {
        let mut values = ids.iter().filter_map(|&i| data.rows[i][j].number());
        if let Some(first) = values.next()
            && values.any(|x| x != first)
        {
            columns.push(Column::Numeric(j));
        }
    }
    for &j in &numeric {
        if ids.iter().any(|&i| data.rows[i][j].is_missing()) {
            columns.push(Column::Missing(j));
        }
    }
    columns
}

/// A fitted LDA leaf: Gaussian class densities on the node's leading
/// discriminant coordinates, with a pooled variance per coordinate. Classes
/// absent from the node have no mean and are never predicted.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Discriminant {
    /// The selected columns as raw (uncentered, unscaled) terms; missing
    /// values take the node mean.
    pub terms: Vec<Term>,
    /// `coefficients[k][t]` is the weight of term `t` in coordinate `k`.
    pub coefficients: Vec<Vec<f64>>,
    /// Class means on each coordinate.
    pub class_means: Vec<Option<Vec<f64>>>,
    /// Pooled within-class variance of each coordinate.
    pub variances: Vec<f64>,
    /// Class priors used for classification.
    pub priors: Vec<f64>,
    /// Training rows and class priors at the root, for GUIDE's zero-posterior
    /// correction of the reported probabilities.
    pub root_rows: usize,
    pub root_priors: Vec<f64>,
}
impl Discriminant {
    /// The coordinate values of a row.
    pub(crate) fn coordinates(&self, row: &[crate::Value]) -> Vec<f64> {
        let x: Vec<f64> = self.terms.iter().map(|t| t.value(row)).collect();
        self.coefficients
            .iter()
            .map(|a| a.iter().zip(&x).map(|(a, x)| a * x).sum())
            .collect()
    }
    /// Posterior class probabilities of a row, with GUIDE's zero-posterior
    /// correction: absent classes keep their share of one prior observation.
    pub(crate) fn posteriors(&self, row: &[crate::Value]) -> Vec<f64> {
        let mut p = self.classifier_posteriors(row);
        if p.contains(&0.0) {
            let n = self.root_rows as f64;
            for (p, prior) in p.iter_mut().zip(&self.root_priors) {
                *p = (n * *p + prior) / (n + 1.0);
            }
        }
        p
    }
    /// Posteriors of the classifier itself; absent classes get zero.
    fn classifier_posteriors(&self, row: &[crate::Value]) -> Vec<f64> {
        let z = self.coordinates(row);
        let scores: Vec<Option<f64>> = self
            .class_means
            .iter()
            .zip(&self.priors)
            .map(|(mean, prior)| {
                mean.as_ref().map(|m| {
                    prior.ln()
                        - 0.5
                            * m.iter()
                                .zip(&z)
                                .zip(&self.variances)
                                .map(|((m, z), v)| (z - m) * (z - m) / v)
                                .sum::<f64>()
                })
            })
            .collect();
        let top = scores
            .iter()
            .flatten()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let mut p: Vec<f64> = scores
            .iter()
            .map(|s| s.map_or(0.0, |s| (s - top).exp()))
            .collect();
        let total: f64 = p.iter().sum();
        for v in &mut p {
            *v /= total;
        }
        p
    }
    /// Whether a deserialized leaf is usable with `classes` classes.
    pub(crate) fn is_valid(&self, schema: &[crate::FeatureSchema], classes: usize) -> bool {
        let dimensions = self.coefficients.len();
        valid_terms(&self.terms, schema)
            && (1..=2).contains(&dimensions)
            && self
                .coefficients
                .iter()
                .all(|a| a.len() == self.terms.len() && a.iter().all(|x| x.is_finite()))
            && self.variances.len() == dimensions
            && self.variances.iter().all(|v| v.is_finite() && *v > 0.0)
            && self.class_means.len() == classes
            && self.class_means.iter().flatten().count() >= 2
            && self
                .class_means
                .iter()
                .flatten()
                .all(|m| m.len() == dimensions && m.iter().all(|x| x.is_finite()))
            && self.priors.len() == classes
            && self.root_priors.len() == classes
            && self.root_rows > 0
            && self
                .priors
                .iter()
                .chain(&self.root_priors)
                .all(|p| p.is_finite() && *p >= 0.0)
            && self
                .class_means
                .iter()
                .zip(&self.priors)
                .all(|(m, p)| m.is_none() || *p > 0.0)
    }
    /// The class with the largest posterior, the first on ties.
    pub(crate) fn predict(&self, row: &[crate::Value]) -> usize {
        let p = self.classifier_posteriors(row);
        (0..p.len()).fold(0, |best, k| if p[k] > p[best] { k } else { best })
    }
}

/// Whether `terms` are raw numeric fit columns of `schema`, as LDA models use.
pub(crate) fn valid_terms(terms: &[Term], schema: &[crate::FeatureSchema]) -> bool {
    !terms.is_empty()
        && terms.iter().all(|t| {
            t.feature < schema.len()
                && matches!(schema[t.feature].feature_type, FeatureType::Numeric)
                && schema[t.feature].role.fits()
                && t.power == 1
                && t.category.is_none()
                && t.center == 0.0
                && t.scale == 1.0
                && t.impute.is_finite()
        })
}

/// Columns kept by a node: ranked by one-way ANOVA F p-value, then counted by
/// GUIDE's thresholds. With K candidates and m of them below 0.4/K, the node
/// keeps m when m >= 2; otherwise it keeps those below 0.8/(K-2) (m = 1) or
/// 0.8/(K-1) (m = 0), and at least two. Columns that are linear combinations
/// of columns ranked before them are then dropped.
pub(crate) fn select(design: &[Vec<f64>], labels: &[usize], classes: usize) -> Vec<usize> {
    let log_p: Vec<(usize, f64)> = design
        .iter()
        .enumerate()
        .filter_map(|(j, x)| anova_log_p(x, labels, classes).map(|p| (j, p)))
        .collect();
    let mut ranked = log_p.clone();
    ranked.sort_by(|a, b| a.1.total_cmp(&b.1));
    let k = ranked.len();
    let below = |t: f64| ranked.iter().filter(|(_, p)| *p < t.ln()).count();
    let significant = below(0.4 / k as f64);
    let count = if significant >= 2 {
        significant
    } else if k <= 2 {
        k
    } else {
        let relaxed = if significant == 1 {
            0.8 / (k - 2) as f64
        } else {
            0.8 / (k - 1) as f64
        };
        below(relaxed).max(2)
    };
    let mut kept: Vec<usize> = vec![];
    let mut basis: Vec<Vec<f64>> = vec![];
    for &(j, _) in &ranked[..count] {
        let mean = stats::mean(&design[j]);
        let mut r: Vec<f64> = design[j].iter().map(|x| x - mean).collect();
        let scale: f64 = r.iter().map(|x| x * x).sum();
        for q in &basis {
            let dot: f64 = r.iter().zip(q).map(|(a, b)| a * b).sum();
            for (a, b) in r.iter_mut().zip(q) {
                *a -= dot * b;
            }
        }
        let rss: f64 = r.iter().map(|x| x * x).sum();
        if rss > COLLINEAR_TOLERANCE * scale {
            let norm = rss.sqrt();
            basis.push(r.iter().map(|x| x / norm).collect());
            kept.push(j);
        }
    }
    kept
}

/// Natural log of the one-way ANOVA F-test p-value, or None for a column that
/// is constant in the node or a node without residual degrees of freedom.
fn anova_log_p(x: &[f64], labels: &[usize], classes: usize) -> Option<f64> {
    let mut sum = vec![0.0; classes];
    let mut count = vec![0usize; classes];
    for (&v, &c) in x.iter().zip(labels) {
        sum[c] += v;
        count[c] += 1;
    }
    let groups = count.iter().filter(|&&n| n > 0).count();
    let n = x.len();
    if groups < 2 || n <= groups {
        return None;
    }
    let mean = stats::mean(x);
    let between: f64 = (0..classes)
        .filter(|&c| count[c] > 0)
        .map(|c| {
            let m = sum[c] / count[c] as f64;
            count[c] as f64 * (m - mean) * (m - mean)
        })
        .sum();
    let within: f64 = x
        .iter()
        .zip(labels)
        .map(|(&v, &c)| {
            let m = sum[c] / count[c] as f64;
            (v - m) * (v - m)
        })
        .sum();
    if between + within <= 0.0 {
        return None;
    }
    let f = (between / (groups - 1) as f64) / (within / (n - groups) as f64);
    Some(stats::f_log_survival(f, groups - 1, n - groups))
}

/// Discriminant coordinates of `design` (one column per variable): the
/// generalized eigenvectors of the between-class scatter against the total
/// scatter, in decreasing eigenvalue order, scaled so that `a'Ta = 1`. The
/// columns must be linearly independent. Signs are arbitrary.
pub(crate) fn coordinates(design: &[Vec<f64>], labels: &[usize], classes: usize) -> Vec<Vec<f64>> {
    let p = design.len();
    let n = labels.len();
    let means: Vec<f64> = design.iter().map(|x| stats::mean(x)).collect();
    let mut count = vec![0.0; classes];
    let mut class_sums = vec![vec![0.0; p]; classes];
    for (i, &c) in labels.iter().enumerate() {
        count[c] += 1.0;
        for j in 0..p {
            class_sums[c][j] += design[j][i];
        }
    }
    let mut total = vec![vec![0.0; p]; p];
    let mut between = vec![vec![0.0; p]; p];
    for a in 0..p {
        for b in 0..p {
            total[a][b] = (0..n)
                .map(|i| (design[a][i] - means[a]) * (design[b][i] - means[b]))
                .sum();
            between[a][b] = (0..classes)
                .filter(|&c| count[c] > 0.0)
                .map(|c| {
                    let da = class_sums[c][a] / count[c] - means[a];
                    let db = class_sums[c][b] / count[c] - means[b];
                    count[c] * da * db
                })
                .sum();
        }
    }
    // Whiten by T^{-1/2} = Q diag(1/sqrt(l)) Q', then diagonalize the whitened B.
    let (values, vectors) = stats::symmetric_eigen(&total);
    let root_inverse: Vec<Vec<f64>> = (0..p)
        .map(|a| {
            (0..p)
                .map(|b| {
                    (0..p)
                        .map(|k| vectors[k][a] * vectors[k][b] / values[k].sqrt())
                        .sum()
                })
                .collect()
        })
        .collect();
    let product = |x: &[Vec<f64>], y: &[Vec<f64>]| -> Vec<Vec<f64>> {
        (0..p)
            .map(|a| {
                (0..p)
                    .map(|b| (0..p).map(|k| x[a][k] * y[k][b]).sum())
                    .collect()
            })
            .collect()
    };
    let whitened = product(&product(&root_inverse, &between), &root_inverse);
    let (_, directions) = stats::symmetric_eigen(&whitened);
    directions
        .iter()
        .map(|v| {
            (0..p)
                .map(|a| (0..p).map(|b| root_inverse[a][b] * v[b]).sum())
                .collect()
        })
        .collect()
}

/// Fits an LDA leaf to rows `ids` with class `labels` (indices below
/// `classes`). `priors` are the class priors used for classification and
/// `root_rows` and `root_priors` those of the zero-posterior correction.
/// Returns None when fewer than two classes are present, no column can be
/// used or a coordinate has no within-class variance.
pub(crate) fn fit(
    data: &Dataset,
    ids: &[usize],
    labels: &[usize],
    classes: usize,
    columns: &[Column],
    priors: &[f64],
    root: (usize, &[f64]),
) -> Option<Discriminant> {
    let present: Vec<usize> = (0..classes).filter(|c| labels.contains(c)).collect();
    if present.len() < 2 {
        return None;
    }
    let terms: Vec<Term> = columns.iter().map(|&c| node_term(data, ids, c)).collect();
    let design: Vec<Vec<f64>> = terms
        .iter()
        .map(|t| ids.iter().map(|&i| t.value(&data.rows[i])).collect())
        .collect();
    let mut kept = select(&design, labels, classes);
    if kept.is_empty() {
        return None;
    }
    kept.sort_unstable();
    let design: Vec<Vec<f64>> = kept.iter().map(|&j| design[j].clone()).collect();
    let dimensions = if present.len() == 2 {
        1
    } else {
        kept.len().min(2)
    };
    // GUIDE's coordinate signs follow no rule found so far (see
    // docs/discriminant-notes.md). They matter because the cut grid counts
    // from the low end; here the last column's coefficient is positive.
    let coefficients: Vec<Vec<f64>> = coordinates(&design, labels, classes)
        .into_iter()
        .take(dimensions)
        .map(|a| {
            if a[a.len() - 1] < 0.0 {
                a.iter().map(|x| -x).collect()
            } else {
                a
            }
        })
        .collect();
    let projected: Vec<Vec<f64>> = coefficients
        .iter()
        .map(|a| {
            (0..ids.len())
                .map(|i| a.iter().zip(&design).map(|(a, x)| a * x[i]).sum())
                .collect()
        })
        .collect();
    let mut class_means = vec![None; classes];
    let mut variances = vec![0.0; dimensions];
    for &c in &present {
        let rows: Vec<usize> = (0..ids.len()).filter(|&i| labels[i] == c).collect();
        let means: Vec<f64> = projected
            .iter()
            .map(|z| rows.iter().map(|&i| z[i]).sum::<f64>() / rows.len() as f64)
            .collect();
        for (k, z) in projected.iter().enumerate() {
            variances[k] += rows.iter().map(|&i| (z[i] - means[k]).powi(2)).sum::<f64>();
        }
        class_means[c] = Some(means);
    }
    let residual_df = (ids.len() - present.len()) as f64;
    for v in &mut variances {
        *v /= residual_df;
    }
    if variances.iter().any(|v| !(v.is_finite() && *v > 0.0)) {
        return None;
    }
    Some(Discriminant {
        terms: kept.iter().map(|&j| terms[j].clone()).collect(),
        coefficients,
        class_means,
        variances,
        priors: priors.to_vec(),
        root_rows: root.0,
        root_priors: root.1.to_vec(),
    })
}

/// The raw term of a candidate column; numeric values are imputed by their
/// mean over the node's observed values.
fn node_term(data: &Dataset, ids: &[usize], column: Column) -> Term {
    let (feature, missing_indicator) = match column {
        Column::Numeric(j) => (j, false),
        Column::Missing(j) => (j, true),
    };
    let observed: Vec<f64> = ids
        .iter()
        .filter_map(|&i| data.rows[i][feature].number())
        .collect();
    Term {
        feature,
        power: 1,
        category: None,
        center: 0.0,
        scale: 1.0,
        impute: if observed.is_empty() {
            0.0
        } else {
            stats::mean(&observed)
        },
        missing_indicator,
    }
}

/// The LDA leaf of rows `ids`, or a constant leaf when LDA does not apply.
/// Search-time child leaves (`search`) classify with the node's raw class
/// probabilities as priors; fitted leaves use GUIDE's corrected priors.
fn leaf(ctx: &Context<'_>, ids: &[usize], search: bool) -> LeafModel {
    let mut counts = vec![0.0; ctx.classes.len()];
    let labels: Vec<usize> = ids.iter().map(|&i| ctx.y[i] as usize).collect();
    for &c in &labels {
        counts[c] += 1.0;
    }
    let mut priors = counts.clone();
    ctx.class_probabilities(&mut priors);
    if !search {
        ctx.correct_zero_posteriors(&mut priors);
    }
    fit(
        ctx.data,
        ids,
        &labels,
        ctx.classes.len(),
        &ctx.discriminant_columns,
        &priors,
        (ctx.root_samples, &ctx.class_priors),
    )
    .map_or_else(|| ctx.classification_leaf(counts), LeafModel::Discriminant)
}

/// GUIDE's cut sizes for a node of `n` rows and minimum node size `s`: every
/// left size when there are at most 102 positions, otherwise 102 evenly
/// indexed ones, in both cases without the end sizes `s` and `n - s`.
fn cut_sizes(n: usize, s: usize) -> Vec<usize> {
    if n < 2 * s + 2 {
        return vec![];
    }
    let m = n - 2 * s + 1;
    let mut sizes: Vec<usize> = if m <= 102 {
        (s..=n - s).collect()
    } else {
        (0..102).map(|j| s + j * (m - 1) / 101).collect()
    };
    sizes.dedup();
    sizes.retain(|&i| s < i && i < n - s);
    sizes
}

/// Grows an LDA tree on rows `ids`. A node splits only with at least
/// 2.5 times the minimum node size, below the depth limit, with two or more
/// classes and with training errors under its own leaf. The split is a cut
/// on the leaf's first discriminant coordinate whose two children, fitted as
/// search leaves, misclassify fewest training rows; ties go to the most even
/// split, then the smaller left child.
pub(crate) fn grow(ctx: &Context<'_>, ids: &[usize], depth: usize, next_id: &mut usize) -> Node {
    let model = leaf(ctx, ids, false);
    let risk = ctx.risk(&model, ids);
    let mut node = Node {
        id: *next_id,
        sample_count: ids.len(),
        weight_sum: ids.iter().map(|&i| ctx.weights[i]).sum(),
        risk: risk / ctx.root_weight,
        impurity: ctx.impurity(&model, ids),
        model,
        diagnostics: vec![],
        split: None,
        left: None,
        right: None,
    };
    *next_id += 1;
    let s = ctx.options.min_samples_leaf;
    let LeafModel::Discriminant(d) = &node.model else {
        return node;
    };
    if depth >= ctx.options.max_depth || 2 * ids.len() < 5 * s || risk == 0.0 {
        return node;
    }
    let mut order: Vec<(f64, usize)> = ids
        .iter()
        .map(|&i| (d.coordinates(&ctx.data.rows[i])[0], i))
        .collect();
    order.sort_by(|a, b| a.0.total_cmp(&b.0));
    let sorted: Vec<usize> = order.iter().map(|&(_, i)| i).collect();
    let sizes: Vec<usize> = cut_sizes(ids.len(), s)
        .into_iter()
        .filter(|&i| order[i - 1].0 < order[i].0)
        .collect();
    let errors = crate::tree::ordered_map(
        &sizes,
        ids.len() >= crate::tree::PARALLEL_MIN_ROWS,
        || (),
        |_, &i| {
            let (left, right) = sorted.split_at(i);
            ctx.risk(&leaf(ctx, left, true), left) + ctx.risk(&leaf(ctx, right, true), right)
        },
    );
    let n = ids.len();
    let Some((&best, _)) = sizes.iter().zip(&errors).min_by(|a, b| {
        a.1.total_cmp(b.1)
            .then((n.abs_diff(2 * a.0)).cmp(&n.abs_diff(2 * b.0)))
            .then(a.0.cmp(b.0))
    }) else {
        return node;
    };
    let (below, above) = (order[best - 1].0, order[best].0);
    let mid = below / 2.0 + above / 2.0;
    let split = Split::Discriminant {
        terms: d.terms.clone(),
        coefficients: d.coefficients[0].clone(),
        threshold: if mid >= above { below } else { mid },
    };
    let (left, right): (Vec<usize>, Vec<usize>) = ids
        .iter()
        .partition(|&&i| split.goes_left(&ctx.data.rows[i]));
    let l = grow(ctx, &left, depth + 1, next_id);
    let r = grow(ctx, &right, depth + 1, next_id);
    node.split = Some(split);
    node.left = Some(Box::new(l));
    node.right = Some(Box::new(r));
    node
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FeatureSchema, Role, Value};
    use serde_json::Value as Json;
    use std::path::Path;

    fn close(a: f64, b: f64, relative: f64, scale: f64) -> bool {
        (a - b).abs() <= relative * scale.max(b.abs())
    }

    /// Replays one frozen single-node native LDA capture.
    fn check_capture(dir: &Path) {
        let read = |name: &str| -> Json {
            serde_json::from_str(&std::fs::read_to_string(dir.join(name)).unwrap()).unwrap()
        };
        let case = read("case.json");
        let manifest = read("manifest.json");
        assert_eq!(manifest["completed"], true, "{dir:?}");
        let result = &manifest["result"];
        let names: Vec<String> = case["features"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| n.as_str().unwrap().to_string())
            .collect();
        let data = Dataset {
            schema: names
                .iter()
                .map(|n| FeatureSchema {
                    name: n.clone(),
                    feature_type: FeatureType::Numeric,
                    role: Role::Both,
                    flags: None,
                })
                .collect(),
            rows: case["rows"]
                .as_array()
                .unwrap()
                .iter()
                .map(|r| {
                    r.as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_f64().map_or(Value::Missing, Value::Number))
                        .collect()
                })
                .collect(),
            targets: vec![],
            weights: vec![],
            offsets: vec![],
        };
        let targets: Vec<&str> = case["targets"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t.as_str().unwrap())
            .collect();
        let mut classes: Vec<&str> = targets.clone();
        classes.sort();
        classes.dedup();
        let labels: Vec<usize> = targets
            .iter()
            .map(|t| classes.binary_search(t).unwrap())
            .collect();
        let ids: Vec<usize> = (0..labels.len()).collect();
        let priors: Vec<f64> = (0..classes.len())
            .map(|c| labels.iter().filter(|&&l| l == c).count() as f64 / labels.len() as f64)
            .collect();
        let columns = candidate_columns(&data, &ids);
        let model = fit(
            &data,
            &ids,
            &labels,
            classes.len(),
            &columns,
            &priors,
            (ids.len(), &priors),
        )
        .unwrap();
        let term_name = |t: &Term| {
            let base = &names[t.feature];
            if t.missing_indicator {
                format!("{base}.NA")
            } else {
                base.clone()
            }
        };

        // Selected variables, node means and coefficients up to sign.
        let native = result["coefficients"].as_array().unwrap();
        let mut ours: Vec<String> = model.terms.iter().map(term_name).collect();
        let mut theirs: Vec<String> = native
            .iter()
            .map(|c| c["variable"].as_str().unwrap().to_string())
            .collect();
        ours.sort();
        theirs.sort();
        assert_eq!(ours, theirs, "{dir:?}: selected variables");
        let mut signs = vec![];
        for (k, a) in model.coefficients.iter().enumerate() {
            let pairs: Vec<(f64, f64)> = model
                .terms
                .iter()
                .zip(a)
                .map(|(t, &ours)| {
                    let row = native
                        .iter()
                        .find(|c| c["variable"] == term_name(t).as_str())
                        .unwrap();
                    (ours, row["coefficients"][k].as_f64().unwrap())
                })
                .collect();
            let scale = pairs.iter().map(|p| p.1.abs()).fold(0.0, f64::max);
            let sign = if pairs.iter().map(|p| p.0 * p.1).sum::<f64>() < 0.0 {
                -1.0
            } else {
                1.0
            };
            for (ours, theirs) in pairs {
                assert!(
                    close(sign * ours, theirs, 1e-6, scale),
                    "{dir:?}: coordinate {k} coefficient {ours} vs {theirs}"
                );
            }
            signs.push(sign);
        }
        for t in model.terms.iter().filter(|t| !t.missing_indicator) {
            let row = native
                .iter()
                .find(|c| c["variable"] == term_name(t).as_str())
                .unwrap();
            assert!(close(t.impute, row["mean"].as_f64().unwrap(), 1e-6, 0.0));
        }

        // Leaf statistics on each coordinate.
        for (k, sign) in signs.iter().enumerate() {
            let stats = &result["leaf"][format!("coordinate{}", k + 1)];
            let spread = model.variances[k].sqrt();
            for (c, name) in classes.iter().enumerate() {
                let theirs = &stats[*name];
                let mean = model.class_means[c].as_ref().unwrap()[k] * sign;
                assert!(
                    close(mean, theirs["mean"].as_f64().unwrap(), 5e-6, spread),
                    "{dir:?}: class {name} mean on coordinate {k}"
                );
                assert!(close(
                    model.variances[k],
                    theirs["variance"].as_f64().unwrap(),
                    5e-6,
                    0.0
                ));
            }
        }

        // Every training row's coordinates and predicted class.
        for (i, row) in result["fitted"].as_array().unwrap().iter().enumerate() {
            let z = model.coordinates(&data.rows[i]);
            for (k, sign) in signs.iter().enumerate() {
                let theirs = row["coordinates"][k].as_f64().unwrap();
                assert!(
                    close(z[k] * sign, theirs, 5e-4, 0.0),
                    "{dir:?}: row {i} coordinate {k}: {} vs {theirs}",
                    z[k] * sign
                );
            }
            assert_eq!(
                classes[model.predict(&data.rows[i])],
                row["predicted"].as_str().unwrap(),
                "{dir:?}: row {i} prediction"
            );
        }
    }

    #[test]
    fn single_node_lda_reproduces_native_captures() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/discriminant");
        let mut checked = 0;
        for version in ["guide-46.2", "guide-46.3"] {
            let mut dirs: Vec<_> = std::fs::read_dir(root.join(version))
                .unwrap()
                .map(|e| e.unwrap().path())
                .filter(|p| {
                    p.file_name()
                        .unwrap()
                        .to_str()
                        .unwrap()
                        .starts_with("lda-leaf-")
                })
                .collect();
            dirs.sort();
            for dir in dirs {
                check_capture(&dir);
                checked += 1;
            }
        }
        assert_eq!(checked, 24);
    }

    #[test]
    fn selection_drops_collinear_columns() {
        let labels = [0, 0, 0, 1, 1, 1];
        let x = vec![1.0, 2.0, 3.0, 6.0, 7.0, 9.0];
        let design = vec![x.clone(), x.iter().map(|v| 2.0 * v + 1.0).collect()];
        assert_eq!(select(&design, &labels, 2), vec![0]);
    }

    #[test]
    fn coordinates_have_unit_total_scatter() {
        let labels = [0, 0, 0, 1, 1, 1, 2, 2, 2];
        let design = vec![
            vec![1.0, 2.0, 1.5, 4.0, 5.0, 4.2, 8.0, 7.5, 9.0],
            vec![3.0, 1.0, 2.0, 2.5, 3.5, 1.0, 0.5, 2.0, 1.0],
        ];
        for a in coordinates(&design, &labels, 3) {
            let z: Vec<f64> = (0..9)
                .map(|i| a[0] * design[0][i] + a[1] * design[1][i])
                .collect();
            let m = stats::mean(&z);
            let ss: f64 = z.iter().map(|v| (v - m) * (v - m)).sum();
            assert!((ss - 1.0).abs() < 1e-12);
        }
    }
}
