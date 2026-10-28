//! GUIDE's linear discriminant (LDA) node models for classification.
//!
//! A node selects its own columns, projects them on its leading discriminant
//! coordinates and classifies with class-specific Gaussian densities on those
//! coordinates. The rules follow black-box probes of GUIDE 46.3; see
//! `docs/discriminant-notes.md`.
use crate::node::Term;
use crate::{Dataset, FeatureType, stats};

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

/// A fitted LDA leaf. Classes absent from the node have no mean.
#[derive(Clone, Debug)]
pub(crate) struct Discriminant {
    /// The selected columns as raw (uncentered, unscaled) terms.
    pub terms: Vec<Term>,
    /// `coefficients[k][t]` is the weight of term `t` in coordinate `k`.
    pub coefficients: Vec<Vec<f64>>,
    /// Class means on each coordinate.
    pub class_means: Vec<Option<Vec<f64>>>,
    /// Pooled within-class variance of each coordinate.
    pub variances: Vec<f64>,
    /// Class priors used for classification.
    pub priors: Vec<f64>,
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
    /// Posterior class probabilities of a row; absent classes get zero.
    pub(crate) fn posteriors(&self, row: &[crate::Value]) -> Vec<f64> {
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
    /// The class with the largest posterior, the first on ties.
    pub(crate) fn predict(&self, row: &[crate::Value]) -> usize {
        let p = self.posteriors(row);
        (0..p.len()).fold(0, |best, k| if p[k] > p[best] { k } else { best })
    }
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
/// `classes`). `priors` are the class priors used for classification. Returns
/// None when fewer than two classes are present or no column can be used.
pub(crate) fn fit(
    data: &Dataset,
    ids: &[usize],
    labels: &[usize],
    classes: usize,
    columns: &[Column],
    priors: &[f64],
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
    let kept = select(&design, labels, classes);
    if kept.is_empty() {
        return None;
    }
    let design: Vec<Vec<f64>> = kept.iter().map(|&j| design[j].clone()).collect();
    let dimensions = if present.len() == 2 {
        1
    } else {
        kept.len().min(2)
    };
    let coefficients: Vec<Vec<f64>> = coordinates(&design, labels, classes)
        .into_iter()
        .take(dimensions)
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
        let model = fit(&data, &ids, &labels, classes.len(), &columns, &priors).unwrap();
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
