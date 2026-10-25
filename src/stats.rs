//! Numerical primitives implemented without native libraries.

pub use crate::quantile::{QuantileFit, quantile_regression};

// Compare objectives in their own units. An absolute epsilon changes tree
// structure when responses or observation weights are rescaled.
pub(crate) fn less_or_close(a: f64, b: f64) -> bool {
    a <= b
        || (a.is_finite() && b.is_finite() && a - b <= 64.0 * f64::EPSILON * a.abs().max(b.abs()))
}
pub(crate) fn strictly_less(a: f64, b: f64) -> bool {
    a < b && !less_or_close(b, a)
}

pub fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len().max(1) as f64
}
pub fn variance(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let m = mean(values);
    values.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (values.len() - 1) as f64
}
/// Type-7 sample quantile (linear interpolation between order statistics).
pub fn quantile(values: &[f64], p: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut x = values.to_vec();
    x.sort_by(f64::total_cmp);
    sorted_quantile(&x, p)
}
/// `quantile` of values already sorted by `f64::total_cmp`.
pub fn sorted_quantile(x: &[f64], p: f64) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    let a = p.clamp(0.0, 1.0) * (x.len() - 1) as f64;
    let i = a.floor() as usize;
    x[i] + (a - i as f64) * (x[a.ceil() as usize] - x[i])
}
pub fn weighted_quantile(values: &[(f64, f64)], p: f64) -> f64 {
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.0.total_cmp(&b.0));
    let total: f64 = v.iter().map(|x| x.1).sum();
    let mut sum = 0.0;
    for &(x, w) in &v {
        sum += w;
        if sum >= p * total {
            return x;
        }
    }
    v.last().map(|x| x.0).unwrap_or(0.0)
}
pub fn log_gamma(z: f64) -> f64 {
    let c = [
        676.5203681218851,
        -1259.1392167224028,
        771.3234287776531,
        -176.6150291621406,
        12.507343278686905,
        -0.13857109526572012,
        9.984369578019572e-6,
        1.5056327351493116e-7,
    ];
    if z < 0.5 {
        return std::f64::consts::PI.ln()
            - (std::f64::consts::PI * z).sin().ln()
            - log_gamma(1.0 - z);
    }
    let z = z - 1.0;
    let mut x = 0.9999999999998099;
    for (i, c) in c.iter().enumerate() {
        x += c / (z + i as f64 + 1.0);
    }
    let t = z + 7.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (z + 0.5) * t.ln() - t + x.ln()
}
/// Regularized upper incomplete gamma, using a series or continued fraction.
pub fn gamma_q(a: f64, x: f64) -> f64 {
    log_gamma_q(a, x).exp()
}
/// Natural logarithm of the regularized upper incomplete gamma.
/// The upper-tail fraction is combined in log space, avoiding exp underflow.
pub fn log_gamma_q(a: f64, x: f64) -> f64 {
    if !a.is_finite() || a <= 0.0 || x.is_nan() {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }
    if !x.is_finite() {
        return f64::NEG_INFINITY;
    }
    let log_factor = -x + a * x.ln() - log_gamma(a);
    if x < a + 1.0 {
        let mut sum = 1.0 / a;
        let mut delta = sum;
        let mut ap = a;
        for _ in 0..10000 {
            ap += 1.0;
            delta *= x / ap;
            sum += delta;
            if delta.abs() < sum.abs() * 1e-14 {
                break;
            }
        }
        let log_p = (sum.ln() + log_factor).clamp(f64::NEG_INFINITY, 0.0);
        if log_p < -std::f64::consts::LN_2 {
            (-log_p.exp()).ln_1p()
        } else {
            (-log_p.exp_m1()).ln()
        }
    } else {
        let tiny = 1e-300;
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / tiny;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..10000 {
            let an = -(i as f64) * (i as f64 - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < tiny {
                d = tiny;
            }
            c = b + an / c;
            if c.abs() < tiny {
                c = tiny;
            }
            d = 1.0 / d;
            let delta = d * c;
            h *= delta;
            if (delta - 1.0).abs() < 1e-14 {
                break;
            }
        }
        (log_factor + h.ln()).clamp(f64::NEG_INFINITY, 0.0)
    }
}
pub fn chi_square_survival(x: f64, df: usize) -> f64 {
    chi_square_log_survival(x, df).exp()
}
/// Natural logarithm of the chi-square upper tail; remains finite after the
/// corresponding ordinary probability has underflowed to zero.
pub fn chi_square_log_survival(x: f64, df: usize) -> f64 {
    if df == 0 {
        0.0
    } else {
        log_gamma_q(df as f64 / 2.0, x / 2.0)
    }
}
/// Quantile specified by upper-tail probability; avoids subtraction near one.
pub fn chi_square_upper_quantile(p: f64, df: usize) -> f64 {
    if !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    chi_square_log_upper_quantile(p.ln(), df)
}
/// Quantile specified by log upper-tail probability, including tails too small
/// to represent as f64 probabilities. log_p = -infinity gives +infinity.
pub fn chi_square_log_upper_quantile(log_p: f64, df: usize) -> f64 {
    if log_p.is_nan() || log_p > 0.0 {
        return f64::NAN;
    }
    if log_p == 0.0 || df == 0 {
        return 0.0;
    }
    if log_p == f64::NEG_INFINITY {
        return f64::INFINITY;
    }
    let mut lo = 0.0;
    let mut hi = df.max(1) as f64;
    while chi_square_log_survival(hi, df) > log_p {
        if hi == f64::MAX {
            return f64::INFINITY;
        }
        hi = (hi * 2.0).min(f64::MAX);
    }
    // Ordinary quantiles converge in about 54 iterations. The larger bound
    // also handles representable quantiles near zero across the f64 exponent range.
    for _ in 0..1100 {
        let mid = lo / 2.0 + hi / 2.0;
        if mid == lo || mid == hi {
            return mid;
        }
        if chi_square_log_survival(mid, df) > log_p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo / 2.0 + hi / 2.0
}
pub fn pearson(table: &[Vec<f64>]) -> (f64, usize) {
    if table.is_empty() || table[0].is_empty() {
        return (0.0, 0);
    }
    let rows: Vec<f64> = table.iter().map(|r| r.iter().sum()).collect();
    let cols: Vec<f64> = (0..table[0].len())
        .map(|j| table.iter().map(|r| r[j]).sum())
        .collect();
    let total: f64 = rows.iter().sum();
    if total <= 0.0 {
        return (0.0, 0);
    }
    let nr = rows.iter().filter(|&&x| x > 0.0).count();
    let nc = cols.iter().filter(|&&x| x > 0.0).count();
    let mut stat = 0.0;
    for (i, row) in table.iter().enumerate() {
        for (j, &obs) in row.iter().enumerate() {
            let e = rows[i] * cols[j] / total;
            if e > 0.0 {
                stat += (obs - e).powi(2) / e;
            }
        }
    }
    (stat, nr.saturating_sub(1) * nc.saturating_sub(1))
}
pub fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Log of the regularized incomplete beta, retaining tiny lower tails.
pub fn beta_log_regularized(x: f64, a: f64, b: f64) -> f64 {
    if a <= 0.0 || b <= 0.0 || !a.is_finite() || !b.is_finite() || !(0.0..=1.0).contains(&x) {
        return f64::NAN;
    }
    beta_log_from_logs(x.ln(), (-x).ln_1p(), a, b)
}
/// Regularized incomplete beta, evaluated with a continued fraction.
pub fn beta_regularized(x: f64, a: f64, b: f64) -> f64 {
    beta_log_regularized(x, a, b).exp()
}
fn beta_fraction(x: f64, a: f64, b: f64) -> f64 {
    let tiny = 1e-300;
    let mut c = 1.0;
    let mut d = 1.0 - (a + b) * x / (a + 1.0);
    if d.abs() < tiny {
        d = tiny;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..10000 {
        let previous = h;
        let m = m as f64;
        let m2 = 2.0 * m;
        for aa in [
            m * (b - m) * x / ((a + m2 - 1.0) * (a + m2)),
            -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1.0)),
        ] {
            d = 1.0 + aa * d;
            if d.abs() < tiny {
                d = tiny;
            }
            c = 1.0 + aa / c;
            if c.abs() < tiny {
                c = tiny;
            }
            d = 1.0 / d;
            h *= d * c;
        }
        if (h - previous).abs() <= h.abs() * 1e-14 {
            break;
        }
    }
    h
}
fn log_complement(log_p: f64) -> f64 {
    if log_p < -std::f64::consts::LN_2 {
        (-log_p.exp()).ln_1p()
    } else {
        (-log_p.exp_m1()).ln()
    }
}
fn beta_log_from_logs(log_x: f64, log_one_minus_x: f64, a: f64, b: f64) -> f64 {
    if log_x == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    if log_one_minus_x == f64::NEG_INFINITY {
        return 0.0;
    }
    let log_factor =
        log_gamma(a + b) - log_gamma(a) - log_gamma(b) + a * log_x + b * log_one_minus_x;
    if log_x < ((a + 1.0) / (a + b + 2.0)).ln() {
        (log_factor + beta_fraction(log_x.exp(), a, b).ln() - a.ln()).min(0.0)
    } else {
        log_complement(
            (log_factor + beta_fraction(log_one_minus_x.exp(), b, a).ln() - b.ln()).min(0.0),
        )
    }
}
/// Natural logarithm of the F upper tail. Computing the beta argument in log
/// space also avoids overflow of numerator_df * f for large finite statistics.
pub fn f_log_survival(f: f64, numerator_df: usize, denominator_df: usize) -> f64 {
    if f.is_nan() {
        return f64::NAN;
    }
    if numerator_df == 0 || denominator_df == 0 || f <= 0.0 {
        return 0.0;
    }
    if f == f64::INFINITY {
        return f64::NEG_INFINITY;
    }
    f_log_survival_from_log(f.ln(), numerator_df, denominator_df)
}
pub(crate) fn f_log_survival_from_log(
    log_f: f64,
    numerator_df: usize,
    denominator_df: usize,
) -> f64 {
    let log_ratio = (numerator_df as f64).ln() + log_f - (denominator_df as f64).ln();
    let softplus = |x: f64| x.max(0.0) + (-x.abs()).exp().ln_1p();
    beta_log_from_logs(
        -softplus(log_ratio),
        -softplus(-log_ratio),
        denominator_df as f64 / 2.0,
        numerator_df as f64 / 2.0,
    )
}
pub fn f_survival(f: f64, numerator_df: usize, denominator_df: usize) -> f64 {
    f_log_survival(f, numerator_df, denominator_df).exp()
}

/// Weighted least squares using reorthogonalized, column-pivoted QR.
/// Rank-deficient columns receive zero coefficients instead of solving normal equations.
/// `out[c] = sum over i of weight(i, x) * x` with `x = columns[c][i]`, for
/// columns of equal length. Each sum accumulates in index order from -0.0,
/// exactly as `Iterator::sum`; four columns at a time only overlaps their
/// floating-point latency.
fn grouped_dots(columns: &[Vec<f64>], out: &mut [f64], weight: impl Fn(usize, f64) -> f64) {
    fn group<const L: usize>(
        columns: [&[f64]; L],
        out: &mut [f64],
        weight: &impl Fn(usize, f64) -> f64,
    ) {
        let n = columns[0].len();
        let columns = columns.map(|c| &c[..n]);
        let mut sums = [-0.0; L];
        for i in 0..n {
            for (sum, column) in sums.iter_mut().zip(&columns) {
                let x = column[i];
                *sum += weight(i, x) * x;
            }
        }
        out.copy_from_slice(&sums);
    }
    for (columns, out) in columns.chunks(4).zip(out.chunks_mut(4)) {
        match columns {
            [a] => group([a.as_slice()], out, &weight),
            [a, b] => group([a.as_slice(), b], out, &weight),
            [a, b, c] => group([a.as_slice(), b, c], out, &weight),
            [a, b, c, d] => group([a.as_slice(), b, c, d], out, &weight),
            _ => unreachable!("chunks of at most four"),
        }
    }
}
pub fn least_squares(x: &[Vec<f64>], y: &[f64], weights: &[f64]) -> Option<Vec<f64>> {
    least_squares_ranked(x, y, weights).map(|(beta, _)| beta)
}
pub(crate) fn least_squares_ranked(
    x: &[Vec<f64>],
    y: &[f64],
    weights: &[f64],
) -> Option<(Vec<f64>, usize)> {
    let n = x.len();
    if n == 0 || x[0].is_empty() {
        return None;
    }
    let p = x[0].len();
    if y.len() != n
        || weights.len() != n
        || x.iter()
            .any(|r| r.len() != p || r.iter().any(|v| !v.is_finite()))
        || y.iter().any(|v| !v.is_finite())
        || weights.iter().any(|w| !w.is_finite() || *w < 0.0)
    {
        return None;
    }
    let mut a: Vec<Vec<f64>> = (0..p)
        .map(|j| (0..n).map(|i| x[i][j] * weights[i].sqrt()).collect())
        .collect();
    let b: Vec<f64> = (0..n).map(|i| y[i] * weights[i].sqrt()).collect();
    let mut order: Vec<usize> = (0..p).collect();
    let mut r = vec![vec![0.0; p]; p];
    let mut qty = vec![0.0; p];
    let mut rank = 0;
    let norm = |v: &Vec<f64>| v.iter().map(|z| z * z).sum::<f64>().sqrt();
    let scale = a.iter().map(&norm).fold(0.0, f64::max);
    if !scale.is_finite() || scale == 0.0 {
        return None;
    }
    let mut norms = vec![0.0; p];
    let mut dots = vec![0.0; p];
    for k in 0..p.min(n) {
        grouped_dots(&a[k..], &mut norms[k..], |_, x| x);
        for v in &mut norms[k..] {
            *v = v.sqrt();
        }
        // k < p, so the candidate pivot range is nonempty.
        let pivot = (k..p)
            .max_by(|&i, &j| norms[i].total_cmp(&norms[j]))
            .unwrap();
        let nk = norms[pivot];
        a.swap(k, pivot);
        order.swap(k, pivot);
        for row in r.iter_mut().take(k) {
            row.swap(k, pivot);
        }
        if nk < scale * 1e-10 {
            break;
        }
        r[k][k] = nk;
        for v in &mut a[k] {
            *v /= nk;
        }
        qty[k] = a[k].iter().zip(&b).map(|(u, v)| u * v).sum();
        let (before, after) = a.split_at_mut(k + 1);
        let basis = &before[k];
        let dots = &mut dots[..after.len()];
        // Two Gram-Schmidt passes against the new basis column. Each column's
        // steps are independent of the others', so all columns take each
        // pass together.
        for _ in 0..2 {
            grouped_dots(after, dots, |i, _| basis[i]);
            for ((column, &dot), r) in after.iter_mut().zip(&*dots).zip(&mut r[k][k + 1..]) {
                *r += dot;
                for (target, basis) in column.iter_mut().zip(basis) {
                    *target -= dot * basis;
                }
            }
        }
        rank += 1;
    }
    let mut beta = vec![0.0; p];
    for k in (0..rank).rev() {
        beta[k] = (qty[k] - (k + 1..rank).map(|j| r[k][j] * beta[j]).sum::<f64>()) / r[k][k];
    }
    let mut answer = vec![0.0; p];
    for k in 0..p {
        answer[order[k]] = beta[k];
    }
    answer
        .iter()
        .all(|v| v.is_finite())
        .then_some((answer, rank))
}

/// Eigenvalues (descending) and unit eigenvectors (`vectors[k]` pairs with
/// `values[k]`) of a small symmetric matrix, by cyclic Jacobi rotations.
pub fn symmetric_eigen(matrix: &[Vec<f64>]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = matrix.len();
    let mut a = matrix.to_vec();
    let mut v: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| f64::from(u8::from(i == j))).collect())
        .collect();
    for _ in 0..100 {
        let off: f64 = (0..n)
            .flat_map(|i| (0..n).filter(move |&j| j != i).map(move |j| (i, j)))
            .map(|(i, j)| a[i][j] * a[i][j])
            .sum();
        let scale: f64 = (0..n).map(|i| a[i][i] * a[i][i]).sum();
        if off <= f64::EPSILON * f64::EPSILON * scale.max(f64::MIN_POSITIVE) {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                if a[p][q] == 0.0 {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let sign = if theta >= 0.0 { 1.0 } else { -1.0 };
                let t = sign / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for row in &mut a {
                    let (akp, akq) = (row[p], row[q]);
                    row[p] = c * akp - s * akq;
                    row[q] = s * akp + c * akq;
                }
                let (head, tail) = a.split_at_mut(q);
                for (apk, aqk) in head[p].iter_mut().zip(tail[0].iter_mut()) {
                    let (x, y) = (*apk, *aqk);
                    *apk = c * x - s * y;
                    *aqk = s * x + c * y;
                }
                for row in &mut v {
                    let (vp, vq) = (row[p], row[q]);
                    row[p] = c * vp - s * vq;
                    row[q] = s * vp + c * vq;
                }
            }
        }
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| a[j][j].total_cmp(&a[i][i]));
    (
        order.iter().map(|&k| a[k][k]).collect(),
        order
            .iter()
            .map(|&k| (0..n).map(|i| v[i][k]).collect())
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The column-at-a-time pivoted Gram-Schmidt that `least_squares_ranked`
    /// must reproduce bit for bit.
    fn reference(x: &[Vec<f64>], y: &[f64], weights: &[f64]) -> (Vec<f64>, usize) {
        let (n, p) = (x.len(), x[0].len());
        let mut a: Vec<Vec<f64>> = (0..p)
            .map(|j| (0..n).map(|i| x[i][j] * weights[i].sqrt()).collect())
            .collect();
        let b: Vec<f64> = (0..n).map(|i| y[i] * weights[i].sqrt()).collect();
        let mut order: Vec<usize> = (0..p).collect();
        let mut r = vec![vec![0.0; p]; p];
        let mut qty = vec![0.0; p];
        let mut rank = 0;
        let norm = |v: &Vec<f64>| v.iter().map(|z| z * z).sum::<f64>().sqrt();
        let scale = a.iter().map(&norm).fold(0.0, f64::max);
        for k in 0..p.min(n) {
            let pivot = (k..p)
                .max_by(|&i, &j| norm(&a[i]).total_cmp(&norm(&a[j])))
                .unwrap();
            a.swap(k, pivot);
            order.swap(k, pivot);
            for row in r.iter_mut().take(k) {
                row.swap(k, pivot);
            }
            let nk = norm(&a[k]);
            if nk < scale * 1e-10 {
                break;
            }
            r[k][k] = nk;
            for v in &mut a[k] {
                *v /= nk;
            }
            qty[k] = a[k].iter().zip(&b).map(|(u, v)| u * v).sum();
            for j in k + 1..p {
                for _ in 0..2 {
                    let dot: f64 = a[k].iter().zip(&a[j]).map(|(u, v)| u * v).sum();
                    r[k][j] += dot;
                    let (before, after) = a.split_at_mut(j);
                    for (target, basis) in after[0].iter_mut().zip(&before[k]) {
                        *target -= dot * basis;
                    }
                }
            }
            rank += 1;
        }
        let mut beta = vec![0.0; p];
        for k in (0..rank).rev() {
            beta[k] = (qty[k] - (k + 1..rank).map(|j| r[k][j] * beta[j]).sum::<f64>()) / r[k][k];
        }
        let mut answer = vec![0.0; p];
        for k in 0..p {
            answer[order[k]] = beta[k];
        }
        (answer, rank)
    }

    #[test]
    fn grouped_least_squares_matches_column_at_a_time() {
        let value = |i: usize, j: usize| ((i * 7919 + j * 104_729) % 1009) as f64 / 97.0 - 5.0;
        for p in 1..=9 {
            let x: Vec<Vec<f64>> = (0..60)
                .map(|i| {
                    (0..p)
                        // Column 3 repeats column 1, so pivoting and rank deficiency run.
                        .map(|j| {
                            if j == 3 {
                                value(i, 1) * 2.0
                            } else {
                                value(i, j)
                            }
                        })
                        .collect()
                })
                .collect();
            let y: Vec<f64> = (0..60).map(|i| value(i, 11) + x[i][0]).collect();
            let w: Vec<f64> = (0..60).map(|i| 0.5 + (i % 4) as f64).collect();
            let (beta, rank) = least_squares_ranked(&x, &y, &w).unwrap();
            let (expected, expected_rank) = reference(&x, &y, &w);
            assert_eq!(rank, expected_rank);
            assert_eq!(rank, if p > 3 { p - 1 } else { p });
            let bits = |v: &[f64]| v.iter().map(|b| b.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&beta), bits(&expected), "p = {p}");
        }
    }
}
