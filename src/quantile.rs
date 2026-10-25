//! Weighted quantile regression as a primal-dual linear program.
//!
//! Minimize w'(tau*u + (1-tau)*v), subject to X*b + u - v = y,
//! u,v >= 0. The dual is maximize y'a subject to X'a = 0 and
//! -(1-tau)*w <= a <= tau*w. Newton directions use pivoted QR,
//! avoiding an explicit normal-equation matrix or native solver.

use crate::stats::least_squares;

#[derive(Clone, Debug)]
pub struct QuantileFit {
    pub coefficients: Vec<f64>,
    /// Dual variables, in original row order; zero for zero-weight rows.
    pub dual: Vec<f64>,
    pub objective: f64,
    /// Absolute primal-dual objective difference, in original units.
    pub duality_gap: f64,
    pub converged: bool,
    pub iterations: usize,
}

struct Direction {
    beta: Vec<f64>,
    u: Vec<f64>,
    v: Vec<f64>,
    a: Vec<f64>,
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

fn step(values: &[f64], changes: &[f64]) -> f64 {
    values
        .iter()
        .zip(changes)
        .filter(|(_, d)| **d < 0.0)
        .map(|(v, d)| -v / d)
        .fold(1.0, f64::min)
}

/// Minimize weighted pinball loss at `probability` using a Mehrotra
/// predictor-corrector iteration (maximum 100 iterations).
///
/// Returns `None` for invalid input or numerical initialization failure.
/// A finite iterate can be returned with `converged == false`; callers must
/// inspect that flag. Nonunique solutions may differ in coefficients while
/// having the same optimal loss. No intercept is added automatically.
pub fn quantile_regression(
    design: &[Vec<f64>],
    response: &[f64],
    weights: &[f64],
    probability: f64,
) -> Option<QuantileFit> {
    let rows = design.len();
    let p = design.first()?.len();
    if p == 0
        || response.len() != rows
        || weights.len() != rows
        || !probability.is_finite()
        || probability <= 0.0
        || probability >= 1.0
        || design
            .iter()
            .any(|r| r.len() != p || r.iter().any(|x| !x.is_finite()))
        || response.iter().any(|y| !y.is_finite())
        || weights.iter().any(|w| !w.is_finite() || *w < 0.0)
    {
        return None;
    }
    let ids: Vec<usize> = (0..rows).filter(|&i| weights[i] > 0.0).collect();
    let n = ids.len();
    if n == 0 {
        return None;
    }
    let weight_scale = ids.iter().map(|&i| weights[i]).fold(0.0, f64::max);
    // An intercept permits exact translation of the response. Without this,
    // convergence is measured against the origin (e.g. 1e9) rather than the
    // variation being fitted. Never center a model without an intercept.
    let intercept = (0..p).find(|&j| {
        design[ids[0]][j] != 0.0 && ids.iter().all(|&i| design[i][j] == design[ids[0]][j])
    });
    let origin = if intercept.is_some() {
        response[ids[0]]
    } else {
        0.0
    };
    let y_scale = ids
        .iter()
        .map(|&i| (response[i] - origin).abs())
        .fold(0.0, f64::max);
    let y_scale = if y_scale == 0.0 { 1.0 } else { y_scale };
    let scales: Vec<f64> = (0..p)
        .map(|j| {
            let scale = ids.iter().map(|&i| design[i][j].abs()).fold(0.0, f64::max);
            if scale == 0.0 { 1.0 } else { scale }
        })
        .collect();
    let x: Vec<Vec<f64>> = ids
        .iter()
        .map(|&i| design[i].iter().zip(&scales).map(|(v, s)| v / s).collect())
        .collect();
    let y: Vec<f64> = ids
        .iter()
        .map(|&i| (response[i] - origin) / y_scale)
        .collect();
    let w: Vec<f64> = ids.iter().map(|&i| weights[i] / weight_scale).collect();
    if w.iter().any(|w| *w < 1e-300) {
        return None;
    }
    let mut beta = least_squares(&x, &y, &w)?;
    let mut u: Vec<f64> = (0..n)
        .map(|i| (y[i] - dot(&x[i], &beta)).max(0.0) + 1.0)
        .collect();
    let mut v: Vec<f64> = (0..n)
        .map(|i| (dot(&x[i], &beta) - y[i]).max(0.0) + 1.0)
        .collect();
    let mut a = vec![0.0; n];
    let mut converged = false;
    let mut iterations = 0;
    for iteration in 0..=100 {
        iterations = iteration;
        let lu: Vec<f64> = (0..n).map(|i| probability * w[i] - a[i]).collect();
        let lv: Vec<f64> = (0..n).map(|i| (1.0 - probability) * w[i] + a[i]).collect();
        let primal: f64 = (0..n)
            .map(|i| {
                let residual = y[i] - dot(&x[i], &beta);
                w[i] * if residual >= 0.0 {
                    probability * residual
                } else {
                    (probability - 1.0) * residual
                }
            })
            .sum();
        let dual = dot(&y, &a);
        let stationarity = (0..p)
            .map(|j| (0..n).map(|i| x[i][j] * a[i]).sum::<f64>().abs())
            .fold(0.0, f64::max);
        let feasibility = (0..n)
            .map(|i| (y[i] - dot(&x[i], &beta) - u[i] + v[i]).abs())
            .fold(0.0, f64::max);
        let mu = (dot(&u, &lu) + dot(&v, &lv)) / (2 * n) as f64;
        if (primal - dual).abs() <= 1e-9 * (1.0 + primal.abs())
            && stationarity <= 1e-9 * (1.0 + w.iter().sum::<f64>())
            && feasibility <= 1e-9
            && mu <= 1e-9 * (1.0 + primal.abs())
            && lu
                .iter()
                .chain(&lv)
                .chain(&u)
                .chain(&v)
                .all(|s| s.is_finite() && *s >= 0.0)
        {
            converged = true;
            break;
        }
        if iteration == 100
            || lu.iter().chain(&lv).any(|s| !s.is_finite() || *s <= 0.0)
            || !mu.is_finite()
        {
            break;
        }
        let direction = |ru: &[f64], rv: &[f64]| -> Option<Direction> {
            let diagonal: Vec<f64> = (0..n).map(|i| u[i] / lu[i] + v[i] / lv[i]).collect();
            let h: Vec<f64> = (0..n)
                .map(|i| y[i] - dot(&x[i], &beta) - u[i] + v[i] - ru[i] / lu[i] + rv[i] / lv[i])
                .collect();
            let iw: Vec<f64> = diagonal.iter().map(|d| 1.0 / d).collect();
            let db = least_squares(&x, &h, &iw)?;
            let da: Vec<f64> = (0..n)
                .map(|i| (h[i] - dot(&x[i], &db)) / diagonal[i])
                .collect();
            let du: Vec<f64> = (0..n).map(|i| (ru[i] + u[i] * da[i]) / lu[i]).collect();
            let dv: Vec<f64> = (0..n).map(|i| (rv[i] - v[i] * da[i]) / lv[i]).collect();
            if db
                .iter()
                .chain(&da)
                .chain(&du)
                .chain(&dv)
                .any(|v| !v.is_finite())
            {
                return None;
            }
            Some(Direction {
                beta: db,
                u: du,
                v: dv,
                a: da,
            })
        };
        let ru: Vec<f64> = (0..n).map(|i| -u[i] * lu[i]).collect();
        let rv: Vec<f64> = (0..n).map(|i| -v[i] * lv[i]).collect();
        let Some(affine) = direction(&ru, &rv) else {
            break;
        };
        let ap = step(&u, &affine.u).min(step(&v, &affine.v));
        let minus_a: Vec<f64> = affine.a.iter().map(|v| -v).collect();
        let ad = step(&lu, &minus_a).min(step(&lv, &affine.a));
        let mu_affine: f64 = (0..n)
            .map(|i| {
                (u[i] + ap * affine.u[i]) * (lu[i] - ad * affine.a[i])
                    + (v[i] + ap * affine.v[i]) * (lv[i] + ad * affine.a[i])
            })
            .sum::<f64>()
            / (2 * n) as f64;
        let sigma = (mu_affine / mu).clamp(0.0, 1.0).powi(3);
        let ru: Vec<f64> = (0..n)
            .map(|i| sigma * mu - u[i] * lu[i] + affine.u[i] * affine.a[i])
            .collect();
        let rv: Vec<f64> = (0..n)
            .map(|i| sigma * mu - v[i] * lv[i] - affine.v[i] * affine.a[i])
            .collect();
        let Some(d) = direction(&ru, &rv) else {
            break;
        };
        let ap = (0.995 * step(&u, &d.u).min(step(&v, &d.v))).min(1.0);
        let minus_a: Vec<f64> = d.a.iter().map(|v| -v).collect();
        let ad = (0.995 * step(&lu, &minus_a).min(step(&lv, &d.a))).min(1.0);
        for (b, db) in beta.iter_mut().zip(&d.beta) {
            *b += ap * db;
        }
        for i in 0..n {
            u[i] += ap * d.u[i];
            v[i] += ap * d.v[i];
            a[i] += ad * d.a[i];
        }
    }
    let mut coefficients: Vec<f64> = beta
        .iter()
        .zip(&scales)
        .map(|(b, s)| b * y_scale / s)
        .collect();
    if let Some(j) = intercept {
        coefficients[j] += origin / design[ids[0]][j];
    }
    let mut dual = vec![0.0; rows];
    for (k, &i) in ids.iter().enumerate() {
        dual[i] = a[k] * weight_scale;
    }
    let objective: f64 = ids
        .iter()
        .map(|&i| {
            let residual = response[i] - dot(&design[i], &coefficients);
            weights[i]
                * if residual >= 0.0 {
                    probability * residual
                } else {
                    (probability - 1.0) * residual
                }
        })
        .sum();
    let duality_gap = (objective - dot(response, &dual)).abs();
    if coefficients.iter().chain(&dual).any(|x| !x.is_finite())
        || !objective.is_finite()
        || !duality_gap.is_finite()
    {
        return None;
    }
    Some(QuantileFit {
        coefficients,
        dual,
        objective,
        duality_gap,
        converged,
        iterations,
    })
}
