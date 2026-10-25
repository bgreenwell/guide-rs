use guide_rs::stats::*;

fn close(a: f64, b: f64, tolerance: f64) {
    assert!((a - b).abs() < tolerance, "{a} != {b}");
}
#[test]
fn distribution_reference_values() {
    close(log_gamma(5.0), 24.0_f64.ln(), 1e-12);
    close(chi_square_survival(3.841458820694124, 1), 0.05, 1e-12);
    close(chi_square_survival(9.487729036781154, 4), 0.05, 1e-12);
    close(chi_square_survival(20.0, 2), (-10.0_f64).exp(), 1e-14);
    close(chi_square_upper_quantile(0.05, 1), 3.841458820694124, 1e-10);
    for df in [1, 2, 5, 30, 200] {
        for p in [1e-12, 0.001, 0.1, 0.5, 0.99] {
            let q = chi_square_upper_quantile(p, df);
            close(chi_square_survival(q, df), p, 1e-11);
        }
    }
}

#[test]
fn upper_quantiles_preserve_zero_and_subnormal_tail_semantics() {
    assert_eq!(chi_square_upper_quantile(0.0, 1), f64::INFINITY);
    assert!(chi_square_upper_quantile(f64::from_bits(1), 1) > chi_square_upper_quantile(1e-300, 1));
}

#[test]
fn log_chi_square_tails_and_quantiles_have_no_probability_floor() {
    // R stats::pchisq(6000, df, lower.tail=FALSE, log.p=TRUE).
    for (df, expected) in [
        (1, -3_004.575_715_324_029),
        (2, -3000.0),
        (3, -2995.8758673535194),
        (999, -1610.6534221874708),
    ] {
        close(chi_square_log_survival(6000.0, df), expected, 1e-9);
    }
    // Independent exponential distribution identity for two degrees of freedom.
    for x in [1e-16, 1.0, 20.0, 2000.0, 1e6] {
        close(chi_square_log_survival(x, 2), -x / 2.0, 1e-10 * (1.0 + x));
    }
    close(
        chi_square_log_upper_quantile(-1e6, 1),
        1999985.0397660367,
        1e-7,
    );
    for df in [1, 2, 5, 30, 200, 2000] {
        for log_p in [-1e-12, -0.01, -0.69, -10.0, -100.0, -1000.0, -1e6] {
            let q = chi_square_log_upper_quantile(log_p, df);
            assert!(q.is_finite() && q > 0.0);
            close(
                chi_square_log_survival(q, df),
                log_p,
                5e-10 * (1.0 + log_p.abs()),
            );
        }
    }
    assert_eq!(chi_square_log_upper_quantile(0.0, 1), 0.0);
    assert_eq!(
        chi_square_log_upper_quantile(f64::NEG_INFINITY, 1),
        f64::INFINITY
    );
    assert!(chi_square_log_upper_quantile(0.1, 1).is_nan());
    assert!(chi_square_upper_quantile(-0.1, 1).is_nan());
    assert!(log_gamma_q(0.0, 1.0).is_nan());
}

#[test]
fn beta_and_f_distribution_reference_values() {
    close(beta_regularized(0.5, 2.0, 2.0), 0.5, 1e-12);
    close(beta_regularized(0.3, 1.0, 1.0), 0.3, 1e-12);
    close(beta_regularized(0.3, 2.0, 1.0), 0.09, 1e-12);
    close(f_survival(1.0, 1, 1), 0.5, 1e-12);
    // R qf(0.95, 5, 10).
    close(f_survival(3.3258345304130112, 5, 10), 0.05, 1e-10);
}
#[test]
fn f_log_tails_preserve_extreme_probabilities_and_avoid_product_overflow() {
    // F(2,2) survival is exactly 1/(1+f), including extreme finite f.
    for f in [1e-300, 1e-12, 1.0, 1e10, 1e100, f64::MAX] {
        let expected = -f.ln_1p();
        close(
            f_log_survival(f, 2, 2),
            expected,
            1e-12 * (1.0 + expected.abs()),
        );
    }
    assert!(f_log_survival(3198.0 / 1.2101, 1, 3198).is_finite());
    assert!(
        f_log_survival(3198.0 * 1.21 / 1.0001, 1, 3198) < f_log_survival(3198.0 / 1.2101, 1, 3198)
    );
    assert_eq!(f_survival(3198.0 / 1.2101, 1, 3198), 0.0);
    assert_eq!(f_log_survival(0.0, 1, 10), 0.0);
    assert_eq!(f_log_survival(f64::INFINITY, 1, 10), f64::NEG_INFINITY);
    assert!(f_log_survival(f64::NAN, 1, 10).is_nan());
    assert_eq!(beta_log_regularized(0.0, 1.0, 1.0), f64::NEG_INFINITY);
    assert_eq!(beta_log_regularized(1.0, 1.0, 1.0), 0.0);
    assert!(beta_log_regularized(-1.0, 1.0, 1.0).is_nan());
}
#[test]
fn contingency_table_drops_empty_margins() {
    let (stat, df) = pearson(&[
        vec![10.0, 20.0, 0.0],
        vec![20.0, 10.0, 0.0],
        vec![0.0, 0.0, 0.0],
    ]);
    close(stat, 20.0 / 3.0, 1e-12);
    assert_eq!(df, 1);
    assert_eq!(pearson(&[vec![3.0, 3.0]]).1, 0);
}
#[test]
fn weighted_quantiles_and_sample_quantiles() {
    close(quantile(&[1.0, 2.0, 3.0, 4.0], 0.25), 1.75, 1e-12);
    close(
        weighted_quantile(&[(1.0, 1.0), (3.0, 4.0), (9.0, 1.0)], 0.5),
        3.0,
        1e-12,
    );
}
#[test]
fn pivoted_qr_handles_collinearity_and_weights() {
    let x: Vec<Vec<f64>> = (0..10)
        .map(|i| vec![1.0, i as f64, 2.0 * i as f64])
        .collect();
    let y: Vec<f64> = (0..10).map(|i| 3.0 + 4.0 * i as f64).collect();
    let w: Vec<f64> = (0..10).map(|i| (i + 1) as f64).collect();
    let b = least_squares(&x, &y, &w).unwrap();
    for (row, y) in x.iter().zip(y) {
        close(row.iter().zip(&b).map(|(x, b)| x * b).sum(), y, 1e-9);
    }
}

#[test]
fn quantile_optimizer_matches_exhaustive_line_vertices() {
    let x: Vec<Vec<f64>> = (0..13).map(|i| vec![1.0, i as f64 - 6.0]).collect();
    let y: Vec<f64> = (0..13)
        .map(|i| 3.0 + 0.4 * i as f64 + ((i * 7) % 11) as f64 - 5.0)
        .collect();
    let w: Vec<f64> = (0..13).map(|i| (1 + i % 3) as f64).collect();
    for tau in [0.01, 0.1, 0.25, 0.5, 0.9, 0.99] {
        let objective = |b: &[f64]| {
            x.iter()
                .zip(&y)
                .zip(&w)
                .map(|((x, y), w)| {
                    let r = y - b[0] - b[1] * x[1];
                    w * if r >= 0.0 { tau * r } else { (tau - 1.0) * r }
                })
                .sum::<f64>()
        };
        let mut optimum = f64::INFINITY;
        for i in 0..x.len() {
            for j in i + 1..x.len() {
                let slope = (y[j] - y[i]) / (x[j][1] - x[i][1]);
                optimum = optimum.min(objective(&[y[i] - slope * x[i][1], slope]));
            }
        }
        let fit = quantile_regression(&x, &y, &w, tau).unwrap();
        assert!(fit.converged, "tau={tau}: {fit:?}");
        close(fit.objective, optimum, 1e-6);
        assert!(fit.duality_gap < 1e-6);
        for j in 0..2 {
            close(
                x.iter().zip(&fit.dual).map(|(x, a)| x[j] * a).sum(),
                0.0,
                1e-7,
            );
        }
        for (a, w) in fit.dual.iter().zip(&w) {
            assert!(*a >= -(1.0 - tau) * w - 1e-10 && *a <= tau * w + 1e-10);
        }
        let scaled_x: Vec<Vec<f64>> = x.iter().map(|r| vec![r[0], r[1] * 1e6]).collect();
        let scaled_y: Vec<f64> = y.iter().map(|v| v * 1e3).collect();
        let scaled_w: Vec<f64> = w.iter().map(|v| v * 0.25).collect();
        let scaled = quantile_regression(&scaled_x, &scaled_y, &scaled_w, tau).unwrap();
        assert!(scaled.converged, "{scaled:?}");
        close(scaled.objective / (1e3 * 0.25), optimum, 1e-6);
    }
}

#[test]
fn quantile_optimizer_handles_degeneracy_weights_and_scaling() {
    let x: Vec<Vec<f64>> = (0..21)
        .map(|i| vec![1.0, i as f64, 2.0 * i as f64])
        .collect();
    let y: Vec<f64> = (0..21)
        .map(|i| if i == 20 { 1e5 } else { 2.0 + 3.0 * i as f64 })
        .collect();
    let w: Vec<f64> = (0..21).map(|i| if i == 20 { 0.0 } else { 1.0 }).collect();
    let fit = quantile_regression(&x, &y, &w, 0.5).unwrap();
    assert!(fit.converged, "{fit:?}");
    close(fit.objective, 0.0, 1e-6);
    assert_eq!(fit.dual[20], 0.0);
    let intercept = quantile_regression(
        &vec![vec![1.0]; 5],
        &[1.0, 3.0, 7.0, 9.0, 11.0],
        &[1.0, 5.0, 1.0, 1.0, 1.0],
        0.5,
    )
    .unwrap();
    assert!(intercept.converged);
    close(intercept.coefficients[0], 3.0, 1e-7);
    assert!(quantile_regression(&x, &y, &w, 0.0).is_none());
    assert!(quantile_regression(&x, &y, &[], 0.5).is_none());
    assert!(quantile_regression(&x, &y, &[0.0; 21], 0.5).is_none());
}

#[test]
fn symmetric_eigen_recovers_known_spectrum() {
    // Eigenvalues 6, 3 and 1 with orthonormal eigenvectors.
    let m: Vec<Vec<f64>> = [[22.0, 4.0, 10.0], [4.0, 22.0, 10.0], [10.0, 10.0, 16.0]]
        .iter()
        .map(|row| row.iter().map(|x| x / 6.0).collect())
        .collect();
    let (values, vectors) = guide_rs::stats::symmetric_eigen(&m);
    for (v, e) in values.iter().zip([6.0, 3.0, 1.0]) {
        assert!((v - e).abs() < 1e-12, "{values:?}");
    }
    for (value, vector) in values.iter().zip(&vectors) {
        for (i, row) in m.iter().enumerate() {
            let mv: f64 = row.iter().zip(vector).map(|(a, b)| a * b).sum();
            assert!((mv - value * vector[i]).abs() < 1e-12);
        }
        let norm: f64 = vector.iter().map(|x| x * x).sum();
        assert!((norm - 1.0).abs() < 1e-12);
    }
}
