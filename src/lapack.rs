//! The reference LAPACK 3.12 path GUIDE 46 takes for discriminant
//! coordinates: `dgesvd` (JOBU = 'N', JOBVT = 'A') of the total scatter and
//! `dsygvx` (all eigenpairs, lower triangle, positive ABSTOL) in its singular
//! axes. Eigenvector signs are fixed by these algorithms, and GUIDE's cut grid
//! and tie-breaks depend on them, so the routines are ported operation for
//! operation rather than replaced by an equivalent decomposition. Only the
//! unblocked code paths that LAPACK selects for small matrices are included.
//!
//! Matrices are square, column-major, and indexed from 1 as in the Fortran.
//! Loops index arrays the way the reference does, so that each statement can
//! be checked against its Fortran line.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

/// `dlamch('E')`, the unit roundoff.
const EPS: f64 = f64::EPSILON / 2.0;
/// `dlamch('P')`, the precision.
const ULP: f64 = f64::EPSILON;
/// `dlamch('S')`, the safe minimum.
const SAFMIN: f64 = f64::MIN_POSITIVE;

/// A square column-major matrix with 1-based indexing.
#[derive(Clone)]
struct Mat {
    n: usize,
    v: Vec<f64>,
}

impl Mat {
    fn from_rows(rows: &[Vec<f64>]) -> Self {
        let n = rows.len();
        let mut v = vec![0.0; n * n];
        for (i, row) in rows.iter().enumerate() {
            for (j, &x) in row.iter().enumerate() {
                v[i + j * n] = x;
            }
        }
        Self { n, v }
    }

    fn at(&self, i: usize, j: usize) -> f64 {
        self.v[(i - 1) + (j - 1) * self.n]
    }

    fn set(&mut self, i: usize, j: usize, x: f64) {
        self.v[(i - 1) + (j - 1) * self.n] = x;
    }
}

fn sign(a: f64, b: f64) -> f64 {
    if b >= 0.0 { a.abs() } else { -a.abs() }
}

/// `dnrm2` for vectors of moderate magnitude (no scaling branch is reached).
fn nrm2(x: &[f64]) -> f64 {
    x.iter().map(|v| v * v).sum::<f64>().sqrt()
}

fn lapy2(x: f64, y: f64) -> f64 {
    let (w, z) = (x.abs().max(y.abs()), x.abs().min(y.abs()));
    if z == 0.0 {
        w
    } else {
        w * (1.0 + (z / w).powi(2)).sqrt()
    }
}

/// `dlartg` (3.10 and later): plane rotation with `r` carrying the sign of `f`.
fn lartg(f: f64, g: f64) -> (f64, f64, f64) {
    let safmax = 1.0 / SAFMIN;
    let rtmin = SAFMIN.sqrt();
    let rtmax = (safmax / 2.0).sqrt();
    let (f1, g1) = (f.abs(), g.abs());
    if g == 0.0 {
        (1.0, 0.0, f)
    } else if f == 0.0 {
        (0.0, sign(1.0, g), g1)
    } else if f1 > rtmin && f1 < rtmax && g1 > rtmin && g1 < rtmax {
        let d = (f * f + g * g).sqrt();
        let r = sign(d, f);
        (f1 / d, g / r, r)
    } else {
        let u = safmax.min(SAFMIN.max(f1).max(g1));
        let (fs, gs) = (f / u, g / u);
        let d = (fs * fs + gs * gs).sqrt();
        let r = sign(d, f);
        (fs.abs() / d, gs / r, r * u)
    }
}

/// `dlarfg` on `alpha` and the vector at `x` (positions into a buffer).
/// Returns tau; `alpha` becomes beta and `x` the reflector's tail.
fn larfg(buf: &mut [f64], alpha_at: usize, x_at: &[usize]) -> f64 {
    if x_at.is_empty() {
        return 0.0;
    }
    let xs: Vec<f64> = x_at.iter().map(|&k| buf[k]).collect();
    let mut xnorm = nrm2(&xs);
    if xnorm == 0.0 {
        return 0.0;
    }
    let mut alpha = buf[alpha_at];
    let mut beta = -sign(lapy2(alpha, xnorm), alpha);
    let safmin = SAFMIN / EPS;
    let mut knt = 0;
    if beta.abs() < safmin {
        let rsafmn = 1.0 / safmin;
        loop {
            knt += 1;
            for &k in x_at {
                buf[k] *= rsafmn;
            }
            beta *= rsafmn;
            alpha *= rsafmn;
            if beta.abs() >= safmin || knt >= 20 {
                break;
            }
        }
        let xs: Vec<f64> = x_at.iter().map(|&k| buf[k]).collect();
        xnorm = nrm2(&xs);
        beta = -sign(lapy2(alpha, xnorm), alpha);
    }
    let tau = (beta - alpha) / beta;
    let scale = 1.0 / (alpha - beta);
    for &k in x_at {
        buf[k] *= scale;
    }
    for _ in 0..knt {
        beta *= safmin;
    }
    buf[alpha_at] = beta;
    tau
}

/// `dlarf` applying `H = I - tau v v'` to rows `r0..r0+m` and columns
/// `c0..c0+n` of `c`, from the left or the right. `v` holds the vector.
fn larf(left: bool, v: &[f64], tau: f64, c: &mut Mat, r0: usize, c0: usize, m: usize, n: usize) {
    if tau == 0.0 {
        return;
    }
    // Trailing zeros of v and zero columns (rows) of C do not change the
    // result, so the reference's scans for them are not needed.
    if left {
        for j in 0..n {
            let mut w = 0.0;
            for i in 0..m {
                w += c.at(r0 + i, c0 + j) * v[i];
            }
            if w != 0.0 {
                let t = -tau * w;
                for i in 0..m {
                    let x = c.at(r0 + i, c0 + j) + v[i] * t;
                    c.set(r0 + i, c0 + j, x);
                }
            }
        }
    } else {
        let mut w = vec![0.0; m];
        for j in 0..n {
            let t = v[j];
            if t != 0.0 {
                for i in 0..m {
                    w[i] += t * c.at(r0 + i, c0 + j);
                }
            }
        }
        for j in 0..n {
            if v[j] != 0.0 {
                let t = -tau * v[j];
                for i in 0..m {
                    let x = c.at(r0 + i, c0 + j) + w[i] * t;
                    c.set(r0 + i, c0 + j, x);
                }
            }
        }
    }
}

/// Positions in `a.v` of row `i`, columns `j0..=j1`, and column `j`, rows
/// `i0..=i1`.
fn row_at(a: &Mat, i: usize, j0: usize, j1: usize) -> Vec<usize> {
    (j0..=j1).map(|j| (i - 1) + (j - 1) * a.n).collect()
}

fn col_at(a: &Mat, j: usize, i0: usize, i1: usize) -> Vec<usize> {
    (i0..=i1).map(|i| (i - 1) + (j - 1) * a.n).collect()
}

fn pos(a: &Mat, i: usize, j: usize) -> usize {
    (i - 1) + (j - 1) * a.n
}

/// Right singular vectors of a square matrix, as `dgesvd('N', 'A')` returns
/// them: singular values in decreasing order and the rows of `V'`.
pub(crate) fn right_singular_vectors(matrix: &[Vec<f64>]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let mut a = Mat::from_rows(matrix);
    let n = a.n;
    // dgebd2 (m = n): A = Q B P' with B upper bidiagonal.
    let mut d = vec![0.0; n + 1];
    let mut e = vec![0.0; n + 1];
    let mut taup = vec![0.0; n + 1];
    for i in 1..=n {
        let alpha = pos(&a, i, i);
        let x = if i < n {
            col_at(&a, i, i + 1, n)
        } else {
            vec![]
        };
        let tauq = larfg(&mut a.v, alpha, &x);
        d[i] = a.at(i, i);
        a.set(i, i, 1.0);
        if i < n {
            let v: Vec<f64> = (i..=n).map(|r| a.at(r, i)).collect();
            larf(true, &v, tauq, &mut a, i, i + 1, n - i + 1, n - i);
        }
        a.set(i, i, d[i]);
        if i < n {
            let alpha = pos(&a, i, i + 1);
            let x = if i + 1 < n {
                row_at(&a, i, i + 2, n)
            } else {
                vec![]
            };
            taup[i] = larfg(&mut a.v, alpha, &x);
            e[i] = a.at(i, i + 1);
            a.set(i, i + 1, 1.0);
            let v: Vec<f64> = (i + 1..=n).map(|c| a.at(i, c)).collect();
            larf(false, &v, taup[i], &mut a, i + 1, i + 1, n - i, n - i);
            a.set(i, i + 1, e[i]);
        }
    }
    // dlacpy('U') then dorgbr('P', n, n, n): shift the reflectors one row
    // down and generate P' with dorgl2 on the trailing n - 1 block.
    let mut vt = Mat {
        n,
        v: vec![0.0; n * n],
    };
    for j in 1..=n {
        for i in 1..=j {
            vt.set(i, j, a.at(i, j));
        }
    }
    vt.set(1, 1, 1.0);
    for i in 2..=n {
        vt.set(i, 1, 0.0);
    }
    for j in 2..=n {
        for i in (2..j).rev() {
            let x = vt.at(i - 1, j);
            vt.set(i, j, x);
        }
        vt.set(1, j, 0.0);
    }
    if n > 1 {
        // dorgl2(m = k = n - 1) on rows and columns 2..n; tau(i) = taup[i].
        let m = n - 1;
        for i in (1..=m).rev() {
            let (r, c) = (i + 1, i + 1);
            if i < m {
                vt.set(r, c, 1.0);
                let v: Vec<f64> = (c..=n).map(|cc| vt.at(r, cc)).collect();
                larf(false, &v, taup[i], &mut vt, r + 1, c, m - i, m - i + 1);
            }
            if i < m {
                for cc in c + 1..=n {
                    let x = vt.at(r, cc) * -taup[i];
                    vt.set(r, cc, x);
                }
            }
            vt.set(r, c, 1.0 - taup[i]);
            for cc in 2..c {
                vt.set(r, cc, 0.0);
            }
        }
    }
    bdsqr_upper(&mut d, &mut e, &mut vt);
    let rows = (1..=n)
        .map(|i| (1..=n).map(|j| vt.at(i, j)).collect())
        .collect();
    (d[1..].to_vec(), rows)
}

/// `dbdsqr('U', n, ncvt = n, nru = 0, ncc = 0)`: singular values of the
/// upper bidiagonal (d, e), applying the right rotations to the rows of `vt`.
fn bdsqr_upper(d: &mut [f64], e: &mut [f64], vt: &mut Mat) {
    let n = vt.n;
    let ncvt = n;
    let rot = |vt: &mut Mat, i: usize, k: usize, c: f64, s: f64| {
        for j in 1..=ncvt {
            let (x, y) = (vt.at(i, j), vt.at(k, j));
            vt.set(i, j, c * x + s * y);
            vt.set(k, j, c * y - s * x);
        }
    };
    // dlasr('L', 'V', direction, rows ll..ll+m-1): rotation j acts on rows
    // ll+j-1 and ll+j.
    let lasr = |vt: &mut Mat, forward: bool, ll: usize, m: usize, c: &[f64], s: &[f64]| {
        let order: Vec<usize> = if forward {
            (1..m).collect()
        } else {
            (1..m).rev().collect()
        };
        for j in order {
            let (ct, st) = (c[j], s[j]);
            if ct != 1.0 || st != 0.0 {
                for i in 1..=ncvt {
                    let temp = vt.at(ll + j, i);
                    let above = vt.at(ll + j - 1, i);
                    vt.set(ll + j, i, ct * temp - st * above);
                    vt.set(ll + j - 1, i, st * temp + ct * above);
                }
            }
        }
    };
    if n > 1 {
        const MAXITR: usize = 6;
        let tolmul = 10f64.max(100f64.min(EPS.powf(-0.125)));
        let tol = tolmul * EPS;
        let unfl = SAFMIN;
        let mut sminoa = d[1].abs();
        if sminoa != 0.0 {
            let mut mu = sminoa;
            for i in 2..=n {
                mu = d[i].abs() * (mu / (mu + e[i - 1].abs()));
                sminoa = sminoa.min(mu);
                if sminoa == 0.0 {
                    break;
                }
            }
        }
        sminoa /= (n as f64).sqrt();
        let thresh = (tol * sminoa).max(MAXITR as f64 * (n as f64 * (n as f64 * unfl)));
        let maxitdivn = MAXITR * n;
        let mut iterdivn = 0;
        let mut iter: isize = -1;
        let mut oldll: isize = -1;
        let mut oldm: isize = -1;
        let mut idir = 0;
        let mut m = n;
        // Rotation storage: cs at [i], sn at [i] of separate arrays (1-based).
        let mut c1 = vec![0.0; n + 1];
        let mut s1 = vec![0.0; n + 1];
        let mut c2 = vec![0.0; n + 1];
        let mut s2 = vec![0.0; n + 1];
        'outer: loop {
            if m <= 1 {
                break;
            }
            if iter >= n as isize {
                iter -= n as isize;
                iterdivn += 1;
                if iterdivn >= maxitdivn {
                    break;
                }
            }
            let mut smax = d[m].abs();
            let mut ll = 0;
            let mut split = false;
            for lll in 1..m {
                ll = m - lll;
                let abss = d[ll].abs();
                let abse = e[ll].abs();
                if abse <= thresh {
                    split = true;
                    break;
                }
                smax = smax.max(abss).max(abse);
            }
            if split {
                e[ll] = 0.0;
                if ll == m - 1 {
                    m -= 1;
                    continue 'outer;
                }
            } else {
                ll = 0;
            }
            ll += 1;
            if ll == m - 1 {
                let (sigmn, sigmx, sinr, cosr, _, _) = lasv2(d[m - 1], e[m - 1], d[m]);
                d[m - 1] = sigmx;
                e[m - 1] = 0.0;
                d[m] = sigmn;
                rot(vt, m - 1, m, cosr, sinr);
                m -= 2;
                continue 'outer;
            }
            if ll as isize > oldm || (m as isize) < oldll {
                idir = if d[ll].abs() >= d[m].abs() { 1 } else { 2 };
            }
            let mut smin;
            if idir == 1 {
                if e[m - 1].abs() <= tol.abs() * d[m].abs() {
                    e[m - 1] = 0.0;
                    continue 'outer;
                }
                let mut mu = d[ll].abs();
                smin = mu;
                for lll in ll..m {
                    if e[lll].abs() <= tol * mu {
                        e[lll] = 0.0;
                        continue 'outer;
                    }
                    mu = d[lll + 1].abs() * (mu / (mu + e[lll].abs()));
                    smin = smin.min(mu);
                }
            } else {
                if e[ll].abs() <= tol.abs() * d[ll].abs() {
                    e[ll] = 0.0;
                    continue 'outer;
                }
                let mut mu = d[m].abs();
                smin = mu;
                for lll in (ll..m).rev() {
                    if e[lll].abs() <= tol * mu {
                        e[lll] = 0.0;
                        continue 'outer;
                    }
                    mu = d[lll].abs() * (mu / (mu + e[lll].abs()));
                    smin = smin.min(mu);
                }
            }
            oldll = ll as isize;
            oldm = m as isize;
            let mut shift;
            if n as f64 * tol * (smin / smax) <= EPS.max(0.01 * tol) {
                shift = 0.0;
            } else {
                let sll;
                if idir == 1 {
                    sll = d[ll].abs();
                    shift = las2(d[m - 1], e[m - 1], d[m]).0;
                } else {
                    sll = d[m].abs();
                    shift = las2(d[ll], e[ll], d[ll + 1]).0;
                }
                if sll > 0.0 && (shift / sll).powi(2) < EPS {
                    shift = 0.0;
                }
            }
            iter += (m - ll) as isize;
            let k = m - ll + 1;
            if shift == 0.0 {
                if idir == 1 {
                    let mut cs = 1.0;
                    let mut oldcs = 1.0;
                    let mut oldsn = 0.0;
                    for i in ll..m {
                        let (c, s, r) = lartg(d[i] * cs, e[i]);
                        cs = c;
                        let sn = s;
                        if i > ll {
                            e[i - 1] = oldsn * r;
                        }
                        let (oc, os, di) = lartg(oldcs * r, d[i + 1] * sn);
                        oldcs = oc;
                        oldsn = os;
                        d[i] = di;
                        c1[i - ll + 1] = cs;
                        s1[i - ll + 1] = sn;
                        c2[i - ll + 1] = oldcs;
                        s2[i - ll + 1] = oldsn;
                    }
                    let h = d[m] * cs;
                    d[m] = h * oldcs;
                    e[m - 1] = h * oldsn;
                    lasr(vt, true, ll, k, &c1, &s1);
                    if e[m - 1].abs() <= thresh {
                        e[m - 1] = 0.0;
                    }
                } else {
                    let mut cs = 1.0;
                    let mut oldcs = 1.0;
                    let mut oldsn = 0.0;
                    for i in (ll + 1..=m).rev() {
                        let (c, s, r) = lartg(d[i] * cs, e[i - 1]);
                        cs = c;
                        let sn = s;
                        if i < m {
                            e[i] = oldsn * r;
                        }
                        let (oc, os, di) = lartg(oldcs * r, d[i - 1] * sn);
                        oldcs = oc;
                        oldsn = os;
                        d[i] = di;
                        c1[i - ll] = cs;
                        s1[i - ll] = -sn;
                        c2[i - ll] = oldcs;
                        s2[i - ll] = -oldsn;
                    }
                    let h = d[ll] * cs;
                    d[ll] = h * oldcs;
                    e[ll] = h * oldsn;
                    lasr(vt, false, ll, k, &c2, &s2);
                    if e[ll].abs() <= thresh {
                        e[ll] = 0.0;
                    }
                }
            } else if idir == 1 {
                let mut f = (d[ll].abs() - shift) * (sign(1.0, d[ll]) + shift / d[ll]);
                let mut g = e[ll];
                for i in ll..m {
                    let (cosr, sinr, r) = lartg(f, g);
                    if i > ll {
                        e[i - 1] = r;
                    }
                    f = cosr * d[i] + sinr * e[i];
                    e[i] = cosr * e[i] - sinr * d[i];
                    g = sinr * d[i + 1];
                    d[i + 1] *= cosr;
                    let (cosl, sinl, r) = lartg(f, g);
                    d[i] = r;
                    f = cosl * e[i] + sinl * d[i + 1];
                    d[i + 1] = cosl * d[i + 1] - sinl * e[i];
                    if i < m - 1 {
                        g = sinl * e[i + 1];
                        e[i + 1] *= cosl;
                    }
                    c1[i - ll + 1] = cosr;
                    s1[i - ll + 1] = sinr;
                    c2[i - ll + 1] = cosl;
                    s2[i - ll + 1] = sinl;
                }
                e[m - 1] = f;
                lasr(vt, true, ll, k, &c1, &s1);
                if e[m - 1].abs() <= thresh {
                    e[m - 1] = 0.0;
                }
            } else {
                let mut f = (d[m].abs() - shift) * (sign(1.0, d[m]) + shift / d[m]);
                let mut g = e[m - 1];
                for i in (ll + 1..=m).rev() {
                    let (cosr, sinr, r) = lartg(f, g);
                    if i < m {
                        e[i] = r;
                    }
                    f = cosr * d[i] + sinr * e[i - 1];
                    e[i - 1] = cosr * e[i - 1] - sinr * d[i];
                    g = sinr * d[i - 1];
                    d[i - 1] *= cosr;
                    let (cosl, sinl, r) = lartg(f, g);
                    d[i] = r;
                    f = cosl * e[i - 1] + sinl * d[i - 1];
                    d[i - 1] = cosl * d[i - 1] - sinl * e[i - 1];
                    if i > ll + 1 {
                        g = sinl * e[i - 2];
                        e[i - 2] *= cosl;
                    }
                    c1[i - ll] = cosr;
                    s1[i - ll] = -sinr;
                    c2[i - ll] = cosl;
                    s2[i - ll] = -sinl;
                }
                e[ll] = f;
                if e[ll].abs() <= thresh {
                    e[ll] = 0.0;
                }
                lasr(vt, false, ll, k, &c2, &s2);
            }
        }
    }
    // Make singular values positive, then sort them into decreasing order.
    for i in 1..=n {
        if d[i] < 0.0 {
            d[i] = -d[i];
            for j in 1..=ncvt {
                let x = -vt.at(i, j);
                vt.set(i, j, x);
            }
        }
    }
    for i in 1..n {
        let mut isub = 1;
        let mut smin = d[1];
        for j in 2..=n + 1 - i {
            if d[j] <= smin {
                isub = j;
                smin = d[j];
            }
        }
        if isub != n + 1 - i {
            d[isub] = d[n + 1 - i];
            d[n + 1 - i] = smin;
            for j in 1..=ncvt {
                let (x, y) = (vt.at(isub, j), vt.at(n + 1 - i, j));
                vt.set(isub, j, y);
                vt.set(n + 1 - i, j, x);
            }
        }
    }
}

/// `dlas2`: singular values (min, max) of [[f, g], [0, h]].
fn las2(f: f64, g: f64, h: f64) -> (f64, f64) {
    let (fa, ga, ha) = (f.abs(), g.abs(), h.abs());
    let fhmn = fa.min(ha);
    let fhmx = fa.max(ha);
    if fhmn == 0.0 {
        let ssmax = if fhmx == 0.0 {
            ga
        } else {
            fhmx.max(ga) * (1.0 + (fhmx.min(ga) / fhmx.max(ga)).powi(2)).sqrt()
        };
        (0.0, ssmax)
    } else if ga < fhmx {
        let as_ = 1.0 + fhmn / fhmx;
        let at = (fhmx - fhmn) / fhmx;
        let au = (ga / fhmx).powi(2);
        let c = 2.0 / ((as_ * as_ + au).sqrt() + (at * at + au).sqrt());
        (fhmn * c, fhmx / c)
    } else {
        let au = fhmx / ga;
        if au == 0.0 {
            ((fhmn * fhmx) / ga, ga)
        } else {
            let as_ = 1.0 + fhmn / fhmx;
            let at = (fhmx - fhmn) / fhmx;
            let c = 1.0 / ((1.0 + (as_ * au).powi(2)).sqrt() + (1.0 + (at * au).powi(2)).sqrt());
            let ssmin = (fhmn * c) * au;
            (ssmin + ssmin, ga / (c + c))
        }
    }
}

/// `dlasv2`: SVD of [[f, g], [0, h]]. Returns (ssmin, ssmax, snr, csr, snl, csl).
fn lasv2(f: f64, g: f64, h: f64) -> (f64, f64, f64, f64, f64, f64) {
    let mut ft = f;
    let mut fa = ft.abs();
    let mut ht = h;
    let mut ha = h.abs();
    let mut pmax = 1;
    let swap = ha > fa;
    if swap {
        pmax = 3;
        std::mem::swap(&mut ft, &mut ht);
        std::mem::swap(&mut fa, &mut ha);
    }
    let gt = g;
    let ga = gt.abs();
    let (mut ssmin, mut ssmax, clt, crt, slt, srt);
    if ga == 0.0 {
        ssmin = ha;
        ssmax = fa;
        clt = 1.0;
        crt = 1.0;
        slt = 0.0;
        srt = 0.0;
    } else {
        let mut gasmal = true;
        let mut tmp = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        if ga > fa {
            pmax = 2;
            if fa / ga < EPS {
                gasmal = false;
                let smax = ga;
                let smin = if ha > 1.0 {
                    fa / (ga / ha)
                } else {
                    (fa / ga) * ha
                };
                tmp = (smin, smax, 1.0, ft / gt, ht / gt, 1.0);
            }
        }
        if gasmal {
            let d = fa - ha;
            let mut l = if d == fa { 1.0 } else { d / fa };
            let m = gt / ft;
            let mut t = 2.0 - l;
            let mm = m * m;
            let tt = t * t;
            let s = (tt + mm).sqrt();
            let r = if l == 0.0 {
                m.abs()
            } else {
                (l * l + mm).sqrt()
            };
            let a = 0.5 * (s + r);
            let smin = ha / a;
            let smax = fa * a;
            if mm == 0.0 {
                if l == 0.0 {
                    t = sign(2.0, ft) * sign(1.0, gt);
                } else {
                    t = gt / sign(d, ft) + m / t;
                }
            } else {
                t = (m / (s + t) + m / (r + l)) * (1.0 + a);
            }
            l = (t * t + 4.0).sqrt();
            let crt_ = 2.0 / l;
            let srt_ = t / l;
            let clt_ = (crt_ + srt_ * m) / a;
            let slt_ = (ht / ft) * srt_ / a;
            tmp = (smin, smax, clt_, crt_, slt_, srt_);
        }
        ssmin = tmp.0;
        ssmax = tmp.1;
        clt = tmp.2;
        crt = tmp.3;
        slt = tmp.4;
        srt = tmp.5;
    }
    let (csl, snl, csr, snr) = if swap {
        (srt, crt, slt, clt)
    } else {
        (clt, slt, crt, srt)
    };
    let tsign = match pmax {
        1 => sign(1.0, csr) * sign(1.0, csl) * sign(1.0, f),
        2 => sign(1.0, snr) * sign(1.0, csl) * sign(1.0, g),
        _ => sign(1.0, snr) * sign(1.0, snl) * sign(1.0, h),
    };
    ssmax = sign(ssmax, tsign);
    ssmin = sign(ssmin, tsign * sign(1.0, f) * sign(1.0, h));
    (ssmin, ssmax, snr, csr, snl, csl)
}

/// `dpotrf2` (lower): recursive Cholesky of the block at (o, o) of size n.
fn potrf2(a: &mut Mat, o: usize, n: usize) -> bool {
    if n == 1 {
        let x = a.at(o, o);
        if x <= 0.0 || x.is_nan() {
            return false;
        }
        a.set(o, o, x.sqrt());
        return true;
    }
    let n1 = n / 2;
    let n2 = n - n1;
    if !potrf2(a, o, n1) {
        return false;
    }
    // dtrsm('R', 'L', 'T', 'N'): A21 := A21 inv(L11').
    let (r0, c0) = (o + n1, o);
    for k in 1..=n1 {
        let temp = 1.0 / a.at(c0 + k - 1, c0 + k - 1);
        for i in 1..=n2 {
            let x = temp * a.at(r0 + i - 1, c0 + k - 1);
            a.set(r0 + i - 1, c0 + k - 1, x);
        }
        for j in k + 1..=n1 {
            let ajk = a.at(c0 + j - 1, c0 + k - 1);
            if ajk != 0.0 {
                for i in 1..=n2 {
                    let x = a.at(r0 + i - 1, c0 + j - 1) - ajk * a.at(r0 + i - 1, c0 + k - 1);
                    a.set(r0 + i - 1, c0 + j - 1, x);
                }
            }
        }
    }
    // dsyrk('L', 'N'): A22 := A22 - A21 A21'.
    for j in 1..=n2 {
        for l in 1..=n1 {
            let ajl = a.at(r0 + j - 1, c0 + l - 1);
            if ajl != 0.0 {
                let temp = -ajl;
                for i in j..=n2 {
                    let x = a.at(r0 + i - 1, r0 + j - 1) + temp * a.at(r0 + i - 1, c0 + l - 1);
                    a.set(r0 + i - 1, r0 + j - 1, x);
                }
            }
        }
    }
    potrf2(a, o + n1, n2)
}

/// `dsyr2('L')` on the trailing block from (o, o): A += alpha (x y' + y x').
fn syr2_lower(a: &mut Mat, o: usize, alpha: f64, x: &[f64], y: &[f64]) {
    let n = x.len();
    for j in 0..n {
        if x[j] != 0.0 || y[j] != 0.0 {
            let temp1 = alpha * y[j];
            let temp2 = alpha * x[j];
            for i in j..n {
                let v = a.at(o + i, o + j) + x[i] * temp1 + y[i] * temp2;
                a.set(o + i, o + j, v);
            }
        }
    }
}

/// Eigenvalues (increasing) and B-orthonormal eigenvectors (columns) of the
/// symmetric-definite problem A z = lambda B z, as `dsygvx(1, 'V', 'A', 'L')`
/// with a positive ABSTOL returns them. Only lower triangles are read.
/// Returns None when B is not positive definite.
pub(crate) fn generalized_eigen(
    a: &[Vec<f64>],
    b: &[Vec<f64>],
) -> Option<(Vec<f64>, Vec<Vec<f64>>)> {
    let mut a = Mat::from_rows(a);
    let mut b = Mat::from_rows(b);
    let n = a.n;
    if !potrf2(&mut b, 1, n) {
        return None;
    }
    // dsygs2 (itype 1, lower): A := inv(L) A inv(L').
    for k in 1..=n {
        let bkk = b.at(k, k);
        let akk = a.at(k, k) / (bkk * bkk);
        a.set(k, k, akk);
        if k < n {
            for i in k + 1..=n {
                let x = a.at(i, k) * (1.0 / bkk);
                a.set(i, k, x);
            }
            let ct = -0.5 * akk;
            for i in k + 1..=n {
                let x = a.at(i, k) + ct * b.at(i, k);
                a.set(i, k, x);
            }
            let x: Vec<f64> = (k + 1..=n).map(|i| a.at(i, k)).collect();
            let y: Vec<f64> = (k + 1..=n).map(|i| b.at(i, k)).collect();
            syr2_lower(&mut a, k + 1, -1.0, &x, &y);
            for i in k + 1..=n {
                let x = a.at(i, k) + ct * b.at(i, k);
                a.set(i, k, x);
            }
            // dtrsv('L', 'N', 'N') with B(k+1.., k+1..).
            for j in k + 1..=n {
                let mut xj = a.at(j, k);
                if xj != 0.0 {
                    xj /= b.at(j, j);
                    a.set(j, k, xj);
                    for i in j + 1..=n {
                        let x = a.at(i, k) - xj * b.at(i, j);
                        a.set(i, k, x);
                    }
                }
            }
        }
    }
    let (w, z) = symmetric_eigen_x(&mut a);
    // dtrsm('L', 'L', 'T', 'N'): Z := inv(L') Z.
    let mut z = z;
    for j in 1..=n {
        for i in (1..=n).rev() {
            let mut temp = z.at(i, j);
            for k in i + 1..=n {
                temp -= b.at(k, i) * z.at(k, j);
            }
            temp /= b.at(i, i);
            z.set(i, j, temp);
        }
    }
    let vectors = (1..=n)
        .map(|j| (1..=n).map(|i| z.at(i, j)).collect())
        .collect();
    Some((w, vectors))
}

/// `dsyevx('V', 'A', 'L')` with a positive ABSTOL: tridiagonalization,
/// bisection, inverse iteration and back-transformation.
fn symmetric_eigen_x(a: &mut Mat) -> (Vec<f64>, Mat) {
    let n = a.n;
    let mut z = Mat {
        n,
        v: vec![0.0; n * n],
    };
    if n == 1 {
        z.set(1, 1, 1.0);
        return (vec![a.at(1, 1)], z);
    }
    // dsytd2 (lower).
    let mut d = vec![0.0; n + 1];
    let mut e = vec![0.0; n + 1];
    let mut tau = vec![0.0; n + 1];
    for i in 1..n {
        let alpha = pos(a, i + 1, i);
        let x = if i + 2 <= n {
            col_at(a, i, i + 2, n)
        } else {
            vec![]
        };
        let taui = larfg(&mut a.v, alpha, &x);
        e[i] = a.at(i + 1, i);
        if taui != 0.0 {
            a.set(i + 1, i, 1.0);
            let m = n - i;
            let v: Vec<f64> = (i + 1..=n).map(|r| a.at(r, i)).collect();
            // dsymv('L'): t := taui * A22 v.
            let mut t = vec![0.0; m];
            for j in 0..m {
                let temp1 = taui * v[j];
                let mut temp2 = 0.0;
                t[j] += temp1 * a.at(i + 1 + j, i + 1 + j);
                for r in j + 1..m {
                    t[r] += temp1 * a.at(i + 1 + r, i + 1 + j);
                    temp2 += a.at(i + 1 + r, i + 1 + j) * v[r];
                }
                t[j] += taui * temp2;
            }
            let dot: f64 = t.iter().zip(&v).map(|(x, y)| x * y).sum();
            let alpha = -0.5 * taui * dot;
            for r in 0..m {
                t[r] += alpha * v[r];
            }
            syr2_lower(a, i + 1, -1.0, &v, &t);
            a.set(i + 1, i, e[i]);
        }
        d[i] = a.at(i, i);
        tau[i] = taui;
    }
    d[n] = a.at(n, n);
    let (w, iblock, isplit) = stebz(&d, &e, n);
    stein(&d, &e, &w, &iblock, &isplit, &mut z);
    // dormtr('L', 'L', 'N') = dorm2r on rows 2..n with reflectors i = n-1..1.
    for i in (1..n).rev() {
        let v: Vec<f64> = std::iter::once(1.0)
            .chain((i + 2..=n).map(|r| a.at(r, i)))
            .collect();
        larf(true, &v, tau[i], &mut z, i + 1, 1, n - i, n);
    }
    // Selection sort into increasing order, as dsyevx does.
    let mut w = w;
    for j in 1..n {
        let mut k = 0;
        let mut tmp = w[j];
        for jj in j + 1..=n {
            if w[jj] < tmp {
                k = jj;
                tmp = w[jj];
            }
        }
        if k != 0 {
            w[k] = w[j];
            w[j] = tmp;
            for r in 1..=n {
                let (x, y) = (z.at(r, k), z.at(r, j));
                z.set(r, k, y);
                z.set(r, j, x);
            }
        }
    }
    (w[1..].to_vec(), z)
}

/// `dstebz('A', 'B')`: all eigenvalues of the tridiagonal (d, e) by
/// bisection, grouped by split block. Returns 1-based w, iblock and isplit.
fn stebz(d: &[f64], e: &[f64], n: usize) -> (Vec<f64>, Vec<usize>, Vec<usize>) {
    const FUDGE: f64 = 2.1;
    const RELFAC: f64 = 2.0;
    let rtoli = ULP * RELFAC;
    let mut e2 = vec![0.0; n + 1];
    let mut isplit = vec![0];
    let mut pivmin = 1.0f64;
    for j in 2..=n {
        let tmp1 = e[j - 1] * e[j - 1];
        if (d[j] * d[j - 1]).abs() * ULP * ULP + SAFMIN > tmp1 {
            isplit.push(j - 1);
            e2[j - 1] = 0.0;
        } else {
            e2[j - 1] = tmp1;
            pivmin = pivmin.max(tmp1);
        }
    }
    isplit.push(n);
    pivmin *= SAFMIN;
    // ABSTOL is positive but far below pivmin, so it never binds.
    let mut w = vec![0.0];
    let mut iblock = vec![0];
    let mut iend = 0;
    for jb in 1..isplit.len() {
        let ibegin = iend + 1;
        iend = isplit[jb];
        let size = iend - ibegin + 1;
        if size == 1 {
            w.push(d[ibegin]);
            iblock.push(jb);
            continue;
        }
        let mut gu = d[ibegin];
        let mut gl = d[ibegin];
        let mut tmp1 = 0.0;
        for j in ibegin..iend {
            let tmp2 = e[j].abs();
            gu = gu.max(d[j] + tmp1 + tmp2);
            gl = gl.min(d[j] - tmp1 - tmp2);
            tmp1 = tmp2;
        }
        gu = gu.max(d[iend] + tmp1);
        gl = gl.min(d[iend] - tmp1);
        let bnorm = gl.abs().max(gu.abs());
        gl = gl - FUDGE * bnorm * ULP * size as f64 - FUDGE * pivmin;
        gu = gu + FUDGE * bnorm * ULP * size as f64 + FUDGE * pivmin;
        // Number of eigenvalues below x (dlaebz's Sturm count).
        let count = |x: f64| -> usize {
            let mut c = 0;
            let mut tmp2 = d[ibegin] - x;
            if tmp2 <= pivmin {
                c += 1;
                tmp2 = tmp2.min(-pivmin);
            }
            for j in ibegin + 1..=iend {
                tmp2 = d[j] - e2[j - 1] / tmp2 - x;
                if tmp2 <= pivmin {
                    c += 1;
                    tmp2 = tmp2.min(-pivmin);
                }
            }
            c
        };
        let itmax = (((gu - gl + pivmin).ln() - pivmin.ln()) / 2f64.ln()) as usize + 2;
        // Bisecting each eigenvalue from the common starting interval follows
        // the same intervals as dlaebz's simultaneous bisection.
        for k in 1..=size {
            let (mut lo, mut hi) = (gl, gu);
            for _ in 0..itmax {
                let width = (hi - lo).abs();
                if width < pivmin.max(rtoli * hi.abs().max(lo.abs())) {
                    break;
                }
                let mid = 0.5 * (lo + hi);
                if count(mid) >= k {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            w.push(0.5 * (lo + hi));
            iblock.push(jb);
        }
    }
    (w, iblock, isplit)
}

/// `dstein`: eigenvectors of the tridiagonal by inverse iteration from
/// `dlarnv` starting vectors, each with its largest component positive.
fn stein(d: &[f64], e: &[f64], w: &[f64], iblock: &[usize], isplit: &[usize], z: &mut Mat) {
    const MAXITS: usize = 5;
    const EXTRA: usize = 2;
    let n = z.n;
    let m = w.len() - 1;
    let mut seed: u64 = 1 + (1 << 12) + (1 << 24) + (1 << 36);
    let mut j1 = 1;
    let mut xjm = 0.0;
    for nblk in 1..=iblock[m] {
        let b1 = isplit[nblk - 1] + 1;
        let bn = isplit[nblk];
        let blksiz = bn - b1 + 1;
        let mut gpind = j1;
        let mut onenrm = 0.0f64;
        let mut ortol = 0.0;
        let mut dtpcrt = 0.0;
        if blksiz > 1 {
            onenrm = (d[b1].abs() + e[b1].abs()).max(d[bn].abs() + e[bn - 1].abs());
            for i in b1 + 1..bn {
                onenrm = onenrm.max(d[i].abs() + e[i - 1].abs() + e[i].abs());
            }
            ortol = 1e-3 * onenrm;
            dtpcrt = (0.1 / blksiz as f64).sqrt();
        }
        let mut jblk = 0;
        for j in j1..=m {
            if iblock[j] != nblk {
                j1 = j;
                break;
            }
            jblk += 1;
            let mut xj = w[j];
            let mut x = vec![0.0; blksiz];
            if blksiz == 1 {
                x[0] = 1.0;
            } else {
                if jblk > 1 {
                    let eps1 = (ULP * xj).abs();
                    let pertol = 10.0 * eps1;
                    if xj - xjm < pertol {
                        xj = xjm + pertol;
                    }
                }
                x = larnv_uniform(&mut seed, blksiz);
                let mut a: Vec<f64> = d[b1..=bn].to_vec();
                let mut bsup: Vec<f64> = e[b1..bn].to_vec();
                let mut csub: Vec<f64> = e[b1..bn].to_vec();
                let (dd, inn) = lagtf(&mut a, xj, &mut bsup, &mut csub);
                let mut tol = 0.0;
                let mut nrmchk = 0;
                let mut its = 0;
                loop {
                    its += 1;
                    if its > MAXITS {
                        break;
                    }
                    let jmax = idamax(&x);
                    let scl = blksiz as f64 * onenrm * ULP.max(a[blksiz - 1].abs()) / x[jmax].abs();
                    for v in &mut x {
                        *v *= scl;
                    }
                    lagts(&a, &bsup, &csub, &dd, &inn, &mut x, &mut tol);
                    if jblk > 1 {
                        if (xj - xjm).abs() > ortol {
                            gpind = j;
                        }
                        if gpind != j {
                            for i in gpind..j {
                                let ztr: f64 =
                                    -(0..blksiz).map(|r| x[r] * z.at(b1 + r, i)).sum::<f64>();
                                for r in 0..blksiz {
                                    x[r] += ztr * z.at(b1 + r, i);
                                }
                            }
                        }
                    }
                    let jmax = idamax(&x);
                    let nrm = x[jmax].abs();
                    if nrm < dtpcrt {
                        continue;
                    }
                    nrmchk += 1;
                    if nrmchk < EXTRA + 1 {
                        continue;
                    }
                    break;
                }
                let mut scl = 1.0 / nrm2(&x);
                let jmax = idamax(&x);
                if x[jmax] < 0.0 {
                    scl = -scl;
                }
                for v in &mut x {
                    *v *= scl;
                }
            }
            for i in 1..=n {
                z.set(i, j, 0.0);
            }
            for (i, &v) in x.iter().enumerate() {
                z.set(b1 + i, j, v);
            }
            xjm = xj;
        }
    }
}

/// First index of the largest absolute value (`idamax`).
fn idamax(x: &[f64]) -> usize {
    let mut k = 0;
    for i in 1..x.len() {
        if x[i].abs() > x[k].abs() {
            k = i;
        }
    }
    k
}

/// `dlarnv(2)`: uniform(-1, 1) values from `dlaruv`'s 48-bit generator
/// (multiplier 33952834046453; the i-th value uses the seed times its i-th
/// power, and the seed advances to the last product).
fn larnv_uniform(seed: &mut u64, n: usize) -> Vec<f64> {
    const A: u64 = 33_952_834_046_453;
    const MASK: u64 = (1 << 48) - 1;
    let r = 1.0 / 4096.0;
    let mut out = Vec::with_capacity(n);
    let mut power = 1u64;
    let mut last = *seed;
    for _ in 0..n.min(128) {
        power = power.wrapping_mul(A) & MASK;
        last = seed.wrapping_mul(power) & MASK;
        let it = |k: u32| ((last >> (12 * k)) & 4095) as f64;
        let u = r * (it(3) + r * (it(2) + r * (it(1) + r * it(0))));
        out.push(2.0 * u - 1.0);
    }
    *seed = last;
    out
}

/// `dlagtf` with TOL = 0: LU factorization with partial pivoting of
/// T - lambda I. Returns the second superdiagonal and pivot flags.
fn lagtf(a: &mut [f64], lambda: f64, b: &mut [f64], c: &mut [f64]) -> (Vec<f64>, Vec<usize>) {
    let n = a.len();
    let mut d = vec![0.0; n.saturating_sub(2)];
    let mut inn = vec![0usize; n];
    a[0] -= lambda;
    let tl = EPS;
    let mut scale1 = a[0].abs() + b[0].abs();
    for k in 0..n - 1 {
        a[k + 1] -= lambda;
        let mut scale2 = c[k].abs() + a[k + 1].abs();
        if k + 1 < n - 1 {
            scale2 += b[k + 1].abs();
        }
        let piv1 = if a[k] == 0.0 {
            0.0
        } else {
            a[k].abs() / scale1
        };
        let piv2;
        if c[k] == 0.0 {
            inn[k] = 0;
            piv2 = 0.0;
            scale1 = scale2;
            if k + 1 < n - 1 {
                d[k] = 0.0;
            }
        } else {
            piv2 = c[k].abs() / scale2;
            if piv2 <= piv1 {
                inn[k] = 0;
                scale1 = scale2;
                c[k] /= a[k];
                a[k + 1] -= c[k] * b[k];
                if k + 1 < n - 1 {
                    d[k] = 0.0;
                }
            } else {
                inn[k] = 1;
                let mult = a[k] / c[k];
                a[k] = c[k];
                let temp = a[k + 1];
                a[k + 1] = b[k] - mult * temp;
                if k + 1 < n - 1 {
                    d[k] = b[k + 1];
                    b[k + 1] = -mult * d[k];
                }
                b[k] = temp;
                c[k] = mult;
            }
        }
        if piv1.max(piv2) <= tl && inn[n - 1] == 0 {
            inn[n - 1] = k + 1;
        }
    }
    if a[n - 1].abs() <= scale1 * tl && inn[n - 1] == 0 {
        inn[n - 1] = n;
    }
    (d, inn)
}

/// `dlagts(-1)`: solves (T - lambda I) x = y with the dlagtf factors,
/// perturbing small pivots. `tol` is computed on the first call.
fn lagts(a: &[f64], b: &[f64], c: &[f64], d: &[f64], inn: &[usize], y: &mut [f64], tol: &mut f64) {
    let n = a.len();
    let sfmin = SAFMIN;
    let bignum = 1.0 / sfmin;
    if *tol <= 0.0 {
        *tol = a[0].abs();
        if n > 1 {
            *tol = tol.max(a[1].abs()).max(b[0].abs());
        }
        for k in 2..n {
            *tol = tol.max(a[k].abs()).max(b[k - 1].abs()).max(d[k - 2].abs());
        }
        *tol *= EPS;
        if *tol == 0.0 {
            *tol = EPS;
        }
    }
    for k in 1..n {
        if inn[k - 1] == 0 {
            y[k] -= c[k - 1] * y[k - 1];
        } else {
            let temp = y[k - 1];
            y[k - 1] = y[k];
            y[k] = temp - c[k - 1] * y[k];
        }
    }
    for k in (0..n).rev() {
        let mut temp = if k + 2 < n {
            y[k] - b[k] * y[k + 1] - d[k] * y[k + 2]
        } else if k + 1 < n {
            y[k] - b[k] * y[k + 1]
        } else {
            y[k]
        };
        let mut ak = a[k];
        let mut pert = sign(*tol, ak);
        loop {
            let absak = ak.abs();
            if absak < 1.0 {
                if absak < sfmin {
                    if absak == 0.0 || temp.abs() * sfmin > absak {
                        ak += pert;
                        pert *= 2.0;
                        continue;
                    }
                    temp *= bignum;
                    ak *= bignum;
                } else if temp.abs() > absak * bignum {
                    ak += pert;
                    pert *= 2.0;
                    continue;
                }
            }
            break;
        }
        y[k] = temp / ak;
    }
}
