//! Optimization algorithms module
//!
//! Implements DFP, BFGS and other optimization algorithms

use std::f64;
use std::ops::{Add, Sub, Mul, Div, Neg, Index, IndexMut};
use anyhow::{Error, bail};

const EPS: f64 = 1.0e-10;
const TAB: usize = 10;

/// Default tolerance for DFP optimization algorithm (consistent with Python implementation)
/// 
/// Note: Ensure the same tolerance value (1e-3) is used as Python version's dfpmin
/// If Python version uses different tolerance, this constant needs to be updated accordingly
pub const DFP_TOLERANCE: f64 = 1e-3;

/// Maximum number of iterations for DFP optimization algorithm (consistent with Python implementation)
/// 
/// Note: Ensure the same maximum number of iterations (200) is used as Python version's dfpmin
pub const DFP_MAX_ITER: usize = 200;

/// Vector type alias
pub type OptVec = Vec<f64>;

/// Matrix class
#[derive(Clone, Debug)]
pub struct Mat {
    cols: Vec<Vec<f64>>,
    nrow: usize,
    ncol: usize,
}

impl Mat {
    /// Create matrix from array
    /// 
    /// # Errors
    /// 
    /// Returns an error if nrow or ncol is zero.
    pub fn new(mut a: Vec<f64>, nrow: usize, ncol: usize) -> Result<Self, Error> {
        if nrow == 0 || ncol == 0 {
            bail!("matrix dimensions must be positive");
        }
        if a.len() < nrow * ncol {
            a.extend(vec![0.0; nrow * ncol - a.len()]);
        }

        let mut cols = vec![vec![0.0; nrow]; ncol];
        for i in 0..ncol {
            for j in 0..nrow {
                cols[i][j] = a[i + j * ncol];
            }
        }

        Ok(Self { cols, nrow, ncol })
    }
    
    /// Create matrix from array (panics on error)
    /// 
    /// # Panics
    /// 
    /// Panics if nrow or ncol is zero.
    pub fn new_or_panic(a: Vec<f64>, nrow: usize, ncol: usize) -> Self {
        Self::new(a, nrow, ncol).expect("Failed to create matrix")
    }

    /// Create matrix from 2D array
    /// 
    /// # Errors
    /// 
    /// Returns an error if the array is empty or columns have different lengths.
    pub fn from_array(ar: Vec<Vec<f64>>) -> Result<Self, Error> {
        if ar.is_empty() || ar[0].is_empty() {
            bail!("array must be non-empty");
        }
        let ncol = ar.len();
        let nrow = ar[0].len();
        for col in &ar {
            if col.len() != nrow {
                bail!("all columns must have the same length");
            }
        }
        Ok(Self {
            cols: ar,
            nrow,
            ncol,
        })
    }
    
    /// Create matrix from 2D array (panics on error)
    /// 
    /// # Panics
    /// 
    /// Panics if the array is empty or columns have different lengths.
    pub fn from_array_or_panic(ar: Vec<Vec<f64>>) -> Self {
        Self::from_array(ar).expect("Failed to create matrix from array")
    }

    /// Create identity matrix
    /// 
    /// # Errors
    /// 
    /// Returns an error if n is zero.
    pub fn id(n: usize) -> Result<Self, Error> {
        if n == 0 {
            bail!("matrix dimension must be positive");
        }
        let mut cols = vec![vec![0.0; n]; n];
        for i in 0..n {
            cols[i][i] = 1.0;
        }
        Ok(Self {
            cols,
            nrow: n,
            ncol: n,
        })
    }
    
    /// Create identity matrix (panics on error)
    /// 
    /// # Panics
    /// 
    /// Panics if n is zero.
    pub fn id_or_panic(n: usize) -> Self {
        Self::id(n).expect("Failed to create identity matrix")
    }

    /// Create zero matrix
    /// 
    /// # Errors
    /// 
    /// Returns an error if nrow or ncol is zero.
    pub fn zero(nrow: usize, ncol: usize) -> Result<Self, Error> {
        if nrow == 0 || ncol == 0 {
            bail!("matrix dimensions must be positive");
        }
        Ok(Self {
            cols: vec![vec![0.0; nrow]; ncol],
            nrow,
            ncol,
        })
    }
    
    /// Create zero matrix (panics on error)
    /// 
    /// # Panics
    /// 
    /// Panics if nrow or ncol is zero.
    pub fn zero_or_panic(nrow: usize, ncol: usize) -> Self {
        Self::zero(nrow, ncol).expect("Failed to create zero matrix")
    }

    /// Get row
    /// 
    /// # Errors
    /// 
    /// Returns an error if the row index is out of bounds.
    pub fn row(&self, i: usize) -> Result<Vec<f64>, Error> {
        if i >= self.nrow {
            bail!("row index {} out of bounds (nrow = {})", i, self.nrow);
        }
        Ok(self.cols.iter().map(|col| col[i]).collect())
    }

    /// Get column
    /// 
    /// # Errors
    /// 
    /// Returns an error if the column index is out of bounds.
    pub fn col(&self, i: usize) -> Result<&Vec<f64>, Error> {
        if i >= self.ncol {
            bail!("column index {} out of bounds (ncol = {})", i, self.ncol);
        }
        Ok(&self.cols[i])
    }

    /// Transpose
    pub fn t(&self) -> Self {
        let mut cols = vec![vec![0.0; self.ncol]; self.nrow];
        for i in 0..self.nrow {
            for j in 0..self.ncol {
                cols[i][j] = self.cols[j][i];
            }
        }
        Self {
            cols,
            nrow: self.ncol,
            ncol: self.nrow,
        }
    }

    /// Trace
    /// 
    /// # Errors
    /// 
    /// Returns an error if the matrix is not square.
    pub fn tr(&self) -> Result<f64, Error> {
        if self.nrow != self.ncol {
            bail!("trace requires a square matrix");
        }
        Ok((0..self.nrow).map(|i| self.cols[i][i]).sum())
    }

    /// p-norm
    pub fn norm(&self, p: f64) -> f64 {
        let s: f64 = self
            .cols
            .iter()
            .flat_map(|col| col.iter())
            .map(|x| x.abs().powf(p))
            .sum();
        if p > 1.0 {
            s.powf(1.0 / p)
        } else {
            s
        }
    }

    /// Matrix size
    pub fn size(&self) -> (usize, usize) {
        (self.nrow, self.ncol)
    }

    /// Submatrix
    pub fn submatrix(&self, rowlist: &[usize], collist: &[usize]) -> Self {
        let mut ar = vec![vec![0.0; rowlist.len()]; collist.len()];
        for (i, &col_idx) in collist.iter().enumerate() {
            for (j, &row_idx) in rowlist.iter().enumerate() {
                ar[i][j] = self.cols[col_idx][row_idx];
            }
        }
        Self::from_array_or_panic(ar)
    }
}

impl Index<(usize, usize)> for Mat {
    type Output = f64;

    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        if row >= self.nrow || col >= self.ncol {
            panic!("Mat::index: index out of bounds");
        }
        &self.cols[col][row]
    }
}

impl IndexMut<(usize, usize)> for Mat {
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
        if row >= self.nrow || col >= self.ncol {
            panic!("Mat::index_mut: index out of bounds");
        }
        &mut self.cols[col][row]
    }
}

impl Add for &Mat {
    type Output = Mat;

    fn add(self, other: &Mat) -> Mat {
        if self.nrow != other.nrow || self.ncol != other.ncol {
            panic!("Mat::add: dimensions must match");
        }
        let mut cols = vec![vec![0.0; self.nrow]; self.ncol];
        for i in 0..self.ncol {
            for j in 0..self.nrow {
                cols[i][j] = self.cols[i][j] + other.cols[i][j];
            }
        }
        Mat {
            cols,
            nrow: self.nrow,
            ncol: self.ncol,
        }
    }
}

impl Add<&Mat> for Mat {
    type Output = Mat;

    fn add(self, other: &Mat) -> Mat {
        &self + other
    }
}

impl Add<Mat> for &Mat {
    type Output = Mat;

    fn add(self, other: Mat) -> Mat {
        self + &other
    }
}

impl Add<Mat> for Mat {
    type Output = Mat;

    fn add(self, other: Mat) -> Mat {
        &self + &other
    }
}

impl Sub for &Mat {
    type Output = Mat;

    fn sub(self, other: &Mat) -> Mat {
        if self.nrow != other.nrow || self.ncol != other.ncol {
            panic!("Mat::sub: dimensions must match");
        }
        let mut cols = vec![vec![0.0; self.nrow]; self.ncol];
        for i in 0..self.ncol {
            for j in 0..self.nrow {
                cols[i][j] = self.cols[i][j] - other.cols[i][j];
            }
        }
        Mat {
            cols,
            nrow: self.nrow,
            ncol: self.ncol,
        }
    }
}

impl Mul<f64> for &Mat {
    type Output = Mat;

    fn mul(self, scalar: f64) -> Mat {
        let mut cols = self.cols.clone();
        for col in &mut cols {
            for val in col.iter_mut() {
                *val *= scalar;
            }
        }
        Mat {
            cols,
            nrow: self.nrow,
            ncol: self.ncol,
        }
    }
}

impl Mul<&Mat> for &Mat {
    type Output = Mat;

    fn mul(self, other: &Mat) -> Mat {
        if self.ncol != other.nrow {
            panic!("Mat::mul: dimensions must match for multiplication");
        }
        let mut m = Mat::zero_or_panic(self.nrow, other.ncol);
        for i in 0..m.nrow {
            for j in 0..m.ncol {
                for k in 0..self.ncol {
                    m[(i, j)] += self[(i, k)] * other[(k, j)];
                }
            }
        }
        m
    }
}

impl Mul<&Mat> for Mat {
    type Output = Mat;

    fn mul(self, other: &Mat) -> Mat {
        &self * other
    }
}

impl Mul<Mat> for &Mat {
    type Output = Mat;

    fn mul(self, other: Mat) -> Mat {
        self * &other
    }
}

impl Mul<Mat> for Mat {
    type Output = Mat;

    fn mul(self, other: Mat) -> Mat {
        &self * &other
    }
}

impl Div<f64> for &Mat {
    type Output = Mat;

    fn div(self, scalar: f64) -> Mat {
        let mut cols = self.cols.clone();
        for col in &mut cols {
            for val in col.iter_mut() {
                *val /= scalar;
            }
        }
        Mat {
            cols,
            nrow: self.nrow,
            ncol: self.ncol,
        }
    }
}

impl Neg for &Mat {
    type Output = Mat;

    fn neg(self) -> Mat {
        self * -1.0
    }
}

impl PartialEq for Mat {
    fn eq(&self, other: &Self) -> bool {
        if self.nrow != other.nrow || self.ncol != other.ncol {
            return false;
        }
        for i in 0..self.ncol {
            if self.cols[i] != other.cols[i] {
                return false;
            }
        }
        true
    }
}

/// Vector class (extends OptVec)
pub trait VecOps {
    fn zero(n: usize) -> OptVec;
    fn t(&self, other: &[f64]) -> Mat;
}

impl VecOps for OptVec {
    fn zero(n: usize) -> OptVec {
        if n == 0 {
            panic!("VecOps::zero: dimension must be positive");
        }
        vec![0.0; n]
    }

    fn t(&self, other: &[f64]) -> Mat {
        let mut m = Mat::zero_or_panic(self.len(), other.len());
        for i in 0..self.len() {
            for j in 0..other.len() {
                m[(i, j)] = self[i] * other[j];
            }
        }
        m
    }
}

/// Matrix-vector multiplication
pub fn apply(m: &Mat, v: &[f64]) -> OptVec {
    if m.ncol != v.len() {
        panic!("apply: matrix columns must match vector length");
    }
    let mut u = vec![0.0; m.nrow];
    for i in 0..m.nrow {
        for j in 0..m.ncol {
            u[i] += m[(i, j)] * v[j];
        }
    }
    u
}

/// Transpose
pub fn t_mat(m: &Mat) -> Mat {
    m.t()
}

/// Norm
pub fn norm_vec(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Trace
pub fn tr(m: &Mat) -> f64 {
    if m.nrow != m.ncol {
        panic!("Mat::tr: matrix must be square");
    }
    (0..m.nrow).map(|i| m.cols[i][i]).sum()
}

/// 2x2 matrix determinant
pub fn det(m: &Mat) -> f64 {
    if m.nrow != 2 || m.ncol != 2 {
        panic!("det: matrix must be 2x2");
    }
    m[(0, 0)] * m[(1, 1)] - m[(0, 1)] * m[(1, 0)]
}

/// 2x2 matrix inverse
pub fn inv(m: &Mat, det_val: f64) -> Mat {
    if m.nrow != 2 || m.ncol != 2 {
        panic!("inv: matrix must be 2x2");
    }
    if det_val == 0.0 {
        panic!("inv: determinant is zero");
    }
    Mat::from_array_or_panic(vec![
        vec![m[(1, 1)] / det_val, -m[(0, 1)] / det_val],
        vec![-m[(1, 0)] / det_val, m[(0, 0)] / det_val],
    ])
}

/// Direction function
pub fn direct<F>(fn_: F, p: &[f64], u: &[f64]) -> impl Fn(f64) -> f64
where
    F: Fn(&[f64]) -> f64 + Clone,
{
    let fn_clone = fn_.clone();
    let p_clone = p.to_vec();
    let u_clone = u.to_vec();
    move |t: f64| {
        let x: Vec<f64> = p_clone.iter().zip(u_clone.iter()).map(|(a, b)| a + t * b).collect();
        fn_clone(&x)
    }
}

/// Calculate derivative of function at x (using numerical differentiation)
pub fn der<F>(mut fn_: F, x: f64, h: f64) -> f64
where
    F: FnMut(f64) -> f64,
{
    // dfridr algorithm
    const CON: f64 = 1.4;
    const CON2: f64 = CON * CON;
    const BIG: f64 = 1e10;
    const SAFE: f64 = 2.0;

    let mut der_a = [[0.0; TAB]; TAB];
    let mut hh = h;
    der_a[0][0] = (fn_(x + hh) - fn_(x - hh)) / (2.0 * hh);
    let mut ans = der_a[0][0];
    let mut err = BIG;

    for i in 1..TAB {
        hh /= CON;
        der_a[0][i] = (fn_(x + hh) - fn_(x - hh)) / (2.0 * hh);
        let mut fac = CON2;
        for j in 1..=i {
            der_a[j][i] = (der_a[j - 1][i] * fac - der_a[j - 1][i - 1]) / (fac - 1.0);
            fac = CON2 * fac;
            let errt = (der_a[j][i] - der_a[j - 1][i])
                .abs()
                .max((der_a[j][i] - der_a[j - 1][i - 1]).abs());
            if errt <= err {
                err = errt;
                ans = der_a[j][i];
            }
        }
        if (der_a[i][i] - der_a[i - 1][i - 1]).abs() >= SAFE * err {
            break;
        }
    }

    ans
}

/// Calculate gradient
pub fn grad<F>(mut fn_: F, p: &[f64], h: f64) -> OptVec
where
    F: FnMut(&[f64]) -> f64,
{
    let n = p.len();
    let mut gr = vec![0.0; n];
    let mut e = vec![0.0; n];

    for i in 0..n {
        e[i] = 1.0;
        gr[i] = pder(&mut fn_, p, &e, h);
        e[i] = 0.0;
    }

    gr
}

/// Directional derivative
fn pder<F>(fn_: &mut F, p: &[f64], u: &[f64], h: f64) -> f64
where
    F: FnMut(&[f64]) -> f64,
{
    let hh = h;
    let mut x1 = p.to_vec();
    let mut x2 = p.to_vec();
    for i in 0..x1.len() {
        x1[i] += hh * u[i];
        x2[i] -= hh * u[i];
    }
    (fn_(&x1) - fn_(&x2)) / (2.0 * hh)
}

/// Calculate function value and gradient
fn df<F>(func: &mut F, p: &[f64], fold: f64, g: &mut [f64])
where
    F: FnMut(&[f64]) -> f64,
{
    let mut ph = p.to_vec();
    for j in 0..p.len() {
        let temp = p[j];
        let mut h = EPS * temp.abs();
        if h < EPS {
            h = EPS;
        }
        ph[j] = temp + h;
        h = ph[j] - temp;
        let fh = func(&ph);
        ph[j] = temp;
        g[j] = (fh - fold) / h;
    }
}

/// Interpolation function
fn ip(a1: f64, fa1: f64, da1: f64, a2: f64, fa2: f64, da2: f64) -> f64 {
    let d = a2 - a1;
    let sgn = if d > 0.0 { 1.0 } else { -1.0 };

    let d1 = da1 + da2 - 3.0 * (fa1 - fa2) / (a1 - a2);
    let d2 = sgn * (d1 * d1 - da1 * da2).abs().sqrt();
    let mut a = a2 - d * (da2 + d2 - d1) / (da2 - da1 + 2.0 * d2);
    let small = 0.00001;
    if (a - a1).abs() < small || (a - a2).abs() < small {
        a = (a1 + a2) / 2.0;
    }
    a
}

/// Line search zoom function
fn zoom<F>(
    mut fn_: F,
    f0: f64,
    df0: f64,
    mut a_lo: f64,
    mut a_hi: f64,
    c1: f64,
    c2: f64,
) -> f64
where
    F: FnMut(f64) -> f64,
{
    let small = 0.00001;
    let h = 0.001;
    let mut df_lo = der(&mut fn_, a_lo, h);
    if df_lo.abs() < small {
        return a_lo;
    }
    let mut df_hi = der(&mut fn_, a_hi, h);
    if df_hi.abs() < small {
        return a_hi;
    }
    let mut f_lo = fn_(a_lo);
    let mut f_hi = fn_(a_hi);

    let mut iter = 1;
    loop {
        let a = ip(a_lo, f_lo, df_lo, a_hi, f_hi, df_hi);
        let fa = fn_(a);
        let dfa = der(&mut fn_, a, h);
        if fa >= f_lo || fa > f0 + c1 * a * df0 {
            a_hi = a;
            f_hi = fa;
            df_hi = dfa;
        } else {
            if dfa.abs() <= -c2 * df0 {
                return a;
            }
            if (dfa >= 0.0 && a_hi >= a_lo) || (dfa <= 0.0 && a_hi <= a_lo) {
                a_hi = a_lo;
                f_hi = f_lo;
                df_hi = df_lo;
            }
            a_lo = a;
            f_lo = fa;
            df_lo = dfa;
        }
        iter += 1;
        if iter > 10 {
            return a;
        }
    }
}

/// Line search
fn line_search<F>(mut fn_: F, c1: f64, c2: f64, amax: f64) -> f64
where
    F: FnMut(f64) -> f64,
{
    if amax <= 1.0 {
        panic!("line_search: amax must be > 1");
    }
    let h = 0.001;
    let f0 = fn_(0.0);
    let df0 = der(&mut fn_, 0.0, h);
    let famax = fn_(amax);
    let dfamax = der(&mut fn_, amax, h);
    let mut a0 = 0.0;
    let mut fa0 = f0;
    let mut a1 = 1.0;
    let mut i = 1;

    loop {
        let fa1 = fn_(a1);
        if (fa1 >= fa0 && i > 1) || fa1 > f0 + c1 * a1 * df0 {
            return zoom(&mut fn_, f0, df0, a0, a1, c1, c2);
        }
        let dfa1 = der(&mut fn_, a1, h);
        if dfa1.abs() <= -c2 * df0 {
            return a1;
        }
        if dfa1 >= 0.0 {
            return zoom(&mut fn_, f0, df0, a0, a1, c1, c2);
        }
        a0 = a1;
        fa0 = fa1;
        a1 = ip(a1, fa1, dfa1, amax, famax, dfamax);
        i += 1;
    }
}

/// Line search (lnsrch)
fn lnsrch<F>(
    func: &mut F,
    xold: &[f64],
    fold: f64,
    g: &[f64],
    p: &mut [f64],
    x: &mut [f64],
    stpmax: f64,
) -> f64
where
    F: FnMut(&[f64]) -> f64,
{
    const ALF: f64 = 1.0e-4;
    const TOLX: f64 = 2.22045e-16;

    let mut alam2 = 0.0;
    let mut f2 = 0.0;
    let n = xold.len();
    let mut sum = 0.0;
    for i in 0..n {
        sum += p[i] * p[i];
    }
    sum = sum.sqrt();

    if sum > stpmax {
        for i in 0..n {
            p[i] *= stpmax / sum;
        }
    }

    let mut slope = 0.0;
    for i in 0..n {
        slope += p[i] * g[i];
    }
    if slope >= 0.0 {
        panic!("Roundoff problem in lnsrch: slope >= 0");
    }

    let mut test = 0.0;
    for i in 0..n {
        let temp = p[i].abs() / xold[i].abs().max(1.0);
        if temp > test {
            test = temp;
        }
    }

    let alamin = TOLX / test;
    let mut alam = 1.0;

    loop {
        for i in 0..n {
            x[i] = xold[i] + alam * p[i];
        }

        let f = func(x);

        if alam < alamin {
            for i in 0..n {
                x[i] = xold[i];
            }
            return f;
        } else if f <= fold + ALF * alam * slope {
            return f;
        } else {
            let tmplam = if alam == 1.0 {
                -slope / (2.0 * (f - fold - slope))
            } else {
                let rhs1 = f - fold - alam * slope;
                let rhs2 = f2 - fold - alam2 * slope;
                let a = (rhs1 / (alam * alam) - rhs2 / (alam2 * alam2)) / (alam - alam2);
                let b = (-alam2 * rhs1 / (alam * alam) + alam * rhs2 / (alam2 * alam2))
                    / (alam - alam2);
                if a == 0.0 {
                    -slope / (2.0 * b)
                } else {
                    let disc = b * b - 3.0 * a * slope;
                    if disc < 0.0 {
                        0.5 * alam
                    } else if b <= 0.0 {
                        (-b + disc.sqrt()) / (3.0 * a)
                    } else {
                        -slope / (b + disc.sqrt())
                    }
                }
            };
            let tmplam = tmplam.min(0.5 * alam);
            alam2 = alam;
            f2 = f;
            alam = tmplam.max(0.1 * alam);
        }
    }
}

/// DFP optimization algorithm
/// 
/// # Parameters
/// - `func`: Function to minimize
/// - `p`: Initial parameter vector (will be modified to optimal value)
/// - `gtol`: Gradient tolerance (default uses DFP_TOLERANCE = 1e-3, consistent with Python implementation)
/// 
/// # Returns
/// - `(fval, iter)`: Function value and number of iterations
/// 
/// # Notes
/// - Maximum number of iterations is fixed at 200 (consistent with Python implementation)
/// - If Python version uses different iteration limit, ITMAX constant needs to be updated accordingly
pub fn dfpmin<F>(mut func: F, p: &mut OptVec, gtol: f64) -> (f64, usize)
where
    F: FnMut(&[f64]) -> f64,
{
    const ITMAX: usize = 200; // Consistent with Python implementation: maximum 200 iterations
    const TOLX: f64 = 4.0 * EPS;
    const STPMX: f64 = 100.0;

    let n = p.len();
    let mut g = vec![0.0; n];
    let mut dg = vec![0.0; n];
    let mut hdg = vec![0.0; n];
    let mut pnew = vec![0.0; n];
    let mut hessin = vec![vec![0.0; n]; n];
    
    // Initialize identity matrix
    for i in 0..n {
        hessin[i][i] = 1.0;
    }

    let mut fp = func(&p);
    df(&mut func, &p, fp, &mut g);
    
    if g.iter().map(|x| x * x).sum::<f64>().sqrt() == 0.0 {
        return (fp, 0);
    }

    let mut xi: OptVec = g.iter().map(|x| -x).collect();
    let mut sum = 0.0;
    for i in 0..n {
        sum += p[i] * p[i];
    }
    let stpmax = STPMX * sum.sqrt().max(n as f64);

    for its in 0..ITMAX {
        let iter = its;
        let fret = lnsrch(&mut func, &p, fp, &g, &mut xi, &mut pnew, stpmax);
        fp = fret;
        for i in 0..n {
            xi[i] = pnew[i] - p[i];
            p[i] = pnew[i];
        }

        let mut test = 0.0;
        for i in 0..n {
            let temp = xi[i].abs() / p[i].abs().max(1.0);
            if temp > test {
                test = temp;
            }
        }

        if test < TOLX {
            return (fret, iter);
        }

        for i in 0..n {
            dg[i] = g[i];
        }

        let fp_new = func(&p);
        df(&mut func, &p, fp_new, &mut g);
        let mut test = 0.0;
        let den = fret.max(1.0);
        for i in 0..n {
            let temp = g[i].abs() * p[i].abs().max(1.0) / den;
            if temp > test {
                test = temp;
            }
        }

        if test < gtol {
            return (fret, iter);
        }

        for i in 0..n {
            dg[i] = g[i] - dg[i];
        }

        for i in 0..n {
            hdg[i] = 0.0;
            for j in 0..n {
                hdg[i] += hessin[i][j] * dg[j];
            }
        }

        let fac = (0..n).map(|i| dg[i] * xi[i]).sum::<f64>();
        let fae = (0..n).map(|i| dg[i] * hdg[i]).sum::<f64>();
        let sumdg = (0..n).map(|i| dg[i] * dg[i]).sum::<f64>();
        let sumxi = (0..n).map(|i| xi[i] * xi[i]).sum::<f64>();

        if fac > (EPS * sumdg * sumxi).sqrt() {
            let fac = 1.0 / fac;
            let fad = 1.0 / fae;
            let mut dg_new = vec![0.0; n];
            for i in 0..n {
                dg_new[i] = fac * xi[i] - fad * hdg[i];
            }

            for i in 0..n {
                for j in i..n {
                    hessin[i][j] += fac * xi[i] * xi[j] - fad * hdg[i] * hdg[j]
                        + fae * dg_new[i] * dg_new[j];
                    hessin[j][i] = hessin[i][j];
                }
            }
        }

        for i in 0..n {
            xi[i] = 0.0;
            for j in 0..n {
                xi[i] -= hessin[i][j] * g[j];
            }
        }
    }

    (fp, ITMAX)
}

/// BFGS optimization algorithm
pub fn bfgs<F>(mut fn_: F, x0: OptVec, er: f64) -> OptVec
where
    F: FnMut(&[f64]) -> f64,
{
    let n = x0.len();
    let i = Mat::id(n).expect("Failed to create identity matrix");
    let mut h = i.clone();
    let mut x1 = x0;
    let mut grad1 = grad(&mut fn_, &x1, 0.001);
    let mut iter = 0;
    let mut ct = 0;

    while norm_vec(&grad1) > er {
        let p = apply(&h, &grad1);
        let x1_clone = x1.clone();
        let p_clone = p.clone();
        let fn_mut = &mut fn_;
        let mut line_search_fn = move |t: f64| {
            let x_new: OptVec = x1_clone.iter().zip(p_clone.iter()).map(|(a, b)| a + t * b).collect();
            fn_mut(&x_new)
        };
        let alpha = line_search(&mut line_search_fn, er, 0.9, 1.1);
        let s: OptVec = p.iter().map(|x| alpha * x).collect();
        let x2: OptVec = x1.iter().zip(s.iter()).map(|(a, b)| a + b).collect();
        let grad2 = grad(&mut fn_, &x2, 0.001);
        let y: OptVec = grad2.iter().zip(grad1.iter()).map(|(a, b)| a - b).collect();
        let mu: f64 = y.iter().zip(s.iter()).map(|(a, b)| a * b).sum();
        if mu.abs() < 0.000001 {
            x1 = x2;
            break;
        }
        let rho = 1.0 / mu;

        let s_t_y = <OptVec as VecOps>::t(&s, &y);
        let u = &i - &(&s_t_y * rho);
        let s_t_s = <OptVec as VecOps>::t(&s, &s);
        h = &u * &h * &u.t() + &(&s_t_s * rho);
        x1 = x2;
        grad1 = grad2;
        iter += 1;
        if norm_vec(&y) < er {
            if ct >= 3 {
                break;
            } else {
                ct += 1;
            }
        } else {
            ct = 0;
        }
        if iter > 25 {
            break;
        }
    }
    x1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_der() {
        // Test derivative of x^2 at x=2 should be 4
        let fn_ = |x: f64| x * x;
        let deriv = der(fn_, 2.0, 0.001);
        assert!((deriv - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_grad() {
        // Test gradient of x^2 + y^2
        let fn_ = |p: &[f64]| p[0] * p[0] + p[1] * p[1];
        let p = vec![2.0, 3.0];
        let gr = grad(fn_, &p, 0.001);
        assert!((gr[0] - 4.0).abs() < 0.1);
        assert!((gr[1] - 6.0).abs() < 0.1);
    }

    #[test]
    fn test_dfpmin() {
        // Test minimizing x^2 + y^2
        let fn_ = |p: &[f64]| p[0] * p[0] + p[1] * p[1];
        let mut p0 = vec![5.0, 5.0];
        let (fval, _iter) = dfpmin(fn_, &mut p0, 1e-3);
        assert!(fval < 0.1);
    }

    #[test]
    fn test_mat() {
        // Test matrix creation
        let m = Mat::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2).expect("Failed to create matrix");
        assert_eq!(m[(0, 0)], 1.0);
        assert_eq!(m[(0, 1)], 2.0);
        assert_eq!(m[(1, 0)], 3.0);
        assert_eq!(m[(1, 1)], 4.0);

        // Test identity matrix
        let id = Mat::id(3).expect("Failed to create identity matrix");
        assert_eq!(id[(0, 0)], 1.0);
        assert_eq!(id[(1, 1)], 1.0);
        assert_eq!(id[(2, 2)], 1.0);
        assert_eq!(id[(0, 1)], 0.0);

        // Test transpose
        let m_t = m.t();
        assert_eq!(m_t[(0, 0)], 1.0);
        assert_eq!(m_t[(1, 0)], 2.0);
        assert_eq!(m_t[(0, 1)], 3.0);
        assert_eq!(m_t[(1, 1)], 4.0);

        // Test trace (method returns Result)
        assert!((m.tr().expect("tr failed") - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_apply() {
        let m = Mat::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2).expect("Failed to create matrix");
        let v = vec![1.0, 2.0];
        let result = apply(&m, &v);
        assert_eq!(result.len(), 2);
        assert!((result[0] - 5.0).abs() < 1e-10); // 1*1 + 2*2 = 5
        assert!((result[1] - 11.0).abs() < 1e-10); // 3*1 + 4*2 = 11
    }

    #[test]
    fn test_norm_vec() {
        let v = vec![3.0, 4.0];
        let norm = norm_vec(&v);
        assert!((norm - 5.0).abs() < 1e-10); // sqrt(3^2 + 4^2) = 5
    }

    #[test]
    fn test_tr() {
        let m = Mat::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 3, 3).expect("Failed to create matrix");
        let trace = tr(&m);
        assert!((trace - 15.0).abs() < 1e-10); // 1 + 5 + 9 = 15
    }

    #[test]
    fn test_det() {
        // Test determinant of 2x2 matrix
        let m = Mat::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2).expect("Failed to create matrix");
        let det_val = det(&m);
        assert!((det_val - (-2.0)).abs() < 1e-10); // 1*4 - 2*3 = -2

        // Test determinant of identity matrix (2x2)
        let id = Mat::id(2).expect("Failed to create identity matrix");
        let det_id = det(&id);
        assert!((det_id - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_inv() {
        // Test inverse of 2x2 matrix
        // Use a simple invertible matrix [2, 1; 1, 1], its inverse is [1, -1; -1, 2]
        let m = Mat::new(vec![2.0, 1.0, 1.0, 1.0], 2, 2).expect("Failed to create matrix");
        let det_val = det(&m);
        assert!((det_val - 1.0).abs() < 1e-10); // 2*1 - 1*1 = 1
        let m_inv = inv(&m, det_val);
        // Verify inverse matrix values: should be [1, -1; -1, 2]
        assert!((m_inv[(0, 0)] - 1.0).abs() < 1e-10);
        assert!((m_inv[(0, 1)] - (-1.0)).abs() < 1e-10);
        assert!((m_inv[(1, 0)] - (-1.0)).abs() < 1e-10);
        assert!((m_inv[(1, 1)] - 2.0).abs() < 1e-10);
        // Verify m * m_inv = I (by checking first column)
        let col1 = vec![m_inv[(0, 0)], m_inv[(1, 0)]];
        let product = apply(&m, &col1);
        assert!((product[0] - 1.0).abs() < 1e-10);
        assert!((product[1] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_bfgs() {
        // Test minimizing x^2 + y^2
        let fn_ = |p: &[f64]| p[0] * p[0] + p[1] * p[1];
        let x0 = vec![5.0, 5.0];
        let result = bfgs(fn_, x0, 1e-3);
        assert!(result[0].abs() < 0.1);
        assert!(result[1].abs() < 0.1);
    }
}

