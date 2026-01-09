//! Univariate state space models module
//!
//! Implements various univariate time series state space models

use crate::models::{StateSpaceModel, is_multivariate_algorithm, is_supported_algorithm};
use crate::optimize::{DFP_TOLERANCE, dfpmin};
use crate::utils::{Datafeed, MAX_LAG};
use anyhow::{Context, Error, bail};
use std::f64;

/// Helper function: Get value from p array (handles NaN and out-of-bounds cases)
/// Consistent with Python's self.algos[ts_idx].p[i]
fn get_p_value(p: &[f64], i: usize) -> f64 {
    if i < p.len() {
        let val = p[i];
        if val.is_nan() { f64::NAN } else { val }
    } else {
        f64::NAN
    }
}

/// Local Level model (LL0)
pub struct LL0 {
    datafeed: Datafeed,
    fc: Vec<f64>, // Filtered state
    p: Vec<f64>,  // Prediction variance
    fcstart: usize,
    pstart: usize,
    state_step: usize,
    forecast_len: usize,
    sigma: f64,
    nu: f64,
    var: f64,
    cur_state_end: usize,
    fc_init_set: bool,
}

impl LL0 {
    pub fn new(datafeed: Datafeed, forecast_len: usize) -> anyhow::Result<Self> {
        Self::new_with_params(
            datafeed,
            None, // fc
            None, // fcstart
            None, // fcend
            None, // fcinitval
            None, // P
            None, // Pstart
            None, // Pend
            None, // Pinitval
            1,    // state_step
            forecast_len,
        )
        .context("Failed to create LL0 model")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_params(
        mut datafeed: Datafeed,
        mut fc: Option<&mut Vec<f64>>,
        fcstart: Option<usize>,
        fcend: Option<usize>,
        fcinitval: Option<f64>,
        mut p: Option<&mut Vec<f64>>,
        pstart: Option<usize>,
        pend: Option<usize>,
        pinitval: Option<f64>,
        state_step: usize,
        forecast_len: usize,
    ) -> Result<Self, Error> {
        // Handle missing values: skip leading None values
        let mut i = datafeed.get_start();
        let mut skipped = 0;
        datafeed.reset();
        for val in datafeed.by_ref() {
            if val.is_nan() {
                i += state_step;
                skipped += 1;
            } else {
                break;
            }
        }
        if skipped > 0 {
            datafeed.set_start(i);
        }
        if datafeed.is_empty() {
            bail!("invalid data: no valid data for LL0 model");
        }

        let data_len = datafeed.len();
        let fcstart = fcstart.unwrap_or_else(|| datafeed.get_start());
        let fcend = fcend.unwrap_or_else(|| data_len + forecast_len);
        let pstart = pstart.unwrap_or(fcstart);
        let pend = pend.unwrap_or(fcend);

        // Use provided fc and p arrays, or create new ones
        // If external arrays are provided, we need to ensure they are large enough
        let mut fc_vec = if let Some(ref mut fc_ref) = fc {
            // Ensure array size is sufficient
            if fc_ref.len() < fcend {
                fc_ref.resize(fcend, f64::NAN);
            }
            // Create temporary copy for computation, will write back later
            fc_ref.clone()
        } else {
            vec![f64::NAN; fcend]
        };
        let mut p_vec = if let Some(ref mut p_ref) = p {
            // Ensure array size is sufficient
            if p_ref.len() < pend {
                p_ref.resize(pend, f64::NAN);
            }
            // Create temporary copy for computation, will write back later
            p_ref.clone()
        } else {
            vec![f64::NAN; pend]
        };

        // If initial values are provided, set them
        let fc_init_set = fcinitval.is_some();
        if let Some(init_val) = fcinitval
            && fcstart < fc_vec.len()
        {
            fc_vec[fcstart] = init_val;
        }
        if let Some(init_val) = pinitval
            && pstart < p_vec.len()
        {
            p_vec[pstart] = init_val;
        }

        let mut ll = Self {
            datafeed,
            fc: fc_vec,
            p: p_vec,
            fcstart,
            pstart,
            state_step,
            forecast_len,
            sigma: 0.0,
            nu: 0.0,
            var: 0.0,
            cur_state_end: 0,
            fc_init_set,
        };

        ll.compute_states();

        // If external arrays are provided, write results back (only write relevant range)
        if let Some(ref mut fc_ref) = fc {
            let end = fcend.min(fc_ref.len()).min(ll.fc.len());
            fc_ref[fcstart..end].copy_from_slice(&ll.fc[fcstart..end]);
        }
        if let Some(ref mut p_ref) = p {
            let end = pend.min(p_ref.len()).min(ll.p.len());
            p_ref[pstart..end].copy_from_slice(&ll.p[pstart..end]);
        }

        Ok(ll)
    }

    fn compute_states(&mut self) {
        self.sigma = 0.0;
        self.nu = 0.0;
        let mut psi = vec![1.0];
        let mut func = |x: &[f64]| {
            let q = x[0].exp();
            self.llh(q)
        };
        let (_, _) = dfpmin(&mut func, &mut psi, DFP_TOLERANCE);
        let q = psi[0].exp();
        self.update_all(q);
        // Fix index out of bounds: ensure index is valid
        let idx = if self.cur_state_end >= self.state_step {
            self.cur_state_end - self.state_step
        } else if self.cur_state_end > 0 {
            self.cur_state_end - 1
        } else {
            0
        };
        if idx < self.p.len() {
            self.var = self.p[idx] + self.sigma;
        } else if !self.p.is_empty() {
            self.var = self.p[self.p.len() - 1] + self.sigma;
        } else {
            self.var = self.sigma;
        }

        // Calculate predictions beyond data range
        let fcend = self.cur_state_end;
        let pend = self.cur_state_end;
        let fclast = fcend - self.state_step;
        let plast = pend - self.state_step;
        for i in (0..(self.forecast_len * self.state_step)).step_by(self.state_step) {
            if fcend + i < self.fc.len() {
                self.fc[fcend + i] = self.fc[fclast];
            }
            if pend + i < self.p.len() {
                self.p[pend + i] = self.p[plast + i] + self.nu;
            }
        }
    }

    fn update_kalman(&self, y: f64, a: f64, p: f64, q: f64) -> (f64, f64, f64, f64) {
        // Handle missing values: see Durbin-Koopman, page 24
        if y.is_nan() {
            return (a, p + q, 0.0, p + 1.0);
        }
        let f = p + 1.0;
        let k = p / f;
        let p_new = k + q;
        let v = y - a;
        let a_new = a + k * v;
        (a_new, p_new, v, f)
    }

    fn llh(&self, q: f64) -> f64 {
        let mut datafeed = self.datafeed.clone();
        datafeed.reset();

        // Prefer using self.fc[fcstart] as initial state (if explicitly set), otherwise take first value from data stream
        // This is consistent with Python implementation: check self.fc[self.fcstart] to determine initial state
        let (mut a, mut p, numdatataken) =
            if self.fc_init_set && self.fcstart < self.fc.len() && self.pstart < self.p.len() {
                // If fc is set, use initial values from fc and p
                let a_val = self.fc[self.fcstart];
                let p_val = if !self.p[self.pstart].is_nan() {
                    self.p[self.pstart]
                } else {
                    1.0 + q
                };
                (a_val, p_val, 0)
            } else {
                // Otherwise take first value from data stream
                (datafeed.next().unwrap_or(0.0), 1.0 + q, 1)
            };

        let mut t1 = 0.0;
        let mut t2 = 0.0;
        for x in datafeed {
            let (a_new, p_new, v, f) = self.update_kalman(x, a, p, q);
            t1 += v * v / f;
            t2 += f.ln();
            a = a_new;
            p = p_new;
        }

        if t1 == 0.0 {
            t2
        } else {
            (self.datafeed.len() - numdatataken) as f64 * t1.ln() + t2
        }
    }

    fn update_all(&mut self, q: f64) {
        let mut datafeed = self.datafeed.clone();
        datafeed.reset();

        // Prefer using self.fc[fcstart] as initial state (if explicitly set), otherwise take first value from data stream
        // This is consistent with Python implementation: check self.fc[self.fcstart] to determine initial state
        let (mut a, mut p, mut i, b) =
            if self.fc_init_set && self.fcstart < self.fc.len() && self.pstart < self.p.len() {
                // If fc is set, use initial values from fc and p
                let a_val = self.fc[self.fcstart];
                let p_val = if !self.p[self.pstart].is_nan() {
                    self.p[self.pstart]
                } else {
                    1.0 + q
                };
                (a_val, p_val, 0, 0)
            } else {
                // Otherwise take first value from data stream
                let first_val = datafeed.next().unwrap_or(0.0);
                // If obtained from data stream, also set to fc[fcstart]
                if self.fcstart < self.fc.len() {
                    self.fc[self.fcstart] = first_val;
                }
                let p_val = 1.0 + q;
                if self.pstart < self.p.len() {
                    self.p[self.pstart] = p_val;
                }
                (first_val, p_val, self.state_step, self.state_step)
            };

        let mut sigma = 0.0;

        for x in datafeed {
            let (a_new, p_new, v, f) = self.update_kalman(x, a, p, q);
            sigma += v * v / f;
            if self.fcstart + i < self.fc.len() {
                self.fc[self.fcstart + i] = a_new;
            }
            if self.pstart + i < self.p.len() {
                self.p[self.pstart + i] = p_new;
            }
            a = a_new;
            p = p_new;
            i += self.state_step;
        }

        // Use floating point division, consistent with Python's old_div
        let denom = ((i - b) as f64 / self.state_step as f64).max(1.0);
        sigma /= denom;
        self.sigma = sigma;
        self.nu = q * sigma;
        for j in (self.pstart..(self.pstart + i)).step_by(self.state_step) {
            if j < self.p.len() {
                self.p[j] = (self.p[j] + 1.0) * sigma;
            }
        }
        self.cur_state_end = self.pstart + i;
    }

    pub fn state(&self, i: usize) -> f64 {
        if i < self.fc.len() { self.fc[i] } else { 0.0 }
    }

    pub fn var(&self, i: usize) -> f64 {
        // In LLP, the passed i is a local index (relative to period), needs to be converted to global index
        // But in Python, the variance method accesses self.p[i], where i is the passed parameter
        // If state_step > 1 (e.g., state_step=period in LLP), need to convert index
        let global_idx = if self.state_step > 1 {
            self.pstart + i * self.state_step
        } else {
            i
        };
        let n = self.datafeed.get_end();
        if i < n {
            if global_idx < self.p.len() {
                let p_val = self.p[global_idx];
                // If p[i] is NaN (representing None), return NaN (consistent with Python)
                // In Python, if p[i] is None, the variance method throws an exception, test code catches and returns None
                if p_val.is_nan() {
                    f64::NAN
                } else {
                    p_val + self.sigma
                }
            } else {
                f64::NAN
            }
        } else {
            self.var + ((i as i32 - n as i32 + 1) as f64) * self.nu
        }
    }

    pub fn datalen(&self) -> usize {
        self.datafeed.len()
    }
}

/// Local Level model (LL) - Chunked version for large datasets
///
/// - Large datasets (>2000 points) are automatically chunked, with at most 2000 points per chunk
/// - Chunking improves numerical stability and avoids numerical issues from long sequences
pub struct LL {
    ll0: LL0,
    chunked: bool,
    chunks: Vec<LL0>,
    data_len: usize,
}

impl LL {
    /// Create LL model, supports shared fc and p arrays (for LLP and other models)
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_params(
        datafeed: Datafeed,
        fc: Option<&mut Vec<f64>>,
        fcstart: Option<usize>,
        fcend: Option<usize>,
        fcinitval: Option<f64>,
        p: Option<&mut Vec<f64>>,
        pstart: Option<usize>,
        pend: Option<usize>,
        pinitval: Option<f64>,
        state_step: usize,
        forecast_len: usize,
    ) -> Result<Self, Error> {
        let data_len = datafeed.len();
        let ll0 = LL0::new_with_params(
            datafeed,
            fc,
            fcstart,
            fcend,
            fcinitval,
            p,
            pstart,
            pend,
            pinitval,
            state_step,
            forecast_len,
        )?;
        Ok(Self {
            ll0,
            chunked: false,
            chunks: vec![],
            data_len,
        })
    }

    /// Create LL model, automatically handles large dataset chunking
    ///
    /// # Chunking Strategy
    /// - Data length <= 2000: Use LL0 directly
    /// - Data length > 2000: Automatically chunk, at most 2000 points per chunk
    ///
    /// # Errors
    ///
    /// Returns an error if the data is invalid or insufficient.
    pub fn new(datafeed: Datafeed, forecast_len: usize) -> Result<Self, Error> {
        const SUBLEN: usize = 2000; // Consistent with Python implementation: 2000 points per chunk
        let data_len = datafeed.len();

        // If data length is less than or equal to SUBLEN, use LL0 directly
        if data_len <= SUBLEN {
            let ll0 = LL0::new(datafeed, forecast_len)?;
            return Ok(Self {
                ll0,
                chunked: false,
                chunks: vec![],
                data_len,
            });
        }

        // Chunk large dataset
        let divide = (data_len / SUBLEN).max(1);
        let chunk_size = data_len / divide;
        let mut chunks = Vec::new();
        let start = datafeed.get_start();
        let end = datafeed.get_end();
        let step = datafeed.get_step();

        // Create chunked models
        for i in 0..divide {
            let chunk_start = start + i * chunk_size * step;
            let chunk_end = if i == divide - 1 {
                end
            } else {
                start + (i + 1) * chunk_size * step
            };

            let chunk_datafeed =
                datafeed.clone_with_params(Some(chunk_start), Some(chunk_end), Some(step));

            // Each chunk's forecast length needs to be calculated based on remaining data
            let chunk_forecast_len = if i == divide - 1 {
                forecast_len
            } else {
                0 // Intermediate chunks don't need forecasting
            };

            let chunk_ll0 = LL0::new(chunk_datafeed, chunk_forecast_len)?;
            chunks.push(chunk_ll0);
        }

        // Create an empty LL0 as placeholder (for compatibility)
        let empty_datafeed = Datafeed::new(vec![0.0], 0, 1, 1);
        let ll0 = LL0::new(empty_datafeed, forecast_len)?;

        Ok(Self {
            ll0,
            chunked: true,
            chunks,
            data_len,
        })
    }

    pub fn state(&self, i: usize) -> f64 {
        if !self.chunked {
            return self.ll0.state(i);
        }

        // In chunked case, find corresponding chunk
        const SUBLEN: usize = 2000;
        let divide = (self.data_len / SUBLEN).max(1);
        let chunk_size = self.data_len / divide;
        let chunk_idx = (i / chunk_size).min(divide - 1);

        if chunk_idx < self.chunks.len() {
            // Calculate index within chunk
            let local_idx = i % chunk_size;
            self.chunks[chunk_idx].state(local_idx)
        } else {
            // Out of data range, use last chunk
            if let Some(last_chunk) = self.chunks.last() {
                last_chunk.state(i - (divide - 1) * chunk_size)
            } else {
                0.0
            }
        }
    }

    pub fn var(&self, i: usize) -> f64 {
        if !self.chunked {
            return self.ll0.var(i);
        }

        // In chunked case, find corresponding chunk
        const SUBLEN: usize = 2000;
        let divide = (self.data_len / SUBLEN).max(1);
        let chunk_size = self.data_len / divide;
        let chunk_idx = (i / chunk_size).min(divide - 1);

        if chunk_idx < self.chunks.len() {
            // Calculate index within chunk
            let local_idx = i % chunk_size;
            self.chunks[chunk_idx].var(local_idx)
        } else {
            // Out of data range, use last chunk
            if let Some(last_chunk) = self.chunks.last() {
                last_chunk.var(i - (divide - 1) * chunk_size)
            } else {
                0.0
            }
        }
    }

    // Get fc array value (for LLP2 and other scenarios that need direct array access)
    pub fn get_fc(&self, i: usize) -> Option<f64> {
        if !self.chunked {
            if i < self.ll0.fc.len() {
                let val = self.ll0.fc[i];
                if val.is_nan() { None } else { Some(val) }
            } else {
                None
            }
        } else {
            // In chunked case, use state method
            let val = self.state(i);
            if val.is_nan() || val == 0.0 && i >= self.data_len {
                None
            } else {
                Some(val)
            }
        }
    }

    // Get p array value (for LLP2 and other scenarios that need direct array access)
    // Note: This returns the p array value, excluding sigma (consistent with Python's self.model1.p[i])
    pub fn get_p(&self, i: usize) -> Option<f64> {
        if !self.chunked {
            if i < self.ll0.p.len() {
                let val = self.ll0.p[i];
                if val.is_nan() { None } else { Some(val) }
            } else {
                None
            }
        } else {
            // In chunked case, find corresponding chunk
            const SUBLEN: usize = 2000;
            let divide = (self.data_len / SUBLEN).max(1);
            let chunk_size = self.data_len / divide;
            let chunk_idx = (i / chunk_size).min(divide - 1);

            if chunk_idx < self.chunks.len() {
                let local_idx = i % chunk_size;
                if local_idx < self.chunks[chunk_idx].p.len() {
                    let val = self.chunks[chunk_idx].p[local_idx];
                    if val.is_nan() { None } else { Some(val) }
                } else {
                    None
                }
            } else {
                None
            }
        }
    }

    pub fn datalen(&self) -> usize {
        self.data_len
    }
}

/// Local Linear Trend model (LLT)
pub struct LLT {
    datafeed: Datafeed,
    fc: Vec<f64>,
    p: Vec<f64>,
    trend: f64,
    zeta: f64,
    forecast_len: usize,
    var: f64,
    cur_state_end: usize,
    state_step: usize,
}

impl LLT {
    /// Create a new Local Linear Trend (LLT) model
    ///
    /// # Errors
    ///
    /// Returns an error if the data has fewer than 2 data points.
    pub fn new(datafeed: Datafeed, forecast_len: usize) -> Result<Self, Error> {
        if datafeed.len() < 2 {
            bail!(
                "Insufficient data: LLT model requires at least 2 data points for trend estimation (required: 2, actual: {})",
                datafeed.len()
            );
        }

        let data_len = datafeed.len();
        let mut llt = Self {
            datafeed,
            fc: vec![0.0; data_len + forecast_len],
            p: vec![0.0; data_len + forecast_len],
            trend: 0.0,
            zeta: 0.0,
            forecast_len,
            var: 0.0,
            cur_state_end: 0,
            state_step: 1,
        };

        llt.compute_states();
        Ok(llt)
    }

    fn compute_states(&mut self) {
        let mut psi = vec![1.0];
        let mut func = |x: &[f64]| {
            let zeta = x[0].exp();
            -self.llh(zeta)
        };
        // dfpmin will modify the passed vector
        let (_, _) = dfpmin(&mut func, &mut psi, DFP_TOLERANCE);
        self.zeta = psi[0].exp();
        self.update_all(self.zeta);

        // Calculate predictions
        // Python code uses cur_state_end
        let fcend = self.cur_state_end;
        let pend = self.cur_state_end;
        let fclast = fcend - self.state_step;
        let plast = pend - self.state_step;
        for i in (0..(self.forecast_len * self.state_step)).step_by(self.state_step) {
            if fcend + i < self.fc.len() {
                // Python: self.fc[fcend+i] = self.fc[fclast+i] + self.trend
                // Note: In Python code, fclast+i may be out of bounds, but here we assume it won't be
                if fclast + i < self.fc.len() {
                    self.fc[fcend + i] = self.fc[fclast + i] + self.trend;
                } else {
                    // If out of bounds, use fclast value plus trend
                    self.fc[fcend + i] = self.fc[fclast] + self.trend;
                }
            }
            if pend + i < self.p.len() {
                // Python: self.p[Pend+i] = self.p[Plast+i] + self.zeta
                if plast + i < self.p.len() {
                    self.p[pend + i] = self.p[plast + i] + self.zeta;
                } else {
                    // If out of bounds, use last value
                    self.p[pend + i] = self.p[plast] + self.zeta;
                }
            }
        }
    }

    fn update_kalman(
        &self,
        y: f64,
        a: &mut [f64],
        p: &mut [f64],
        k: &mut [f64],
        zeta: f64,
    ) -> (f64, f64, f64) {
        // Handle missing values
        if y.is_nan() {
            a[0] += a[1]; // a[1] remains the same
            p[0] += zeta;
            let f = p[0] + 1.0;
            return (0.0, f, f.ln());
        }
        let x1 = p[0] + p[2];
        let x2 = p[1] + p[3];
        let f = p[0] + 1.0;
        k[0] = x1 / f;
        k[1] = p[2] / f;
        let v = y - a[0] - a[1];
        a[0] += a[1] + v * k[0];
        a[1] += v * k[1];
        let l0 = 1.0 - k[0];
        // Save p2 and p3 values first, because updating p[2] requires the old p[2] value
        let p2 = p[2];
        let p3 = p[3];
        p[0] = x1 * l0 + x2 + zeta;
        p[1] = x2 - x1 * k[1];
        p[2] = p2 * l0 + p3;
        p[3] = p3 - p2 * k[1];
        (v, f, f.ln())
    }

    fn llh(&self, zeta: f64) -> f64 {
        let mut datafeed = self.datafeed.clone();
        datafeed.reset();
        let pt1 = datafeed.next().unwrap();
        let pt2 = datafeed.next().unwrap();
        let mut a = vec![pt2, pt2 - pt1];
        let mut p = vec![5.0 + 2.0 * zeta, 3.0 + zeta, 3.0 + zeta, 2.0 + zeta];
        let f = p[0] + 1.0;
        let _lf = f.ln();
        let mut k = vec![(p[0] + p[2]) / f, p[2] / f];
        let mut t1 = 0.0;
        let mut t2 = 0.0;

        for x in datafeed {
            let (v, f, lf_val) = self.update_kalman(x, &mut a, &mut p, &mut k, zeta);
            t1 += v * v / f;
            t2 += lf_val;
        }

        if t1 == 0.0 {
            -t2
        } else {
            -(self.datafeed.len() as f64 - 2.0) * t1.ln() - t2
        }
    }

    fn update_all(&mut self, zeta: f64) {
        let mut datafeed = self.datafeed.clone();
        datafeed.reset();
        let pt1 = datafeed.next().unwrap();
        let pt2 = datafeed.next().unwrap();
        let mut a = vec![pt2, pt2 - pt1];
        let mut p = vec![5.0 + 2.0 * zeta, 3.0 + zeta, 3.0 + zeta, 2.0 + zeta];
        let f = p[0] + 1.0;
        let mut k = vec![(p[0] + p[2]) / f, p[2] / f];
        let mut epsilon = 0.0;

        // In Python code, self.epsilon = 1.0 is set first, but here we calculate it later
        let fcstart = 0;
        let pstart = 0;
        // Note: In Python code, initial values are set at the end, but here we set temporary values first
        // Actually in Python code, self.fc[fcstart] and self.p[pstart] are set at the end

        let mut i = fcstart + 2 * self.state_step;
        let b = i;
        for x in datafeed {
            let (v, f, _) = self.update_kalman(x, &mut a, &mut p, &mut k, zeta);
            epsilon += v * v / f;
            if i < self.fc.len() {
                self.fc[i] = a[0];
            }
            if i < self.p.len() {
                self.p[i] = p[0] + 1.0;
            }
            i += self.state_step;
        }

        // Calculate epsilon (In Python code, epsilon=1.0 is set first, then calculated)
        if self.datafeed.len() > 2 {
            epsilon /= (self.datafeed.len() - 2) as f64;
        }

        self.zeta = zeta * epsilon;
        self.trend = a[1];
        self.var = (p[0] + 1.0) * epsilon;

        // In Python code, initial values are set at the end
        self.fc[fcstart] = pt1;
        self.p[pstart] = epsilon * (1.0 + zeta);
        self.fc[fcstart + self.state_step] = pt2;
        self.p[pstart + self.state_step] = epsilon * (5.0 + 2.0 * zeta);

        // Then update subsequent values
        for j in (b..i).step_by(self.state_step) {
            if j < self.p.len() {
                self.p[j] = (self.p[j] + 1.0) * epsilon;
            }
        }
        self.cur_state_end = i;
    }

    pub fn state(&self, i: usize) -> f64 {
        if i < self.fc.len() { self.fc[i] } else { 0.0 }
    }

    pub fn var(&self, i: usize) -> f64 {
        let n = self.datafeed.get_end();
        if i < n - 2 {
            if i < self.p.len() { self.p[i] } else { 0.0 }
        } else {
            self.var + ((i as i32 - n as i32 + 3) as f64) * self.zeta
        }
    }

    pub fn datalen(&self) -> usize {
        self.datafeed.len()
    }
}

/// Univariate model wrapper
pub struct Univar {
    algos: Vec<Box<dyn StateSpaceModel>>,
    data_len: usize,
    period: i32,
}

// StateSpaceModel trait is defined in models/mod.rs

impl StateSpaceModel for LL {
    fn state(&self, i: usize) -> f64 {
        self.state(i)
    }

    fn var(&self, i: usize) -> f64 {
        self.var(i)
    }

    fn p(&self, i: usize) -> f64 {
        // Directly return p[i], consistent with Python's self.algos[ts_idx].p[i]
        if let Some(p_val) = self.get_p(i) {
            p_val
        } else {
            f64::NAN
        }
    }

    fn datalen(&self) -> usize {
        self.datalen()
    }

    fn least_num_data(&self) -> usize {
        1
    }

    fn first_forecast_index(&self) -> usize {
        0
    }
}

impl StateSpaceModel for LLT {
    fn state(&self, i: usize) -> f64 {
        self.state(i)
    }

    fn var(&self, i: usize) -> f64 {
        self.var(i)
    }

    fn p(&self, i: usize) -> f64 {
        get_p_value(&self.p, i)
    }

    fn datalen(&self) -> usize {
        self.datalen()
    }

    fn least_num_data(&self) -> usize {
        2
    }

    fn first_forecast_index(&self) -> usize {
        0
    }
}

impl Univar {
    /// Create a new Univar model
    ///
    /// # Errors
    ///
    /// Returns an error if the algorithm is unsupported or data is invalid.
    pub fn new(
        algorithm: &str,
        data: Vec<Vec<f64>>,
        data_start: usize,
        data_end: usize,
        period: i32,
        forecast_len: usize,
    ) -> Result<Self, Error> {
        Self::new_with_missing(
            algorithm,
            data,
            data_start,
            data_end,
            period,
            forecast_len,
            false, // missingValued
            None,  // correlate
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_missing(
        algorithm: &str,
        data: Vec<Vec<f64>>,
        data_start: usize,
        data_end: usize,
        period: i32,
        forecast_len: usize,
        missing_valued: bool,
        _correlate: Option<&[f64]>,
    ) -> Result<Self, Error> {
        let data_len = data_end - data_start;
        if period > MAX_LAG as i32 {
            bail!(
                "Invalid parameter 'period': {} - period cannot be greater than {}",
                period,
                MAX_LAG
            );
        }

        if !is_supported_algorithm(algorithm) {
            let _supported_univariate = [
                "LL".to_string(),
                "LLT".to_string(),
                "LLP".to_string(),
                "LLP1".to_string(),
                "LLP2".to_string(),
                "LLP5".to_string(),
            ];
            bail!(
                "Unsupported algorithm {}. For multivariate algorithms (LLB, LLBmv, BiLL, BiLLmv), use Multivar instead",
                algorithm
            );
        }

        let mut algos: Vec<Box<dyn StateSpaceModel>> = vec![];

        if algorithm.starts_with("LLP") {
            for row in &data {
                let mut datafeed = Datafeed::new(row.clone(), data_start, data_end, 1);
                if missing_valued {
                    datafeed = datafeed.with_missing_values(true);
                }
                let model: Box<dyn StateSpaceModel> = match algorithm {
                    "LLP" => Box::new(LLP::new(datafeed, period, forecast_len)?),
                    "LLP1" => Box::new(LLP1::new(datafeed, period, forecast_len)?),
                    "LLP2" => Box::new(LLP2::new(datafeed, period, forecast_len)?),
                    "LLP5" => Box::new(LLP5::new(datafeed, period, forecast_len)?),
                    _ => {
                        // Multivariate algorithms cannot be used through Univar
                        if is_multivariate_algorithm(algorithm) {
                            bail!(
                                "Algorithm {} is a multivariate, but was used as a univariate",
                                algorithm
                            );
                        }
                        bail!("Unknown LLP algorithm: {}", algorithm);
                    }
                };
                algos.push(model);
            }
        } else {
            for row in &data {
                let mut datafeed = Datafeed::new(row.clone(), data_start, data_end, 1);
                if missing_valued {
                    datafeed = datafeed.with_missing_values(true);
                }
                let model: Box<dyn StateSpaceModel> = match algorithm {
                    "LL" => Box::new(LL::new(datafeed, forecast_len)?),
                    "LLT" => Box::new(LLT::new(datafeed, forecast_len)?),
                    _ => {
                        // Multivariate algorithms cannot be used through Univar, as they require different parameters
                        if is_multivariate_algorithm(algorithm) {
                            bail!(
                                "Algorithm {} is a multivariate, but was used as a univariate",
                                algorithm
                            );
                        }
                        bail!("Unknown or unsupported algorithm: {}", algorithm);
                    }
                };
                algos.push(model);
            }
        }

        Ok(Self {
            algos,
            data_len,
            period,
        })
    }

    pub fn state(&self, ts_idx: usize, i: usize) -> f64 {
        if ts_idx < self.algos.len() {
            self.algos[ts_idx].state(i)
        } else {
            0.0
        }
    }

    pub fn var(&self, ts_idx: usize, i: usize) -> f64 {
        // Consistent with Python implementation: return p[i] instead of variance(i)
        // Python: return self.algos[ts_idx].p[i]
        if ts_idx < self.algos.len() {
            self.algos[ts_idx].p(i)
        } else {
            0.0
        }
    }

    pub fn datalen(&self) -> usize {
        self.data_len
    }

    pub fn period(&self) -> i32 {
        self.period
    }

    pub fn multivariate(&self) -> bool {
        false
    }

    pub fn least_num_data(&self) -> usize {
        if !self.algos.is_empty() {
            self.algos[0].least_num_data()
        } else {
            1
        }
    }

    pub fn first_forecast_index(&self) -> usize {
        if !self.algos.is_empty() {
            self.algos[0].first_forecast_index()
        } else {
            0
        }
    }
}
/// Periodic Local Level model (LLP)
pub struct LLP {
    datafeed: Datafeed,
    fc: Vec<f64>,
    p: Vec<f64>,
    models: Vec<LL>,
    period: usize,
    data_len: usize,
}

impl LLP {
    /// Create a new Periodic Local Level (LLP) model
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The detected period is less than 2
    /// - The data length is less than the period
    pub fn new(datafeed: Datafeed, period: i32, forecast_len: usize) -> Result<Self, Error> {
        let data_len = datafeed.len();
        let period = if period >= 2 {
            period as usize
        } else {
            let detected = datafeed.period();
            if detected < 2 {
                bail!("Period detection failed: LLP model requires periodic data with period >= 2");
            }
            detected as usize
        };

        if data_len < period {
            bail!(
                "Insufficient data: LLP model requires at least {} data points for period {} (required: {}, actual: {})",
                period,
                period,
                period,
                data_len
            );
        }

        let forecast_len = forecast_len.max(period);
        // Initialize fc and p arrays to NaN (representing None, consistent with Python)
        let mut llp = Self {
            datafeed: datafeed.clone(),
            fc: vec![f64::NAN; data_len + forecast_len],
            p: vec![f64::NAN; data_len + forecast_len],
            models: vec![],
            period,
            data_len,
        };

        llp.set_models()?;
        Ok(llp)
    }

    fn set_models(&mut self) -> Result<(), Error> {
        // Fill missing values in first period
        for i in 0..self.period.min(self.data_len) {
            if self.datafeed.get_val(i).is_nan() {
                let mut j = i + 1;
                while j < self.data_len && self.datafeed.get_val(j).is_nan() {
                    j += 1;
                }
                let denom = (j - i + 1) as f64;
                let right = if j < self.data_len {
                    self.datafeed.get_val(j) / denom
                } else {
                    self.datafeed.get_val(i - 1) / denom
                };
                let left = if i > 0 {
                    self.datafeed.get_val(i - 1) / denom
                } else {
                    right
                };
                let mut w1 = (denom - 1.0) * left;
                let mut w2 = right;
                for k in i..j {
                    self.datafeed.set_val(k, w1 + w2);
                    w1 -= left;
                    w2 += right;
                }
            }
        }

        let start = self.datafeed.get_start();
        let end = self.datafeed.get_end();
        self.models = Vec::new();

        for i in 0..self.period {
            let model_start = start + i;
            let mut model_end = end - ((end - i) % self.period);
            if model_end < end {
                model_end += self.period;
            }
            let datafeed_clone = self.datafeed.clone_with_params(
                Some(model_start),
                Some(model_end),
                Some(self.period),
            );
            let diff = self.fc.len() - model_end;
            let model_forecast_len = diff / self.period;
            let model_forecast_len = if !diff.is_multiple_of(self.period) {
                model_forecast_len + 1
            } else {
                model_forecast_len
            };

            // Create LL model with shared fc and p arrays (consistent with Python implementation)
            // Note: fcend and pend should be self.fc.len() so that LL model can fill all prediction values
            let fcend = self.fc.len();
            let pend = self.p.len();
            let model = LL::new_with_params(
                datafeed_clone,
                Some(&mut self.fc),
                Some(model_start),
                Some(fcend),
                None, // fcinitval
                Some(&mut self.p),
                Some(model_start),
                Some(pend),
                None, // Pinitval
                self.period,
                model_forecast_len,
            )?;
            self.models.push(model);
        }

        Ok(())
    }

    pub fn least_num_data(&self) -> usize {
        self.period * LL::least_num_data()
    }

    pub fn first_forecast_index(&self) -> usize {
        self.models
            .iter()
            .enumerate()
            .map(|(i, m)| i + m.first_forecast_index() * self.period)
            .min()
            .unwrap_or(0)
    }

    pub fn variance(&self, i: usize) -> f64 {
        // Call LL model's variance method (consistent with Python implementation)
        // Python: return self.models[i%self.period].variance(old_div(i,self.period))
        let model_idx = i % self.period;
        let period_idx = i / self.period;
        if model_idx < self.models.len() {
            self.models[model_idx].var(period_idx)
        } else {
            f64::NAN
        }
    }

    pub fn datalen(&self) -> usize {
        self.data_len
    }

    pub fn state(&self, i: usize) -> f64 {
        // Get state from fc array (consistent with Python implementation)
        if i < self.fc.len() {
            let val = self.fc[i];
            // If value is NaN, return NaN (representing None)
            if val.is_nan() { f64::NAN } else { val }
        } else {
            f64::NAN
        }
    }

    pub fn var(&self, i: usize) -> f64 {
        self.variance(i)
    }
}

impl StateSpaceModel for LLP {
    fn state(&self, i: usize) -> f64 {
        self.state(i)
    }

    fn var(&self, i: usize) -> f64 {
        self.var(i)
    }

    fn p(&self, i: usize) -> f64 {
        get_p_value(&self.p, i)
    }

    fn datalen(&self) -> usize {
        self.datalen()
    }

    fn least_num_data(&self) -> usize {
        self.least_num_data()
    }

    fn first_forecast_index(&self) -> usize {
        self.first_forecast_index()
    }
}

impl LL {
    pub fn least_num_data() -> usize {
        1
    }

    pub fn first_forecast_index() -> usize {
        0
    }
}

/// LLP1 - Variant of LLP, variance takes maximum value
pub struct LLP1 {
    llp: LLP,
}

impl LLP1 {
    /// Create a new LLP1 model (variant of LLP with maximum variance)
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying LLP model creation fails.
    pub fn new(datafeed: Datafeed, period: i32, forecast_len: usize) -> Result<Self, Error> {
        let mut llp = LLP::new(datafeed, period, forecast_len)?;
        // LLP1's compute_states: set self.p[i] = max(self.p[i-1], self.p[i])
        for i in 1..llp.p.len() {
            let prev = if !llp.p[i - 1].is_nan() {
                llp.p[i - 1]
            } else {
                f64::NEG_INFINITY
            };
            let curr = if !llp.p[i].is_nan() {
                llp.p[i]
            } else {
                f64::NEG_INFINITY
            };
            if prev > curr && !prev.is_infinite() {
                llp.p[i] = prev;
            } else if !curr.is_infinite() {
                llp.p[i] = curr;
            }
        }
        Ok(Self { llp })
    }
}

impl LLP {
    pub fn compute_states(&mut self) {
        // LLP1 overrides this method: update variance to maximum value
        for i in 1..self.p.len() {
            self.p[i] = self.p[i - 1].max(self.p[i]);
        }
    }
}

impl StateSpaceModel for LLP1 {
    fn state(&self, i: usize) -> f64 {
        self.llp.state(i)
    }

    fn var(&self, i: usize) -> f64 {
        self.llp.var(i)
    }

    fn p(&self, i: usize) -> f64 {
        // Directly return p[i], consistent with Python's self.algos[ts_idx].p[i]
        self.llp.p(i)
    }

    fn datalen(&self) -> usize {
        self.llp.datalen()
    }

    fn least_num_data(&self) -> usize {
        self.llp.least_num_data()
    }

    fn first_forecast_index(&self) -> usize {
        self.llp.first_forecast_index()
    }
}

/// LLP2 - Combines LL and LLP
pub struct LLP2 {
    model1: LL,
    model2: LLP,
    fc: Vec<f64>,
    p: Vec<f64>,
    data_len: usize,
}

impl LLP2 {
    /// Create a new LLP2 model (combines LL and LLP)
    ///
    /// # Errors
    ///
    /// Returns an error if the period is less than 2.
    pub fn new(datafeed: Datafeed, period: i32, forecast_len: usize) -> Result<Self, Error> {
        if period < 2 {
            bail!(
                "Invalid parameter 'period': {} - period must be >= 2 for LLP2 model",
                period
            );
        }
        let period = period as usize;
        let data_len = datafeed.len();
        let model1 = LL::new(datafeed.clone(), forecast_len)?;
        let model2 = LLP::new(datafeed, period as i32, forecast_len)?;
        let fc_len = data_len + forecast_len;
        let mut llp2 = Self {
            model1,
            model2,
            fc: vec![0.0; fc_len],
            p: vec![0.0; fc_len],
            data_len,
        };
        llp2.compute_states();
        Ok(llp2)
    }

    fn compute_states(&mut self) {
        for i in 0..self.fc.len() {
            self.combine(i);
        }
    }

    fn combine(&mut self, i: usize) {
        // Get fc and p values from model1 (direct array access, consistent with Python)
        let fc1_val = self.model1.get_fc(i);
        let p1_val = self.model1.get_p(i);

        // Get fc and p values from model2 (direct array access, consistent with Python)
        let fc2_val = if i < self.model2.fc.len() {
            let val = self.model2.fc[i];
            if val.is_nan() { None } else { Some(val) }
        } else {
            None
        };
        let p2_val = if i < self.model2.p.len() {
            let val = self.model2.p[i];
            if val.is_nan() { None } else { Some(val) }
        } else {
            None
        };

        // Consistent with Python implementation: check if None
        if fc1_val.is_none() || p1_val.is_none() {
            if let (Some(fc2), Some(p2)) = (fc2_val, p2_val) {
                self.p[i] = p2;
                self.fc[i] = fc2;
            } else {
                // If both are None, keep NaN
                self.p[i] = f64::NAN;
                self.fc[i] = f64::NAN;
            }
        } else if fc2_val.is_none() || p2_val.is_none() {
            if let (Some(fc1), Some(p1)) = (fc1_val, p1_val) {
                self.p[i] = p1;
                self.fc[i] = fc1;
            } else {
                self.p[i] = f64::NAN;
                self.fc[i] = f64::NAN;
            }
        } else {
            // Both pairs have Some values
            let (fc1, p1, fc2, p2) = match (fc1_val, p1_val, fc2_val, p2_val) {
                (Some(fc1), Some(p1), Some(fc2), Some(p2)) => (fc1, p1, fc2, p2),
                _ => {
                    self.p[i] = f64::NAN;
                    self.fc[i] = f64::NAN;
                    return;
                }
            };

            if p1 == 0.0 && p2 == 0.0 {
                self.p[i] = 0.0;
                self.fc[i] = (fc1 + fc2) / 2.0;
            } else {
                let k = p1 / (p1 + p2);
                self.fc[i] = fc1 + k * (fc2 - fc1);
                self.p[i] = (1.0 - k) * p1;
            }
        }
    }

    pub fn first_forecast_index(&self) -> usize {
        self.model1
            .first_forecast_index()
            .min(self.model2.first_forecast_index())
    }

    pub fn variance(&self, i: usize) -> f64 {
        if i < self.p.len() { self.p[i] } else { 0.0 }
    }

    pub fn state(&self, i: usize) -> f64 {
        if i < self.fc.len() { self.fc[i] } else { 0.0 }
    }

    pub fn var(&self, i: usize) -> f64 {
        self.variance(i)
    }

    pub fn datalen(&self) -> usize {
        self.data_len
    }
}

impl StateSpaceModel for LLP2 {
    fn state(&self, i: usize) -> f64 {
        self.state(i)
    }

    fn var(&self, i: usize) -> f64 {
        self.var(i)
    }

    fn p(&self, i: usize) -> f64 {
        get_p_value(&self.p, i)
    }

    fn datalen(&self) -> usize {
        self.datalen()
    }

    fn least_num_data(&self) -> usize {
        self.model2.least_num_data()
    }

    fn first_forecast_index(&self) -> usize {
        self.first_forecast_index()
    }
}

/// LLP5 - Combines LLT and LLP1
pub struct LLP5 {
    model1: LLT,
    model2: Option<LLP1>,
    fc: Vec<f64>,
    p: Vec<f64>,
    data_len: usize,
}

impl LLP5 {
    /// Create a new LLP5 model (combines LLT and LLP1)
    ///
    /// # Errors
    ///
    /// Returns an error if model creation fails.
    pub fn new(datafeed: Datafeed, period: i32, forecast_len: usize) -> Result<Self, Error> {
        let data_len = datafeed.len();
        let model1 = LLT::new(datafeed.clone(), forecast_len)?;
        let mut period = period;
        let model2 = if period < 2 {
            let detected = datafeed.period();
            if detected >= 2 && data_len >= (detected as usize) * LL::least_num_data() {
                period = detected;
                Some(LLP1::new(datafeed, period, forecast_len)?)
            } else {
                None
            }
        } else if data_len >= (period as usize) * LL::least_num_data() {
            Some(LLP1::new(datafeed, period, forecast_len)?)
        } else {
            None
        };

        let fc_len = data_len + forecast_len;
        let mut llp5 = Self {
            model1,
            model2,
            fc: vec![0.0; fc_len],
            p: vec![0.0; fc_len],
            data_len,
        };
        llp5.compute_states();
        Ok(llp5)
    }

    fn compute_states(&mut self) {
        let first_idx = self.model1.first_forecast_index();
        for i in first_idx..self.fc.len() {
            self.combine(i);
        }
    }

    fn combine(&mut self, i: usize) {
        // Get fc and p values from model1 (direct array access, consistent with Python)
        let fc1_val = if i < self.model1.fc.len() {
            let val = self.model1.fc[i];
            if val.is_nan() { None } else { Some(val) }
        } else {
            None
        };
        let p1_val = if i < self.model1.p.len() {
            let val = self.model1.p[i];
            if val.is_nan() { None } else { Some(val) }
        } else {
            None
        };

        // Get fc and p values from model2 (direct array access, consistent with Python)
        if let Some(ref m2) = self.model2 {
            let p2_val = if i < m2.llp.p.len() {
                let val = m2.llp.p[i];
                if val.is_nan() { None } else { Some(val) }
            } else {
                None
            };
            let fc2_val = if i < m2.llp.fc.len() {
                let val = m2.llp.fc[i];
                if val.is_nan() { None } else { Some(val) }
            } else {
                None
            };

            if p1_val.is_none() || fc1_val.is_none() {
                if let (Some(fc2), Some(p2)) = (fc2_val, p2_val) {
                    self.p[i] = p2;
                    self.fc[i] = fc2;
                } else {
                    self.p[i] = f64::NAN;
                    self.fc[i] = f64::NAN;
                }
            } else if p2_val.is_none() || fc2_val.is_none() {
                if let (Some(fc1), Some(p1)) = (fc1_val, p1_val) {
                    self.p[i] = p1;
                    self.fc[i] = fc1;
                } else {
                    self.p[i] = f64::NAN;
                    self.fc[i] = f64::NAN;
                }
            } else {
                // Both pairs have Some values
                let (fc1, p1, fc2, p2) = match (fc1_val, p1_val, fc2_val, p2_val) {
                    (Some(fc1), Some(p1), Some(fc2), Some(p2)) => (fc1, p1, fc2, p2),
                    _ => {
                        self.p[i] = f64::NAN;
                        self.fc[i] = f64::NAN;
                        return;
                    }
                };

                if p1 == 0.0 && p2 == 0.0 {
                    self.p[i] = 0.0;
                    self.fc[i] = (fc1 + fc2) / 2.0;
                } else {
                    let k = p1 / (p1 + p2);
                    self.fc[i] = fc1 + k * (fc2 - fc1);
                    self.p[i] = (1.0 - k) * p1;
                }
            }
        } else if let (Some(fc1), Some(p1)) = (fc1_val, p1_val) {
            self.p[i] = p1;
            self.fc[i] = fc1;
        } else {
            self.p[i] = f64::NAN;
            self.fc[i] = f64::NAN;
        }
    }

    pub fn first_forecast_index(&self) -> usize {
        self.model1.first_forecast_index()
    }

    pub fn variance(&self, i: usize) -> f64 {
        if i < self.p.len() { self.p[i] } else { 0.0 }
    }

    pub fn state(&self, i: usize) -> f64 {
        if i < self.fc.len() { self.fc[i] } else { 0.0 }
    }

    pub fn var(&self, i: usize) -> f64 {
        self.variance(i)
    }

    pub fn datalen(&self) -> usize {
        self.data_len
    }
}

impl StateSpaceModel for LLP5 {
    fn state(&self, i: usize) -> f64 {
        self.state(i)
    }

    fn var(&self, i: usize) -> f64 {
        self.var(i)
    }

    fn p(&self, i: usize) -> f64 {
        get_p_value(&self.p, i)
    }

    fn datalen(&self) -> usize {
        self.datalen()
    }

    fn least_num_data(&self) -> usize {
        self.model1.least_num_data()
    }

    fn first_forecast_index(&self) -> usize {
        self.first_forecast_index()
    }
}

/// LLT2 - Local Linear Trend model with non-constant trend
pub struct LLT2 {
    datafeed: Datafeed,
    fc: Vec<f64>,
    p: Vec<f64>,
    trend: Vec<f64>,
    zeta: f64,
    eta: f64,
    forecast_len: usize,
    var: f64,
    nu: f64,
    epsilon: f64,
    cur_state_end: usize,
    state_step: usize,
    fcstart: usize,
    pstart: usize,
}

impl LLT2 {
    pub fn new(mut datafeed: Datafeed, forecast_len: usize) -> Self {
        let data_start = datafeed.get_start();
        let step = datafeed.get_step();
        let i = data_start + step;

        // If second data point is missing, fill it
        if i < datafeed.get_end() && datafeed.get_val(i).is_nan() {
            let mut j = i + step;
            while j < datafeed.get_end() && datafeed.get_val(j).is_nan() {
                j += step;
            }
            let fillval = if j < datafeed.get_end() {
                (datafeed.get_val(data_start) + datafeed.get_val(j)) / 2.0
            } else {
                datafeed.get_val(data_start)
            };
            datafeed.set_val(i, fillval);
        }

        let data_len = datafeed.len();
        let mut llt2 = Self {
            datafeed,
            fc: vec![0.0; data_len + forecast_len],
            p: vec![0.0; data_len + forecast_len],
            trend: vec![0.0; data_len + forecast_len],
            zeta: 0.0,
            eta: 0.0,
            forecast_len,
            var: 0.0,
            nu: 0.0,
            epsilon: 0.0,
            cur_state_end: 0,
            state_step: 1,
            fcstart: 0,
            pstart: 0,
        };

        llt2.compute_states();
        llt2
    }

    fn compute_states(&mut self) {
        self.fcstart = self.datafeed.get_start();
        self.pstart = self.fcstart;
        let mut psi = vec![1.0, 1.0];
        let mut func = |x: &[f64]| {
            let zeta = x[0].exp();
            let eta = x[1].exp();
            -self.llh(zeta, eta)
        };
        let (_, _) = dfpmin(&mut func, &mut psi, DFP_TOLERANCE);
        self.zeta = psi[0].exp();
        self.eta = psi[1].exp();
        self.update_all(self.zeta, self.eta);

        // Calculate predictions
        let fcend = self.cur_state_end;
        let pend = self.cur_state_end;
        let fclast = fcend - self.state_step;
        let plast = pend - self.state_step;
        let tr = self.trend[fclast];
        for i in (0..(self.forecast_len * self.state_step)).step_by(self.state_step) {
            if fcend + i < self.fc.len() {
                self.fc[fcend + i] = self.fc[fclast + i] + tr;
            }
            if pend + i < self.p.len() {
                self.p[pend + i] = self.p[plast + i] + self.var + self.nu;
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn update_kalman(
        &self,
        y: f64,
        a: &mut [f64],
        p: &mut [f64],
        p2: &mut [f64],
        k: &mut [f64],
        zeta: f64,
        eta: f64,
    ) -> (f64, f64, f64) {
        if y.is_nan() {
            let f = p[0] + 1.0;
            a[0] += a[1];
            p2[0] = p[0] + zeta;
            p2[1] = p[1];
            p2[2] = p[2];
            p2[3] = p[3] + eta;
            return (0.0, f, f.ln());
        }

        let x1 = p[0] + p[2];
        let x2 = p[1] + p[3];
        let f = p[0] + 1.0;
        k[0] = x1 / f;
        k[1] = p[2] / f;
        let v = y - a[0];
        a[0] += a[1] + v * k[0];
        a[1] += v * k[1];
        let l0 = 1.0 - k[0];
        p2[0] = x1 * l0 + x2 + zeta;
        p2[1] = x2 - x1 * k[1];
        p2[2] = p[2] * l0 + p[3];
        p2[3] = p[3] - p[2] * k[1] + eta;
        (v, f, f.ln())
    }

    fn steady_update(&self, y: f64, a: &mut [f64], k: &[f64]) -> f64 {
        if y.is_nan() {
            a[0] += a[1];
            return 0.0;
        }
        let v = y - a[0];
        a[0] += a[1] + v * k[0];
        a[1] += v * k[1];
        v
    }

    fn llh(&self, zeta: f64, eta: f64) -> f64 {
        let mut datafeed = self.datafeed.clone();
        datafeed.reset();
        let pt1 = datafeed.next().unwrap();
        let pt2 = datafeed.next().unwrap();
        let mut a = vec![2.0 * pt2 - pt1, pt2 - pt1];
        let mut p = vec![
            5.0 + 2.0 * zeta + eta,
            3.0 + zeta + eta,
            3.0 + zeta + eta,
            2.0 + zeta + 2.0 * eta,
        ];
        let mut p2 = vec![0.0; 4];
        let f = p[0] + 1.0;
        let lf = f.ln();
        let mut k = vec![(p[0] + p[2]) / f, p[2] / f];
        let mut t1 = 0.0;
        let mut t2 = 0.0;
        let mut steady = false;

        for y in datafeed {
            if !steady {
                let (v, f_val, lf_val) =
                    self.update_kalman(y, &mut a, &mut p, &mut p2, &mut k, zeta, eta);
                let mut norm = 0.0;
                for i in 0..4 {
                    norm += (p2[i] - p[i]).abs();
                }
                if norm < 0.001 {
                    steady = true;
                }
                p[..4].copy_from_slice(&p2[..4]);
                t1 += v * v / f_val;
                t2 += lf_val;
            } else {
                let v = self.steady_update(y, &mut a, &k);
                t1 += v * v / f;
                t2 += lf;
            }
        }

        if t1 == 0.0 {
            -t2
        } else {
            -(self.datafeed.len() as f64 - 2.0) * t1.ln() - t2
        }
    }

    fn update_all(&mut self, zeta: f64, eta: f64) {
        let mut datafeed = self.datafeed.clone();
        datafeed.reset();
        let pt1 = datafeed.next().unwrap();
        let pt2 = datafeed.next().unwrap();
        let mut a = vec![2.0 * pt2 - pt1, pt2 - pt1];
        let mut p = vec![
            5.0 + 2.0 * zeta + eta,
            3.0 + zeta + eta,
            3.0 + zeta + eta,
            2.0 + zeta + 2.0 * eta,
        ];
        let mut p2 = vec![0.0; 4];
        let f = p[0] + 1.0;
        let mut k = vec![(p[0] + p[2]) / f, p[2] / f];
        let mut epsilon = 0.0;
        let mut steady = false;
        let mut i = self.fcstart + 2 * self.state_step;
        let b = i;

        self.fc[self.fcstart] = pt1;
        self.trend[self.fcstart] = 0.0;
        self.p[self.pstart] = 1.0 + zeta;
        self.fc[self.fcstart + self.state_step] = pt2;
        self.trend[self.fcstart + self.state_step] = pt2 - pt1;
        self.p[self.pstart + self.state_step] = 5.0 + 2.0 * zeta;

        for y in datafeed.skip(2) {
            if !steady {
                let (v, f_val, _) =
                    self.update_kalman(y, &mut a, &mut p, &mut p2, &mut k, zeta, eta);
                let mut norm = 0.0;
                for j in 0..4 {
                    norm += (p2[j] - p[j]).abs();
                }
                if norm < 0.001 {
                    steady = true;
                }
                p[..4].copy_from_slice(&p2[..4]);
                self.fc[i] = a[0];
                self.trend[i] = a[1];
                epsilon += v * v / f_val;
                self.p[i] = p[0] + 1.0;
            } else {
                let v = self.steady_update(y, &mut a, &k);
                self.fc[i] = a[0];
                self.trend[i] = a[1];
                epsilon += v * v / f;
                self.p[i] = p[0] + 1.0;
            }
            i += self.state_step;
        }

        self.epsilon = epsilon / ((i - b) / self.state_step).max(1) as f64;
        self.zeta = zeta * self.epsilon;
        self.eta = eta * self.epsilon;
        self.var = (p[0] + 1.0) * self.epsilon;
        self.nu = p[3] * self.epsilon;
        for j in (b..i).step_by(self.state_step) {
            if j < self.p.len() {
                self.p[j] = (self.p[j] + 1.0) * self.epsilon;
            }
        }
        self.cur_state_end = i;
    }

    pub fn variance(&self, i: usize) -> f64 {
        let n = self.datafeed.get_end();
        if i < n - 2 {
            if i < self.p.len() { self.p[i] } else { 0.0 }
        } else {
            self.var + ((i as i32 - n as i32 + 3) as f64) * self.zeta
        }
    }

    pub fn state(&self, i: usize) -> f64 {
        if i < self.fc.len() { self.fc[i] } else { 0.0 }
    }

    pub fn var(&self, i: usize) -> f64 {
        self.variance(i)
    }

    pub fn datalen(&self) -> usize {
        self.datafeed.len()
    }
}

impl StateSpaceModel for LLT2 {
    fn state(&self, i: usize) -> f64 {
        self.state(i)
    }

    fn var(&self, i: usize) -> f64 {
        self.var(i)
    }

    fn p(&self, i: usize) -> f64 {
        get_p_value(&self.p, i)
    }

    fn datalen(&self) -> usize {
        self.datalen()
    }

    fn least_num_data(&self) -> usize {
        2
    }

    fn first_forecast_index(&self) -> usize {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::Datafeed;

    #[test]
    fn test_ll0_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let df = Datafeed::new(data, 0, 5, 1);
        let ll0 = LL0::new(df, 3).unwrap();

        // Test basic functionality
        assert!(ll0.state(0).is_finite());
        assert!(ll0.var(0) >= 0.0);
        assert_eq!(ll0.datalen(), 5);
    }

    #[test]
    fn test_ll_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let df = Datafeed::new(data, 0, 5, 1);
        let ll = LL::new(df, 3).expect("Failed to create LL model");

        assert!(ll.state(0).is_finite());
        assert!(ll.var(0) >= 0.0);
        assert_eq!(ll.datalen(), 5);
    }

    #[test]
    fn test_ll_large_dataset() {
        // Test large dataset chunking functionality
        let data: Vec<f64> = (0..2500).map(|i| i as f64).collect();
        let df = Datafeed::new(data, 0, 2500, 1);
        let ll = LL::new(df, 3).expect("Failed to create LL model");

        assert!(ll.state(0).is_finite());
        assert!(ll.state(1000).is_finite());
        assert!(ll.state(2000).is_finite());
        assert_eq!(ll.datalen(), 2500);
    }

    #[test]
    fn test_llt_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let df = Datafeed::new(data, 0, 6, 1);
        let llt = LLT::new(df, 3).expect("Failed to create LLT model");

        assert!(llt.state(0).is_finite());
        assert!(llt.var(0) >= 0.0);
        assert_eq!(llt.datalen(), 6);
    }

    #[test]
    fn test_llt_insufficient_data() {
        let data = vec![1.0];
        let df = Datafeed::new(data, 0, 1, 1);
        // Using new which should return error
        assert!(LLT::new(df, 3).is_err());
    }

    #[test]
    fn test_llp_basic() {
        let data = vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0];
        let df = Datafeed::new(data, 0, 8, 1);
        let llp = LLP::new(df, 3, 3).expect("Failed to create LLP model");

        assert!(llp.state(0).is_finite());
        assert!(llp.var(0) >= 0.0);
        assert_eq!(llp.datalen(), 8);
    }

    #[test]
    fn test_llp1_basic() {
        let data = vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0];
        let df = Datafeed::new(data, 0, 6, 1);
        let llp1 = LLP1::new(df, 3, 3).expect("Failed to create LLP1 model");

        assert!(llp1.state(0).is_finite());
        assert_eq!(llp1.datalen(), 6);
    }

    #[test]
    fn test_llp2_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let df = Datafeed::new(data, 0, 6, 1);
        let llp2 = LLP2::new(df, 2, 3).expect("Failed to create LLP2 model");

        assert!(llp2.state(0).is_finite());
        assert_eq!(llp2.datalen(), 6);
    }

    #[test]
    fn test_llp5_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let df = Datafeed::new(data, 0, 6, 1);
        let llp5 = LLP5::new(df, 2, 3).expect("Failed to create LLP5 model");

        assert!(llp5.state(0).is_finite());
        assert_eq!(llp5.datalen(), 6);
    }

    #[test]
    fn test_llt2_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let df = Datafeed::new(data, 0, 6, 1);
        let llt2 = LLT2::new(df, 3);

        assert!(llt2.state(0).is_finite());
        assert!(llt2.var(0) >= 0.0);
        assert!(llt2.var(0) >= 0.0);
        assert_eq!(llt2.datalen(), 6);
    }

    #[test]
    fn test_univar_ll() {
        let data = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        let univar = Univar::new("LL", data, 0, 5, -1, 3).unwrap();

        assert!(univar.state(0, 0).is_finite());
        assert!(univar.var(0, 0) >= 0.0);
        assert_eq!(univar.datalen(), 5);
        assert!(!univar.multivariate());
    }

    #[test]
    fn test_univar_llt() {
        let data = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]];
        let univar = Univar::new("LLT", data, 0, 6, -1, 3).unwrap();

        assert!(univar.state(0, 0).is_finite());
        assert_eq!(univar.datalen(), 6);
    }

    #[test]
    fn test_univar_llp() {
        let data = vec![vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]];
        let univar = Univar::new("LLP", data, 0, 6, 3, 3).unwrap();

        assert!(univar.state(0, 0).is_finite());
        assert_eq!(univar.period(), 3);
    }

    #[test]
    fn test_univar_rejects_multivariate() {
        let data = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        // Using new which should return error for multivariate algorithm
        assert!(Univar::new("BiLL", data, 0, 3, -1, 3).is_err());
    }

    #[test]
    fn test_state_space_model_trait_ll() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let df = Datafeed::new(data, 0, 5, 1);
        let ll = LL::new(df, 3).expect("Failed to create LL model");

        // Test trait methods
        assert_eq!(ll.least_num_data(), 1);
        assert_eq!(ll.first_forecast_index(), 0);
    }

    #[test]
    fn test_state_space_model_trait_llt() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let df = Datafeed::new(data, 0, 6, 1);
        let llt = LLT::new(df, 3).expect("Failed to create LLT model");

        assert_eq!(llt.least_num_data(), 2);
        assert_eq!(llt.first_forecast_index(), 0);
    }

    #[test]
    fn test_ll_state_consistency() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let df = Datafeed::new(data.clone(), 0, 5, 1);
        let ll = LL::new(df, 3).expect("Failed to create LL model");

        // State values should be in reasonable range
        for i in 0..5 {
            let state = ll.state(i);
            assert!(state.is_finite());
            // State values should be close to data values
            assert!((state - data[i]).abs() < 10.0);
        }
    }

    #[test]
    fn test_ll_var_consistency() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let df = Datafeed::new(data, 0, 5, 1);
        let ll = LL::new(df, 3).expect("Failed to create LL model");

        // Variance should be non-negative
        for i in 0..5 {
            let var = ll.var(i);
            assert!(var >= 0.0 || var.is_nan());
        }
    }
}
