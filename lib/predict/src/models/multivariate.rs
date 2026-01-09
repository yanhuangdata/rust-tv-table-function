//! Multivariate state space models module
//!
//! Implements various multivariate time series state space models

use crate::optimize::{dfpmin, DFP_TOLERANCE};
use crate::utils::MAX_LAG;
use crate::models::MultivarModel;
use anyhow::{Error, bail};
use std::f64;

/// Bivariate Local Level model (BiLL)
pub struct BiLL {
    data: Vec<Vec<f64>>,
    datalength: usize,
    forecast_len: usize,
    q: [f64; 3],
    p: Vec<[f64; 3]>,
    a: Vec<[f64; 2]>,
    epsilon: f64,
}

impl BiLL {
    const EPS: f64 = 1e-12;
    
    /// Create a new BiLL (Bivariate Local Level) model
    /// 
    /// # Errors
    /// 
    /// Returns an error if the data is empty.
    pub fn new(data: Vec<Vec<f64>>, data_len: usize, forecast_len: usize) -> Result<Self, Error> {
        if data.is_empty() {
            bail!("Invalid data: BiLL model requires non-empty data")
        }
        let mut bill = Self {
            data: data.clone(),
            datalength: data_len,
            forecast_len,
            q: [0.0; 3],
            p: vec![[0.0; 3]; data_len + forecast_len],
            a: vec![[0.0; 2]; data_len + forecast_len],
            epsilon: 0.0,
        };
        bill.a[0] = [data[0][0], data[1][0]];
        let scale = 0.1;
        let mut psi = vec![1.0 / scale, 0.5 / scale, 1.0 / scale];
        let mut func = |x: &[f64]| {
            bill.llh(x[0], x[1], x[2], bill.datalength)
        };
        let (_, _) = dfpmin(&mut func, &mut psi, DFP_TOLERANCE);
        bill.update_all(&psi);
        
        // Calculate predictions
        let n = bill.datalength;
        for i in 0..bill.forecast_len {
            if n + i < bill.a.len() {
                bill.a[n + i] = bill.a[n - 1];
            }
            if n + i < bill.p.len() {
                bill.p[n + i] = bill.p[n + i - 1];
                for j in 0..3 {
                    bill.p[n + i][j] += bill.q[j];
                    if j != 1 {
                        bill.p[n + i][j] += bill.epsilon;
                    }
                }
            }
        }
        Ok(bill)
    }
    
    /// Create a new BiLL model (panics on error)
    /// 
    /// # Panics
    /// 
    /// Panics if the data is empty.
    pub fn new_or_panic(data: Vec<Vec<f64>>, data_len: usize, forecast_len: usize) -> Self {
        Self::new(data, data_len, forecast_len)
            .expect("Failed to create BiLL model: data is empty")
    }
    
    pub fn instance(var1: &[f64], var2: &[f64], forecast_len: usize) -> Self {
        let data = vec![var1.to_vec(), var2.to_vec()];
        let data_len = var1.len().min(var2.len());
        Self::new(data, data_len, forecast_len).expect("Failed to create BiLL model")
    }
    
    pub fn least_num_data() -> usize {
        1
    }
    
    pub fn first_forecast_index() -> usize {
        0
    }
    
    fn update_kalman(&self, y: &[f64], a: &mut [f64], v: &mut [f64], p: &mut [f64], k: &mut [f64], fi: &mut [f64]) -> (f64, bool) {
        let detp = p[0] * p[2] - p[1] * p[1];
        let trp = p[0] + p[2];
        let detf = detp + trp + 1.0;
        
        fi[0] = (p[2] + 1.0) / detf;
        fi[1] = -p[1] / detf;
        fi[2] = (p[0] + 1.0) / detf;
        
        k[0] = (p[0] + detp) / detf;
        k[1] = -fi[1];
        k[2] = (p[2] + detp) / detf;
        
        v[0] = y[0] - a[0];
        v[1] = y[1] - a[1];
        
        a[0] += k[0] * v[0] + k[1] * v[1];
        a[1] += k[1] * v[0] + k[2] * v[1];
        
        let pp1 = k[0] + self.q[0];
        let pp2 = k[1] + self.q[1];
        let pp3 = k[2] + self.q[2];
        let diff = (pp1 - p[0]).abs() + (pp2 - p[1]).abs() + (pp3 - p[2]).abs();
        let steady = diff < 1e-5;
        p[0] = pp1;
        p[1] = pp2;
        p[2] = pp3;
        
        (detf, steady)
    }
    
    fn update_steady(&self, y: &[f64], a: &mut [f64], v: &mut [f64], k: &[f64]) {
        v[0] = y[0] - a[0];
        v[1] = y[1] - a[1];
        a[0] += k[0] * v[0] + k[1] * v[1];
        a[1] += k[1] * v[0] + k[2] * v[1];
    }
    
    fn llh(&self, t1: f64, t2: f64, t3: f64, datarange: usize) -> f64 {
        let eps = Self::EPS;
        let q0 = t1 * t1 + eps;
        let q1 = (t1.abs() + eps) * t2;
        let q2 = t3 * t3 + eps + t2 * t2;
        
        let mut a = [self.data[0][0], self.data[1][0]];
        let mut p = [q0 + 1.0, q1, q2 + 1.0];
        let mut k = [0.0; 3];
        let mut fi = [1.0, 0.0, 1.0];
        let mut v = [0.0; 2];
        let mut t1_sum = 0.0;
        let mut t2_sum = 0.0;
        let mut steady = false;
        
        for i in 1..datarange.min(self.datalength) {
            let y = [self.data[0][i], self.data[1][i]];
            if !steady {
                let (detf, steady_flag) = self.update_kalman(&y, &mut a, &mut v, &mut p, &mut k, &mut fi);
                steady = steady_flag;
                t1_sum += v[0] * fi[0] * v[0] + 2.0 * v[0] * fi[1] * v[1] + v[1] * fi[2] * v[1];
                t2_sum += detf.ln();
            } else {
                self.update_steady(&y, &mut a, &mut v, &k);
                t1_sum += v[0] * fi[0] * v[0] + 2.0 * v[0] * fi[1] * v[1] + v[1] * fi[2] * v[1];
            }
        }
        
        if t1_sum == 0.0 {
            t2_sum
        } else {
            (datarange - 1) as f64 * t1_sum.ln() + t2_sum
        }
    }
    
    fn update_all(&mut self, psi: &[f64]) {
        let eps = Self::EPS;
        let t1 = psi[0];
        let t2 = psi[1];
        let t3 = psi[2];
        self.q[0] = t1 * t1 + eps;
        self.q[1] = (t1.abs() + eps) * t2;
        self.q[2] = t3 * t3 + eps + t2 * t2;
        
        let mut a = [self.data[0][0], self.data[1][0]];
        self.a[0] = a;
        let mut p = [self.q[0] + 1.0, self.q[1], self.q[2] + 1.0];
        self.p[0] = p;
        let mut k = [0.0; 3];
        let mut fi = [1.0, 0.0, 1.0];
        let mut v = [0.0; 2];
        let mut epsilon = 0.0;
        let mut steady = false;
        
        for i in 1..self.datalength {
            let y = [self.data[0][i], self.data[1][i]];
            if !steady {
                let (_detf, steady_flag) = self.update_kalman(&y, &mut a, &mut v, &mut p, &mut k, &mut fi);
                steady = steady_flag;
                self.a[i] = a;
                self.p[i] = [p[0] + 1.0, p[1], p[2] + 1.0];
                epsilon += v[0] * fi[0] * v[0] + 2.0 * v[0] * fi[1] * v[1] + v[1] * fi[2] * v[1];
            } else {
                self.update_steady(&y, &mut a, &mut v, &k);
                self.a[i] = a;
                self.p[i] = [p[0] + 1.0, p[1], p[2] + 1.0];
                epsilon += v[0] * fi[0] * v[0] + 2.0 * v[0] * fi[1] * v[1] + v[1] * fi[2] * v[1];
            }
        }
        
        epsilon /= self.datalength as f64;
        self.epsilon = epsilon;
        for i in 0..self.p.len() {
            for j in 0..3 {
                self.p[i][j] *= epsilon;
            }
        }
        for i in 0..3 {
            self.q[i] *= epsilon;
        }
    }
    
    pub fn state(&self, ts_idx: usize, i: usize) -> f64 {
        if ts_idx > 1 {
            return 0.0;
        }
        if i < self.a.len() {
            self.a[i][ts_idx]
        } else {
            0.0
        }
    }
    
    pub fn var(&self, ts_idx: usize, i: usize) -> f64 {
        if ts_idx > 1 {
            return 0.0;
        }
        if i < self.datalength {
            if i < self.p.len() {
                self.p[i][ts_idx] + self.q[ts_idx]
            } else {
                0.0
            }
        } else {
            if i < self.p.len() {
                self.p[i][ts_idx]
            } else {
                0.0
            }
        }
    }
    
    pub fn datalen(&self) -> usize {
        self.datalength
    }
}

impl MultivarModel for BiLL {
    fn state(&self, ts_idx: usize, i: usize) -> f64 {
        self.state(ts_idx, i)
    }
    
    fn var(&self, ts_idx: usize, i: usize) -> f64 {
        self.var(ts_idx, i)
    }
    
    fn variance(&self, i: usize) -> f64 {
        if i < self.p.len() {
            self.p[i][0]
        } else {
            0.0
        }
    }
    
    fn datalen(&self) -> usize {
        self.datalen()
    }
    
    fn least_num_data(&self) -> usize {
        Self::least_num_data()
    }
    
    fn first_forecast_index(&self) -> usize {
        Self::first_forecast_index()
    }
    
    fn predict(&mut self, _predict_var: usize, _start: usize) -> Result<(), Error> {
        // BiLL does not support predict method
        Ok(())
    }
}

/// BiLL missing values version
pub struct BiLLmv {
    data: Vec<Vec<Option<f64>>>,
    datalength: usize,
    forecast_len: usize,
    q: [f64; 3],
    p: Vec<[f64; 3]>,
    a: Vec<[f64; 2]>,
    epsilon: f64,
}

impl BiLLmv {
    /// Create a new BiLLmv (BiLL with missing values) model
    /// 
    /// # Errors
    /// 
    /// Returns an error if the data is empty.
    pub fn new(data: Vec<Vec<Option<f64>>>, data_len: usize, forecast_len: usize) -> Result<Self, Error> {
        if data.is_empty() {
            bail!("Invalid data: BiLLmv model requires non-empty data")
        }
        let mut billmv = Self {
            data: data.clone(),
            datalength: data_len,
            forecast_len,
            q: [0.0; 3],
            p: vec![[0.0; 3]; data_len + forecast_len],
            a: vec![[0.0; 2]; data_len + forecast_len],
            epsilon: 0.0,
        };
        
        // Get first non-None value as initial value
        let mut init_val0 = 0.0;
        let mut init_val1 = 0.0;
        for (row_idx, row) in data.iter().enumerate() {
            for val in row {
                if let Some(v) = val {
                    if init_val0 == 0.0 && row_idx == 0 {
                        init_val0 = *v;
                    }
                    if init_val1 == 0.0 && row_idx == 1 && data.len() > 1 {
                        init_val1 = *v;
                    }
                    if init_val0 != 0.0 && (data.len() <= 1 || init_val1 != 0.0) {
                        break;
                    }
                }
            }
        }
        billmv.a[0] = [init_val0, init_val1];
        
        let scale = 0.1;
        let mut psi = vec![1.0 / scale, 0.5 / scale, 1.0 / scale];
        let mut func = |x: &[f64]| {
            billmv.llh(x[0], x[1], x[2], billmv.datalength)
        };
        let (_, _) = dfpmin(&mut func, &mut psi, DFP_TOLERANCE);
        billmv.update_all(&psi);
        
        // Calculate predictions
        let n = billmv.datalength;
        for i in 0..billmv.forecast_len {
            if n + i < billmv.a.len() {
                billmv.a[n + i] = billmv.a[n - 1];
            }
            if n + i < billmv.p.len() {
                billmv.p[n + i] = billmv.p[n + i - 1];
                for j in 0..3 {
                    billmv.p[n + i][j] += billmv.q[j];
                    if j != 1 {
                        billmv.p[n + i][j] += billmv.epsilon;
                    }
                }
            }
        }
        Ok(billmv)
    }
    
    /// Create a new BiLLmv model (panics on error)
    /// 
    /// # Panics
    /// 
    /// Panics if the data is empty.
    pub fn new_or_panic(data: Vec<Vec<Option<f64>>>, data_len: usize, forecast_len: usize) -> Self {
        Self::new(data, data_len, forecast_len)
            .expect("Failed to create BiLLmv model: data is empty")
    }
    
    pub fn instance(var1: &[Option<f64>], var2: &[Option<f64>], forecast_len: usize) -> Self {
        let data = vec![var1.to_vec(), var2.to_vec()];
        let data_len = var1.len().min(var2.len());
        Self::new(data, data_len, forecast_len).expect("Failed to create BiLLmv model")
    }
    
    pub fn least_num_data() -> usize {
        1
    }
    
    pub fn first_forecast_index() -> usize {
        0
    }
    
    fn update_kalman(&self, y: &[Option<f64>], a: &mut [f64], v: &mut [f64], p: &mut [f64], k: &mut [f64], fi: &mut [f64]) -> f64 {
        let detp = p[0] * p[2] - p[1] * p[1];
        let trp = p[0] + p[2];
        let detf = detp + trp + 1.0;
        
        fi[0] = (p[2] + 1.0) / detf;
        fi[1] = -p[1] / detf;
        fi[2] = (p[0] + 1.0) / detf;
        
        k[0] = (p[0] + detp) / detf;
        k[1] = -fi[1];
        k[2] = (p[2] + detp) / detf;
        
        if y[0].is_none() {
            v[0] = 0.0;
            p[0] += self.q[0];
        } else {
            v[0] = y[0].unwrap() - a[0];
            a[0] += k[0] * v[0] + k[1] * v[1];
            p[0] = k[0] + self.q[0];
        }
        if y[1].is_none() {
            v[1] = 0.0;
            p[2] += self.q[2];
        } else {
            v[1] = y[1].unwrap() - a[1];
            a[1] += k[1] * v[0] + k[2] * v[1];
            p[2] = k[2] + self.q[2];
        }
        p[1] = k[1] + self.q[1];
        
        detf
    }
    
    fn next_state(&self, data: &[Vec<Option<f64>>], start: usize, end: usize, a: &mut [f64], v: &mut [f64], p: &mut [f64], k: &mut [f64], fi: &mut [f64]) -> Vec<f64> {
        let mut detf_vec = Vec::new();
        for i in start..end.min(data[0].len()).min(data[1].len()) {
            let y = [data[0].get(i).copied().flatten(), data[1].get(i).copied().flatten()];
            let y_opt = [y[0], y[1]];
            let detf = self.update_kalman(&y_opt, a, v, p, k, fi);
            detf_vec.push(detf);
        }
        detf_vec
    }
    
    fn llh(&self, t1: f64, t2: f64, t3: f64, datarange: usize) -> f64 {
        let eps = BiLL::EPS;
        let q0 = t1 * t1 + eps;
        let q1 = (t1.abs() + eps) * t2;
        let q2 = t3 * t3 + eps + t2 * t2;
        
        let mut a = [self.a[0][0], self.a[0][1]];
        let mut p = [q0 + 1.0, q1, q2 + 1.0];
        let mut k = [0.0; 3];
        let mut fi = [1.0, 0.0, 1.0];
        let mut v = [0.0; 2];
        let mut t1_sum = 0.0;
        let mut t2_sum = 0.0;
        
        for detf in self.next_state(&self.data, 1, datarange.min(self.datalength), &mut a, &mut v, &mut p, &mut k, &mut fi) {
            t1_sum += v[0] * fi[0] * v[0] + 2.0 * v[0] * fi[1] * v[1] + v[1] * fi[2] * v[1];
            t2_sum += detf.ln();
        }
        
        if t1_sum == 0.0 {
            t2_sum
        } else {
            (datarange - 1) as f64 * t1_sum.ln() + t2_sum
        }
    }
    
    fn update_all(&mut self, psi: &[f64]) {
        let eps = BiLL::EPS;
        let t1 = psi[0];
        let t2 = psi[1];
        let t3 = psi[2];
        self.q[0] = t1 * t1 + eps;
        self.q[1] = (t1.abs() + eps) * t2;
        self.q[2] = t3 * t3 + eps + t2 * t2;
        
        let mut a = [self.a[0][0], self.a[0][1]];
        self.a[0] = a;
        let mut p = [self.q[0] + 1.0, self.q[1], self.q[2] + 1.0];
        self.p[0] = p;
        let mut k = [0.0; 3];
        let mut fi = [1.0, 0.0, 1.0];
        let mut v = [0.0; 2];
        let mut epsilon = 0.0;
        
        for (i, _detf) in self.next_state(&self.data, 0, self.datalength, &mut a, &mut v, &mut p, &mut k, &mut fi).iter().enumerate() {
            self.a[i] = a;
            self.p[i] = [p[0] + 1.0, p[1], p[2] + 1.0];
            epsilon += v[0] * fi[0] * v[0] + 2.0 * v[0] * fi[1] * v[1] + v[1] * fi[2] * v[1];
        }
        
        epsilon /= self.datalength as f64;
        self.epsilon = epsilon;
        for i in 0..self.p.len() {
            for j in 0..3 {
                self.p[i][j] *= epsilon;
            }
        }
        for i in 0..3 {
            self.q[i] *= epsilon;
        }
    }
    
    pub fn state(&self, ts_idx: usize, i: usize) -> f64 {
        if ts_idx > 1 {
            return 0.0;
        }
        if i < self.a.len() {
            self.a[i][ts_idx]
        } else {
            0.0
        }
    }
    
    pub fn var(&self, ts_idx: usize, i: usize) -> f64 {
        if ts_idx > 1 {
            return 0.0;
        }
        if i < self.datalength {
            if i < self.p.len() {
                self.p[i][ts_idx] + self.q[ts_idx]
            } else {
                0.0
            }
        } else {
            if i < self.p.len() {
                self.p[i][ts_idx]
            } else {
                0.0
            }
        }
    }
    
    pub fn datalen(&self) -> usize {
        self.datalength
    }
}

impl MultivarModel for BiLLmv {
    fn state(&self, ts_idx: usize, i: usize) -> f64 {
        self.state(ts_idx, i)
    }
    
    fn var(&self, ts_idx: usize, i: usize) -> f64 {
        self.var(ts_idx, i)
    }
    
    fn variance(&self, i: usize) -> f64 {
        self.var(0, i)
    }
    
    fn datalen(&self) -> usize {
        self.datalen()
    }
    
    fn least_num_data(&self) -> usize {
        Self::least_num_data()
    }
    
    fn first_forecast_index(&self) -> usize {
        Self::first_forecast_index()
    }
    
    fn predict(&mut self, _predict_var: usize, _start: usize) -> Result<(), Error> {
        // BiLLmv does not support predict method
        Ok(())
    }
}

/// BiLLmv2 - Another missing values version (independent implementation)
pub struct BiLLmv2 {
    data: Vec<Vec<Option<f64>>>,
    datalength: usize,
    forecast_len: usize,
    q: [f64; 3],
    p: Vec<[f64; 3]>,
    a: Vec<[f64; 2]>,
    epsilon: f64,
}

impl BiLLmv2 {
    const EPS: f64 = 1e-12;
    
    /// Create a new BiLLmv2 model
    /// 
    /// # Errors
    /// 
    /// Returns an error if the data is empty.
    pub fn new(data: Vec<Vec<Option<f64>>>, data_len: usize, forecast_len: usize) -> Result<Self, Error> {
        if data.is_empty() {
            bail!("Invalid data: BiLLmv2 model requires non-empty data")
        }
        let mut billmv2 = Self {
            data: data.clone(),
            datalength: data_len,
            forecast_len,
            q: [0.0; 3],
            p: vec![[0.0; 3]; data_len + forecast_len],
            a: vec![[0.0; 2]; data_len + forecast_len],
            epsilon: 0.0,
        };
        
        // Get first non-None value as initial value
        let mut init_val0 = 0.0;
        let mut init_val1 = 0.0;
        for (row_idx, row) in data.iter().enumerate() {
            for val in row {
                if let Some(v) = val {
                    if init_val0 == 0.0 && row_idx == 0 {
                        init_val0 = *v;
                    }
                    if init_val1 == 0.0 && row_idx == 1 && data.len() > 1 {
                        init_val1 = *v;
                    }
                    if init_val0 != 0.0 && (data.len() <= 1 || init_val1 != 0.0) {
                        break;
                    }
                }
            }
        }
        billmv2.a[0] = [init_val0, init_val1];
        
        let scale = 0.1;
        let mut psi = vec![1.0 / scale, 0.5 / scale, 1.0 / scale];
        let mut func = |x: &[f64]| {
            billmv2.llh(x[0], x[1], x[2], billmv2.datalength)
        };
        let (_, _) = dfpmin(&mut func, &mut psi, DFP_TOLERANCE);
        billmv2.update_all(&psi);
        
        // Calculate predictions
        let n = billmv2.datalength;
        for i in 0..billmv2.forecast_len {
            if n + i < billmv2.a.len() {
                billmv2.a[n + i] = billmv2.a[n - 1];
            }
            if n + i < billmv2.p.len() {
                billmv2.p[n + i] = billmv2.p[n + i - 1];
                for j in 0..3 {
                    billmv2.p[n + i][j] += billmv2.q[j];
                    if j != 1 {
                        billmv2.p[n + i][j] += billmv2.epsilon;
                    }
                }
            }
        }
        Ok(billmv2)
    }
    
    /// Create a new BiLLmv2 model (panics on error)
    /// 
    /// # Panics
    /// 
    /// Panics if the data is empty.
    pub fn new_or_panic(data: Vec<Vec<Option<f64>>>, data_len: usize, forecast_len: usize) -> Self {
        Self::new(data, data_len, forecast_len)
            .expect("Failed to create BiLLmv2 model: data is empty")
    }
    
    pub fn instance(var1: &[Option<f64>], var2: &[Option<f64>], forecast_len: usize) -> Self {
        let data = vec![var1.to_vec(), var2.to_vec()];
        let data_len = var1.len().min(var2.len());
        Self::new(data, data_len, forecast_len).expect("Failed to create BiLLmv2 model")
    }
    
    pub fn least_num_data() -> usize {
        1
    }
    
    pub fn first_forecast_index() -> usize {
        0
    }
    
    fn update_kalman(&self, y: &[Option<f64>], a: &mut [f64], v: &mut [f64], p: &mut [f64], k: &mut [f64], fi: &mut [f64]) -> f64 {
        let detp = p[0] * p[2] - p[1] * p[1];
        let trp = p[0] + p[2];
        let detf = detp + trp + 1.0;
        
        fi[0] = (p[2] + 1.0) / detf;
        fi[1] = -p[1] / detf;
        fi[2] = (p[0] + 1.0) / detf;
        
        k[0] = (p[0] + detp) / detf;
        k[1] = -fi[1];
        k[2] = (p[2] + detp) / detf;
        
        if y[0].is_none() {
            v[0] = 0.0;
            p[0] += self.q[0];
        } else {
            v[0] = y[0].unwrap() - a[0];
            a[0] += k[0] * v[0] + k[1] * v[1];
            p[0] = k[0] + self.q[0];
        }
        if y[1].is_none() {
            v[1] = 0.0;
            p[2] += self.q[2];
        } else {
            v[1] = y[1].unwrap() - a[1];
            a[1] += k[1] * v[0] + k[2] * v[1];
            p[2] = k[2] + self.q[2];
        }
        p[1] = k[1] + self.q[1];
        
        detf
    }
    
    fn next_state(&self, data: &[Vec<Option<f64>>], start: usize, end: usize, a: &mut [f64], v: &mut [f64], p: &mut [f64], k: &mut [f64], fi: &mut [f64]) -> Vec<f64> {
        let mut detf_vec = Vec::new();
        for i in start..end.min(data[0].len()).min(data[1].len()) {
            let y = [data[0].get(i).copied().flatten(), data[1].get(i).copied().flatten()];
            let y_opt = [y[0], y[1]];
            let detf = self.update_kalman(&y_opt, a, v, p, k, fi);
            detf_vec.push(detf);
        }
        detf_vec
    }
    
    fn llh(&self, t1: f64, t2: f64, t3: f64, datarange: usize) -> f64 {
        let eps = BiLL::EPS;
        let q0 = t1 * t1 + eps;
        let q1 = (t1.abs() + eps) * t2;
        let q2 = t3 * t3 + eps + t2 * t2;
        
        let mut a = [self.a[0][0], self.a[0][1]];
        let mut p = [q0 + 1.0, q1, q2 + 1.0];
        let mut k = [0.0; 3];
        let mut fi = [1.0, 0.0, 1.0];
        let mut v = [0.0; 2];
        let mut t1_sum = 0.0;
        let mut t2_sum = 0.0;
        
        for detf in self.next_state(&self.data, 1, datarange.min(self.datalength), &mut a, &mut v, &mut p, &mut k, &mut fi) {
            t1_sum += v[0] * fi[0] * v[0] + 2.0 * v[0] * fi[1] * v[1] + v[1] * fi[2] * v[1];
            t2_sum += detf.ln();
        }
        
        if t1_sum == 0.0 {
            t2_sum
        } else {
            (datarange - 1) as f64 * t1_sum.ln() + t2_sum
        }
    }
    
    fn update_all(&mut self, psi: &[f64]) {
        let eps = Self::EPS;
        let t1 = psi[0];
        let t2 = psi[1];
        let t3 = psi[2];
        self.q[0] = t1 * t1 + eps;
        self.q[1] = (t1.abs() + eps) * t2;
        self.q[2] = t3 * t3 + eps + t2 * t2;
        
        let mut a = [self.a[0][0], self.a[0][1]];
        self.a[0] = a;
        let mut p = [self.q[0] + 1.0, self.q[1], self.q[2] + 1.0];
        self.p[0] = p;
        let mut k = [0.0; 3];
        let mut fi = [1.0, 0.0, 1.0];
        let mut v = [0.0; 2];
        let mut epsilon = 0.0;
        
        for (i, _detf) in self.next_state(&self.data, 0, self.datalength, &mut a, &mut v, &mut p, &mut k, &mut fi).iter().enumerate() {
            self.a[i] = a;
            self.p[i] = [p[0] + 1.0, p[1], p[2] + 1.0];
            epsilon += v[0] * fi[0] * v[0] + 2.0 * v[0] * fi[1] * v[1] + v[1] * fi[2] * v[1];
        }
        
        epsilon /= self.datalength as f64;
        self.epsilon = epsilon;
        for i in 0..self.p.len() {
            for j in 0..3 {
                self.p[i][j] *= epsilon;
            }
        }
        for i in 0..3 {
            self.q[i] *= epsilon;
        }
    }
    
    pub fn state(&self, ts_idx: usize, i: usize) -> f64 {
        if ts_idx > 1 {
            return 0.0;
        }
        if i < self.a.len() {
            self.a[i][ts_idx]
        } else {
            0.0
        }
    }
    
    pub fn var(&self, ts_idx: usize, i: usize) -> f64 {
        if ts_idx > 1 {
            return 0.0;
        }
        if i < self.datalength {
            if i < self.p.len() {
                self.p[i][ts_idx] + self.q[ts_idx]
            } else {
                0.0
            }
        } else {
            if i < self.p.len() {
                self.p[i][ts_idx]
            } else {
                0.0
            }
        }
    }
    
    pub fn datalen(&self) -> usize {
        self.datalength
    }
}

impl MultivarModel for BiLLmv2 {
    fn state(&self, ts_idx: usize, i: usize) -> f64 {
        self.state(ts_idx, i)
    }
    
    fn var(&self, ts_idx: usize, i: usize) -> f64 {
        self.var(ts_idx, i)
    }
    
    fn variance(&self, i: usize) -> f64 {
        self.var(0, i)
    }
    
    fn datalen(&self) -> usize {
        self.datalen()
    }
    
    fn least_num_data(&self) -> usize {
        Self::least_num_data()
    }
    
    fn first_forecast_index(&self) -> usize {
        Self::first_forecast_index()
    }
    
    fn predict(&mut self, _predict_var: usize, _start: usize) -> Result<(), Error> {
        // BiLLmv2 does not support predict method
        Ok(())
    }
}

/// Multivariate Local Level model (LLB)
pub struct LLB {
    data: Vec<Vec<f64>>,
    data_len: usize,
    #[allow(dead_code)]
    forecast_len: usize,
    q: [f64; 3],
    p: Vec<[f64; 3]>,
    a: Vec<[f64; 2]>,
    epsilon: f64,
    fc: Option<Vec<f64>>,
    var_arr: Option<Vec<f64>>,
}

impl LLB {
    const EPS: f64 = 1e-12;
    
    /// Create a new LLB model
    /// 
    /// # Errors
    /// 
    /// Returns an error if the data is empty or data_len is 0.
    pub fn new(data: Vec<Vec<f64>>, data_len: usize, forecast_len: usize) -> Result<Self, Error> {
        if data.is_empty() || data_len == 0 {
            bail!("Invalid data: LLB model requires non-empty data with positive data_len")
        }
        let mut llb = Self {
            data: data.clone(),
            data_len,
            forecast_len,
            q: [0.0; 3],
            p: vec![[0.0; 3]; data_len + forecast_len],
            a: vec![[0.0; 2]; data_len + forecast_len],
            epsilon: 0.0,
            fc: None,
            var_arr: None,
        };
        llb.a[0] = [data[0][0], data[1][0]];
        let scale = 0.1;
        let mut psi = vec![1.0 / scale, 0.5 / scale, 1.0 / scale];
        let mut func = |x: &[f64]| {
            llb.llh(x[0], x[1], x[2], data_len)
        };
        let (_, _) = dfpmin(&mut func, &mut psi, DFP_TOLERANCE);
        llb.update_all(&psi);
        Ok(llb)
    }
    
    /// Create a new LLB model (panics on error)
    /// 
    /// # Panics
    /// 
    /// Panics if the data is empty or data_len is 0.
    pub fn new_or_panic(data: Vec<Vec<f64>>, data_len: usize, forecast_len: usize) -> Self {
        Self::new(data, data_len, forecast_len)
            .expect("Failed to create LLB model: invalid data")
    }
    
    pub fn instance(var1: &[f64], var2: &[f64], data_len: usize, forecast_len: usize) -> Self {
        let data = vec![var1.to_vec(), var2.to_vec()];
        Self::new(data, data_len, forecast_len).expect("Failed to create LLB model")
    }
    
    pub fn instance_with_opt(var1: &[Option<f64>], var2: &[Option<f64>], data_len: usize, forecast_len: usize) -> Self {
        let data_opt = vec![var1.to_vec(), var2.to_vec()];
        // Convert to Vec<Vec<f64>>, fill missing values with 0
        let data: Vec<Vec<f64>> = data_opt
            .iter()
            .map(|row| row.iter().map(|x| x.unwrap_or(0.0)).collect())
            .collect();
        Self::new(data, data_len, forecast_len).expect("Failed to create LLB model")
    }
    
    pub fn least_num_data() -> usize {
        2
    }
    
    pub fn first_forecast_index() -> usize {
        1
    }
    
    fn update_kalman(&self, y: &[f64], a: &mut [f64], v: &mut [f64], p: &mut [f64], k: &mut [f64], fi: &mut [f64]) -> (f64, bool) {
        let detp = p[0] * p[2] - p[1] * p[1];
        let trp = p[0] + p[2];
        let detf = detp + trp + 1.0;
        
        fi[0] = (p[2] + 1.0) / detf;
        fi[1] = -p[1] / detf;
        fi[2] = (p[0] + 1.0) / detf;
        
        k[0] = (p[0] + detp) / detf;
        k[1] = -fi[1];
        k[2] = (p[2] + detp) / detf;
        
        v[0] = y[0] - a[0];
        v[1] = y[1] - a[1];
        
        a[0] += k[0] * v[0] + k[1] * v[1];
        a[1] += k[1] * v[0] + k[2] * v[1];
        
        let pp1 = k[0] + self.q[0];
        let pp2 = k[1] + self.q[1];
        let pp3 = k[2] + self.q[2];
        let pp = (pp1 - p[0]).abs() + (pp2 - p[1]).abs() + (pp3 - p[2]).abs();
        let steady = pp < 1e-5;
        p[0] = pp1;
        p[1] = pp2;
        p[2] = pp3;
        
        (detf, steady)
    }
    
    fn update_steady(&self, y: &[f64], a: &mut [f64], v: &mut [f64], k: &[f64]) {
        v[0] = y[0] - a[0];
        v[1] = y[1] - a[1];
        a[0] += k[0] * v[0] + k[1] * v[1];
        a[1] += k[1] * v[0] + k[2] * v[1];
    }
    
    fn llh(&self, t1: f64, t2: f64, t3: f64, datarange: usize) -> f64 {
        let eps = Self::EPS;
        let q0 = t1 * t1 + eps;
        let q1 = (t1.abs() + eps) * t2;
        let q2 = t3 * t3 + eps + t2 * t2;
        
        let mut a = [self.data[0][0], self.data[1][0]];
        let mut p = [q0 + 1.0, q1, q2 + 1.0];
        let mut k = [0.0; 3];
        let mut fi = [1.0, 0.0, 1.0];
        let mut v = [0.0; 2];
        let mut t1_sum = 0.0;
        let mut t2_sum = 0.0;
        let mut steady = false;
        
        for i in 1..datarange.min(self.data_len) {
            let y = [self.data[0][i], self.data[1][i]];
            if !steady {
                let (detf, steady_flag) = self.update_kalman(&y, &mut a, &mut v, &mut p, &mut k, &mut fi);
                steady = steady_flag;
                t1_sum += v[0] * fi[0] * v[0] + 2.0 * v[0] * fi[1] * v[1] + v[1] * fi[2] * v[1];
                t2_sum += detf.ln();
            } else {
                self.update_steady(&y, &mut a, &mut v, &k);
                t1_sum += v[0] * fi[0] * v[0] + 2.0 * v[0] * fi[1] * v[1] + v[1] * fi[2] * v[1];
            }
        }
        
        if t1_sum == 0.0 {
            t2_sum / 2.0
        } else {
            (self.data_len - 1) as f64 * t1_sum.ln() + t2_sum / 2.0
        }
    }
    
    fn update_all(&mut self, psi: &[f64]) {
        let eps = Self::EPS;
        let t1 = psi[0];
        let t2 = psi[1];
        let t3 = psi[2];
        self.q[0] = t1 * t1 + eps;
        self.q[1] = (t1.abs() + eps) * t2;
        self.q[2] = t3 * t3 + eps + t2 * t2;
        
        let mut a = [self.data[0][0], self.data[1][0]];
        self.a[0] = a;
        let mut p = [self.q[0] + 1.0, self.q[1], self.q[2] + 1.0];
        self.p[0] = p;
        let mut k = [0.0; 3];
        let mut fi = [1.0, 0.0, 1.0];
        let mut v = [0.0; 2];
        let mut epsilon = 0.0;
        let mut steady = false;
        
        for i in 1..self.data_len {
            let y = [self.data[0][i], self.data[1][i]];
            if !steady {
                let (_detf, steady_flag) = self.update_kalman(&y, &mut a, &mut v, &mut p, &mut k, &mut fi);
                steady = steady_flag;
                self.a[i] = a;
                self.p[i] = [p[0] + 1.0, p[1], p[2] + 1.0];
                epsilon += v[0] * fi[0] * v[0] + 2.0 * v[0] * fi[1] * v[1] + v[1] * fi[2] * v[1];
            } else {
                self.update_steady(&y, &mut a, &mut v, &k);
                self.a[i] = a;
                self.p[i] = [p[0] + 1.0, p[1], p[2] + 1.0];
                epsilon += v[0] * fi[0] * v[0] + 2.0 * v[0] * fi[1] * v[1] + v[1] * fi[2] * v[1];
            }
        }
        
        epsilon /= 2.0 * self.data_len as f64;
        self.epsilon = epsilon;
        for i in 0..self.p.len() {
            for j in 0..3 {
                self.p[i][j] *= epsilon;
            }
        }
    }
    
    /// Make predictions using the LLB model
    /// 
    /// # Errors
    /// 
    /// Returns an error if predict_var is not 0 or 1.
    pub fn predict(&mut self, predict_var: usize, start: usize) -> Result<(), Error> {
        if predict_var != 0 && predict_var != 1 {
            bail!("Invalid parameter 'predict_var': {} - predict_var must be 0 or 1", predict_var);
        }
        let n = self.data_len;
        let corvar = 1 - predict_var;
        let mut fc = vec![0.0; n];
        let mut var_arr = vec![0.0; n];
        
        for i in start..n {
            let cov = self.p[i];
            let sigma = if cov[2 * corvar] != 0.0 {
                cov[1] / cov[2 * corvar]
            } else {
                0.0
            };
            var_arr[i] = if cov[2 * corvar] != 0.0 {
                cov[2 * predict_var] - (cov[1] * cov[1]) / cov[2 * corvar]
            } else {
                cov[2 * predict_var]
            };
            
            if i < self.data[corvar].len() && !self.data[corvar][i].is_nan() {
                fc[i] = self.a[i - 1][predict_var] + sigma * (self.data[corvar][i] - self.a[i - 1][corvar]);
            } else {
                fc[i] = self.a[i - 1][predict_var];
            }
        }
        
        // Store prediction results
        self.fc = Some(fc);
        self.var_arr = Some(var_arr);
        Ok(())
    }
    
    pub fn state(&self, ts_idx: usize, i: usize) -> f64 {
        if ts_idx > 1 {
            return 0.0;
        }
        if i < self.a.len() {
            self.a[i][ts_idx]
        } else {
            0.0
        }
    }
    
    pub fn var(&self, ts_idx: usize, i: usize) -> f64 {
        if ts_idx > 1 {
            return 0.0;
        }
        if i < self.p.len() {
            self.p[i][ts_idx]
        } else {
            0.0
        }
    }
    
    pub fn variance(&self, i: usize) -> f64 {
        // If predict has been called, return value from VAR array
        if let Some(ref var_arr) = self.var_arr {
            if i < var_arr.len() {
                return var_arr[i];
            }
        }
        // Otherwise return value from p array
        if i < self.p.len() {
            self.p[i][0]
        } else {
            0.0
        }
    }
    
    pub fn datalen(&self) -> usize {
        self.data_len
    }
}

/// LLB missing values version
pub struct LLBmv {
    llb: LLB,
}

impl LLBmv {
    /// Create a new LLBmv model (LLB with missing values)
    /// 
    /// # Errors
    /// 
    /// Returns an error if the underlying LLB model creation fails.
    pub fn new(data: Vec<Vec<Option<f64>>>, data_len: usize, forecast_len: usize) -> Result<Self, Error> {
        let data_f64: Vec<Vec<f64>> = data
            .iter()
            .map(|row| {
                row.iter()
                    .map(|x| x.unwrap_or(0.0))
                    .collect()
            })
            .collect();
        let llb = LLB::new(data_f64, data_len, forecast_len)?;
        Ok(Self { llb })
    }
    
    /// Create a new LLBmv model (panics on error)
    /// 
    /// # Panics
    /// 
    /// Panics if the underlying LLB model creation fails.
    pub fn new_or_panic(data: Vec<Vec<Option<f64>>>, data_len: usize, forecast_len: usize) -> Self {
        Self::new(data, data_len, forecast_len)
            .expect("Failed to create LLBmv model")
    }
    
    pub fn instance(var1: &[Option<f64>], var2: &[Option<f64>], data_len: usize, forecast_len: usize) -> Self {
        let data = vec![var1.to_vec(), var2.to_vec()];
        Self::new(data, data_len, forecast_len).expect("Failed to create LLBmv model")
    }
    
    pub fn state(&self, ts_idx: usize, i: usize) -> f64 {
        self.llb.state(ts_idx, i)
    }
    
    pub fn var(&self, ts_idx: usize, i: usize) -> f64 {
        self.llb.var(ts_idx, i)
    }
    
    pub fn variance(&self, i: usize) -> f64 {
        self.llb.variance(i)
    }
    
    pub fn datalen(&self) -> usize {
        self.llb.datalen()
    }
}

impl MultivarModel for LLBmv {
    fn state(&self, ts_idx: usize, i: usize) -> f64 {
        self.state(ts_idx, i)
    }
    
    fn var(&self, ts_idx: usize, i: usize) -> f64 {
        self.var(ts_idx, i)
    }
    
    fn variance(&self, i: usize) -> f64 {
        self.variance(i)
    }
    
    fn datalen(&self) -> usize {
        self.datalen()
    }
    
    fn least_num_data(&self) -> usize {
        LLB::least_num_data()
    }
    
    fn first_forecast_index(&self) -> usize {
        LLB::first_forecast_index()
    }
    
    fn predict(&mut self, predict_var: usize, start: usize) -> Result<(), Error> {
        self.llb.predict(predict_var, start)
    }
}

/// Multivariate model wrapper
pub struct Multivar {
    algo: Box<dyn MultivarModel>,
    #[allow(dead_code)]
    datalength: usize,
    period: Option<i32>,
    #[allow(dead_code)]
    forecast_len: usize,
    algorithm: String,
}

// MultivarModel trait is defined in models/mod.rs

// Add a helper method for LLB to access internal data
impl LLB {
    pub fn get_state(&self, ts_idx: usize, i: usize) -> f64 {
        self.state(ts_idx, i)
    }
}

impl MultivarModel for LLB {
    fn state(&self, ts_idx: usize, i: usize) -> f64 {
        self.state(ts_idx, i)
    }
    
    fn var(&self, ts_idx: usize, i: usize) -> f64 {
        self.var(ts_idx, i)
    }
    
    fn variance(&self, i: usize) -> f64 {
        self.variance(i)
    }
    
    fn datalen(&self) -> usize {
        self.datalen()
    }
    
    fn least_num_data(&self) -> usize {
        Self::least_num_data()
    }
    
    fn first_forecast_index(&self) -> usize {
        Self::first_forecast_index()
    }
    
    fn predict(&mut self, predict_var: usize, start: usize) -> Result<(), Error> {
        LLB::predict(self, predict_var, start)
    }
}

impl Multivar {
    /// Create a new Multivar model
    /// 
    /// # Errors
    /// 
    /// Returns an error if:
    /// - The period is greater than MAX_LAG
    /// - The algorithm is unsupported
    /// - Data is insufficient for the selected algorithm
    pub fn new(
        algorithm: &str,
        data: Vec<Vec<f64>>,
        data_end: usize,
        period: Option<i32>,
        forecast_len: usize,
        correlate: Option<&[f64]>,
        missing_valued: bool,
    ) -> Result<Self, Error> {
        if let Some(p) = period {
            if p > MAX_LAG as i32 {
                bail!("Invalid parameter 'period': {} - period cannot be greater than {}", p, MAX_LAG);
            }
        }
        
        let mut algo_name = algorithm.to_string();
        if missing_valued && !algo_name.ends_with("mv") {
            algo_name.push_str("mv");
        }
        
        let algo: Box<dyn MultivarModel> = if algo_name.starts_with("LLB") {
            // LLB algorithm needs to use instance method, pass var1 and var2 (correlate)
            if algo_name == "LLBmv" {
                // LLBmv needs to handle missing values
                if let Some(corr) = correlate {
                    // Convert correlate to Option<f64>
                    let corr_opt: Vec<Option<f64>> = corr.iter().map(|x| Some(*x)).collect();
                    let var1_opt: Vec<Option<f64>> = if !data.is_empty() {
                        data[0].iter().map(|x| Some(*x)).collect()
                    } else {
                        vec![]
                    };
                    Box::new(LLBmv::instance(&var1_opt, &corr_opt, data_end, forecast_len))
                } else {
                    // If no correlate, use first two columns of data
                    if data.len() < 2 {
                        bail!("Insufficient data: required at least 2 data points, got {}", data.len());
                    }
                    let data_opt: Vec<Vec<Option<f64>>> = data
                        .iter()
                        .map(|row| row.iter().map(|x| Some(*x)).collect())
                        .collect();
                    Box::new(LLBmv::new(data_opt, data_end, forecast_len)?)
                }
            } else {
                // LLB algorithm
                if let Some(corr) = correlate {
                    // Use instance method, pass var1 and var2 (correlate)
                    let var1 = if !data.is_empty() { &data[0] } else {
                        bail!("Invalid data: LLB model requires non-empty data")
                    };
                    Box::new(LLB::instance(var1, corr, data_end, forecast_len))
                } else {
                    // If no correlate, use first two columns of data
                    if data.len() < 2 {
                        bail!("Insufficient data: required at least 2 data points, got {}", data.len());
                    }
                    Box::new(LLB::new(data, data_end, forecast_len)?)
                }
            }
        } else if algo_name == "BiLL" {
            Box::new(BiLL::new(data, data_end, forecast_len)?)
        } else if algo_name == "BiLLmv" {
            let data_opt: Vec<Vec<Option<f64>>> = data
                .iter()
                .map(|row| row.iter().map(|x| Some(*x)).collect())
                .collect();
            Box::new(BiLLmv::new(data_opt, data_end, forecast_len)?)
        } else {
            let _supported = vec![
                "LLB".to_string(), "LLBmv".to_string(),
                "BiLL".to_string(), "BiLLmv".to_string()
            ];
            bail!("Unsupported algorithm. Supported: supported")
        };
        
        Ok(Self {
            algo,
            datalength: data_end,
            period,
            forecast_len,
            algorithm: algo_name,
        })
    }
    
    /// Create a new Multivar model (panics on error)
    /// 
    /// # Panics
    /// 
    /// Panics if the algorithm is unsupported or data is invalid.
    pub fn new_or_panic(
        algorithm: &str,
        data: Vec<Vec<f64>>,
        data_end: usize,
        period: Option<i32>,
        forecast_len: usize,
        correlate: Option<&[f64]>,
        missing_valued: bool,
    ) -> Self {
        Self::new(algorithm, data, data_end, period, forecast_len, correlate, missing_valued)
            .expect("Failed to create Multivar model")
    }
    
    /// Make predictions using the Multivar model
    /// 
    /// # Errors
    /// 
    /// Returns an error if the underlying model's predict method fails.
    pub fn predict(&mut self, predict_var: usize, start: usize) -> Result<(), Error> {
        if self.algorithm.starts_with("LLB") {
            self.algo.predict(predict_var, start)?;
        }
        Ok(())
    }
    
    pub fn state(&self, ts_idx: usize, i: usize) -> f64 {
        if self.algorithm.starts_with("LLB") {
            self.algo.state(ts_idx, i)
        } else {
            self.algo.state(ts_idx, i)
        }
    }
    
    pub fn var(&self, ts_idx: usize, i: usize) -> f64 {
        if self.algorithm.starts_with("LLB") {
            self.algo.variance(i)
        } else {
            self.algo.var(ts_idx, i)
        }
    }
    
    pub fn datalen(&self) -> usize {
        self.algo.datalen()
    }
    
    pub fn multivariate(&self) -> bool {
        true
    }
    
    pub fn period(&self) -> Option<i32> {
        self.period
    }
    
    pub fn least_num_data(&self) -> usize {
        self.algo.least_num_data()
    }
    
    pub fn first_forecast_index(&self) -> usize {
        self.algo.first_forecast_index()
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bill_basic() {
        let data = vec![vec![1.0, 2.0, 3.0], vec![2.0, 3.0, 4.0]];
        let bill = BiLL::new(data, 3, 3).expect("Failed to create BiLL model");
        
        assert!(bill.state(0, 0).is_finite());
        assert!(bill.state(1, 0).is_finite());
        assert!(bill.var(0, 0) >= 0.0);
        assert_eq!(bill.datalen(), 3);
    }

    #[test]
    fn test_billmv_basic() {
        let data = vec![vec![Some(1.0), Some(2.0), Some(3.0)], vec![Some(2.0), Some(3.0), Some(4.0)]];
        let billmv = BiLLmv::new(data, 3, 3).expect("Failed to create BiLLmv model");
        
        assert!(billmv.state(0, 0).is_finite());
        assert!(billmv.state(1, 0).is_finite());
        assert_eq!(billmv.datalen(), 3);
    }

    #[test]
    fn test_billmv2_basic() {
        let data = vec![vec![Some(1.0), Some(2.0), Some(3.0)], vec![Some(2.0), Some(3.0), Some(4.0)]];
        let billmv2 = BiLLmv2::new(data, 3, 3).expect("Failed to create BiLLmv2 model");
        
        assert!(billmv2.state(0, 0).is_finite());
        assert!(billmv2.state(1, 0).is_finite());
        assert_eq!(billmv2.datalen(), 3);
    }

    #[test]
    fn test_llb_basic() {
        let data = vec![vec![1.0, 2.0, 3.0], vec![2.0, 3.0, 4.0], vec![3.0, 4.0, 5.0]];
        let llb = LLB::new(data, 3, 3).expect("Failed to create LLB model");
        
        assert!(llb.state(0, 0).is_finite());
        assert!(llb.state(1, 0).is_finite());
        assert!(llb.state(2, 0).is_finite());
        assert_eq!(llb.datalen(), 3);
    }

    #[test]
    fn test_llbmv_basic() {
        let data = vec![
            vec![Some(1.0), Some(2.0), Some(3.0)], 
            vec![Some(2.0), Some(3.0), Some(4.0)], 
            vec![Some(3.0), Some(4.0), Some(5.0)]
        ];
        let llbmv = LLBmv::new(data, 3, 3).expect("Failed to create LLBmv model");
        
        assert!(llbmv.state(0, 0).is_finite());
        assert!(llbmv.state(1, 0).is_finite());
        assert!(llbmv.state(2, 0).is_finite());
        assert_eq!(llbmv.datalen(), 3);
    }

    #[test]
    fn test_multivar_bill() {
        let data = vec![vec![1.0, 2.0, 3.0, 4.0], vec![2.0, 3.0, 4.0, 5.0]];
        let multivar = Multivar::new("BiLL", data, 4, None, 3, None, false).expect("Failed to create Multivar model");
        
        assert!(multivar.state(0, 0).is_finite());
        assert!(multivar.state(1, 0).is_finite());
        assert_eq!(multivar.datalen(), 4);
        assert!(multivar.multivariate());
    }

    #[test]
    fn test_multivar_llb() {
        let data = vec![vec![1.0, 2.0, 3.0], vec![2.0, 3.0, 4.0], vec![3.0, 4.0, 5.0]];
        let multivar = Multivar::new("LLB", data, 3, None, 3, None, false).expect("Failed to create Multivar model");
        
        assert!(multivar.state(0, 0).is_finite());
        assert!(multivar.state(1, 0).is_finite());
        assert!(multivar.state(2, 0).is_finite());
        assert_eq!(multivar.datalen(), 3);
        assert!(multivar.multivariate());
    }

    #[test]
    fn test_multivar_model_trait_bill() {
        let data = vec![vec![1.0, 2.0, 3.0], vec![2.0, 3.0, 4.0]];
        let bill = BiLL::new(data, 3, 3).expect("Failed to create BiLL model");
        
        // Test trait methods
        assert_eq!(bill.least_num_data(), 1);
        assert_eq!(bill.first_forecast_index(), 0);
    }

    #[test]
    fn test_multivar_model_trait_llb() {
        let data = vec![vec![1.0, 2.0, 3.0], vec![2.0, 3.0, 4.0], vec![3.0, 4.0, 5.0]];
        let llb = LLB::new(data, 3, 3).expect("Failed to create LLB model");
        
        // LLB's least_num_data returns 2 (need at least 2 data points to calculate covariance)
        assert_eq!(llb.least_num_data(), 2);
        // first_forecast_index returns 1
        assert_eq!(llb.first_forecast_index(), 1);
    }

    #[test]
    fn test_bill_state_consistency() {
        let data = vec![vec![1.0, 2.0, 3.0], vec![2.0, 3.0, 4.0]];
        let bill = BiLL::new(data, 3, 3).expect("Failed to create BiLL model");
        
        // Test state of two time series
        for ts_idx in 0..2 {
            for i in 0..3 {
                let state = bill.state(ts_idx, i);
                assert!(state.is_finite());
            }
        }
    }

    #[test]
    fn test_bill_var_consistency() {
        let data = vec![vec![1.0, 2.0, 3.0], vec![2.0, 3.0, 4.0]];
        let bill = BiLL::new(data, 3, 3).expect("Failed to create BiLL model");
        
        // Variance should be non-negative
        for ts_idx in 0..2 {
            for i in 0..3 {
                let var = bill.var(ts_idx, i);
                assert!(var >= 0.0 || var.is_nan());
            }
        }
    }

    #[test]
    fn test_llb_multiple_series() {
        let data = vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, 3.0, 4.0],
            vec![3.0, 4.0, 5.0],
            vec![4.0, 5.0, 6.0],
        ];
        let llb = LLB::new(data, 3, 3).expect("Failed to create LLB model");
        
        // Test multiple time series
        for ts_idx in 0..4 {
            assert!(llb.state(ts_idx, 0).is_finite());
            assert!(llb.state(ts_idx, 1).is_finite());
            assert!(llb.state(ts_idx, 2).is_finite());
        }
    }

    #[test]
    fn test_multivar_period() {
        let data = vec![vec![1.0, 2.0, 3.0], vec![2.0, 3.0, 4.0]];
        let multivar = Multivar::new("BiLL", data, 3, Some(2), 3, None, false).expect("Failed to create Multivar model");
        
        // Test period setting
        let period = multivar.period();
        assert_eq!(period, Some(2));
    }

    #[test]
    fn test_multivar_least_num_data() {
        let data = vec![vec![1.0, 2.0, 3.0], vec![2.0, 3.0, 4.0]];
        let multivar = Multivar::new("BiLL", data, 3, None, 3, None, false).expect("Failed to create Multivar model");
        
        assert!(multivar.least_num_data() >= 1);
    }
}
