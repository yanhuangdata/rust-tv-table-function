//! State space models module
//!
//! Provides implementations of univariate and multivariate state space models

use anyhow::Error;

pub mod multivariate;
pub mod univariate;

/// Default forecast length
pub const FORECAST: usize = 3;

/// State space model trait
pub trait StateSpaceModel {
    fn state(&self, i: usize) -> f64;
    fn var(&self, i: usize) -> f64;
    /// Direct access to p[i] array (consistent with Python's self.algos[ts_idx].p[i])
    fn p(&self, i: usize) -> f64;
    fn datalen(&self) -> usize;
    fn least_num_data(&self) -> usize;
    fn first_forecast_index(&self) -> usize;
}

/// Multivariate model trait
pub trait MultivarModel {
    fn state(&self, ts_idx: usize, i: usize) -> f64;
    fn var(&self, ts_idx: usize, i: usize) -> f64;
    fn variance(&self, i: usize) -> f64 {
        // Default implementation, some models can override
        self.var(0, i)
    }
    fn datalen(&self) -> usize;
    fn least_num_data(&self) -> usize;
    fn first_forecast_index(&self) -> usize;
    fn predict(&mut self, predict_var: usize, start: usize) -> Result<(), Error> {
        // Default empty implementation, some models can override
        let _ = (predict_var, start);
        Ok(())
    }
}

/// Algorithm mapping - Check if algorithm is supported (including univariate and multivariate algorithms)
pub fn is_supported_algorithm(algorithm: &str) -> bool {
    matches!(
        algorithm,
        "LL" | "LLT" | "LLP" | "LLP1" | "LLP2" | "LLP5" | "LLB" | "LLBmv" | "BiLL" | "BiLLmv"
    )
}

/// Check if algorithm is univariate
pub fn is_univariate_algorithm(algorithm: &str) -> bool {
    matches!(algorithm, "LL" | "LLT" | "LLP" | "LLP1" | "LLP2" | "LLP5")
}

/// Check if algorithm is multivariate
pub fn is_multivariate_algorithm(algorithm: &str) -> bool {
    matches!(algorithm, "LLB" | "LLBmv" | "BiLL" | "BiLLmv")
}
