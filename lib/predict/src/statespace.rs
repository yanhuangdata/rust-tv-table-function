//! State space models library
//!
//! This library provides various state space models for time series analysis and forecasting
//!
//! # Module Structure
//!
//! - `utils`: Utility functions module, includes missing value handling, autocovariance, correlogram, period finding, etc.
//! - `models`: State space models module
//!   - `univariate`: Univariate models (LL, LLT, LLP series, etc.)
//!   - `multivariate`: Multivariate models (BiLL, LLB, etc.)

// Re-export commonly used types and functions
pub use crate::utils::{
    Datafeed, MAX_LAG, MAX_POINTS, OptionF64, autocovariance, correlogram, f64_to_option_or_none,
    fillin_mv, find_longest_continuous_stretch, find_period, find_period2, find_period3,
    is_missing, option_to_f64_or_nan, prediction_interval, prediction_interval_95,
};

pub use crate::models::{
    FORECAST, MultivarModel, StateSpaceModel, is_multivariate_algorithm, is_supported_algorithm,
    is_univariate_algorithm,
};

pub use crate::models::univariate::{LL, LL0, LLP, LLP1, LLP2, LLP5, LLT, LLT2, Univar};

pub use crate::models::multivariate::{BiLL, BiLLmv, BiLLmv2, LLB, LLBmv, Multivar};
