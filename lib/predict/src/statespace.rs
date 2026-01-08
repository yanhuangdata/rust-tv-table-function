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
    Datafeed, OptionF64, 
    option_to_f64_or_nan, f64_to_option_or_none, is_missing,
    autocovariance, correlogram, find_period, find_period2, find_period3,
    find_longest_continuous_stretch, fillin_mv,
    prediction_interval, prediction_interval_95,
    MAX_LAG, MAX_POINTS,
};

pub use crate::models::{
    StateSpaceModel, MultivarModel,
    is_supported_algorithm, is_univariate_algorithm, is_multivariate_algorithm,
    FORECAST,
};

pub use crate::models::univariate::{
    LL0, LL, LLT, LLP, LLP1, LLP2, LLP5, LLT2, Univar,
};

pub use crate::models::multivariate::{
    BiLL, BiLLmv, BiLLmv2, LLB, LLBmv, Multivar,
};
