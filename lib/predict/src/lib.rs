//! Time series state space models library
//!
//! This library provides various state space models for time series analysis and forecasting
//!
//! # Module Structure
//!
//! - `dist`: Statistical distributions module (Gamma, Chi-square, F distribution, Normal distribution, etc.)
//! - `optimize`: Optimization algorithms module (DFP, BFGS, etc.)
//! - `statespace`: State space models module
//!   - `utils`: Utility functions
//!   - `models`: Model implementations
//!     - `univariate`: Univariate models
//!     - `multivariate`: Multivariate models

pub mod dist;
pub mod optimize;
pub mod statespace;
pub mod utils;
pub mod models;

// Re-export commonly used types and functions for external use
pub use crate::statespace::{Datafeed, Univar, Multivar, is_univariate_algorithm, is_multivariate_algorithm};
pub use anyhow::Error;

