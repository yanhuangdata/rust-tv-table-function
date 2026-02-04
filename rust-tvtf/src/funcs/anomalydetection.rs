use std::collections::HashMap;
use std::sync::Arc;

use crate::TableFunction;
use anyhow::Context;
use arrow::{
    array::{Array, ArrayRef, AsArray, Float64Array, StringArray},
    compute,
    record_batch::RecordBatch,
};
use arrow_schema::{DataType, Field, Schema};
use rust_tvtf_api::arg::{Arg, Args};

/// Field type analysis result
#[derive(Debug, Clone, PartialEq)]
enum FieldType {
    Numerical,
    Categorical,
    Mixed,
}

/// Anomaly detection result for a single record
#[derive(Debug, Clone, PartialEq)]
struct AnomalyRecord {
    log_prob: f64,
    probable_cause: String,
    probable_cause_freq: f64,
    max_freq: f64,
    is_anomaly: bool,
}

/// Summary data for action=summary
#[derive(Debug, Clone)]
struct SummaryData {
    num_anomalies: usize,
    thresh: f64,
    max_logprob: f64,
    min_logprob: f64,
    first_quartile: f64,
    third_quartile: f64,
}

/// AnomalyDetector - Splunk anomalydetection command implementation
///
/// Default: method=histogram, cutoff=true, action=filter (histogram/zscore)
///          method=iqr, action=transform (default for IQR)
#[derive(Debug)]
pub struct AnomalyDetector {
    /// Detection method: "histogram", "zscore", "iqr"
    method: String,
    /// Action: "filter", "annotate", "summary" (histogram/zscore)
    ///         "remove", "transform" (iqr)
    action: String,
    /// Number of bins for histogram method
    n_bins: usize,
    /// Whether to apply cutoff threshold (histogram/zscore)
    cutoff: bool,
    /// Probability threshold for zscore method
    pthresh: Option<f64>,
    /// Sensitivity adjustment: "strict", "default", "lenient", "very_lenient"
    /// Note: This is a custom extension, not in Splunk docs
    sensitivity: String,
    /// Field frequencies (value -> frequency)
    field_frequencies: HashMap<String, HashMap<String, f64>>,
    /// Field types
    field_types: HashMap<String, FieldType>,
    /// Calculated threshold
    threshold: Option<f64>,
    /// IQR parameter: multiplier for IQR (default 1.5)
    /// Note: This is from Splunk's outlier command
    param: Option<f64>,
    /// IQR flag: whether to use lower bound as well
    /// Note: This is from Splunk's outlier command
    uselower: bool,
    /// IQR flag: whether to prefix transformed values with "000"
    /// Note: This is from Splunk's outlier command
    /// LIMITATION: For Int64/Float64 column types, the "000" prefix cannot be stored
    /// as numeric types don't support string prefixes. In a full implementation,
    /// this would require either:
    ///    1. Converting numeric columns to string columns, or
    ///    2. Adding a separate marker/flag column to indicate transformed values
    ///
    /// Current implementation truncates values but cannot add the "000" prefix.
    mark: bool,
}

impl Default for AnomalyDetector {
    fn default() -> Self {
        AnomalyDetector {
            method: "histogram".to_string(),
            action: String::new(), // Will be set based on method in new()
            n_bins: 10,
            cutoff: true,
            pthresh: None,
            sensitivity: "default".to_string(),
            field_frequencies: HashMap::new(),
            field_types: HashMap::new(),
            threshold: None,
            param: Some(1.5),
            uselower: false,
            mark: false,
        }
    }
}

impl AnomalyDetector {
    pub fn new(
        _params: Option<Args>,
        named_params: Vec<(String, Arg)>,
    ) -> anyhow::Result<AnomalyDetector> {
        let mut detector = Self::default();
        let mut action_specified = false;

        // Parse all parameters first without validation
        for (name, arg) in named_params {
            match name.as_str() {
                "method" => {
                    if let Arg::String(method) = arg {
                        detector.method = method.to_lowercase();
                        if !matches!(detector.method.as_str(), "histogram" | "zscore" | "iqr") {
                            return Err(anyhow::anyhow!(
                                "Invalid method '{}'. Must be 'histogram', 'zscore', or 'iqr'",
                                method
                            ));
                        }
                    }
                }
                "action" => {
                    if let Arg::String(action) = arg {
                        detector.action = action.to_lowercase();
                        action_specified = true;
                    }
                }
                "bins" | "n_bins" => {
                    if let Arg::Int(n) = arg {
                        detector.n_bins = n as usize;
                        if detector.n_bins == 0 {
                            return Err(anyhow::anyhow!("Number of bins must be greater than 0"));
                        }
                    }
                }
                "cutoff" => {
                    if let Arg::Bool(cutoff) = arg {
                        detector.cutoff = cutoff;
                    }
                }
                "pthresh" => {
                    if let Arg::Float(p) = arg {
                        if p <= 0.0 || p >= 1.0 {
                            return Err(anyhow::anyhow!(
                                "pthresh must be between 0 and 1, got {}",
                                p
                            ));
                        }
                        detector.pthresh = Some(p);
                    }
                }
                "sensitivity" => {
                    if let Arg::String(sensitivity) = arg {
                        detector.sensitivity = sensitivity.to_lowercase();
                        if !matches!(
                            detector.sensitivity.as_str(),
                            "strict" | "default" | "lenient" | "very_lenient"
                        ) {
                            return Err(anyhow::anyhow!(
                                "Invalid sensitivity '{}'. Must be 'strict', 'default', 'lenient', or 'very_lenient'",
                                sensitivity
                            ));
                        }
                    }
                }
                // IQR-specific parameters
                "mark" => {
                    if let Arg::Bool(mark) = arg {
                        detector.mark = mark;
                    }
                }
                "param" => {
                    if let Arg::Float(p) = arg {
                        if p <= 0.0 {
                            return Err(anyhow::anyhow!("param must be greater than 0, got {}", p));
                        }
                        detector.param = Some(p);
                    }
                }
                "uselower" => {
                    if let Arg::Bool(uselower) = arg {
                        detector.uselower = uselower;
                    }
                }
                _ => {
                    // Ignore unknown parameters
                }
            }
        }

        // Set default action based on method (only if not explicitly specified)
        if !action_specified {
            detector.action = match detector.method.as_str() {
                "iqr" => "transform".to_string(), // IQR default is transform
                _ => "filter".to_string(),        // histogram/zscore default is filter
            };
        }

        // Validate action based on method (after all parameters are parsed)
        match detector.method.as_str() {
            "histogram" | "zscore" => {
                if !matches!(detector.action.as_str(), "filter" | "annotate" | "summary") {
                    return Err(anyhow::anyhow!(
                        "Invalid action '{}' for method '{}'. Must be 'filter', 'annotate', or 'summary'",
                        detector.action,
                        detector.method
                    ));
                }
            }
            "iqr" => {
                // pthresh is not supported for IQR method
                if detector.pthresh.is_some() {
                    return Err(anyhow::anyhow!(
                        "Invalid argument: 'pthresh' is not supported with method 'iqr'"
                    ));
                }
                if !matches!(
                    detector.action.as_str(),
                    "remove" | "transform" | "rm" | "tf"
                ) {
                    return Err(anyhow::anyhow!(
                        "Invalid action '{}' for method 'iqr'. Must be 'remove', 'transform', or their abbreviations 'rm', 'tf'",
                        detector.action
                    ));
                }
            }
            _ => {}
        }

        Ok(detector)
    }

    /// Check if a string value can be converted to a number
    fn is_numeric(value: &str) -> bool {
        let clean_value = value
            .trim()
            .trim_matches('"')
            .trim_matches('\'')
            .to_lowercase();
        clean_value.parse::<f64>().is_ok()
    }

    /// Clean and convert string to numeric value
    fn clean_numeric_value(value: &str) -> f64 {
        let clean_value = value
            .trim()
            .trim_matches('"')
            .trim_matches('\'')
            .to_lowercase();
        clean_value.parse::<f64>().unwrap_or(0.0)
    }

    /// Analyze field type based on value distribution
    fn analyze_field_type(values: &[String]) -> FieldType {
        let numeric_count = values.iter().filter(|v| Self::is_numeric(v)).count();
        let total_count = values.len();

        if total_count == 0 {
            return FieldType::Categorical;
        }

        let numeric_ratio = numeric_count as f64 / total_count as f64;

        if numeric_ratio > 0.8 {
            FieldType::Numerical
        } else if numeric_ratio < 0.2 {
            FieldType::Categorical
        } else {
            FieldType::Mixed
        }
    }

    /// Create histogram bins for numeric values - exactly matching NumPy's histogram
    fn create_histogram_bins(values: &[f64], n_bins: usize) -> HashMap<String, usize> {
        if values.is_empty() {
            return HashMap::new();
        }
        if values.len() == 1 {
            return HashMap::from([(values[0].to_string(), 0)]);
        }

        // Sort values to compute percentiles like NumPy
        let mut sorted_values = values.to_vec();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Compute bin edges using NumPy's percentile-based method
        // NumPy uses: bin_edges[i] = quantiles[i] where quantiles are evenly spaced
        let mut bin_edges: Vec<f64> = (0..=n_bins)
            .map(|i| {
                let rank = (i as f64) * (sorted_values.len() - 1) as f64 / n_bins as f64;
                let lower = rank.floor() as usize;
                let upper = (rank.ceil() as usize).min(sorted_values.len() - 1);
                let weight = rank - lower as f64;
                sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight
            })
            .collect();

        // Ensure unique bin edges
        bin_edges.dedup_by(|a, b| (*a - *b).abs() < 1e-10);

        // If all values are the same, put everything in bin 0
        if bin_edges.len() <= 1 {
            return values.iter().map(|v| (v.to_string(), 0)).collect();
        }

        // Use digitize-like logic: find the right bin for each value
        let mut value_to_bin = HashMap::new();
        for &value in values {
            let mut bin_idx = 0;
            for (i, &edge) in bin_edges.iter().enumerate().skip(1) {
                if value < edge {
                    bin_idx = i - 1;
                    break;
                }
                bin_idx = i;
            }
            bin_idx = bin_idx.min(n_bins - 1);
            value_to_bin.insert(value.to_string(), bin_idx);
        }

        value_to_bin
    }

    /// Build frequency distributions for each field
    fn build_field_frequencies(&mut self, data: &[Vec<(String, String)>], fields: &[String]) {
        self.field_frequencies.clear();
        self.field_types.clear();

        // First, analyze field types
        for field in fields {
            let values: Vec<String> = data
                .iter()
                .filter_map(|row| row.iter().find(|(f, _)| f == field))
                .map(|(_, v)| v.clone())
                .collect();

            let field_type = Self::analyze_field_type(&values);
            self.field_types.insert(field.clone(), field_type);
        }

        // Build frequency distributions
        for field in fields {
            let field_type = self
                .field_types
                .get(field)
                .unwrap_or(&FieldType::Categorical);

            match field_type {
                FieldType::Numerical => {
                    self.build_numerical_field_frequency(field, data);
                }
                _ => {
                    self.build_categorical_field_frequency(field, data);
                }
            }
        }
    }

    /// Build frequency distribution for a numerical field - exactly matching Python
    fn build_numerical_field_frequency(&mut self, field: &str, data: &[Vec<(String, String)>]) {
        // Collect numeric values (matching Python's _clean_numeric_value)
        let values: Vec<f64> = data
            .iter()
            .filter_map(|row| row.iter().find(|(f, _)| f == field))
            .map(|(_, v)| Self::clean_numeric_value(v))
            .collect();

        if values.is_empty() {
            return;
        }

        // Calculate unique count (matching Python)
        let unique_count = {
            let mut unique: Vec<f64> = values.clone();
            unique.sort_by(|a, b| a.partial_cmp(b).unwrap());
            unique.dedup_by(|a, b| (*a - *b).abs() < 1e-10);
            unique.len()
        };

        let n_bins = self.n_bins.min(unique_count);

        if unique_count > n_bins {
            // Use histogram binning - exactly matching NumPy
            let value_to_bin = Self::create_histogram_bins(&values, n_bins);

            // Count values in each bin
            let mut bin_counts = vec![0usize; n_bins];
            for value in &values {
                if let Some(&bin_idx) = value_to_bin.get(&value.to_string()) {
                    bin_counts[bin_idx] += 1;
                }
            }

            let total_count = values.len() as f64;

            // Build frequency map - match Python's logic exactly
            let mut value_frequencies = HashMap::new();
            for (i, row) in data.iter().enumerate() {
                if let Some((_, v)) = row.iter().find(|(f, _)| f == field) {
                    let numeric_val = values[i];
                    if let Some(&bin_idx) = value_to_bin.get(&numeric_val.to_string()) {
                        let bin_freq = bin_counts[bin_idx] as f64 / total_count;
                        value_frequencies.insert(v.clone(), bin_freq);
                    }
                }
            }

            self.field_frequencies
                .insert(field.to_string(), value_frequencies);
        } else {
            // Use direct frequency counting - matching Python
            let mut counter: HashMap<String, usize> = HashMap::new();
            for row in data {
                if let Some((_, v)) = row.iter().find(|(f, _)| f == field) {
                    *counter.entry(v.clone()).or_insert(0) += 1;
                }
            }

            let total = counter.values().sum::<usize>() as f64;
            let value_frequencies: HashMap<String, f64> = counter
                .into_iter()
                .map(|(value, count)| (value, count as f64 / total))
                .collect();

            self.field_frequencies
                .insert(field.to_string(), value_frequencies);
        }
    }

    /// Build frequency distribution for a categorical field
    fn build_categorical_field_frequency(&mut self, field: &str, data: &[Vec<(String, String)>]) {
        let counter: HashMap<String, usize> = data
            .iter()
            .filter_map(|row| row.iter().find(|(f, _)| f == field))
            .map(|(_, v)| v.clone())
            .fold(HashMap::new(), |mut acc, v| {
                *acc.entry(v).or_insert(0) += 1;
                acc
            });

        let total = counter.values().sum::<usize>() as f64;
        let value_frequencies: HashMap<String, f64> = counter
            .into_iter()
            .map(|(value, count)| (value, count as f64 / total))
            .collect();

        self.field_frequencies
            .insert(field.to_string(), value_frequencies);
    }

    /// Calculate event probability as product of field value frequencies
    fn calculate_event_probability(&self, row: &[(String, String)], fields: &[String]) -> f64 {
        let mut probability = 1.0f64;

        for field in fields {
            if let Some((_, value)) = row.iter().find(|(f, _)| f == field)
                && let Some(field_freqs) = self.field_frequencies.get(field)
            {
                let freq = field_freqs.get(value).copied().unwrap_or(1e-10);
                probability *= freq;
            }
        }

        probability.max(1e-50)
    }

    /// Calculate threshold based on the selected method
    fn calculate_threshold(&mut self, log_probabilities: &[f64], fields: &[String]) -> f64 {
        if log_probabilities.is_empty() {
            return -10.0;
        }

        let mut sorted_probs: Vec<f64> = log_probabilities.to_vec();
        sorted_probs.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let threshold = match self.method.as_str() {
            "histogram" => self.calculate_histogram_threshold(&sorted_probs, fields),
            "zscore" => self.calculate_zscore_threshold(log_probabilities),
            "iqr" => self.calculate_iqr_threshold(&sorted_probs),
            _ => self.calculate_histogram_threshold(&sorted_probs, fields),
        };

        // Apply sensitivity adjustment (Python 逻辑)
        self.apply_sensitivity_adjustment(threshold, log_probabilities)
    }

    /// Calculate threshold using histogram method (Splunk default)
    fn calculate_histogram_threshold(&mut self, sorted_probs: &[f64], fields: &[String]) -> f64 {
        let q1 = percentile(sorted_probs, 25.0);
        let q3 = percentile(sorted_probs, 75.0);
        let iqr = q3 - q1;

        if self.cutoff {
            // cutoff=true (default): Modified formula for fewer anomalies
            let categorical_count = fields
                .iter()
                .filter(|f| {
                    self.field_types
                        .get(*f)
                        .map(|t| *t == FieldType::Categorical)
                        .unwrap_or(false)
                })
                .count();
            let numerical_count = fields
                .iter()
                .filter(|f| {
                    self.field_types
                        .get(*f)
                        .map(|t| *t == FieldType::Numerical)
                        .unwrap_or(false)
                })
                .count();
            let mixed_count = fields
                .iter()
                .filter(|f| {
                    self.field_types
                        .get(*f)
                        .map(|t| *t == FieldType::Mixed)
                        .unwrap_or(false)
                })
                .count();

            if categorical_count > 0 && numerical_count == 0 && mixed_count == 0 {
                // Pure categorical: Use very strict threshold (only absolute minimums)
                // Python: unique_probs[0] + 0.001
                let mut unique_probs: Vec<f64> = sorted_probs.to_vec();
                unique_probs.sort_by(|a, b| a.partial_cmp(b).unwrap());
                unique_probs.dedup_by(|a, b| (*a - *b).abs() < 1e-10);
                if unique_probs.len() > 1 {
                    unique_probs[0] + 0.001
                } else {
                    sorted_probs[0] + 0.001
                }
            } else {
                // Mixed or numerical: Use modified approach for better sensitivity
                let base_threshold = q1 - 1.5 * iqr;

                // MAD: Python uses median of absolute deviations from median
                // np.median(np.abs(log_probs - np.median(log_probs)))
                let median = median(sorted_probs);
                let abs_deviations: Vec<f64> =
                    sorted_probs.iter().map(|p| (p - median).abs()).collect();
                let mad = percentile(&abs_deviations, 50.0); // 使用中位数

                let mad_threshold = if mad > 0.0 {
                    median - 3.0 * mad
                } else {
                    base_threshold
                };

                // Z-score constraint
                let mean: f64 = sorted_probs.iter().sum::<f64>() / sorted_probs.len() as f64;
                let variance: f64 = sorted_probs.iter().map(|p| (p - mean).powi(2)).sum::<f64>()
                    / sorted_probs.len() as f64;
                let std = variance.sqrt();

                let z_threshold = if std > 0.0 {
                    mean - 3.0 * std
                } else {
                    base_threshold
                };

                if mixed_count > 0 {
                    // Mixed fields: Use a dual-threshold approach
                    // Python: max(statistical_threshold, rank_based_threshold)
                    let statistical_threshold = base_threshold.max(mad_threshold);

                    // Rank-based threshold for ~1% anomalies (Python default)
                    let target_anomaly_rate = 0.01;
                    let target_count =
                        (sorted_probs.len() as f64 * target_anomaly_rate).max(3.0) as usize;

                    let rank_based_threshold = if target_count < sorted_probs.len() {
                        sorted_probs[target_count - 1] + 0.001
                    } else {
                        statistical_threshold
                    };

                    // Use the more lenient threshold to catch more mixed anomalies
                    statistical_threshold.max(rank_based_threshold)
                } else {
                    // Pure numerical: be more restrictive
                    // Python: min(base_threshold, mad_threshold, z_threshold)
                    base_threshold.min(mad_threshold).min(z_threshold)
                }
            }
        } else {
            // cutoff=false: Pure IQR formula without modification
            q1 - 1.5 * iqr
        }
    }

    /// Calculate threshold using z-score method
    fn calculate_zscore_threshold(&self, log_probs: &[f64]) -> f64 {
        let mean: f64 = log_probs.iter().sum::<f64>() / log_probs.len() as f64;
        let variance: f64 =
            log_probs.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / log_probs.len() as f64;
        let std = variance.sqrt();

        let pthresh = self.pthresh.unwrap_or(0.01);
        // Python: z_score = -np.log(pthresh)
        let z_score = -pthresh.ln();

        if std > 0.0 {
            mean - z_score * std
        } else {
            mean - 3.0
        }
    }

    /// Calculate threshold using IQR method
    /// Uses self.param (default 1.5) as the multiplier for IQR
    fn calculate_iqr_threshold(&self, sorted_probs: &[f64]) -> f64 {
        let q1 = percentile(sorted_probs, 25.0);
        let q3 = percentile(sorted_probs, 75.0);
        let iqr = q3 - q1;
        let param = self.param.unwrap_or(1.5);
        q1 - param * iqr
    }

    /// Apply sensitivity adjustment to threshold
    /// Note: log_probs may not be sorted, so we sort it for percentile calculation
    fn apply_sensitivity_adjustment(&self, threshold: f64, log_probs: &[f64]) -> f64 {
        // Sort log_probs for percentile calculation
        let mut sorted_probs: Vec<f64> = log_probs.to_vec();
        sorted_probs.sort_by(|a, b| a.partial_cmp(b).unwrap());

        match self.sensitivity.as_str() {
            "strict" => {
                // Fewer anomalies: make threshold more restrictive
                // Python: adjustment = -0.3 * iqr if iqr > 0 else -0.5
                let q1 = percentile(&sorted_probs, 25.0);
                let iqr = percentile(&sorted_probs, 75.0) - q1;
                if iqr > 0.0 {
                    threshold - 0.3 * iqr
                } else {
                    threshold - 0.5
                }
            }
            "lenient" => {
                // More anomalies: make threshold less restrictive
                // Python: adjustment = 0.5 * iqr if iqr > 0 else 1.0
                let q1 = percentile(&sorted_probs, 25.0);
                let iqr = percentile(&sorted_probs, 75.0) - q1;
                if iqr > 0.0 {
                    threshold + 0.5 * iqr
                } else {
                    threshold + 1.0
                }
            }
            "very_lenient" => {
                // Many more anomalies: make threshold much less restrictive
                // Python: adjustment = 1.0 * iqr if iqr > 0 else 2.0
                let q1 = percentile(&sorted_probs, 25.0);
                let iqr = percentile(&sorted_probs, 75.0) - q1;
                if iqr > 0.0 {
                    threshold + 1.0 * iqr
                } else {
                    threshold + 2.0
                }
            }
            _ => threshold, // "default"
        }
    }

    /// Detect anomalies in the data
    fn detect_anomalies(
        &mut self,
        data: &[Vec<(String, String)>],
        fields: &[String],
    ) -> Vec<AnomalyRecord> {
        // Build field frequencies
        self.build_field_frequencies(data, fields);

        if data.is_empty() || fields.is_empty() {
            return Vec::new();
        }

        // Calculate event probabilities
        let mut records: Vec<AnomalyRecord> = data
            .iter()
            .map(|row| {
                let probability = self.calculate_event_probability(row, fields);
                let log_prob = probability.ln();

                // Find probable cause
                let mut min_freq = f64::INFINITY;
                let mut probable_cause = fields.first().cloned().unwrap_or_default();
                let mut probable_cause_freq = 1.0;

                // For categorical data, prioritize categorical fields
                let categorical_fields: Vec<&String> = fields
                    .iter()
                    .filter(|f| {
                        self.field_types
                            .get(*f)
                            .map(|t| *t == FieldType::Categorical)
                            .unwrap_or(false)
                    })
                    .collect();

                let check_fields: Vec<&String> = if !categorical_fields.is_empty() {
                    categorical_fields
                } else if fields.contains(&"value".to_string()) {
                    let mut prioritized: Vec<&String> = vec![];
                    prioritized.push(fields.iter().find(|f| *f == "value").unwrap());
                    prioritized.extend(fields.iter().filter(|f| *f != "value"));
                    prioritized
                } else {
                    fields.iter().collect()
                };

                for &field in &check_fields {
                    if let Some((_, value)) = row.iter().find(|(f, _)| f == field)
                        && let Some(field_freqs) = self.field_frequencies.get(field)
                        && let Some(&freq) = field_freqs.get(value)
                        && freq < min_freq
                    {
                        min_freq = freq;
                        probable_cause = field.clone();
                        probable_cause_freq = freq;
                    }
                }

                // Calculate max frequency - Python logic
                // If categorical >= len(fields)//2: use simple max
                // Otherwise: use cumulative method covering 91%
                let all_field_freqs: Vec<f64> = self
                    .field_frequencies
                    .values()
                    .flat_map(|h| h.values().copied())
                    .collect();

                let max_freq = if !all_field_freqs.is_empty() {
                    let categorical_count = fields
                        .iter()
                        .filter(|f| {
                            self.field_types
                                .get(*f)
                                .map(|t| *t == FieldType::Categorical)
                                .unwrap_or(false)
                        })
                        .count();

                    if categorical_count >= fields.len() / 2 {
                        // Primarily categorical - use simple max frequency
                        all_field_freqs.iter().fold(0.0f64, |acc, v| acc.max(*v))
                    } else {
                        // Primarily numerical - use cumulative method
                        let mut sorted_freqs = all_field_freqs;
                        sorted_freqs.sort_by(|a, b| b.partial_cmp(a).unwrap()); // descending
                        let mut cumulative = 0.0f64;
                        let target_coverage = 0.91;

                        for freq in &sorted_freqs {
                            cumulative += freq;
                            if cumulative >= target_coverage {
                                break;
                            }
                        }
                        cumulative
                    }
                } else {
                    1.0
                };

                AnomalyRecord {
                    log_prob,
                    probable_cause,
                    probable_cause_freq,
                    max_freq,
                    is_anomaly: false,
                }
            })
            .collect();

        // Calculate threshold
        let log_probs: Vec<f64> = records.iter().map(|r| r.log_prob).collect();
        self.threshold = Some(self.calculate_threshold(&log_probs, fields));

        // Mark anomalies
        if let Some(threshold) = self.threshold {
            let mut anomalous_probs: Vec<f64> = records
                .iter()
                .filter(|r| r.log_prob < threshold)
                .map(|r| r.log_prob)
                .collect();
            anomalous_probs.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let representative_log_prob = anomalous_probs.first().copied();

            for record in &mut records {
                if record.log_prob < threshold {
                    record.is_anomaly = true;
                    if let Some(rep_prob) = representative_log_prob {
                        record.log_prob = rep_prob;
                    }
                }
            }
        }

        records
    }

    /// Build summary data for action=summary
    fn build_summary(&self, anomaly_records: &[AnomalyRecord], log_probs: &[f64]) -> SummaryData {
        let num_anomalies = anomaly_records.iter().filter(|r| r.is_anomaly).count();

        let thresh = self.threshold.unwrap_or(0.0);

        let max_logprob = log_probs
            .iter()
            .fold(f64::NEG_INFINITY, |acc, &p| acc.max(p));
        let min_logprob = log_probs.iter().fold(f64::INFINITY, |acc, &p| acc.min(p));

        // Sort log_probs for percentile calculation
        let mut sorted_probs: Vec<f64> = log_probs.to_vec();
        sorted_probs.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let first_quartile = percentile(&sorted_probs, 25.0);
        let third_quartile = percentile(&sorted_probs, 75.0);

        SummaryData {
            num_anomalies,
            thresh,
            max_logprob,
            min_logprob,
            first_quartile,
            third_quartile,
        }
    }

    /// Create summary output for action=summary (following Splunk docs - only 6 fields)
    fn create_summary_output_only(
        &self,
        summary: SummaryData,
    ) -> anyhow::Result<Option<RecordBatch>> {
        let fields = vec![
            Arc::new(Field::new("num_anomalies", DataType::Int64, false)),
            Arc::new(Field::new("thresh", DataType::Float64, false)),
            Arc::new(Field::new("max_logprob", DataType::Float64, false)),
            Arc::new(Field::new("min_logprob", DataType::Float64, false)),
            Arc::new(Field::new("1st_quartile", DataType::Float64, false)),
            Arc::new(Field::new("3rd_quartile", DataType::Float64, false)),
        ];

        let schema = Arc::new(Schema::new(fields));

        let columns: Vec<ArrayRef> = vec![
            Arc::new(arrow::array::Int64Array::from(vec![
                summary.num_anomalies as i64,
            ])),
            Arc::new(Float64Array::from(vec![summary.thresh])),
            Arc::new(Float64Array::from(vec![summary.max_logprob])),
            Arc::new(Float64Array::from(vec![summary.min_logprob])),
            Arc::new(Float64Array::from(vec![summary.first_quartile])),
            Arc::new(Float64Array::from(vec![summary.third_quartile])),
        ];

        RecordBatch::try_new(schema, columns)
            .map(Some)
            .context("Failed to create summary output")
    }

    /// Create filter output for action=filter
    /// Returns anomalous events only
    fn create_filter_output(
        &self,
        anomaly_records: &[AnomalyRecord],
        input: &RecordBatch,
    ) -> anyhow::Result<Option<RecordBatch>> {
        let anomalies: Vec<&AnomalyRecord> =
            anomaly_records.iter().filter(|r| r.is_anomaly).collect();

        self.build_filter_output(&anomalies, anomaly_records, input)
    }

    /// Create remove output for action=remove (IQR method)
    /// Returns normal events only (removes outliers)
    fn create_remove_output(
        &self,
        anomaly_records: &[AnomalyRecord],
        input: &RecordBatch,
    ) -> anyhow::Result<Option<RecordBatch>> {
        let normal_events: Vec<&AnomalyRecord> =
            anomaly_records.iter().filter(|r| !r.is_anomaly).collect();

        self.build_filter_output(&normal_events, anomaly_records, input)
    }

    /// Common helper to build filter/remove output
    fn build_filter_output(
        &self,
        records: &[&AnomalyRecord],
        anomaly_records: &[AnomalyRecord],
        input: &RecordBatch,
    ) -> anyhow::Result<Option<RecordBatch>> {
        // Build output schema with additional columns
        let mut output_fields = input.schema().fields().to_vec();
        output_fields.push(Arc::new(Field::new(
            "log_event_prob",
            DataType::Float64,
            false,
        )));
        output_fields.push(Arc::new(Field::new("max_freq", DataType::Float64, false)));
        output_fields.push(Arc::new(Field::new(
            "probable_cause",
            DataType::Utf8,
            false,
        )));
        output_fields.push(Arc::new(Field::new(
            "probable_cause_freq",
            DataType::Float64,
            false,
        )));
        let output_schema = Arc::new(Schema::new(output_fields));

        if records.is_empty() {
            return Ok(Some(RecordBatch::new_empty(output_schema)));
        }

        // Build output columns
        let mut output_columns: Vec<ArrayRef> = Vec::new();

        // Collect the actual row indices from the original input
        // records are references to AnomalyRecords, which correspond to input rows
        let num_output_rows = records.len();

        // Copy original columns for selected rows
        for col_idx in 0..input.num_columns() {
            let col = input.column(col_idx);
            let mut indices: Vec<u32> = Vec::with_capacity(num_output_rows);

            // Build a mapping from record pointer to its index in anomaly_records
            // Then use that index as the row index in input
            for (anomaly_idx, anomaly_record) in anomaly_records.iter().enumerate() {
                if records.contains(&anomaly_record) {
                    indices.push(anomaly_idx as u32);
                }
            }

            if !indices.is_empty() {
                let indices_array = arrow::array::UInt32Array::from(indices);
                let taken_col = compute::take(col.as_ref(), &indices_array, None)
                    .context("Failed to take columns for output")?;
                output_columns.push(taken_col);
            } else {
                let empty_array = match col.data_type() {
                    DataType::Utf8 => Arc::new(StringArray::from(Vec::<&str>::new())) as ArrayRef,
                    DataType::Int64 => {
                        Arc::new(arrow::array::Int64Array::from(Vec::<i64>::new())) as ArrayRef
                    }
                    DataType::Float64 => {
                        Arc::new(Float64Array::from(Vec::<f64>::new())) as ArrayRef
                    }
                    _ => Arc::new(arrow::array::NullArray::new(0)) as ArrayRef,
                };
                output_columns.push(empty_array);
            }
        }

        // Add anomaly information columns
        let log_probs: Vec<f64> = records.iter().map(|r| r.log_prob).collect();
        output_columns.push(Arc::new(Float64Array::from(log_probs)));

        let max_freqs: Vec<f64> = records.iter().map(|r| r.max_freq).collect();
        output_columns.push(Arc::new(Float64Array::from(max_freqs)));

        let probable_causes: Vec<&str> =
            records.iter().map(|r| r.probable_cause.as_str()).collect();
        output_columns.push(Arc::new(StringArray::from(probable_causes)));

        let probable_cause_freqs: Vec<f64> =
            records.iter().map(|r| r.probable_cause_freq).collect();
        output_columns.push(Arc::new(Float64Array::from(probable_cause_freqs)));

        let output_batch = RecordBatch::try_new(output_schema, output_columns)
            .context("Failed to create output RecordBatch")?;

        Ok(Some(output_batch))
    }

    /// Create transform output for IQR method - truncates outlier values
    /// According to Splunk docs:
    /// - Truncates outlying value to the threshold for outliers
    /// - If mark=true, prefixes the value with "000"
    ///
    /// # Mark Parameter Limitation
    /// For Int64/Float64 column types, the "000" prefix cannot be stored as numeric
    /// types don't support string prefixes. In a full implementation, this would require:
    ///    1. Converting numeric columns to string columns, or
    ///    2. Adding a separate marker/flag column to indicate transformed values
    ///
    /// Current implementation truncates values but cannot add the "000" prefix.
    fn create_transform_output(
        &self,
        anomaly_records: &[AnomalyRecord],
        input: &RecordBatch,
    ) -> anyhow::Result<Option<RecordBatch>> {
        let param = self.param.unwrap_or(1.5);

        // Build output schema (same as input for transform)
        let output_schema = input.schema();

        let num_rows = input.num_rows();
        let mut output_columns: Vec<ArrayRef> = Vec::new();

        // Pre-calculate field bounds for each numeric column
        // According to Splunk outlier docs, IQR should be calculated per numeric field
        let mut field_bounds: HashMap<usize, (f64, f64)> = HashMap::new();

        for col_idx in 0..input.num_columns() {
            let col = input.column(col_idx);

            match col.data_type() {
                DataType::Int64 => {
                    let arr = col
                        .as_any()
                        .downcast_ref::<arrow::array::Int64Array>()
                        .unwrap();
                    let values: Vec<i64> = (0..num_rows).map(|i| arr.value(i)).collect();
                    let float_values: Vec<f64> = values.iter().map(|v| *v as f64).collect();
                    let mut sorted_values = float_values.clone();
                    sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let q1 = percentile(&sorted_values, 25.0);
                    let q3 = percentile(&sorted_values, 75.0);
                    let iqr = q3 - q1;
                    let upper_bound = q3 + param * iqr;
                    let lower_bound = q1 - param * iqr;
                    field_bounds.insert(col_idx, (upper_bound, lower_bound));
                }
                DataType::Float64 => {
                    let arr = col.as_any().downcast_ref::<Float64Array>().unwrap();
                    let values: Vec<f64> = (0..num_rows).map(|i| arr.value(i)).collect();
                    let mut sorted_values = values.clone();
                    sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let q1 = percentile(&sorted_values, 25.0);
                    let q3 = percentile(&sorted_values, 75.0);
                    let iqr = q3 - q1;
                    let upper_bound = q3 + param * iqr;
                    let lower_bound = q1 - param * iqr;
                    field_bounds.insert(col_idx, (upper_bound, lower_bound));
                }
                _ => {}
            }
        }

        // Process each column - numeric columns get transformed based on their own bounds
        for col_idx in 0..input.num_columns() {
            let col = input.column(col_idx);

            match col.data_type() {
                DataType::Int64 => {
                    let arr = col
                        .as_any()
                        .downcast_ref::<arrow::array::Int64Array>()
                        .unwrap();
                    let mut values: Vec<i64> = Vec::with_capacity(num_rows);

                    // Get bounds for this field
                    let (upper_bound, lower_bound) = field_bounds
                        .get(&col_idx)
                        .copied()
                        .unwrap_or((f64::MAX, f64::MIN));

                    for row_idx in 0..num_rows {
                        let is_anomaly = if row_idx < anomaly_records.len() {
                            anomaly_records[row_idx].is_anomaly
                        } else {
                            false
                        };

                        if is_anomaly {
                            let original_value = arr.value(row_idx);
                            let _transformed_value = if self.mark {
                                // Mark parameter would prefix with "000" as per Splunk docs
                                // Note: For Int64 type, we cannot store "000" prefix.
                                // In a full implementation, this would require converting to string
                                // or adding a separate marker column. For now, we just truncate.
                                format!("000{}", original_value)
                            } else {
                                original_value.to_string()
                            };
                            // Truncate to upper_bound or lower_bound based on direction and field bounds
                            if original_value > upper_bound as i64 {
                                values.push(upper_bound as i64);
                            } else if original_value < lower_bound as i64 && self.uselower {
                                values.push(lower_bound as i64);
                            } else {
                                values.push(original_value);
                            }
                        } else {
                            values.push(arr.value(row_idx));
                        }
                    }
                    output_columns
                        .push(Arc::new(arrow::array::Int64Array::from(values)) as ArrayRef);
                }
                DataType::Float64 => {
                    let arr = col.as_any().downcast_ref::<Float64Array>().unwrap();
                    let mut values: Vec<f64> = Vec::with_capacity(num_rows);

                    // Get bounds for this field
                    let (upper_bound, lower_bound) = field_bounds
                        .get(&col_idx)
                        .copied()
                        .unwrap_or((f64::MAX, f64::MIN));

                    for row_idx in 0..num_rows {
                        let is_anomaly = if row_idx < anomaly_records.len() {
                            anomaly_records[row_idx].is_anomaly
                        } else {
                            false
                        };

                        if is_anomaly {
                            let original_value = arr.value(row_idx);
                            let _transformed_value = if self.mark {
                                // Mark parameter would prefix with "000" as per Splunk docs
                                // Note: For Float64 type, we cannot store "000" prefix.
                                // In a full implementation, this would require converting to string
                                // or adding a separate marker column. For now, we just truncate.
                                format!("000{}", original_value)
                            } else {
                                upper_bound.to_string()
                            };
                            // Truncate to upper_bound or lower_bound based on direction and field bounds
                            if original_value > upper_bound {
                                values.push(upper_bound);
                            } else if original_value < lower_bound && self.uselower {
                                values.push(lower_bound);
                            } else {
                                values.push(original_value);
                            }
                        } else {
                            values.push(arr.value(row_idx));
                        }
                    }
                    output_columns.push(Arc::new(Float64Array::from(values)) as ArrayRef);
                }
                _ => {
                    // Non-numeric columns: copy as-is
                    output_columns.push(col.clone());
                }
            }
        }

        if num_rows > 0 {
            RecordBatch::try_new(output_schema, output_columns)
                .map(Some)
                .context("Failed to create transform output")
        } else {
            RecordBatch::try_new(output_schema, output_columns)
                .map(Some)
                .context("Failed to create empty transform output")
        }
    }

    /// Create annotated output for action=annotate (returns all events with annotation fields)
    fn create_annotated_output(
        &self,
        anomaly_records: &[AnomalyRecord],
        input: &RecordBatch,
        output_schema: Arc<Schema>,
    ) -> anyhow::Result<Option<RecordBatch>> {
        let num_rows = input.num_rows();
        let mut output_columns: Vec<ArrayRef> = Vec::new();

        // Copy all original columns
        for col_idx in 0..input.num_columns() {
            let col = input.column(col_idx);
            output_columns.push(col.clone());
        }

        // Add annotation fields
        let is_anomaly: Vec<bool> = anomaly_records.iter().map(|r| r.is_anomaly).collect();
        output_columns.push(Arc::new(arrow::array::BooleanArray::from(is_anomaly)));

        let log_probs: Vec<f64> = anomaly_records.iter().map(|r| r.log_prob).collect();
        output_columns.push(Arc::new(Float64Array::from(log_probs)));

        let max_freqs: Vec<f64> = anomaly_records.iter().map(|r| r.max_freq).collect();
        output_columns.push(Arc::new(Float64Array::from(max_freqs)));

        let probable_causes: Vec<&str> = anomaly_records
            .iter()
            .map(|r| r.probable_cause.as_str())
            .collect();
        output_columns.push(Arc::new(StringArray::from(probable_causes)));

        let probable_cause_freqs: Vec<f64> = anomaly_records
            .iter()
            .map(|r| r.probable_cause_freq)
            .collect();
        output_columns.push(Arc::new(Float64Array::from(probable_cause_freqs)));

        if num_rows > 0 {
            RecordBatch::try_new(output_schema, output_columns)
                .map(Some)
                .context("Failed to create annotated output")
        } else {
            RecordBatch::try_new(output_schema, output_columns)
                .map(Some)
                .context("Failed to create empty annotated output")
        }
    }
}

/// Calculate percentile of a sorted array
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }

    let rank = (p / 100.0) * (sorted.len() as f64 - 1.0);
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;

    if lower == upper {
        sorted[lower]
    } else {
        let weight = rank - lower as f64;
        sorted[lower] * (1.0 - weight) + sorted[upper] * weight
    }
}

/// Calculate median of an array
fn median(sorted: &[f64]) -> f64 {
    percentile(sorted, 50.0)
}

impl TableFunction for AnomalyDetector {
    fn process(&mut self, input: RecordBatch) -> anyhow::Result<Option<RecordBatch>> {
        if input.num_rows() == 0 {
            return Ok(Some(input));
        }

        let schema = input.schema();
        let fields: Vec<String> = schema.fields().iter().map(|f| f.name().clone()).collect();

        // Convert RecordBatch to Vec<Vec<(String, String)>>
        let data: Vec<Vec<(String, String)>> = (0..input.num_rows())
            .map(|row_idx| {
                let mut row = Vec::new();
                for col_idx in 0..input.num_columns() {
                    let col = input.column(col_idx);
                    let field_name = schema.field(col_idx).name();

                    let value = if col.is_null(row_idx) {
                        "null".to_string()
                    } else {
                        match col.data_type() {
                            DataType::Utf8 => col.as_string::<i32>().value(row_idx).to_string(),
                            DataType::Int8
                            | DataType::Int16
                            | DataType::Int32
                            | DataType::Int64 => {
                                let arr = col
                                    .as_any()
                                    .downcast_ref::<arrow::array::Int64Array>()
                                    .unwrap();
                                arr.value(row_idx).to_string()
                            }
                            DataType::Float32 | DataType::Float64 => {
                                let arr = col.as_any().downcast_ref::<Float64Array>().unwrap();
                                arr.value(row_idx).to_string()
                            }
                            _ => col.as_string::<i32>().value(row_idx).to_string(),
                        }
                    };

                    row.push((field_name.clone(), value));
                }
                row
            })
            .collect();

        // Detect anomalies
        let anomaly_records = self.detect_anomalies(&data, &fields);
        let log_probs: Vec<f64> = anomaly_records.iter().map(|r| r.log_prob).collect();

        // Handle different actions
        match self.action.as_str() {
            "summary" => {
                // Build summary data
                let summary = self.build_summary(&anomaly_records, &log_probs);
                self.create_summary_output_only(summary)
            }
            "annotate" => {
                // Build output schema with annotation fields
                // Note: Following Splunk docs, annotate adds log_event_prob, probable_cause,
                // probable_cause_freq, max_freq. The is_anomaly field is a custom addition
                // for convenience.
                let mut output_fields = schema.fields().to_vec();
                output_fields.push(Arc::new(Field::new("is_anomaly", DataType::Boolean, false)));
                output_fields.push(Arc::new(Field::new(
                    "log_event_prob",
                    DataType::Float64,
                    false,
                )));
                output_fields.push(Arc::new(Field::new("max_freq", DataType::Float64, false)));
                output_fields.push(Arc::new(Field::new(
                    "probable_cause",
                    DataType::Utf8,
                    false,
                )));
                output_fields.push(Arc::new(Field::new(
                    "probable_cause_freq",
                    DataType::Float64,
                    false,
                )));
                let output_schema = Arc::new(Schema::new(output_fields));
                self.create_annotated_output(&anomaly_records, &input, output_schema)
            }
            "remove" | "rm" => {
                // Remove anomalous events (for IQR method)
                // Returns normal events (non-anomalies), removes outliers
                // "rm" is the abbreviation for "remove"
                self.create_remove_output(&anomaly_records, &input)
            }
            "transform" | "tf" => {
                // Transform outlier values (for IQR method)
                // Truncates outlying values to the threshold
                // "tf" is the abbreviation for "transform"
                self.create_transform_output(&anomaly_records, &input)
            }
            _ => {
                // filter (default for histogram/zscore)
                // Filter anomalies and create output
                self.create_filter_output(&anomaly_records, &input)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Int64Array, StringArray};

    #[test]
    fn test_anomaly_detector_basic() {
        // Create test data with some obvious anomalies
        let schema = Arc::new(Schema::new(vec![
            Field::new("status", DataType::Utf8, false),
            Field::new("code", DataType::Int64, false),
        ]));

        // Most entries are "success" with 200 (100 times)
        // Anomalies: "error" with 500 (3 times), "unknown" with 999 (1 time)
        // This larger dataset ensures the threshold is sensitive enough
        let status_col = Arc::new(StringArray::from(
            std::iter::repeat("success")
                .take(100)
                .chain(std::iter::repeat("error").take(3))
                .chain(std::iter::once("unknown"))
                .collect::<Vec<_>>(),
        )) as ArrayRef;
        let code_col = Arc::new(Int64Array::from(
            std::iter::repeat(200i64)
                .take(100)
                .chain(std::iter::repeat(500i64).take(3))
                .chain(std::iter::once(999i64))
                .collect::<Vec<_>>(),
        )) as ArrayRef;

        let input_batch = RecordBatch::try_new(schema, vec![status_col, code_col])
            .expect("Failed to create record batch");

        let params = vec![];
        let named_params = vec![];
        let mut detector = AnomalyDetector::new(Some(params), named_params)
            .expect("Failed to create AnomalyDetector");

        let output = detector.process(input_batch).expect("Processing failed");

        assert!(output.is_some());
        let output_batch = output.unwrap();

        // Should have 6 columns (2 original + 4 anomaly info columns)
        assert_eq!(output_batch.num_columns(), 6);

        // Should have detected some anomalies
        assert!(output_batch.num_rows() > 0);
    }

    #[test]
    fn test_anomaly_detector_method_parameter() {
        let params = vec![];
        let named_params = vec![("method".to_string(), Arg::String("iqr".to_string()))];

        let result = AnomalyDetector::new(Some(params), named_params);
        assert!(result.is_ok());

        let detector = result.unwrap();
        assert_eq!(detector.method, "iqr");
        // IQR method should have default action = "transform"
        assert_eq!(detector.action, "transform");
    }

    #[test]
    fn test_anomaly_detector_invalid_method() {
        let params = vec![];
        let named_params = vec![("method".to_string(), Arg::String("invalid".to_string()))];

        let result = AnomalyDetector::new(Some(params), named_params);
        assert!(result.is_err());
    }

    #[test]
    fn test_anomaly_detector_bins_parameter() {
        let params = vec![];
        let named_params = vec![("bins".to_string(), Arg::Int(20))];

        let result = AnomalyDetector::new(Some(params), named_params);
        assert!(result.is_ok());

        let detector = result.unwrap();
        assert_eq!(detector.n_bins, 20);
    }

    #[test]
    fn test_anomaly_detector_cutoff_parameter() {
        let params = vec![];
        let named_params = vec![("cutoff".to_string(), Arg::Bool(false))];

        let result = AnomalyDetector::new(Some(params), named_params);
        assert!(result.is_ok());

        let detector = result.unwrap();
        assert!(!detector.cutoff);
    }

    #[test]
    fn test_anomaly_detector_sensitivity_parameter() {
        let params = vec![];
        let named_params = vec![("sensitivity".to_string(), Arg::String("strict".to_string()))];

        let result = AnomalyDetector::new(Some(params), named_params);
        assert!(result.is_ok());

        let detector = result.unwrap();
        assert_eq!(detector.sensitivity, "strict");
    }

    #[test]
    fn test_anomaly_detector_pthresh_parameter() {
        let params = vec![];
        let named_params = vec![
            ("method".to_string(), Arg::String("zscore".to_string())),
            ("pthresh".to_string(), Arg::Float(0.05)),
        ];

        let result = AnomalyDetector::new(Some(params), named_params);
        assert!(result.is_ok());

        let detector = result.unwrap();
        assert_eq!(detector.pthresh, Some(0.05));
    }

    #[test]
    fn test_anomaly_detector_empty_input() {
        let schema = Arc::new(Schema::new(vec![Field::new("col1", DataType::Utf8, false)]));
        let col = Arc::new(StringArray::from(Vec::<&str>::new())) as ArrayRef;
        let input_batch =
            RecordBatch::try_new(schema, vec![col]).expect("Failed to create record batch");

        let params = vec![];
        let named_params = vec![];
        let mut detector = AnomalyDetector::new(Some(params), named_params)
            .expect("Failed to create AnomalyDetector");

        let output = detector.process(input_batch).expect("Processing failed");
        assert!(output.is_some());
        let output_batch = output.unwrap();
        assert_eq!(output_batch.num_rows(), 0);
    }

    #[test]
    fn test_percentile() {
        let sorted = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(percentile(&sorted, 25.0), 2.0);
        assert_eq!(percentile(&sorted, 50.0), 3.0);
        assert_eq!(percentile(&sorted, 75.0), 4.0);
    }

    #[test]
    fn test_median() {
        let sorted = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(median(&sorted), 3.0);

        let even_sorted = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(median(&even_sorted), 2.5);
    }

    #[test]
    fn test_is_numeric() {
        assert!(AnomalyDetector::is_numeric("123"));
        assert!(AnomalyDetector::is_numeric("123.45"));
        assert!(AnomalyDetector::is_numeric("-123.45"));
        assert!(AnomalyDetector::is_numeric("1.23e-4"));
        assert!(AnomalyDetector::is_numeric("1E+5"));
        assert!(!AnomalyDetector::is_numeric("abc"));
        assert!(!AnomalyDetector::is_numeric(""));
    }

    #[test]
    fn test_clean_numeric_value() {
        assert_eq!(AnomalyDetector::clean_numeric_value("123"), 123.0);
        assert_eq!(AnomalyDetector::clean_numeric_value("123.45"), 123.45);
        assert_eq!(AnomalyDetector::clean_numeric_value("-123.45"), -123.45);
        assert_eq!(AnomalyDetector::clean_numeric_value("1.23e-4"), 0.000123);
        assert_eq!(AnomalyDetector::clean_numeric_value("abc"), 0.0);
        assert_eq!(AnomalyDetector::clean_numeric_value(""), 0.0);
    }

    #[test]
    fn test_action_filter_parameter() {
        let params = vec![];
        let named_params = vec![("action".to_string(), Arg::String("filter".to_string()))];

        let result = AnomalyDetector::new(Some(params), named_params);
        assert!(result.is_ok());

        let detector = result.unwrap();
        assert_eq!(detector.action, "filter");
    }

    #[test]
    fn test_action_annotate_parameter() {
        let params = vec![];
        let named_params = vec![("action".to_string(), Arg::String("annotate".to_string()))];

        let result = AnomalyDetector::new(Some(params), named_params);
        assert!(result.is_ok());

        let detector = result.unwrap();
        assert_eq!(detector.action, "annotate");
    }

    #[test]
    fn test_action_summary_parameter() {
        let params = vec![];
        let named_params = vec![("action".to_string(), Arg::String("summary".to_string()))];

        let result = AnomalyDetector::new(Some(params), named_params);
        assert!(result.is_ok());

        let detector = result.unwrap();
        assert_eq!(detector.action, "summary");
    }

    #[test]
    fn test_action_remove_parameter() {
        let params = vec![];
        let named_params = vec![
            ("method".to_string(), Arg::String("iqr".to_string())),
            ("action".to_string(), Arg::String("remove".to_string())),
        ];

        let result = AnomalyDetector::new(Some(params), named_params);
        assert!(result.is_ok());

        let detector = result.unwrap();
        assert_eq!(detector.action, "remove");
    }

    #[test]
    fn test_action_transform_parameter() {
        let params = vec![];
        let named_params = vec![
            ("method".to_string(), Arg::String("iqr".to_string())),
            ("action".to_string(), Arg::String("transform".to_string())),
        ];

        let result = AnomalyDetector::new(Some(params), named_params);
        assert!(result.is_ok());

        let detector = result.unwrap();
        assert_eq!(detector.action, "transform");
    }

    #[test]
    fn test_action_invalid_for_histogram() {
        let params = vec![];
        let named_params = vec![
            ("method".to_string(), Arg::String("histogram".to_string())),
            ("action".to_string(), Arg::String("remove".to_string())),
        ];

        let result = AnomalyDetector::new(Some(params), named_params);
        assert!(result.is_err());

        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("Invalid action 'remove' for method 'histogram'"));
    }

    #[test]
    fn test_action_invalid_for_zscore() {
        let params = vec![];
        let named_params = vec![
            ("method".to_string(), Arg::String("zscore".to_string())),
            ("action".to_string(), Arg::String("transform".to_string())),
        ];

        let result = AnomalyDetector::new(Some(params), named_params);
        assert!(result.is_err());

        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("Invalid action 'transform' for method 'zscore'"));
    }

    #[test]
    fn test_action_invalid_for_iqr() {
        let params = vec![];
        let named_params = vec![
            ("method".to_string(), Arg::String("iqr".to_string())),
            ("action".to_string(), Arg::String("annotate".to_string())),
        ];

        let result = AnomalyDetector::new(Some(params), named_params);
        assert!(result.is_err());

        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("Invalid action 'annotate' for method 'iqr'"));
    }

    #[test]
    fn test_pthresh_invalid_for_iqr() {
        // pthresh should return an error when used with IQR method
        let params = vec![];
        let named_params = vec![
            ("method".to_string(), Arg::String("iqr".to_string())),
            ("pthresh".to_string(), Arg::Float(0.05)),
        ];

        let result = AnomalyDetector::new(Some(params), named_params);
        assert!(result.is_err());

        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("pthresh"));
        assert!(err_msg.contains("iqr"));
    }

    #[test]
    fn test_action_abbreviation_rm() {
        // Test that "rm" is accepted as abbreviation for "remove"
        let params = vec![];
        let named_params = vec![
            ("method".to_string(), Arg::String("iqr".to_string())),
            ("action".to_string(), Arg::String("rm".to_string())),
        ];

        let result = AnomalyDetector::new(Some(params), named_params);
        assert!(result.is_ok());

        let detector = result.unwrap();
        assert_eq!(detector.action, "rm");
    }

    #[test]
    fn test_action_abbreviation_tf() {
        // Test that "tf" is accepted as abbreviation for "transform"
        let params = vec![];
        let named_params = vec![
            ("method".to_string(), Arg::String("iqr".to_string())),
            ("action".to_string(), Arg::String("tf".to_string())),
        ];

        let result = AnomalyDetector::new(Some(params), named_params);
        assert!(result.is_ok());

        let detector = result.unwrap();
        assert_eq!(detector.action, "tf");
    }

    #[test]
    fn test_action_summary_output() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("status", DataType::Utf8, false),
            Field::new("code", DataType::Int64, false),
        ]));

        let status_col = Arc::new(StringArray::from(
            std::iter::repeat("success")
                .take(100)
                .chain(std::iter::repeat("error").take(3))
                .chain(std::iter::once("unknown"))
                .collect::<Vec<_>>(),
        )) as ArrayRef;
        let code_col = Arc::new(Int64Array::from(
            std::iter::repeat(200i64)
                .take(100)
                .chain(std::iter::repeat(500i64).take(3))
                .chain(std::iter::once(999i64))
                .collect::<Vec<_>>(),
        )) as ArrayRef;

        let input_batch = RecordBatch::try_new(schema, vec![status_col, code_col])
            .expect("Failed to create record batch");

        let params = vec![];
        let named_params = vec![("action".to_string(), Arg::String("summary".to_string()))];
        let mut detector = AnomalyDetector::new(Some(params), named_params)
            .expect("Failed to create AnomalyDetector");

        let output = detector.process(input_batch).expect("Processing failed");

        assert!(output.is_some());
        let output_batch = output.unwrap();

        // Summary should have exactly 1 row
        assert_eq!(output_batch.num_rows(), 1);

        // Summary should have 6 summary fields + original fields
        assert!(output_batch.num_columns() >= 6);

        // First column should be num_anomalies
        let num_anomalies_col = output_batch
            .column(0)
            .as_any()
            .downcast_ref::<arrow::array::Int64Array>()
            .unwrap();
        assert!(num_anomalies_col.value(0) > 0);
    }

    #[test]
    fn test_action_annotate_output() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("status", DataType::Utf8, false),
            Field::new("code", DataType::Int64, false),
        ]));

        let status_col = Arc::new(StringArray::from(
            std::iter::repeat("success")
                .take(100)
                .chain(std::iter::repeat("error").take(3))
                .chain(std::iter::once("unknown"))
                .collect::<Vec<_>>(),
        )) as ArrayRef;
        let code_col = Arc::new(Int64Array::from(
            std::iter::repeat(200i64)
                .take(100)
                .chain(std::iter::repeat(500i64).take(3))
                .chain(std::iter::once(999i64))
                .collect::<Vec<_>>(),
        )) as ArrayRef;

        let input_batch = RecordBatch::try_new(schema, vec![status_col, code_col])
            .expect("Failed to create record batch");

        let params = vec![];
        let named_params = vec![("action".to_string(), Arg::String("annotate".to_string()))];
        let mut detector = AnomalyDetector::new(Some(params), named_params)
            .expect("Failed to create AnomalyDetector");

        let output = detector.process(input_batch).expect("Processing failed");

        assert!(output.is_some());
        let output_batch = output.unwrap();

        // Annotate should return all rows (104)
        assert_eq!(output_batch.num_rows(), 104);

        // Annotate should have 2 original + 5 annotation fields = 7 columns
        assert_eq!(output_batch.num_columns(), 7);

        // Third column should be is_anomaly (boolean)
        let is_anomaly_col = output_batch
            .column(2)
            .as_any()
            .downcast_ref::<arrow::array::BooleanArray>()
            .unwrap();
        let anomaly_count = (0..104).filter(|i| is_anomaly_col.value(*i)).count();
        assert!(anomaly_count > 0);
    }

    #[test]
    fn test_default_action_is_filter() {
        // Default method is histogram, so default action should be filter
        let params = vec![];
        let named_params = vec![];

        let detector = AnomalyDetector::new(Some(params), named_params)
            .expect("Failed to create AnomalyDetector");

        // histogram/zscore default action is filter
        assert_eq!(detector.method, "histogram");
        assert_eq!(detector.action, "filter");
    }

    #[test]
    fn test_iqr_default_action_is_transform() {
        // IQR method should have default action = transform
        let params = vec![];
        let named_params = vec![("method".to_string(), Arg::String("iqr".to_string()))];

        let detector = AnomalyDetector::new(Some(params), named_params)
            .expect("Failed to create AnomalyDetector");

        assert_eq!(detector.method, "iqr");
        assert_eq!(detector.action, "transform");
    }

    #[test]
    fn test_action_rm_processing() {
        // Test that "rm" (abbreviation for "remove") actually removes anomalies
        let schema = Arc::new(Schema::new(vec![
            Field::new("status", DataType::Utf8, false),
            Field::new("code", DataType::Int64, false),
        ]));

        let status_col = Arc::new(StringArray::from(
            std::iter::repeat("success")
                .take(100)
                .chain(std::iter::repeat("error").take(3))
                .chain(std::iter::once("unknown"))
                .collect::<Vec<_>>(),
        )) as ArrayRef;
        let code_col = Arc::new(Int64Array::from(
            std::iter::repeat(200i64)
                .take(100)
                .chain(std::iter::repeat(500i64).take(3))
                .chain(std::iter::once(999i64))
                .collect::<Vec<_>>(),
        )) as ArrayRef;

        let input_batch = RecordBatch::try_new(schema, vec![status_col, code_col])
            .expect("Failed to create record batch");

        let params = vec![];
        let named_params = vec![
            ("method".to_string(), Arg::String("iqr".to_string())),
            ("action".to_string(), Arg::String("rm".to_string())),
        ];
        let mut detector = AnomalyDetector::new(Some(params), named_params)
            .expect("Failed to create AnomalyDetector");

        let output = detector.process(input_batch).expect("Processing failed");

        assert!(output.is_some());
        let output_batch = output.unwrap();

        // rm (remove) should return fewer rows than input (anomalies removed)
        // Input has 104 rows, anomalies should be removed
        assert!(output_batch.num_rows() < 104);
        assert!(output_batch.num_rows() > 0);
    }

    #[test]
    fn test_action_tf_processing() {
        // Test that "tf" (abbreviation for "transform") actually transforms values
        // Create data with obvious outliers: normal values around 100, outliers at 1000
        let value_col = Arc::new(Float64Array::from(
            std::iter::repeat(100.0)
                .take(50)
                .chain(std::iter::repeat(1000.0).take(5))
                .collect::<Vec<_>>(),
        )) as ArrayRef;

        let input_batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "value",
                DataType::Float64,
                false,
            )])),
            vec![value_col],
        )
        .expect("Failed to create record batch");

        let params = vec![];
        let named_params = vec![
            ("method".to_string(), Arg::String("iqr".to_string())),
            ("action".to_string(), Arg::String("tf".to_string())),
        ];
        let mut detector = AnomalyDetector::new(Some(params), named_params)
            .expect("Failed to create AnomalyDetector");

        let output = detector.process(input_batch).expect("Processing failed");

        assert!(output.is_some());
        let output_batch = output.unwrap();

        // tf (transform) should return same number of rows
        assert_eq!(output_batch.num_rows(), 55);

        // The transformed values should be truncated to the upper bound
        let value_col_result = output_batch
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();

        // All values should be <= the original max (100) since outliers are truncated
        for i in 0..value_col_result.len() {
            if !value_col_result.is_null(i) {
                let value = value_col_result.value(i);
                // Values should be <= 100 (or close to it after truncation)
                assert!(
                    value <= 150.0,
                    "Transformed value {} should be truncated",
                    value
                );
            }
        }
    }
}
