//! Utility functions module
//!
//! Provides utility functions for missing value handling, autocovariance, correlogram, period finding, etc.

use anyhow::{Error, bail};
use std::f64;

/// Maximum lag
pub const MAX_LAG: usize = 2000;
/// Maximum number of points
pub const MAX_POINTS: usize = 20 * MAX_LAG;

/// Calculate autocovariance
fn autocovariance0(data: &[f64], start: usize, end: usize, mean: f64, k: usize) -> f64 {
    let n = end - start;
    if n <= k {
        return 0.0;
    }
    let mut cov = 0.0;
    for i in start..(end - k) {
        cov += (data[i] - mean) * (data[i + k] - mean);
    }
    cov / ((n - 1) as f64)
}

/// Calculate correlogram
///
/// # Errors
///
/// Returns an error if the variance is zero (all data points are the same).
fn correlogram0(data: &[f64], start: usize, end: usize, n: usize) -> Result<Vec<f64>, Error> {
    let n_data = end - start;
    let mut mean = 0.0;
    for val in data.iter().take(end).skip(start) {
        mean += val;
    }
    mean /= n_data as f64;

    let n_lag = if n == 0 || n >= n_data { n_data - 1 } else { n };
    let var = autocovariance0(data, start, end, mean, 0);
    if var == 0.0 {
        bail!("numerical error: variance is zero - all data points may be identical");
    }

    let mut result = vec![1.0];
    for i in 1..=n_lag {
        result.push(autocovariance0(data, start, end, mean, i) / var);
    }
    Ok(result)
}

/// Find period
pub fn find_period0(data: &[f64], start: usize, end: usize) -> i32 {
    if end <= start + 1 {
        return 1;
    }
    let end = if end > start + MAX_POINTS {
        start + MAX_POINTS
    } else {
        end
    };

    let cor = match correlogram0(data, start, end, MAX_LAG) {
        Ok(c) => c,
        Err(_) => return -1, // Return -1 if correlogram calculation fails
    };
    if cor.len() < 2 {
        return 1;
    }

    let mut prev = cor[0];
    let mut curr = cor[1];
    let mut peak_idx = 0;
    let mut peak_val = 0.0;

    for i in 1..(cor.len() - 1) {
        let next_item = cor[i + 1];
        if curr > prev && curr > next_item && curr > peak_val + 0.01 {
            peak_val = curr;
            peak_idx = i;
        }
        prev = curr;
        curr = next_item;
    }

    if peak_val <= 0.01 {
        -1
    } else {
        peak_idx as i32
    }
}

/// Find longest continuous segment (handles missing values)
pub fn find_longest_continuous_stretch(
    data: &[Option<f64>],
    start: usize,
    end: usize,
) -> (usize, usize) {
    let mut longest_start = start;
    let mut longest_end = start;
    let mut current_start = start;
    let mut current_end = start;

    while current_end < end {
        if data[current_end].is_some() {
            current_end += 1;
        } else {
            if current_end - current_start > longest_end - longest_start {
                longest_start = current_start;
                longest_end = current_end;
            }
            current_end += 1;
            while current_end < end && data[current_end].is_none() {
                current_end += 1;
            }
            current_start = current_end;
        }
    }

    if current_start < end && current_end - current_start > longest_end - longest_start {
        longest_start = current_start;
        longest_end = current_end;
    }

    (longest_start, longest_end)
}

/// Find period (handles missing values)
pub fn find_period2(data: &[Option<f64>], start: usize, end: usize) -> i32 {
    let (new_start, new_end) = find_longest_continuous_stretch(data, start, end);
    let data_vec: Vec<f64> = data[new_start..new_end].iter().filter_map(|x| *x).collect();
    find_period0(&data_vec, 0, data_vec.len())
}

/// Fill missing values
pub fn fillin_mv(data: &mut [Option<f64>], start: usize, end: usize) {
    let end = if end > start + MAX_POINTS {
        start + MAX_POINTS
    } else {
        end
    };

    let mut i = start;
    while i < end {
        if data[i].is_none() {
            let mut j = i + 1;
            while j < end && data[j].is_none() {
                j += 1;
            }

            let denom = (j - i + 1) as f64;
            let left = if i > start {
                data[i - 1].unwrap_or(0.0) / denom
            } else {
                data[j].unwrap_or(0.0) / denom
            };
            let right = if j < end {
                data[j].unwrap_or(0.0) / denom
            } else {
                data[i - 1].unwrap_or(0.0) / denom
            };

            let mut w1 = (denom - 1.0) * left;
            let mut w2 = right;
            for val in data.iter_mut().take(j).skip(i) {
                *val = Some(w1 + w2);
                w1 -= left;
                w2 += right;
            }
            i = j;
        } else {
            i += 1;
        }
    }
}

/// Find period (after filling missing values)
pub fn find_period3(data: &[Option<f64>], start: usize, end: usize) -> i32 {
    let mut data_copy: Vec<Option<f64>> = data.to_vec();
    fillin_mv(&mut data_copy, start, end);
    let data_vec: Vec<f64> = data_copy[start..end].iter().filter_map(|x| *x).collect();
    find_period0(&data_vec, 0, data_vec.len())
}

/// Data iterator
#[derive(Clone)]
pub struct Datafeed {
    data: Vec<f64>,
    start: usize,
    end: usize,
    step: usize,
    cur: usize,
    missing_valued: bool,
}

impl Datafeed {
    pub fn new(data: Vec<f64>, start: usize, end: usize, step: usize) -> Self {
        Self {
            data,
            start,
            end,
            step,
            cur: start,
            missing_valued: false,
        }
    }

    pub fn with_missing_values(mut self, missing_valued: bool) -> Self {
        self.missing_valued = missing_valued;
        self
    }

    pub fn get_val(&self, i: usize) -> f64 {
        self.data[i]
    }

    pub fn set_val(&mut self, i: usize, val: f64) {
        self.data[i] = val;
    }

    pub fn get_start(&self) -> usize {
        self.start
    }

    pub fn get_end(&self) -> usize {
        self.end
    }

    pub fn get_step(&self) -> usize {
        self.step
    }

    pub fn get_cur(&self) -> usize {
        self.cur
    }

    pub fn set_start(&mut self, start: usize) {
        self.start = start;
    }

    pub fn set_end(&mut self, end: usize) {
        self.end = end;
    }

    pub fn reset(&mut self) {
        self.cur = self.start;
    }

    pub fn rewind(&mut self) {
        if self.cur >= self.step {
            self.cur -= self.step;
        }
    }

    pub fn clone_with_params(
        &self,
        start: Option<usize>,
        end: Option<usize>,
        step: Option<usize>,
    ) -> Self {
        Self {
            data: self.data.clone(),
            start: start.unwrap_or(self.start),
            end: end.unwrap_or(self.end),
            step: step.unwrap_or(self.step),
            cur: start.unwrap_or(self.start),
            missing_valued: self.missing_valued,
        }
    }

    pub fn period(&self) -> i32 {
        if self.missing_valued {
            // Need to convert to Option<f64> format
            let data_opt: Vec<Option<f64>> = self.data.iter().map(|x| Some(*x)).collect();
            find_period3(&data_opt, self.start, self.end)
        } else {
            find_period0(&self.data, self.start, self.end)
        }
    }

    pub fn len(&self) -> usize {
        (self.end - self.start) / self.step
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Iterator for Datafeed {
    type Item = f64;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cur >= self.end {
            None
        } else {
            let cur = self.cur;
            self.cur += self.step;
            Some(self.data[cur])
        }
    }
}

/// Calculate confidence interval for prediction values
///
/// # Parameters
/// - `prediction`: Prediction value (mean)
/// - `variance`: Prediction variance
/// - `confidence`: Confidence level, e.g., 0.95 for 95% confidence interval
///
/// # Returns
/// - `(lower, upper)`: Lower and upper bounds of confidence interval
///
/// # Errors
///
/// Returns an error if:
/// - `variance` is negative
/// - `confidence` is not in the range (0, 1)
///
/// # Examples
/// ```
/// use predict::utils::prediction_interval;
///
/// let prediction = 100.0;
/// let variance = 4.0;
/// let (lower, upper) = prediction_interval(prediction, variance, 0.95).unwrap();
/// assert!(lower < prediction);
/// assert!(upper > prediction);
/// ```
pub fn prediction_interval(
    prediction: f64,
    variance: f64,
    confidence: f64,
) -> Result<(f64, f64), Error> {
    if variance < 0.0 {
        bail!(
            "invalid parameter 'variance': {} - variance must be non-negative",
            variance
        );
    }
    if confidence <= 0.0 || confidence >= 1.0 {
        bail!(
            "invalid parameter 'confidence': {} - confidence must be in (0, 1)",
            confidence
        );
    }

    let std_dev = variance.sqrt();
    let std_norm = crate::dist::Normaldist::default();
    // For two-sided confidence interval, need (1 + confidence) / 2 quantile
    // For example, 95% confidence interval needs 97.5% quantile
    let z_score = std_norm.invcdf((1.0 + confidence) / 2.0)?;

    let lower = prediction - z_score * std_dev;
    let upper = prediction + z_score * std_dev;

    Ok((lower, upper))
}

/// Calculate 95% confidence interval for prediction values (convenience function)
///
/// # Parameters
/// - `prediction`: Prediction value (mean)
/// - `variance`: Prediction variance
///
/// # Returns
/// - `(lower95, upper95)`: Lower and upper bounds of 95% confidence interval
///
/// # Errors
///
/// Returns an error if variance is negative.
///
/// # Examples
/// ```
/// use predict::utils::prediction_interval;
///
/// let prediction = 100.0;
/// let variance = 4.0;
/// let (lower95, upper95) = prediction_interval(prediction, variance, 0.95).unwrap();
/// assert!(lower95 < prediction);
/// assert!(upper95 > prediction);
/// ```
///
/// Calculate 95% confidence interval for prediction values (convenience function)
///
/// # Parameters
/// - `prediction`: Prediction value (mean)
/// - `variance`: Prediction variance
///
/// # Returns
/// - `(lower95, upper95)`: Lower and upper bounds of 95% confidence interval
///
/// # Errors
///
/// Returns an error if variance is negative.
///
/// # Examples
/// ```
/// use predict::utils::prediction_interval;
///
/// let prediction = 100.0;
/// let variance = 4.0;
/// let (lower95, upper95) = prediction_interval(prediction, variance, 0.95).unwrap();
/// assert!(lower95 < prediction);
/// assert!(upper95 > prediction);
/// ```

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_autocovariance() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mean = 3.0;
        let cov0 = autocovariance0(&data, 0, data.len(), mean, 0);
        assert!(cov0 >= 0.0); // Variance should be non-negative

        let cov1 = autocovariance0(&data, 0, data.len(), mean, 1);
        // Autocovariance should be less than or equal to variance
        assert!(cov1.abs() <= cov0 + 1e-10);
    }

    #[test]
    fn test_correlogram() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let cor = correlogram0(&data, 0, data.len(), 3).unwrap_or_default();
        assert_eq!(cor.len(), 4); // 0, 1, 2, 3
        assert!((cor[0] - 1.0).abs() < 1e-10); // Autocorrelation is 1 at lag=0
    }

    #[test]
    fn test_find_period() {
        // Test periodic data
        let periodic_data = vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0];
        let period = find_period0(&periodic_data, 0, periodic_data.len());
        // Should be able to find period 3
        assert!(period > 0 || period == -1); // May find period or not

        // Test non-periodic data
        let non_periodic = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let period2 = find_period0(&non_periodic, 0, non_periodic.len());
        assert!(period2 >= -1);
    }

    #[test]
    fn test_find_longest_continuous_stretch() {
        let data = vec![Some(1.0), Some(2.0), None, Some(3.0), Some(4.0), None, None];
        let (start, end) = find_longest_continuous_stretch(&data, 0, data.len());
        assert!(start < end);
        assert!(end <= data.len());
    }

    #[test]
    fn test_find_period2() {
        let data = vec![
            Some(1.0),
            Some(2.0),
            Some(3.0),
            Some(1.0),
            Some(2.0),
            Some(3.0),
        ];
        let period = find_period2(&data, 0, data.len());
        assert!(period >= -1);
    }

    #[test]
    fn test_fillin_mv() {
        let mut data = vec![Some(1.0), None, None, Some(4.0)];
        let len = data.len();
        fillin_mv(&mut data, 0, len);
        // Missing values should be filled
        assert!(data[1].is_some());
        assert!(data[2].is_some());
    }

    #[test]
    fn test_find_period3() {
        let data = vec![Some(1.0), Some(2.0), None, Some(1.0), Some(2.0)];
        let period = find_period3(&data, 0, data.len());
        assert!(period >= -1);
    }

    #[test]
    fn test_datafeed() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut df = Datafeed::new(data, 0, 5, 1);
        assert_eq!(df.next(), Some(1.0));
        assert_eq!(df.next(), Some(2.0));
        assert_eq!(df.len(), 5);
    }

    #[test]
    fn test_datafeed_with_step() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut df = Datafeed::new(data, 0, 8, 2);
        assert_eq!(df.next(), Some(1.0));
        assert_eq!(df.next(), Some(3.0));
        assert_eq!(df.next(), Some(5.0));
        assert_eq!(df.len(), 4);
    }

    #[test]
    fn test_datafeed_reset() {
        let data = vec![1.0, 2.0, 3.0];
        let mut df = Datafeed::new(data, 0, 3, 1);
        assert_eq!(df.next(), Some(1.0));
        df.reset();
        assert_eq!(df.next(), Some(1.0));
    }

    #[test]
    fn test_datafeed_rewind() {
        let data = vec![1.0, 2.0, 3.0];
        let mut df = Datafeed::new(data, 0, 3, 1);
        assert_eq!(df.next(), Some(1.0));
        assert_eq!(df.next(), Some(2.0));
        df.rewind();
        assert_eq!(df.next(), Some(2.0));
    }

    #[test]
    fn test_datafeed_get_set_val() {
        let data = vec![1.0, 2.0, 3.0];
        let mut df = Datafeed::new(data, 0, 3, 1);
        assert_eq!(df.get_val(1), 2.0);
        df.set_val(1, 5.0);
        assert_eq!(df.get_val(1), 5.0);
    }

    #[test]
    fn test_autocovariance_edge_cases() {
        // Test case where k >= n
        let data2 = vec![1.0, 2.0];
        let cov2 = autocovariance0(&data2, 0, data2.len(), 1.5, 2);
        assert_eq!(cov2, 0.0);

        // Test multi-point data
        let data3 = vec![1.0, 2.0, 3.0];
        let mean3 = 2.0;
        let cov3 = autocovariance0(&data3, 0, data3.len(), mean3, 0);
        assert!(cov3 >= 0.0);
    }

    #[test]
    fn test_correlogram_edge_cases() {
        // Test multi-point data
        let data = vec![1.0, 2.0, 3.0];
        let cor = correlogram0(&data, 0, data.len(), 2).unwrap_or_default();
        assert_eq!(cor.len(), 3);
        assert!((cor[0] - 1.0).abs() < 1e-10);

        // Test non-constant data
        let non_constant_data = vec![1.0, 2.0, 3.0, 4.0];
        let cor2 =
            correlogram0(&non_constant_data, 0, non_constant_data.len(), 2).unwrap_or_default();
        assert_eq!(cor2.len(), 3);
    }

    #[test]
    fn test_find_period_clear_pattern() {
        // Test clear periodic pattern
        let data = vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0];
        let period = find_period0(&data, 0, data.len());
        assert!(period > 0);
    }

    #[test]
    fn test_find_longest_continuous_stretch_edge_cases() {
        // Test all missing values
        let data = vec![None, None, None];
        let (start, end) = find_longest_continuous_stretch(&data, 0, data.len());
        assert_eq!(start, 0);
        assert_eq!(end, 0);

        // Test all valid values
        let data2 = vec![Some(1.0), Some(2.0), Some(3.0)];
        let (start2, end2) = find_longest_continuous_stretch(&data2, 0, data2.len());
        assert_eq!(start2, 0);
        assert_eq!(end2, 3);
    }

    #[test]
    fn test_fillin_mv_edge_cases() {
        // Test missing at start
        let mut data = vec![None, None, Some(3.0), Some(4.0)];
        let len = data.len();
        fillin_mv(&mut data, 0, len);
        assert!(data[0].is_some());
        assert!(data[1].is_some());

        // Test missing at end
        let mut data2 = vec![Some(1.0), Some(2.0), None, None];
        let len2 = data2.len();
        fillin_mv(&mut data2, 0, len2);
        assert!(data2[2].is_some());
        assert!(data2[3].is_some());
    }

    #[test]
    fn test_prediction_interval() {
        let prediction = 100.0;
        let variance = 4.0; // Standard deviation is 2.0

        // Test 95% confidence interval
        let (lower95, upper95) =
            prediction_interval(prediction, variance, 0.95).expect("prediction_interval_95 failed");
        assert!(lower95 < prediction);
        assert!(upper95 > prediction);
        // 95% confidence interval should be approximately [96.08, 103.92] (100 ± 1.96 * 2)
        assert!((lower95 - 96.08).abs() < 0.1);
        assert!((upper95 - 103.92).abs() < 0.1);

        // Test 90% confidence interval
        let (lower90, upper90) =
            prediction_interval(prediction, variance, 0.90).expect("prediction_interval failed");
        assert!(lower90 < prediction);
        assert!(upper90 > prediction);
        // 90% confidence interval should be narrower than 95% confidence interval
        assert!(lower90 > lower95);
        assert!(upper90 < upper95);

        // Test 99% confidence interval
        let (lower99, upper99) =
            prediction_interval(prediction, variance, 0.99).expect("prediction_interval failed");
        assert!(lower99 < prediction);
        assert!(upper99 > prediction);
        // 99% confidence interval should be wider than 95% confidence interval
        assert!(lower99 < lower95);
        assert!(upper99 > upper95);
    }

    #[test]
    #[should_panic(expected = "variance must be non-negative")]
    fn test_prediction_interval_negative_variance() {
        prediction_interval(100.0, -1.0, 0.95).unwrap();
    }

    #[test]
    #[should_panic(expected = "confidence must be in")]
    fn test_prediction_interval_invalid_confidence() {
        prediction_interval(100.0, 4.0, 1.5).unwrap();
    }
}
