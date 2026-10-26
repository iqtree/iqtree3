use candle_core::Tensor;

pub fn tensor_full(value: f64, dims: &[usize]) -> Tensor {
    Tensor::full(value, dims, &candle_core::Device::Cpu).unwrap()
}

fn histogram_with_edges(values: &[f64], bin_edges: Vec<f64>) -> (Tensor, Tensor) {
    let num_bins = bin_edges.len() - 1;
    let device = &candle_core::Device::Cpu;
    let mut counts = vec![0i64; num_bins];

    for &value in values {
        // Binary search for the bin: right-inclusive on the last bin
        let idx = bin_edges.partition_point(|&e| e <= value).saturating_sub(1);
        let idx = idx.min(num_bins - 1);
        counts[idx] += 1;
    }

    let bin_edges = Tensor::from_vec(bin_edges, &[num_bins + 1], device).unwrap();
    let counts = Tensor::from_vec(counts, &[num_bins], device).unwrap();
    (bin_edges, counts)
}

/// Linear-spaced histogram.
pub fn histogram(data: &Tensor, num_bins: usize) -> (Tensor, Tensor) {
    assert!(num_bins > 0, "histogram requires at least one bin");

    let values = data.flatten_all().unwrap().to_vec1::<f64>().unwrap();
    assert!(!values.is_empty(), "histogram requires at least one value");

    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    if min == max {
        let device = &candle_core::Device::Cpu;
        let mut counts = vec![0i64; num_bins];
        counts[0] = values.len() as i64;
        let edges = Tensor::from_vec(vec![min; num_bins + 1], &[num_bins + 1], device).unwrap();
        let counts = Tensor::from_vec(counts, &[num_bins], device).unwrap();
        return (edges, counts);
    }

    let width = (max - min) / num_bins as f64;
    let bin_edges = (0..=num_bins)
        .map(|i| min + i as f64 * width)
        .collect::<Vec<f64>>();

    histogram_with_edges(&values, bin_edges)
}

/// Log-spaced histogram. All values must be positive.
pub fn _histogram_log(data: &Tensor, num_bins: usize) -> (Tensor, Tensor) {
    assert!(num_bins > 0, "histogram_log requires at least one bin");

    let values = data.flatten_all().unwrap().to_vec1::<f64>().unwrap();
    assert!(
        !values.is_empty(),
        "histogram_log requires at least one value"
    );
    assert!(
        values.iter().all(|&v| v > 0.0),
        "histogram_log requires all values to be positive"
    );

    let log_min = values
        .iter()
        .copied()
        .map(f64::ln)
        .fold(f64::INFINITY, f64::min);
    let log_max = values
        .iter()
        .copied()
        .map(f64::ln)
        .fold(f64::NEG_INFINITY, f64::max);

    if log_min == log_max {
        let device = &candle_core::Device::Cpu;
        let mut counts = vec![0i64; num_bins];
        counts[0] = values.len() as i64;
        let v = log_min.exp();
        let edges = Tensor::from_vec(vec![v; num_bins + 1], &[num_bins + 1], device).unwrap();
        let counts = Tensor::from_vec(counts, &[num_bins], device).unwrap();
        return (edges, counts);
    }

    let log_width = (log_max - log_min) / num_bins as f64;
    let bin_edges = (0..=num_bins)
        .map(|i| (log_min + i as f64 * log_width).exp())
        .collect::<Vec<f64>>();

    histogram_with_edges(&values, bin_edges)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn histogram_counts_values_in_expected_bins() {
        let data =
            Tensor::from_vec(vec![0.0, 0.2, 0.8, 1.0], &[4], &candle_core::Device::Cpu).unwrap();

        let (edges, counts) = histogram(&data, 2);

        assert_eq!(edges.to_vec1::<f64>().unwrap(), vec![0.0, 0.5, 1.0]);
        assert_eq!(counts.to_vec1::<i64>().unwrap(), vec![2, 2]);
    }

    #[test]
    fn histogram_handles_constant_input() {
        let data = Tensor::from_vec(vec![3.0, 3.0, 3.0], &[3], &candle_core::Device::Cpu).unwrap();

        let (edges, counts) = histogram(&data, 3);

        assert_eq!(edges.to_vec1::<f64>().unwrap(), vec![3.0, 3.0, 3.0, 3.0]);
        assert_eq!(counts.to_vec1::<i64>().unwrap(), vec![3, 0, 0]);
    }

    #[test]
    fn histogram_log_spans_decades() {
        // Values spanning 3 decades: 0.1, 1, 10, 100
        let data =
            Tensor::from_vec(vec![0.1, 1.0, 10.0, 100.0], &[4], &candle_core::Device::Cpu).unwrap();

        let (edges, counts) = _histogram_log(&data, 3);

        let edges_v = edges.to_vec1::<f64>().unwrap();
        let counts_v = counts.to_vec1::<i64>().unwrap();

        // 4 edges spanning [0.1, 100] in 3 log-equal steps
        assert_eq!(edges_v.len(), 4);
        assert!((edges_v[0] - 0.1).abs() < 1e-10);
        assert!((edges_v[3] - 100.0).abs() < 1e-10);

        // All 4 values should be counted
        assert_eq!(counts_v.iter().sum::<i64>(), 4);
    }
}
