use std::cmp::min;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

use crate::common::cluster::{iterative_clustering, Center, Cluster};

///
/// The goal of the summarization below is as follows: on being provided a collection of sampled weighted points
/// represented by a slice &[(Vec<f32>,f32)] where each of the Vec<f32> has the same length/dimension
/// and the f32 in the pair is the corresponding weight.
/// The algorithm uses the philosophy of RCFs, in repeatedly using randomization. It proceeds as follows:
/// 1. It uses an initial sampling which serves as a basis of efficiency as well as denoising, borrowing from
/// https://en.wikipedia.org/wiki/CURE_algorithm, in that algorithm's robustness to outliers.
/// 2. It uses a sampling mechanism to initialize some clusters based on https://en.wikipedia.org/wiki/Data_stream_clustering
/// where the radom sampling achieves half of the the same effects as hierarchical compression.
///3.  It repeatedly merges the most overlapping clusters, failing that, eliminates the least weighted cluster to achieve
/// the same effect as hieararchical compression.
///
/// The algorithm takes a distance function as an input, and tends to produce spherical (measured in the input
/// distance function) clusters. These types of algorithms are unlikely to be useful for large number of output clusters.
/// The output is the SampleSummary, which provides basic statistics of mean, median and deviation
/// in addition it performs a grouping/clustering, assuming that the maximum number of clusters are not large
/// the routine below bounds the number to be max_number_per_dimension times the dimension of Vec<f32>
/// and a smaller number can also be provided in the summarize() function
///
///

const max_number_per_dimension: usize = 5;

const PHASE2_THRESHOLD: usize = 2;

const LENGTH_BOUND: usize = 1000;

const UPPER_FRACTION : f64 = 0.9;

const LOWER_FRACTION : f64 = 0.1;

#[repr(C)]
pub struct SampleSummary {
    pub summary_points: Vec<Vec<f32>>,

    // a measure of comparison among the typical points;
    pub relative_weight: Vec<f32>,

    // number of samples, often the number of summary, but can handle weighted points
    // (possibly indicating confidence or othe measure) in the future
    pub total_weight: f32,

    // the global mean, median
    pub mean: Vec<f32>,
    pub median: Vec<f32>,

    // percentiles and bounds
    pub upper: Vec<f32>,
    pub lower : Vec<f32>,

    // This is the global deviation,
    pub deviation: Vec<f32>,
}

impl SampleSummary {
    pub fn new(
        total_weight: f32,
        summary_points: Vec<Vec<f32>>,
        relative_weight: Vec<f32>,
        median: Vec<f32>,
        mean: Vec<f32>,
        upper: Vec<f32>,
        lower: Vec<f32>,
        deviation: Vec<f32>,
    ) -> Self {
        SampleSummary {
            total_weight,
            summary_points: summary_points.clone(),
            relative_weight: relative_weight.clone(),
            median: median.clone(),
            mean: mean.clone(),
            upper: upper.clone(),
            lower: lower.clone(),
            deviation: deviation.clone(),
        }
    }

    pub fn add_typical(&mut self, summary_points: Vec<Vec<f32>>, relative_weight: Vec<f32>) {
        self.summary_points = summary_points.clone();
        self.relative_weight = relative_weight.clone();
    }

    pub fn pick(weighted_points : &[(f32,f32)], weight: f64, start: usize, initial_weight : f64) -> (usize,f64) {
        let mut running = initial_weight;
        let mut index = start;
        while index + 1 < weighted_points.len() && weighted_points[index].1 as f64 + running < weight {
            running += weighted_points[index].1 as f64;
            index += 1;
        }
        (index, running)
    }


    pub fn from_points(points: &[(Vec<f32>, f32)], lower_fraction: f64, upper_fraction:f64) -> Self {
        assert!(points.len() > 0, "cannot be empty list");
        assert!(lower_fraction < 0.5, " has to be less than half");
        assert!(upper_fraction > 0.5, "has to be larger than half");
        let dimensions = points[0].0.len();
        assert!(dimensions > 0, " cannot have 0 dimensions");
        let total_weight: f64 = points.iter().map(|x| x.1 as f64).sum();
        assert!(total_weight > 0.0, "weights cannot be all zero");
        assert!(total_weight.is_finite(), " cannot have infinite weights");
        let mut mean = vec![0.0f32; dimensions];
        let mut deviation = vec![0.0f32; dimensions];
        let mut sum_values_sq = vec![0.0f64; dimensions];
        let mut sum_values = vec![0.0f64; dimensions];
        for i in 0..points.len() {
            assert!(points[i].0.len() == dimensions, "incorrect dimensions");
            assert!(points[i].1 >= 0.0, "point weights have to be non-negative");
            for j in 0..dimensions {
                assert!(
                    points[i].0[j].is_finite() && !points[i].0[j].is_nan(),
                    " cannot have NaN or infinite values"
                );
                sum_values[j] += points[i].1 as f64 * points[i].0[j] as f64;
                sum_values_sq[j] +=
                    points[i].1 as f64 * points[i].0[j] as f64 * points[i].0[j] as f64;
            }
        }
        for j in 0..dimensions {
            mean[j] = (sum_values[j] / total_weight) as f32;
            let t: f64 = sum_values_sq[j] / total_weight
                - sum_values[j] * sum_values[j] / (total_weight * total_weight);
            deviation[j] = f64::sqrt(if t > 0.0 { t } else { 0.0 }) as f32;
        }
        let mut median = vec![0.0f32; dimensions];
        let mut upper_vec = vec![0.0f32;dimensions];
        let mut lower_vec = vec![0.0f32;dimensions];
        let num = total_weight/2.0;
        let lower = total_weight * lower_fraction;
        let upper = total_weight * upper_fraction;
        for j in 0..dimensions {
            let mut y: Vec<(f32,f32)> = points.iter().map(|x| (x.0[j],x.1)).collect();
            y.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let first = Self::pick(&y,lower,0,0.0);
            lower_vec[j] = y[first.0].0;
            let second = Self::pick(&y,num,first.0,first.1);
            median[j] = y[second.0].0;
            let third = Self::pick(&y,upper,second.0,second.1);
            upper_vec[j] = y[third.0].0;
        }

        SampleSummary {
            summary_points: Vec::new(),
            relative_weight: Vec::new(),
            total_weight: total_weight as f32,
            mean,
            upper: upper_vec,
            lower: lower_vec,
            median,
            deviation,
        }
    }
}

pub fn summarize(
    points: &[(Vec<f32>, f32)],
    distance: fn(&[f32], &[f32]) -> f64,
    max_number: usize,
    parallel_enabled: bool,
) -> SampleSummary {
    assert!(max_number < 51, " for large number of clusters, other methods may be better, consider recursively removing clusters");
    let mut summary = SampleSummary::from_points(&points,LOWER_FRACTION,UPPER_FRACTION);

    if max_number > 0 {
        let dimensions = points[0].0.len();
        let max_allowed = min(dimensions * max_number_per_dimension, max_number);
        let total_weight: f64 = points.iter().map(|x| x.1 as f64).sum();
        let mut rng = ChaCha20Rng::seed_from_u64(max_allowed as u64);

        let mut sampled_points: Vec<(&[f32], f32)> = Vec::new();
        if points.len() < 5 * LENGTH_BOUND {
            for j in 0..points.len() {
                sampled_points.push((&points[j].0, points[j].1));
            }
        } else {
            let mut remainder = 0.0f64;
            for j in 0..points.len() {
                if points[j].1 > (total_weight / LENGTH_BOUND as f64) as f32 {
                    sampled_points.push((&points[j].0, points[j].1));
                } else {
                    remainder += points[j].1 as f64;
                }
            }
            for j in 0..points.len() {
                if points[j].1 <= (total_weight / LENGTH_BOUND as f64) as f32
                    && rng.gen::<f64>() < 5.0 * (LENGTH_BOUND as f64) / (points.len() as f64)
                {
                    let t = points[j].1 as f64
                        * (points.len() as f64 / (5.0 * LENGTH_BOUND as f64))
                        * (remainder / total_weight);
                    sampled_points.push((&points[j].0, t as f32));
                }
            }
        }

        let mut list: Vec<Center> = iterative_clustering(
            max_allowed,
            &sampled_points,
            Center::new,
            distance,
            parallel_enabled,
        );
        list.sort_by(|o1, o2| o2.weight().partial_cmp(&o1.weight()).unwrap()); // decreasing order
        let mut summary_points: Vec<Vec<f32>> = Vec::new();
        let mut relative_weight: Vec<f32> = Vec::new();
        let center_sum: f64 = list.iter().map(|x| x.weight()).sum();
        for i in 0..list.len() {
            summary_points.push(list[i].primary_representative(distance).clone());
            relative_weight.push((list[i].weight() / center_sum) as f32);
        }
        summary.add_typical(summary_points, relative_weight);
    }
    return summary;
}
