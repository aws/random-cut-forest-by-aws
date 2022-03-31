
use num::abs;
use crate::pointstore::PointStore;
use crate::samplesummary::{SampleSummary, summarize};

fn project_missing(point: &Vec<f32>,position : &[usize]) -> Vec<f32> {
    position.iter().map(|i| point[*i]).collect()
}

/// the following function is a conduit that summarizes the conditional samples derived from the trees
/// The samples are denoted by (PointIndex, f32) where the PointIndex(usize) corresponds to the point identifier
/// in the point store and the f32 associated with a scalar value (corresponding to weight)
/// the field missing corresponds to the list of missing fields in the space of the full (potentially shingled) points
/// centrality corresponds to the parameter which was used to derive the samples, and thus provides a mechanism for
/// refined interpretation in summarization
/// project corresponds to a boolean flag, determining whether we wish to focus on the missing fields only (project = true)
/// or we focus on the entire space of (potentially shingled) points (in case of project = false) which have different
/// and complementary uses.
/// max_number corresponds to a parameter that controls the summarization -- in the current version this corresponds to
/// an upper bound on the number of summary points in the SampleSummary construct
///
/// Note that the global, mean and median do not perform any weighting/pruning; whereas the summarize() performs on
/// somewhat denoised data to provide a list of summary. Note further that summarize() is redundant (and skipped)
/// when max_number = 0
/// The combination appears to provide the best of all worlds with little performance overhead and can be
/// used and reconfigured easily. In the fullness of time, it is possible to leverage a dynamic Kernel, since
/// the entire PointStore is present and the PointStore is dynamic.
#[repr(C)]
pub struct FieldSummarizer {
    centrality: f64,
    project : bool,
    max_number : usize,
    distance : fn(&[f32],&[f32]) -> f64
}

impl FieldSummarizer {
    pub fn new(centrality: f64, project: bool, max_number: usize, distance: fn(&[f32], &[f32]) -> f64) -> Self {
        FieldSummarizer {
            centrality,
            project,
            max_number,
            distance
        }
    }

    pub fn summarize_list(&self, pointstore: &dyn PointStore, point_list_with_distance: &[(usize, f32)], missing: &[usize]) -> SampleSummary {
        let mut distance_list: Vec<f32> = point_list_with_distance.iter().map(|a| a.1)
            .collect();
        distance_list.sort_by(|a, b| a.partial_cmp(&b).unwrap());
        let mut threshold = 0.0;
        if self.centrality > 0.0 {
            let mut always_include = 0;
            while always_include < point_list_with_distance.len() && distance_list[always_include] == 0.0 {
                always_include += 1;
            }
            threshold = self.centrality * (distance_list[always_include + (distance_list.len() - always_include) / 3] +
                distance_list[always_include + (distance_list.len() - always_include) / 2]) as f64;
        }
        threshold += (1.0 - self.centrality) * distance_list[point_list_with_distance.len() - 1] as f64;

        let total_weight = point_list_with_distance.len() as f64;
        let dimensions = if !self.project || missing.len() == 0 {
            pointstore.get_copy(point_list_with_distance[0].0).len()
        } else {
            missing.len()
        };
        let mut mean = vec![0.0f32; dimensions];
        let mut deviation = vec![0.0f32; dimensions];
        let mut sum_values_sq = vec![0.0f64; dimensions];
        let mut sum_values = vec![0.0f64; dimensions];
        let mut vec = Vec::new();
        for i in 0..point_list_with_distance.len() {
            let point = if !self.project || missing.len() == 0 {
                pointstore.get_copy(point_list_with_distance[i].0)
            } else {
                project_missing(&pointstore.get_copy(point_list_with_distance[i].0), &missing)
            };
            for j in 0..dimensions {
                sum_values[j] += point[j] as f64;
                sum_values_sq[j] += point[j] as f64 * point[j] as f64;
            }
            /// the else can be filtered further
            let weight: f32 = if point_list_with_distance[i].1 <= threshold as f32 {
                1.0
            } else {
                threshold as f32 / point_list_with_distance[i].1
            };

            vec.push((point, weight));
        };

        for j in 0..dimensions {
            mean[j] = (sum_values[j] / total_weight as f64) as f32;
            let t: f64 = sum_values_sq[j] / total_weight as f64 - sum_values[j] * sum_values[j] / (total_weight as f64 * total_weight as f64);
            deviation[j] = f64::sqrt(if t > 0.0 { t } else { 0.0 }) as f32;
        };
        let mut median = vec![0.0f32; dimensions];
        for j in 0..dimensions {
            let mut v: Vec<f32> = vec.iter().map(|x| x.0[j]).collect();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            median[j] = v[vec.len() / 2];
        };

        let mut summary = summarize(&vec, self.distance, self.max_number);
        SampleSummary {
            summary_points: summary.summary_points.clone(),
            relative_weight: summary.relative_weight.clone(),
            total_weight: summary.total_weight,
            mean,
            median,
            deviation
        }
    }
}