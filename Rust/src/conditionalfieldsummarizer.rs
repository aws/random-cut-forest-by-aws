
use num::abs;
use crate::pointstore::PointStore;
use crate::samplesummary::{SampleSummary, summarize};

fn project_missing(point: &Vec<f32>,position : &[usize]) -> Vec<f32> {
    position.iter().map(|i| point[*i]).collect()
}

pub fn field_summarizer(pointstore: &dyn PointStore, point_list_with_distance : &[(usize, f32)], missing: &[usize], centrality: f64, project: bool, max_number : usize) -> SampleSummary {

    let mut distance_list : Vec<f32> = point_list_with_distance.iter().map(|a| a.1)
        .collect();
    distance_list.sort_by(|a,b| a.partial_cmp(&b).unwrap());
    let mut threshold = 0.0;
    if (centrality>0.0) {
        let mut always_include = 0;
        while (always_include < point_list_with_distance.len() && distance_list[always_include] == 0.0) {
            always_include += 1;
        }
        threshold = centrality * ( distance_list[always_include + (distance_list.len() - always_include)/3] +
            distance_list[always_include + (distance_list.len() - always_include)/2] ) as f64;
    }
    threshold += (1.0 - centrality) * distance_list[point_list_with_distance.len()-1] as f64;

    let total_weight = point_list_with_distance.len() as f64;
    let dimensions = if !project || missing.len() ==0 {
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
        let point = if !project || missing.len() == 0 {
            pointstore.get_copy(point_list_with_distance[i].0)
        } else {
            project_missing(&pointstore.get_copy(point_list_with_distance[i].0), &missing)
        };
        for j in 0..dimensions {
            sum_values[j] += point[j] as f64;
            sum_values_sq[j] += point[j] as f64 * point[j] as f64;
        }
        /// the else can be filtered further
        let weight: f32 = if (point_list_with_distance[i].1 <= threshold as f32) {
            1.0
        } else {
            threshold as f32 / point_list_with_distance[i].1
        };

        vec.push((point, weight));
    };

    for j in 0..dimensions {
        mean[j] = (sum_values[j] / total_weight as f64) as f32;
        let t: f64 = sum_values_sq[j] / total_weight as f64 - sum_values[j] * sum_values[j] / (total_weight as f64 * total_weight as f64);
        deviation[j] = f64::sqrt(if (t > 0.0) { t } else { 0.0 }) as f32;
    };
    let mut median = vec![0.0f32; dimensions];
    for j in 0..dimensions {
        let mut v : Vec<f32>= vec.iter().map(|x| x.0[j]).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        median[j] = v[vec.len() / 2];
    };

    let mut summary = summarize(&vec,max_number);
    SampleSummary {
        summary_points: summary.summary_points.clone(),
        relative_weight: summary.relative_weight.clone(),
        total_weight: summary.total_weight,
        mean,
        median,
        deviation
    }
}