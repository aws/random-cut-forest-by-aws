use std::cmp::min;
use crate::common::divector::DiVector;

/**
 * This class maintains a simple discounted statistics. Setters are avoided
 * except for discount rate which is useful as initialization from raw scores
 */
#[repr(C)]
#[derive(Clone)]
pub struct AnomalyDescriptor {

    // the current input point; can have missing values
    pub current_input: Vec<f32>,

    // current timestamp
    pub current_timestamp: usize,

    // potential missing values in the current input (ideally None)
    pub missing_values: Option<Vec<i32>>,

    // potentially transformed point used by RCF, can have different dimensions than input
    pub rcf_point: Option<Vec<f32>>,

    pub score: f32,

    pub internal_timestamp: usize,

    pub threshold: f32,

    pub anomaly_grade: f32,

    pub data_confidence: f32,

    // present only if grade > 0
    pub attribution: Option<DiVector>,

    pub expected_rcf_point: Option<Vec<f32>>,

    pub relative_index: Option<usize>,

    // useful for time augmented forests
    pub expected_timestamp: Option<usize>,

    pub start_of_anomaly: Option<bool>,

    pub in_high_score_region: Option<bool>,

    pub relevant_attribution: Option<Vec<f32>>,

    pub time_attribution: Option<f32>,

    // the values being replaced; may correspond to past
    pub past_values: Option<Vec<f32>>,

    pub past_timestamp: Option<usize>,

    pub expected_values_list: Option<Vec<Vec<f32>>>,

    pub likelihood_of_values: Option<Vec<f32>>

}

impl AnomalyDescriptor {
    pub fn new(point: Vec<f32>, timestamp: usize) -> Self {
        AnomalyDescriptor {
            current_input: point.clone(),
            current_timestamp: timestamp,
            missing_values: None,
            rcf_point: None,
            score: 0.0,
            internal_timestamp: 0,
            threshold: 0.0,
            anomaly_grade: 0.0,
            data_confidence: 0.0,
            attribution: None,
            expected_rcf_point: None,
            relative_index: None,
            expected_timestamp: None,
            start_of_anomaly: None,
            in_high_score_region: None,
            relevant_attribution: None,
            time_attribution: None,
            past_values: None,
            past_timestamp: None,
            expected_values_list: None,
            likelihood_of_values: None
        }
    }

    pub fn new_with_missing_values(point: Vec<f32>, timestamp: usize, missing_values: Vec<i32>) -> Self {
        assert!(missing_values.len() <= point.len(), "incorrect input");
        for i in &missing_values {
            assert!( *i >=0 && (*i as usize) < point.len(), "incorrect input")
        }
        AnomalyDescriptor {
            current_input: point.clone(),
            current_timestamp: timestamp,
            missing_values: Some(missing_values.clone()),
            rcf_point: None,
            score: 0.0,
            internal_timestamp: 0,
            threshold: 0.0,
            anomaly_grade: 0.0,
            data_confidence: 0.0,
            attribution: None,
            expected_rcf_point: None,
            relative_index: None,
            expected_timestamp: None,
            start_of_anomaly: None,
            in_high_score_region: None,
            relevant_attribution: None,
            time_attribution: None,
            past_values: None,
            past_timestamp: None,
            expected_values_list: None,
            likelihood_of_values: None
        }
    }
}

