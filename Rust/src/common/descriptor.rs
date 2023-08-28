use crate::common::divector::DiVector;
use crate::common::rangevector::RangeVector;
use crate::trcf::types::{CorrectionMode, ImputationMethod, ScoringStrategy, TransformMethod};
use crate::trcf::types::ImputationMethod::USE_RCF;
use crate::trcf::types::ScoringStrategy::EXPECTED_INVERSE_HEIGHT;

/**
 * This class maintains a simple discounted statistics. Setters are avoided
 * except for discount rate which is useful as initialization from raw scores
 */
#[repr(C)]
#[derive(Clone)]
pub struct Descriptor {
    pub id: u64,
    pub current_input: Vec<f32>,
    pub current_timestamp: u64,
    pub missing_values: Option<Vec<usize>>,
    pub rcf_point: Option<Vec<f32>>,
    pub score: f32,
    pub correction_mode: CorrectionMode,
    pub values_seen: usize,
    pub transform_method : TransformMethod,
    pub threshold: f32,
    pub anomaly_grade: f32,
    pub data_confidence: f32,
    pub attribution: Option<DiVector>,
    pub relative_index : i32,
    pub scale : Option<Vec<f32>>,
    pub shift : Option<Vec<f32>>,
    pub difference_deviations: Option<Vec<f32>>,
    pub deviations_post : Option<Vec<f32>>,
    pub time_augmented : bool,
    pub expected_rcf_point: Option<Vec<f32>>,
    pub last_anomaly : Option<AnomalyInformation>,
    pub forecast : Option<RangeVector<f32>>,
    pub error_information : Option<ErrorInformation>,
    pub scoring_strategy : ScoringStrategy,
    pub imputation_method : ImputationMethod,
}

#[repr(C)]
#[derive(Clone)]
pub struct AnomalyInformation {
    // we do not explicitly provide a default so that each of these entires are
    // considered carefully before declaring an anomaly
    pub expected_rcf_point: Vec<f32>,
    pub anomalous_rcf_point: Vec<f32>,
    pub relative_index: i32,
    pub values_seen: usize,
    pub attribution: Option<DiVector>,
    pub score: f32,
    pub grade: f32,
    pub expected_timestamp: u64,
    pub relevant_attribution: Option<Vec<f32>>,
    pub time_attribution: f32,
    pub past_values: Vec<f32>,
    pub past_timestamp: u64,
    pub expected_values_list: Vec<Vec<f32>>,
    pub likelihood_of_values: Vec<f32>
}

#[repr(C)]
#[derive(Clone)]
pub struct ErrorInformation {
    pub interval_precision: Vec<f32>,
    pub error_distribution : RangeVector<f32>,
    pub error_rmse : DiVector,
    pub error_mean : Vec<f32>
}

impl Default for Descriptor {
    fn default() -> Self {
        Descriptor{
            id: 0,
            current_input: vec![],
            current_timestamp: 0,
            missing_values: None,
            rcf_point: None,
            score: 0.0,
            correction_mode: CorrectionMode::NONE,
            values_seen: 0,
            transform_method: TransformMethod::NONE,
            threshold: 0.0,
            anomaly_grade: 0.0,
            data_confidence: 0.0,
            attribution: None,
            relative_index: 0,
            scale: None,
            shift: None,
            difference_deviations: None,
            deviations_post: None,
            time_augmented: false,
            expected_rcf_point: None,
            last_anomaly: None,
            forecast: None,
            error_information: None,
            scoring_strategy: EXPECTED_INVERSE_HEIGHT,
            imputation_method: USE_RCF
        }
    }
}

impl Descriptor {
    pub fn new(id: u64, point: &[f32], current_timestamp: u64,time_augmented: bool, missing_values: Option<Vec<usize>>) -> Self {
        if missing_values.as_ref().is_some(){
            for i in missing_values.as_ref().unwrap() {
                assert!( *i < point.len(), "incorrect input")
            }
        }
        Descriptor { id, current_input: Vec::from(point), current_timestamp, time_augmented, missing_values, ..Default::default()}
    }
}

