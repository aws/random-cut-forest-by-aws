use crate::common::divector::DiVector;
use crate::rcf::RCF;
use crate::trcf::basicthresholder::BasicThresholder;
use crate::common::anomalydescriptor::AnomalyDescriptor;

const DEFAULT_NUMBER_OF_MAX_ATTRIBUTORS: usize = 5;
const DEFAULT_REPEAT_ANOMALY_Z_FACTOR : f32 = 3.5;
const DEFAULT_IGNORE_SIMILAR_FACTOR :f32 = 0.3;
const DEFAULT_IGNORE_SIMILAR : bool = false;

#[repr(C)]
#[derive(Clone)]
pub struct PredictorCorrector {
    basic_thresholder: BasicThresholder,
    ignore_similar : f32,
    trigger_factor : f32,
    max_attributors: usize
}

impl PredictorCorrector {
    // for mappers
    pub fn new(discount: f32) -> Self {
        PredictorCorrector {
            basic_thresholder: BasicThresholder::new_adjustible(discount, false),
            ignore_similar: DEFAULT_IGNORE_SIMILAR_FACTOR,
            trigger_factor: DEFAULT_REPEAT_ANOMALY_Z_FACTOR,
            max_attributors: DEFAULT_NUMBER_OF_MAX_ATTRIBUTORS
        }
    }

    pub fn expected_point(di_vector: &DiVector, max_attributors: usize, position: usize, base_dimension: usize, point: &[f32],
                          forest: &Box<dyn RCF>) -> Vec<f32> {
        let mut likely_missing_indices: Vec<usize>;

        if base_dimension == 1 {
            likely_missing_indices = vec![position; 1];
        } else {
            let mut sum = 0.0;
            let mut values = vec![0.0; base_dimension];
            for i in 0..base_dimension {
                values[i] = di_vector.high_low_sum(i + position);
                sum += values[i];
            }
            // sort decreasing
            values.sort_by(|a, b| b.partial_cmp(a).unwrap());

            let mut pick = 0;
            while pick < base_dimension && values[pick] >= (sum * 0.5 / max_attributors as f64) {
                pick += 1;
            }
            likely_missing_indices = Vec::new();

            if pick != 0 && pick <= max_attributors {
                let cutoff = values[pick - 1];
                for i in 0..base_dimension {
                    if di_vector.high_low_sum(i + position) >= cutoff && likely_missing_indices.len() < max_attributors {
                        likely_missing_indices.push(position + i);
                    }
                }
            }
        }

        let mut answer = Vec::from(point);
        if likely_missing_indices.len() != 0 && (2 * likely_missing_indices.len()  < forest.dimensions()) {
            let prediction = forest.conditional_field(&likely_missing_indices, point, 1.0, false, 0).unwrap().median;
            for i in likely_missing_indices{
                answer[i] = prediction[i];
            }
        }
        answer
    }

    fn trigger(&self, candidate: &DiVector, gap: usize, base_dimension: usize, ideal: Option<DiVector>,
               previous: bool, last_anomaly_descriptor: &AnomalyDescriptor) -> bool {
        match &last_anomaly_descriptor.attribution {
            None => { return true; },
            Some(x) => {
                let last_score = last_anomaly_descriptor.score;
                let dimensions = candidate.dimensions();
                let difference = gap * base_dimension;
                if difference < dimensions {
                    match ideal {
                        None => {
                            let mut remainder = 0.0;
                            for i in (dimensions - difference)..dimensions {
                                remainder += candidate.high_low_sum(i);
                            }
                            return self.basic_thresholder.anomaly_grade_with_factor((remainder as f32) * dimensions as f32 / difference as f32, previous, self.trigger_factor) > 0.0;
                        },
                        Some(ref y) => {
                            let mut differential_remainder = 0.0;
                            for i in (dimensions - difference)..dimensions {
                                let low_diff = candidate.low[i] - ideal.as_ref().unwrap().low[i];
                                differential_remainder += if low_diff > 0.0 { low_diff } else { -low_diff };
                                let high_diff = candidate.high[i] - ideal.as_ref().unwrap().high[i];
                                differential_remainder += if high_diff > 0.0 { high_diff } else { -high_diff };
                            }
                            return (differential_remainder > self.ignore_similar as f64 * last_score as f64) &&
                                self.basic_thresholder.anomaly_grade_with_factor((differential_remainder as f32) * (dimensions as f32) / difference as f32, previous, self.trigger_factor) > 0.0;
                        }
                    }
                } else {
                    return true;
                }
            }
        }
    }

    pub fn apply_basic_corrector(point: &[f32], gap: usize, shingle_size: usize, base_dimension: usize,
                                 last_anomaly_descriptor: &AnomalyDescriptor, use_difference: bool, time_augmented: bool) -> Vec<f32> {
        assert!(gap <= shingle_size, "incorrect invocation");

        let mut corrected_point = Vec::from(point);
        let last_expected_point = last_anomaly_descriptor.expected_rcf_point.as_ref().unwrap();
        let last_anomaly_point = last_anomaly_descriptor.rcf_point.as_ref().unwrap();
        let last_relative_index = last_anomaly_descriptor.relative_index.unwrap();
        if gap < shingle_size {
            for i in gap * base_dimension..point.len() {
                corrected_point[i - gap * base_dimension] = last_expected_point[i];
            }
        }
        if last_relative_index == 0 { // is is possible to fix other cases, but is more complicated
            if use_difference {
                for y in 0..base_dimension {
                    corrected_point[point.len() - gap * base_dimension + y] +=
                        last_anomaly_point[point.len() - base_dimension + y] -
                            last_expected_point[point.len() - base_dimension + y];
                }
            } else if time_augmented {
                // definitely correct the time dimension which is always differenced
                // this applies to the non-differenced cases
                corrected_point[point.len() - (gap - 1) * base_dimension - 1] +=
                    last_anomaly_point[point.len() - 1]
                        - last_expected_point[point.len() - 1];
            }
        }
        return corrected_point;
    }


    pub fn detect_and_modify(&mut self, result: &mut AnomalyDescriptor, last_anomaly_descriptor : &AnomalyDescriptor, shingle_size: usize, forest : &Box<dyn RCF>, use_difference: bool, time_augmented: bool) {
        match &result.rcf_point {
            None => return,
            Some(point) => {
                let score = forest.score(point).unwrap() as f32;
                result.score = score;
                if score == 0.0 {
                    return;
                }
                let internal_timestamp = result.internal_timestamp;
                let base_dimension = forest.dimensions() / shingle_size;
                let start_position = (shingle_size - 1) * base_dimension;
                result.threshold = self.basic_thresholder.threshold();
                let previous = self.basic_thresholder.in_potential_anomaly();

                if self.basic_thresholder.anomaly_grade(score, previous) == 0.0 {
                    result.anomaly_grade = 0.0;
                    result.in_high_score_region = Some(false);
                    self.basic_thresholder.update(score, score, 0.0, false);
                    return;
                }

                // the score is now high enough to be considered an anomaly
                result.in_high_score_region = Some(true);

                let gap = internal_timestamp - last_anomaly_descriptor.internal_timestamp;

                let reasonable_forecast = result.forecast_reasonable;

                if reasonable_forecast && gap > 0 && gap <= shingle_size {
                    if let Some(_expected) = &last_anomaly_descriptor.expected_rcf_point {
                        let corrected_point = Self::apply_basic_corrector(point, gap, shingle_size, base_dimension,
                                                                          last_anomaly_descriptor, use_difference, time_augmented);
                        let corrected_score = forest.score(&corrected_point).unwrap() as f32;
                        // we know we are looking previous anomalies
                        if self.basic_thresholder.anomaly_grade(corrected_score, true) == 0.0 {
                            // fixing the past makes this anomaly go away; nothing to do but process the
                            // score
                            // we will not change inHighScoreRegion however, because the score has been
                            // larger
                            self.basic_thresholder.update(score, corrected_score, 0.0, false);
                            result.expected_rcf_point = Some(corrected_point);
                            result.anomaly_grade = 0.0;
                            return;
                        }
                    }
                }

                let attribution = forest.attribution(point).unwrap();
                // index is 0 .. (shingle_size - 1); this is a departure from java version
                let index = attribution.max_contribution(base_dimension);

                if !previous && self.trigger(&attribution, gap, base_dimension, None, false, last_anomaly_descriptor) {
                    result.anomaly_grade = self.basic_thresholder.anomaly_grade(score, false);
                    result.start_of_anomaly = Some(true);
                    self.basic_thresholder.update(score, score, 0.0, true);
                    let start_position = base_dimension * index;
                    if reasonable_forecast {
                        let new_point = Self::expected_point(&attribution, self.max_attributors, start_position, base_dimension, point, &forest);
                        result.expected_rcf_point = Some(new_point);
                    }
                    result.relative_index = Some(index as i32 - shingle_size as i32 + 1);
                    result.attribution = Some(attribution);
                    return;
                } else if reasonable_forecast {
                    let start_position = base_dimension * index;
                    let new_point = Self::expected_point(&attribution, self.max_attributors, start_position, base_dimension, point, &forest);
                    let new_attribution = forest.attribution(&new_point).unwrap();
                    let new_score = forest.score(&new_point).unwrap() as f32;


                    if self.trigger(&attribution, gap, base_dimension, Some(new_attribution), previous,
                                    &last_anomaly_descriptor) {
                        result.anomaly_grade = self.basic_thresholder.anomaly_grade(score, previous);
                        result.expected_rcf_point = Some(new_point);
                        result.relative_index = Some(0);
                        result.attribution = Some(attribution);
                        self.basic_thresholder.update(score, new_score, 0.0, true);
                    } else {
                        // previousIsPotentialAnomaly is true now, but not calling it anomaly either
                        self.basic_thresholder.update(score, new_score, 0.0, true);
                        result.anomaly_grade = 0.0;
                        return;
                    }
                }
                // previousIsPotentialAnomaly is true now, but not calling it anomaly either
                self.basic_thresholder.update(score, score, 0.0, true);
                result.anomaly_grade = 0.0;
                return;
            },
        }
    }


    pub fn set_z_factor(&mut self, factor: f32) {
        self.basic_thresholder.set_z_factor(factor);
        if factor > self.trigger_factor {
            self.trigger_factor = factor;
        }
    }

    pub fn set_lower_threshold(&mut self, lower :f32) {
        self.basic_thresholder.set_absolute_threshold(lower);
    }

    pub fn set_horizon(&mut self, horizon : f32) {
        self.basic_thresholder.set_horizon(horizon);
    }

    pub fn set_initial_threshold(&mut self, initial : f32) {
        self.basic_thresholder.set_initial_threshold(initial);
    }
}
