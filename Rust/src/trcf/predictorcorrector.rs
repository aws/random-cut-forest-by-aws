use std::f32::consts::E;
use num::abs;
use crate::common::divector::DiVector;
use crate::rcf::{AugmentedRCF, RCF};
use crate::types::{Result};
use crate::trcf::basicthresholder::BasicThresholder;
use crate::common::descriptor::Descriptor;
use crate::common::deviation::Deviation;
use crate::trcf::types::CorrectionMode::{ANOMALY_IN_SHINGLE, CONDITIONAL_FORECAST, DATA_DRIFT, FORECAST, NOISE};
use crate::trcf::types::ScoringStrategy::EXPECTED_INVERSE_HEIGHT;
use crate::trcf::types::TransformMethod::{DIFFERENCE, NORMALIZE_DIFFERENCE};
use crate::util::{absf32, maxf32, minf32};

const DEFAULT_NORMALIZATION_PRECISION:f32 = 1e-3;
const DEFAULT_NUMBER_OF_MAX_ATTRIBUTORS: usize = 5;
const NOISE_FACTOR: f32 = 1.0;
const DEFAULT_SAMPLING_SUPPORT : f32 = 0.1;
const DEFAULT_DIFFERENTIAL_FACTOR : f32 = 0.3;
const DEFAULT_RUN_ALLOWED : usize = 2;

#[repr(C)]
#[derive(Clone)]
pub struct PredictorCorrector {
    basic_thresholder: BasicThresholder,
    auto_adjust : bool,
    run_length : usize,
    deviations_actual : Vec<Deviation>,
    deviations_expected: Vec<Deviation>,
    max_attributors: usize
}

impl PredictorCorrector {
    // for mappers
    pub fn new(discount: f64, auto_adjust:bool, base_dimension : usize) -> Result<Self> {
        let mut a = Vec::new();
        let mut b =Vec::new();
        if auto_adjust {
            for _ in 0..base_dimension {
                a.push(Deviation::new(discount)?);
                b.push(Deviation::new(discount)?);
            }
        }
        Ok(PredictorCorrector {
            basic_thresholder: BasicThresholder::new(discount)?,
            auto_adjust,
            run_length : 0,
            deviations_actual: a,
            deviations_expected: b,
            max_attributors: DEFAULT_NUMBER_OF_MAX_ATTRIBUTORS
        })
    }

    pub fn expected_point<U :?Sized,Label : Sync + Copy, Attributes: Sync + Copy>(di_vector: &DiVector, max_attributors: usize, position: usize, base_dimension: usize, point: &[f32],
                          forest: &Box<U>) -> Result<Vec<f32>>  where U: AugmentedRCF<Label,Attributes> {
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
            let prediction = forest.conditional_field(&likely_missing_indices, point, 1.0, false, 0)?.median;
            for i in likely_missing_indices{
                answer[i] = prediction[i];
            }
        }
        Ok(answer)
    }

    fn trigger(&self, candidate: &DiVector, gap: usize, base_dimension: usize, ideal: &DiVector,
               last_descriptor: &Descriptor, threshold: f32) -> bool {
        match &last_descriptor.last_anomaly {
            None => { return true; },
            Some(y) => {
                match &y.attribution {
                    None => { return true; },
                    Some(_x) => {
                        let last_score = y.score;
                        let dimensions = candidate.dimensions();
                        let difference = gap * base_dimension;
                        if difference < dimensions {
                            let mut differential_remainder = 0.0;
                            for i in (dimensions - difference)..dimensions {
                                let low_diff = candidate.low[i] - ideal.low[i];
                                differential_remainder += absf32(low_diff as f32);
                                let high_diff = candidate.high[i] - ideal.high[i];
                                differential_remainder += absf32(high_diff as f32)
                            }
                            return differential_remainder > DEFAULT_DIFFERENTIAL_FACTOR * last_score
                                && differential_remainder as f32 * (dimensions as f32) / difference as f32 > threshold;
                        } else {
                            return true;
                        }
                    }
                }
            }
        }
    }

    pub fn apply_basic_corrector(point: &[f32], gap: usize, shingle_size: usize, base_dimension: usize,
                                 last_descriptor: &Descriptor, use_difference: bool, time_augmented: bool) -> Vec<f32> {

        let mut corrected_point = Vec::from(point);
        if gap > shingle_size || last_descriptor.last_anomaly.is_none() {
            return corrected_point;
        }
        let last_expected_point = &last_descriptor.last_anomaly.as_ref().unwrap().expected_rcf_point;
        let last_anomaly_point = &last_descriptor.last_anomaly.as_ref().unwrap().anomalous_rcf_point;
        let last_relative_index = last_descriptor.last_anomaly.as_ref().unwrap().relative_index;
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

    fn centered_transform_pass(base_dimensions: usize, result: &Descriptor, point : &[f32]) -> f32 {
        let mut max_factor = 0.0f32;
        let scale = result.scale.as_ref().unwrap();
        let shift = result.shift.as_ref().unwrap();
        let deviations = result.difference_deviations.as_ref().unwrap();
        for i in 0..point.len() {
            if absf32(point[i]) * scale[i%base_dimensions] > DEFAULT_NORMALIZATION_PRECISION * (1.0+absf32(shift[i%base_dimensions])) {
                max_factor = 1.0;
            }
        }

        if max_factor>0.0 {
            for i in 0..base_dimensions {
                let z = absf32(point[point.len() - base_dimensions + i])*scale[i];
                let dev = maxf32(0.0,deviations[i]);
                if z > NOISE_FACTOR * dev {
                    max_factor = minf32(1.0, maxf32(max_factor, z / (3.0 * dev)));
                }
            }
        }
        max_factor
    }

    fn calculate_path_deviation(point : &[f32], start_position: usize, index: usize, base_dimension : usize, differenced : bool) -> f32 {
        let mut position = start_position;
        let mut variation = 0.0;
        let mut observation: usize = 0;
        while position + index + base_dimension < point.len() {
            variation += if differenced { absf32(point[position + index]) } else { absf32(point[position + index] - point[position + base_dimension + index]) };
            position += base_dimension;
            observation += 1;
        }
        if observation == 0 {
            0.0
        } else {
            variation / observation as f32
        }
    }

    fn construct_uncertainty_box(point :&[f32], start_position : usize, base_dimension : usize, result: &Descriptor) -> Vec<f32>{
        let method = result.transform_method;
        let differenced = (method == DIFFERENCE)  || (method == NORMALIZE_DIFFERENCE);
        let scale = result.scale.as_ref().unwrap();
        let shift = result.shift.as_ref().unwrap();
        let mut answer = vec![0.0f32;base_dimension];
        for y in 0..base_dimension {
            let shift_amount = DEFAULT_NORMALIZATION_PRECISION * scale[y] * absf32(shift[y]);
            let path_gap = Self::calculate_path_deviation(point, start_position, y, base_dimension, differenced);
            let noise_gap = NOISE_FACTOR * result.difference_deviations.as_ref().unwrap()[y];
            answer[y] = maxf32(scale[y] * path_gap,noise_gap) + shift_amount;
        }
        answer
    }

    fn within_unertainty_box(uncertainty_box: &[f32], start_position : usize, scale: &[f32], point: &[f32], other_point : &[f32]) -> bool {
        let mut answer = false;
        for y in 0..uncertainty_box.len() {
            let a = scale[y] * point[start_position + y];
            let b = scale[y] * other_point[start_position + y];
            answer = answer || a < b - uncertainty_box[y] || a > b + uncertainty_box[y];
        }
        return !(answer);
    }

    fn explained_by_conditional_field<U :?Sized,Label : Sync + Copy, Attributes: Sync + Copy>(uncertainty_box: &[f32], point: &[f32], corrected_point : &[f32], start_position : usize,
                                      result :&Descriptor, forest :&Box<U>) -> Result<bool>
        where U: AugmentedRCF<Label,Attributes> {
        let list = forest.near_neighbor_list(corrected_point, 50)?;
        let mut weight = 0;
        let total = list.len();
        for e in list {
            if Self::within_unertainty_box(uncertainty_box, start_position, result.scale.as_ref().unwrap(), point, &e.1) {
                weight += 1;
            }
        }
        return Ok(weight as f32 >= DEFAULT_SAMPLING_SUPPORT * total as f32);
    }


    pub fn update_auto_adjust(&mut self, point : &[f32]){
        if self.auto_adjust && self.run_length > 0 {
            for y in 0..self.deviations_actual.len() {
                self.deviations_actual[y].update(point[y] as f64);
            }
            self.run_length +=1;
        }
    }

    fn update_score(&mut self, score: f32, corrected_score:f32, result: &Descriptor, last_descriptor: &Descriptor){
        if result.scoring_strategy == EXPECTED_INVERSE_HEIGHT {
            let last_score = if last_descriptor.scoring_strategy == EXPECTED_INVERSE_HEIGHT {
                last_descriptor.score
            } else {
                0.0
            };
            self.basic_thresholder.update(score,corrected_score,last_score);
        } else {
            self.basic_thresholder.update_primary(score as f64);
        }
    }

    pub fn detect_and_modify<U :?Sized,Label : Sync + Copy, Attributes: Sync + Copy>(&mut self, result: &mut Descriptor, last_descriptor : &Descriptor, shingle_size: usize, forest : &Box<U>)
        -> Result<()>
    where U : AugmentedRCF<Label,Attributes> {
        match &result.rcf_point {
            None => return Ok(()),
            Some(point) => {
                let score = if result.scoring_strategy == EXPECTED_INVERSE_HEIGHT {
                    forest.score(point)? as f32
                } else {
                    forest.density_interpolant(point)?.distance.total() as f32
                };
                let method = result.transform_method;
                result.score = score;
                let internal_timestamp = result.values_seen;
                let gap = internal_timestamp - if last_descriptor.last_anomaly.is_none() {0}
                                          else {last_descriptor.last_anomaly.as_ref().unwrap().values_seen };
                if score == 0.0 {
                    return Ok(());
                }

                let dimension = forest.dimensions();
                let base_dimension = dimension / shingle_size;
                let (threshold, grade) = if result.scoring_strategy == EXPECTED_INVERSE_HEIGHT {
                    self.basic_thresholder.threshold_and_grade(score, method, dimension, shingle_size)
                } else {
                    self.basic_thresholder.primary_threshold_and_grade(score)
                };
                result.threshold = threshold;

                if grade == 0.0 {
                    if self.auto_adjust {
                        self.run_length = 0;
                        for y in 0..base_dimension {
                            self.deviations_actual[y].reset();
                            self.deviations_expected[y].reset();
                        }
                    }
                    result.anomaly_grade = 0.0;
                    self.update_score(score, score, result,last_descriptor);
                    return Ok(());
                }

                let candidate = result.scoring_strategy == last_descriptor.scoring_strategy &&
                    (score > last_descriptor.score
                    || last_descriptor.score - last_descriptor.threshold > score
                    - maxf32(threshold, last_descriptor.threshold)
                    * (1.0 + maxf32(0.2, self.run_length as f32
                    / (2.0 * maxf32(10.0, shingle_size as f32)))));


                let use_difference = method == DIFFERENCE || method == NORMALIZE_DIFFERENCE;
                let corrected_point = Self::apply_basic_corrector(point, gap, shingle_size, base_dimension,
                                                                  last_descriptor, use_difference, result.time_augmented);
                let mut corrected_score = score;
                if gap > 0 && gap <= shingle_size && last_descriptor.last_anomaly.is_some() {
                    corrected_score = if result.scoring_strategy == EXPECTED_INVERSE_HEIGHT {
                        forest.score(&corrected_point)? as f32
                    } else {
                        forest.density_interpolant(&corrected_point)?.distance.total() as f32
                    };
                    let (_,newgrade) = if result.scoring_strategy == EXPECTED_INVERSE_HEIGHT {
                        self.basic_thresholder.threshold_and_grade(corrected_score, method, dimension, shingle_size)
                    } else {
                        self.basic_thresholder.primary_threshold_and_grade(corrected_score)
                    };
                    // we know we are looking previous anomalies
                    if newgrade == 0.0 {
                        self.update_auto_adjust(point);
                        result.correction_mode = ANOMALY_IN_SHINGLE;
                        self.update_score(score, corrected_score, result,last_descriptor);
                        result.anomaly_grade = 0.0;
                        return Ok(());
                    }
                }

                let working_grade = grade * Self::centered_transform_pass(base_dimension, result, &corrected_point);
                if working_grade == 0.0 {
                    self.update_auto_adjust(point);
                    result.correction_mode = NOISE;
                    result.anomaly_grade = 0.0;
                    self.update_score(score, corrected_score, result, last_descriptor);
                    return Ok(());
                }


                let mut attribution = if result.scoring_strategy == EXPECTED_INVERSE_HEIGHT {
                    forest.attribution(&corrected_point)?
                } else {
                    forest.density_interpolant(&corrected_point)?.distance
                };

                let index = attribution.max_gap_contribution(base_dimension, gap)?;
                let start_position = index * point.len() / shingle_size;
                let uncertainty_box = Self::construct_uncertainty_box(point, start_position, base_dimension, result);

                if self.auto_adjust &&
                    Self::explained_by_conditional_field(&uncertainty_box, point, &corrected_point, start_position,
                                                        result, forest)? {
                    self.update_auto_adjust(point);
                    result.correction_mode = CONDITIONAL_FORECAST;
                    result.anomaly_grade = 0.0;
                    self.update_score(score, corrected_score, result, last_descriptor);
                    return Ok(());
                }

                let expected_point = Self::expected_point(&attribution, self.max_attributors, start_position, base_dimension, point, forest)?;
                if gap < shingle_size {
                    let new_attribution = forest.attribution(&expected_point)?;
                    if !self.trigger(&attribution, gap, base_dimension, &new_attribution, last_descriptor, threshold) {
                        result.correction_mode = ANOMALY_IN_SHINGLE;
                        self.update_auto_adjust(point);
                        result.anomaly_grade = 0.0;
                        self.update_score(score, corrected_score, result,last_descriptor);
                        return Ok(());
                    }
                }

                if Self::within_unertainty_box(&uncertainty_box, start_position, result.scale.as_ref().unwrap(), point,
                                    &expected_point) {
                    result.correction_mode = FORECAST;
                    self.update_auto_adjust(point);
                    result.anomaly_grade = 0.0;
                    self.update_score(score, corrected_score, result,last_descriptor);
                    return Ok(());
                }


                if candidate {
                    if self.auto_adjust {
                        for y in 0..base_dimension {
                            self.deviations_actual[y].update(point[dimension - base_dimension + y] as f64);
                            self.deviations_expected[y].update(expected_point[dimension - base_dimension + y] as f64);
                        }
                        if self.run_length > DEFAULT_RUN_ALLOWED {
                            let mut within = true;
                            for y in 0..base_dimension {
                                within =
                                    absf32(self.deviations_actual[y].mean() as f32 - point[dimension - base_dimension + y]) <
                                        maxf32(2.0 * self.deviations_actual[y].deviation() as f32,
                                               NOISE_FACTOR * result.difference_deviations.as_ref().unwrap()[y]);
                                within = within && absf32(self.deviations_expected[y].mean() as f32
                                    - expected_point[dimension - base_dimension + y]) < 2.0
                                    * maxf32(self.deviations_expected[y].deviation() as f32,
                                             self.deviations_actual[y].deviation() as f32)
                                    + 0.1 * absf32(
                                    (self.deviations_actual[y].mean() - self.deviations_expected[y].mean()) as f32);
                            }
                            if within { // already adjusted
                                result.correction_mode = DATA_DRIFT;
                                result.anomaly_grade = 0.0;
                                self.update_score(score, corrected_score,result, last_descriptor);
                                return Ok(());
                            }
                        }
                    }
                }


                self.run_length += 1;
                result.expected_rcf_point=Some(expected_point);
                result.anomaly_grade = working_grade;
                self.update_score(score, corrected_score, result,last_descriptor);
                result.relative_index = index as i32 - shingle_size as i32 + 1;
                attribution.normalize(score as f64);
                result.attribution = Some(attribution);
                return Ok(());
            }
        }
    }


    pub fn set_z_factor(&mut self, factor: f32) {
        self.basic_thresholder.set_z_factor(factor);
    }

    pub fn set_lower_threshold(&mut self, lower :f32) {
        self.basic_thresholder.set_absolute_threshold(lower);
    }

    pub fn set_horizon(&mut self, horizon : f32) {
        self.basic_thresholder.set_score_differencing(horizon);
    }

    pub fn set_initial_threshold(&mut self, initial : f32) {
        self.basic_thresholder.set_initial_threshold(initial);
    }
}
