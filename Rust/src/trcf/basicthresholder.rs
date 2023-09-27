
use crate::common::deviation::Deviation;
use crate::trcf::types::TransformMethod;
use crate::trcf::types::TransformMethod::{DIFFERENCE, NONE, NORMALIZE_DIFFERENCE};
use crate::util::{maxf32, minf32};
use crate::types::Result;

const DEFAULT_ELASTICITY : f32= 0.01;
const DEFAULT_SCORE_DIFFERENCING: f32 = 0.5;

const DEFAULT_MINIMUM_SCORES : i32 = 10;
const DEFAULT_ABSOLUTE_THRESHOLD : f32= 0.8;
const DEFAULT_ABSOLUTE_SCORE_FRACTION :f32 = 0.5;
const DEFAULT_LOWER_THRESHOLD :f32 = 1.0;
const DEFAULT_INITIAL_THRESHOLD :f32= 1.5;
const DEFAULT_Z_FACTOR :f32 = 3.0;
const MINIMUM_Z_FACTOR : f32 = 2.0;
const DEFAULT_FACTOR_ADJUSTMENT_THRESHOLD : f32 = 0.9;

#[repr(C)]
#[derive(Clone)]
pub struct BasicThresholder {
    elasticity: f32,
    count: i32,
    score_differencing: f32,
    last_score: f32,
    primary_deviation: Deviation,
    secondary_deviation: Deviation,
    threshold_deviation: Deviation,
    auto_threshold: bool,
    absolute_threshold:f32,
    absolute_score_fraction: f32,
    lower_threshold: f32,
    factor_adjustment_threshold: f32,
    initial_threshold: f32,
    z_factor: f32,
    minimum_scores: i32,
}

impl BasicThresholder {
    pub fn new_adjustible(discount : f64, adjust:bool) -> Result<Self> {
        Ok(BasicThresholder{
            elasticity: DEFAULT_ELASTICITY,
            count: 0,
            score_differencing: DEFAULT_SCORE_DIFFERENCING,
            last_score: 0.0,
            primary_deviation: Deviation::new(discount as f64)?,
            secondary_deviation: Deviation::new(discount as f64)?,
            threshold_deviation: Deviation::new(discount as f64/2.0)?,
            auto_threshold: adjust,
            absolute_threshold: DEFAULT_ABSOLUTE_THRESHOLD,
            absolute_score_fraction: DEFAULT_ABSOLUTE_SCORE_FRACTION,
            lower_threshold: DEFAULT_LOWER_THRESHOLD,
            factor_adjustment_threshold: DEFAULT_FACTOR_ADJUSTMENT_THRESHOLD,
            initial_threshold: DEFAULT_INITIAL_THRESHOLD,
            z_factor: DEFAULT_Z_FACTOR,
            minimum_scores: DEFAULT_MINIMUM_SCORES
        })
    }

    pub fn new(discount : f64) -> Result<Self>{
        BasicThresholder::new_adjustible(discount,false)
    }

    pub fn is_deviation_ready(&self) -> bool {
        if self.count < self.minimum_scores {
            return false;
        }
        if self.score_differencing != 0.0 {
            return self.primary_deviation.count() >= self.minimum_scores;
        }
        return true;
    }

    fn intermediate_fraction(&self) -> f32 {
        if self.count < self.minimum_scores {
            return 0.0;
        } else if self.count > 2 * self.minimum_scores {
            return 1.0;
        } else {
            return (self.count - self.minimum_scores) as f32 * 1.0 / self.minimum_scores as f32;
        }
    }

    fn adjusted_factor(&self, factor : f32, method:TransformMethod, _dimension:usize) -> f32 {
        let base = self.primary_deviation.mean();
        let corrected_factor = if (base as f32) < self.factor_adjustment_threshold && method != NONE {
            (base as f32) * factor / self.factor_adjustment_threshold
        } else {
            factor
        };
        if corrected_factor < MINIMUM_Z_FACTOR {
            MINIMUM_Z_FACTOR
        } else {
            corrected_factor
        }
    }

    fn long_term_deviation(&self, method : TransformMethod, shingle_size:usize) -> f32{
        if shingle_size == 1 && !(method == DIFFERENCE || method == NORMALIZE_DIFFERENCE) {
            minf32((f64::sqrt(2.0)*self.threshold_deviation.mean()) as f32,self.primary_deviation.mean() as f32)
        } else {
            let mut first = self.primary_deviation.deviation();
            let t = f64::sqrt(2.0) * self.threshold_deviation.deviation();
            if t < first {
                first = t;
            }
            if self.secondary_deviation.deviation() < first {
                first = self.secondary_deviation.deviation();
            }
            self.score_differencing * (first as f32) + (1.0 - self.score_differencing) * (self.secondary_deviation.deviation() as f32)
        }
    }

    pub fn threshold_and_grade(&self, score : f32, method: TransformMethod, dimension : usize,
                               shingle_size : usize)  -> ( f32, f32) {
        self.threshold_and_grade_with_factor(score,self.z_factor,method,dimension,shingle_size)
    }

    pub fn threshold_and_grade_with_factor(&self, score : f32, factor : f32, method: TransformMethod, dimension : usize,
                                   shingle_size : usize)  -> ( f32, f32) {
        let intermediate_fraction = self.intermediate_fraction();
        let new_factor = self.adjusted_factor(factor, method, dimension);
        let long_term = self.long_term_deviation(method, shingle_size);
        let scaled_deviation = (new_factor - 1.0) * long_term + self.primary_deviation.deviation() as f32;

        let mut absolute = self.absolute_threshold;
        let t = self.primary_deviation.mean() as f32;
        if self.auto_threshold && intermediate_fraction >= 1.0 && t <
            self.factor_adjustment_threshold {
            absolute = t * absolute / self.factor_adjustment_threshold;
        }
        let threshold = if !self.is_deviation_ready() {
            maxf32(self.initial_threshold,absolute)
        } else {
            let t = intermediate_fraction * (self.primary_deviation.mean() as f32 + scaled_deviation) +
                (1.0 - intermediate_fraction) * self.initial_threshold;
            maxf32(t,absolute)
        };

        if   (score as f32) < threshold || threshold == 0.0 {
            return (threshold, 0.0);
        } else {
            let mut t = self.surprise_index(score, threshold, new_factor, scaled_deviation / new_factor);
            t = minf32(f32::floor(t * 20.0) / 16.0,1.0);
            if t > 0.0 { (threshold, t) } else {
                (score as f32, 0.0)
            }
        }
    }

    fn surprise_index(&self, score: f32, base : f32, factor: f32, deviation : f32)  -> f32 {
        if self.is_deviation_ready() {
            let mut t_factor = 2.0 * factor;
            if deviation > 0.0 {
                let z = (score as f32 - base) / deviation;
                t_factor = minf32(z,factor);
            }
            t_factor = t_factor/factor;
            maxf32(t_factor,0.0)
        } else {
            let t = ((score as f32) - self.absolute_threshold) / self.absolute_threshold;
            minf32(1.0,maxf32(t,0.0))
        }
    }


    pub fn threshold(&self) -> f32 {
        self.primary_deviation.mean() as f32 + self.z_factor * self.primary_deviation.deviation() as f32
    }

    pub fn primary_grade(&self, score : f32) -> f32 {
        if !self.is_deviation_ready() {
            return 0.0;
        }
        let threshold = self.threshold();
        let mut t = score - threshold;
        let deviation = self.primary_deviation.deviation() as f32;
        if t>0.0 {
            if deviation > 0.0 {
                t = t/(deviation);
                return minf32(t,1.0);
            } else {
                return 0.1;
            }
        } else {
            return 0.0;
        }
    }

    pub fn primary_threshold_and_grade(&self, score: f32) -> (f32,f32) {
        (self.threshold(),self.primary_grade(score))
    }

    pub fn update_threshold(&mut self, score:f32) {
        let gap : f32 = score - self.primary_deviation.mean() as f32;
        if gap>0.0 {
            self.threshold_deviation.update(gap as f64);
        }
    }

    pub fn update_primary(&mut self, score : f64) {
        self.last_score = score as f32;
        self.primary_deviation.update(score);
        self.update_threshold(score as f32);
        self.count += 1;
    }

    pub fn update_both(&mut self, primary : f32, secondary : f32) {
        self.last_score = primary;
        self.primary_deviation.update(primary as f64);
        self.secondary_deviation.update(secondary as f64);
        self.update_threshold(primary as f32);
        self.count += 1;
    }

    pub fn update(&mut self, primary: f32, secondary: f32, last_score: f32){
        self.update_both(minf32(2.0,primary),secondary - last_score);
    }


    pub fn z_factor(&self) -> f32 {
        self.z_factor
    }

    pub fn set_z_factor(&mut self, factor : f32){
        self.z_factor = factor;
    }

    // the next set of functions maintain the invariant that
    // absolute_threshold <= lower_threshold < initial_threshold <= upper_threshold
    // absolute_threshold <= lower_threshold < 2.0 *lower_threshold <= upper_threshold
    // to increase proceed as upper_threshold, initial_threshold, lower_threshold, absolute_threshold
    // to decrease proceed in reverse of the above order

    pub fn set_lower_threshold(&mut self, lower : f32) {
        self.lower_threshold = lower;
    }

    pub fn set_absolute_threshold(&mut self, value:f32) {
        self.absolute_threshold = value;
    }

    pub fn set_initial_threshold(&mut self, initial : f32) {
        self.initial_threshold =  initial;
    }

    pub fn set_score_differencing(&mut self, horizon:f32) {
        assert!(horizon >= 0.0 && horizon <= 1.0, "incorrect horizon parameter");
        self.score_differencing = horizon;
    }

    pub fn last_score(&self) -> f32{
        self.last_score
    }

    pub fn primary_mean(&self) -> f64 {
        self.primary_deviation.mean()
    }

    pub fn primary_deviation(&self) -> f64 {
        self.primary_deviation.deviation()
    }

}
