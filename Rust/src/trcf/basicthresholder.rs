use crate::common::deviation::Deviation;

const DEFAULT_ELASTICITY : f32= 0.01;
const DEFAULT_HORIZON : f32 = 0.5;
const DEFAULT_HORIZON_ONED : f32 = 0.75;
const DEFAULT_MINIMUM_SCORES : i32 = 10;
const DEFAULT_ABSOLUTE_THRESHOLD : f32= 0.8;
const DEFAULT_ABSOLUTE_SCORE_FRACTION :f32 = 0.5;
const DEFAULT_UPPER_THRESHOLD :f32= 2.0;
const DEFAULT_LOWER_THRESHOLD :f32 = 1.0;
const DEFAULT_LOWER_THRESHOLD_ONED :f32 = 1.1;
const DEFAULT_LOWER_THRESHOLD_NORMALIZED : f32 = 0.9;
const DEFAULT_INITIAL_THRESHOLD :f32= 1.5;
const DEFAULT_Z_FACTOR :f32 = 2.0;
const DEFAULT_UPPER_FACTOR :f32 = 5.0;
const DEFAULT_AUTO_ADJUST_LOWER_THRESHOLD :bool = false;
const DEFAULT_THRESHOLD_STEP :f32 = 0.1;

#[repr(C)]
#[derive(Clone)]
pub struct BasicThresholder {
    elasticity: f32,
    count: i32,
    horizon: f32,
    last_score: f32,
    primary_deviation: Deviation,
    secondary_deviation: Deviation,
    threshold_deviation: Deviation,
    auto_threshold: bool,
    absolute_threshold:f32,
    absolute_score_fraction: f32,
    upper_threshold: f32,
    lower_threshold: f32,
    initial_threshold: f32,
    z_factor: f32,
    upper_z_factor: f32,
    in_potential_anomaly: bool,
    minimum_scores: i32,
}

impl BasicThresholder {
    pub fn new_adjustible(discount : f32, adjust:bool) -> Self {
        BasicThresholder{
            elasticity: DEFAULT_ELASTICITY,
            count: 0,
            horizon: DEFAULT_HORIZON,
            last_score: 0.0,
            primary_deviation: Deviation::new(discount as f64),
            secondary_deviation: Deviation::new(discount as f64),
            threshold_deviation: Deviation::new(discount as f64/2.0),
            auto_threshold: adjust,
            absolute_threshold: DEFAULT_ABSOLUTE_THRESHOLD,
            absolute_score_fraction: DEFAULT_ABSOLUTE_SCORE_FRACTION,
            upper_threshold: DEFAULT_UPPER_THRESHOLD,
            lower_threshold: DEFAULT_LOWER_THRESHOLD,
            initial_threshold: DEFAULT_INITIAL_THRESHOLD,
            z_factor: DEFAULT_Z_FACTOR,
            upper_z_factor: DEFAULT_UPPER_FACTOR,
            in_potential_anomaly: false,
            minimum_scores: DEFAULT_MINIMUM_SCORES
        }
    }

    pub fn new(discount : f32) -> Self{
        BasicThresholder::new_adjustible(discount,false)
    }

    pub fn is_deviation_ready(&self) -> bool {
        if self.count < self.minimum_scores {
            return false;
        }

        if self.horizon == 0.0 {
            return self.secondary_deviation.count() >= self.minimum_scores;
        } else if self.horizon == 1.0 {
            return self.primary_deviation.count() >= self.minimum_scores;
        } else {
            return self.secondary_deviation.count() >= self.minimum_scores
                && self.primary_deviation.count() >= self.minimum_scores;
        }
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

    fn short_term_threshold(&self, factor : f32, intermediate_fraction : f32) -> f32{
        if !self.is_deviation_ready() { // count < minimumScore is this branch
            if self.initial_threshold>self.lower_threshold{
                self.initial_threshold
            } else {
                self.lower_threshold
            }
        } else {
            let t = intermediate_fraction *self.longterm_threshold(factor) +
                (1.0 -intermediate_fraction)*self.initial_threshold;
            if t>self.lower_threshold{
                t
            } else {
                self.lower_threshold
            }
        }
    }

    pub fn threshold(&self) -> f32{
        return self.longterm_threshold(self.z_factor);
    }

    pub fn longterm_threshold(&self, factor:f32) -> f32 {
        let t= (self.primary_deviation.mean() as f32) + factor * self.longterm_deviation();
        if t >self.lower_threshold {
            t
        } else {
            self.lower_threshold
        }
    }

    fn longterm_deviation(&self) -> f32{
        self.horizon * (self.primary_deviation.deviation() as f32) + (1.0 - self.horizon) * (self.secondary_deviation.deviation() as f32)
    }

    pub fn anomaly_grade_with_factor(&self, score : f32, previous : bool, factor: f32) -> f32{
        assert!(factor >= self.z_factor, "incorrect call");

        let elastic_addition = if previous {self.elasticity} else {0.0};
        let intermediate_fraction = self.intermediate_fraction();
        if intermediate_fraction == 1.0 {
            if score < self.longterm_threshold(factor) - elastic_addition {
                return 0.0;
            }
            let mut t_factor = self.upper_z_factor;
            let longterm_deviation = self.longterm_deviation();
            if longterm_deviation > 0.0 {
                let t =  (score - self.primary_deviation.mean() as f32) / longterm_deviation;
                if (t as f32) < t_factor {
                    t_factor = t as f32;
                }
            }
            return (t_factor - self.z_factor) / (self.upper_z_factor - self.z_factor);
        } else {
            let t = self.short_term_threshold(factor, intermediate_fraction);
            if score < t - elastic_addition {
                return 0.0;
            }
            let upper = if self.upper_threshold > 2.0 * t {self.upper_threshold} else {2.0*t};
            let quasi_score = if score <upper {score} else {upper};
            return (quasi_score - t) / (upper - t);
        }
    }

    pub fn anomaly_grade(&self, score : f32, previous : bool) -> f32{
        return self.anomaly_grade_with_factor(score, previous, self.z_factor);
    }

    pub fn update_threshold(&mut self, score:f32) {
        let gap : f32 = if score > self.lower_threshold {1.0} else {0.0};
        self.threshold_deviation.update(gap as f64);
        if self.auto_threshold && self.threshold_deviation.count() > self.minimum_scores {
            // note the rate is set at half the anomaly rate
            if self.threshold_deviation.mean() > self.threshold_deviation.discount() {
                let t = self.lower_threshold + DEFAULT_THRESHOLD_STEP;
                self.set_lower_threshold(t, self.auto_threshold);
                self.threshold_deviation.set_count(0);
            } else if self.threshold_deviation.mean() < self.threshold_deviation.discount() / 4.0 {
                let t = self.lower_threshold - DEFAULT_THRESHOLD_STEP;
                if t > self.absolute_threshold {
                    self.set_lower_threshold(t, self.auto_threshold);
                    self.threshold_deviation.set_count(0);
                }
            }
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

    pub fn update(&mut self, primary: f32, secondary: f32, last_score: f32, previous : bool){
        self.update_both(primary,secondary - last_score);
        self.in_potential_anomaly = previous;
    }
    // the next set of functions maintain the invariant that
    // DEFAULT_Z_FACTOR <= z_FACTOR < 2.0 * z_factor <= upper_z_factor

    pub fn set_z_factor(&mut self, factor : f32) {
        let z_factor = if factor > DEFAULT_Z_FACTOR {factor} else {DEFAULT_Z_FACTOR};
        if self.upper_z_factor < 2.0 * z_factor {
            self.upper_z_factor = 2.0 * z_factor;
        }
    }

    pub fn set_upper_z_factor(&mut self, factor : f32) {
        let t = if factor> 2.0*self.z_factor {factor} else {2.0*self.z_factor};
        self.upper_z_factor = t
    }

    // the next set of functions maintain the invariant that
    // absolute_threshold <= lower_threshold < initial_threshold <= upper_threshold
    // absolute_threshold <= lower_threshold < 2.0 *lower_threshold <= upper_threshold
    // to increase proceed as upper_threshold, initial_threshold, lower_threshold, absolute_threshold
    // to decrease proceed in reverse of the above order

    pub fn set_lower_threshold(&mut self, lower : f32, adjust: bool) {
        let lower_t = if lower>self.absolute_threshold {lower} else {self.absolute_threshold};
        self.lower_threshold = lower_t;
        self.auto_threshold = adjust;
        if self.initial_threshold < lower_t {
            self.initial_threshold = lower_t;
        };
        if self.upper_threshold < 2.0 * lower_t {
            self.upper_threshold = 2.0 * lower_t;
        }
    }

    pub fn set_absolute_threshold(&mut self, value:f32) {
        self.absolute_threshold = value;
        self.set_lower_threshold(self.absolute_threshold, self.auto_threshold);
    }

    pub fn set_initial_threshold(&mut self, initial : f32) {
        self.initial_threshold = if initial< self.lower_threshold {self.lower_threshold} else {initial};
        if self.upper_threshold < initial {
            self.upper_threshold = initial;
        }
    }

    pub fn set_upper_threshold(&mut self, upper : f32) {
        self.upper_threshold = if upper < self.initial_threshold {self.initial_threshold} else {upper};
        if self.upper_threshold < 2.0 * self.lower_threshold {
            self.upper_threshold = 2.0 * self.lower_threshold;
        }
    }

    pub fn in_potential_anomaly(&self) -> bool {
        self.in_potential_anomaly
    }

    pub fn set_horizon(&mut self, horizon:f32) {
        assert!(horizon >= 0.0 && horizon <= 1.0, "incorrect horizon parameter");
        self.horizon = horizon;
    }

    pub fn last_score(&self) -> f32{
        self.last_score
    }

}
