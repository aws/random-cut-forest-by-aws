
use crate::common::divector::DiVector;
use crate::samplerplustree::boundingbox::BoundingBox;

#[repr(C)]
#[derive(Clone)]
pub struct InterpolationMeasure {
    pub measure: DiVector,
    pub distance: DiVector,
    pub probability_mass: DiVector,
    pub sample_size: f32
}

impl InterpolationMeasure {
    pub fn empty(dimension: usize, sample_size: f32)  -> Self {
        InterpolationMeasure {
            measure: DiVector::empty(dimension),
            distance: DiVector::empty(dimension),
            probability_mass: DiVector::empty(dimension),
            sample_size,
        }
    }

    pub fn new(measure : DiVector, distance: DiVector, prob_mass:DiVector,sample_size: f32) -> Self{
        assert!(measure.dimensions() == distance.dimensions(), " incorrect lengths");
        assert!(measure.dimensions() == prob_mass.dimensions(), " incorrect lengths");
        InterpolationMeasure{
            measure: measure,
            distance: distance,
            probability_mass: prob_mass,
            sample_size
        }
    }

    pub fn add_to(&self, other: &mut InterpolationMeasure) {
        self.probability_mass.add_to(&mut other.probability_mass);
        self.distance.add_to(& mut other.distance);
        self.measure.add_to(&mut other.measure);
        other.sample_size += self.sample_size;
    }

    pub fn divide(&mut self, num: usize){
        self.scale(1.0/num as f64);
        self.scale_samples(1.0/num as f64);
    }


    pub fn scale(&mut self, factor : f64) {
        self.distance.scale(factor);
        self.probability_mass.scale(factor);
        self.measure.scale(factor);
    }

    pub fn scale_samples(&mut self, factor : f64) {
        self.sample_size =  (self.sample_size as f64 * factor) as f32;
    }

    pub fn update(&mut self, point: &[f32], bounding_box: &BoundingBox, measure: f64) -> f64{
        let min_values = bounding_box.get_min_values();
        let max_values = bounding_box.get_max_values();
        let minsum: f32 = min_values
            .iter()
            .zip(point)
            .map(|(&x, &y)| if x - y > 0.0 { x - y } else { 0.0 })
            .sum();
        let maxsum: f32 = point
            .iter()
            .zip(max_values)
            .map(|(&x, &y)| if x - y > 0.0 { x - y } else { 0.0 })
            .sum();
        let sum = maxsum + minsum;
        let new_range = sum as f64 + bounding_box.get_range_sum();
        let prob = sum as f64/(new_range);
        if prob > 0.0 {
            self.scale( 1.0 - prob);
            for i in 0.. point.len() {
                if point[i] > max_values[i] {
                    let t = (point[i] - max_values[i]) as f64/new_range;
                    self.distance.high[i] += t * (point[i] - min_values[i]) as f64;
                    self.probability_mass.high[i] += t;
                    self.measure.high[i] += measure * t;
                } else if point[i] < min_values[i] {
                    let t = (min_values[i] - point[i]) as f64/new_range;
                    self.distance.low[i] += t * (max_values[i] - point[i]) as f64;
                    self.probability_mass.low[i] += t;
                    self.measure.low[i] += measure * t;
                }
            }
        }
        prob
    }

    pub fn directional_measure(&self, threshold: f64, manifold_dimension: f64) -> DiVector{
        assert!(self.sample_size >= 0.0 && self.measure.total() >= 0.0, " cannot have negative samples or measure");
        if self.sample_size == 0.0f32 || self.measure.total() == 0.0{
            return DiVector::empty(self.measure.dimensions());
        }

        let mut sum_of_factors = 0.0;

        for i in 0..self.measure.dimensions() {
            let mut t = if self.probability_mass.high_low_sum(i) > 0.0 {
                self.distance.high_low_sum(i) / self.probability_mass.high_low_sum(i)
            } else {
                0.0
            };

            if t > 0.0 {
                t = f64::exp(f64::ln(t) * manifold_dimension)
                    * self.probability_mass.high_low_sum(i);
            }
            sum_of_factors += t;
        }

        let density_factor = 1.0 / (threshold + sum_of_factors);
        let mut answer = self.measure.clone();
        answer.scale(density_factor);
        answer
    }

    pub fn directional_density(&self) -> DiVector {
        self.directional_measure(1e-3, self.measure.dimensions() as f64)
    }

    pub fn density(&self) -> f64 {
       self.directional_density().total()
    }
}