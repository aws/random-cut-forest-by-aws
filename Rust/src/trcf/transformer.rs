use crate::common::deviation::Deviation;
use crate::common::rangevector::RangeVector;
use crate::trcf;
use crate::trcf::preprocessor::Preprocessor;
use crate::trcf::types::TransformMethod;
use crate::trcf::types::TransformMethod::{DIFFERENCE, NONE, NORMALIZE, NORMALIZE_DIFFERENCE, SUBTRACT_MA,WEIGHTED};
use crate::util::check_argument;
use crate::trcf::preprocessor::DEFAULT_DEVIATION_STATES;
use crate::types::Result;

#[repr(C)]
#[derive(Clone)]
pub struct WeightedTransformer {
    transform_method : TransformMethod,
    deviations : Vec<Deviation>,
    input_length : usize,
    weights : Vec<f32>,
}

impl WeightedTransformer {

    pub fn new(transform_method : TransformMethod, input_length : usize, transform_decay: f64, weights : &[f32]) -> Result<Self> {
        check_argument(input_length == weights.len(), "incorrect lengths")?;
        let mut deviations : Vec<Deviation> = Vec::new();
        for _i in 0..2*input_length {
            deviations.push(Deviation::new(transform_decay)?);
        }
        for _i in 0..(DEFAULT_DEVIATION_STATES - 2) * input_length {
            deviations.push(Deviation::new(0.1 * transform_decay)?);
        }
        if transform_method == NONE{
            for w in weights {
                check_argument( *w ==1.0, "incorrect setting for NONE transformation")?;
            }
        }
        Ok(WeightedTransformer{
            transform_method,
            deviations,
            input_length,
            weights: Vec::from(weights)
        })
    }

     pub fn update(&mut self, input: &[f32], previous: &[f32]) -> Result<()> {
         check_argument(input.len() == self.input_length, " incorrect length")?;
         check_argument(previous.len() == self.input_length, " incorrect length")?;
         for i in 0..self.input_length {
             self.deviations[i].update(input[i] as f64);
             let deviation = self.deviations[i].deviation();
             self.deviations[i + self.input_length].update((input[i] - previous[i]) as f64);
             let difference_mean = self.deviations[i + self.input_length].mean();
             let difference_deviation = self.deviations[i + self.input_length].deviation();
             self.deviations[i + 2 * self.input_length].update(deviation);
             self.deviations[i + 3 * self.input_length].update(difference_mean);
             self.deviations[i + 4 * self.input_length].update(difference_deviation);
         }
         Ok(())
     }

    fn normalized_scale(&self, i: usize) -> f32 {
        (self.deviations[ i + 2 * self.input_length].mean() + 1.0) as f32
    }

    fn basic_shift(&self, i: usize) -> f32 {
        self.deviations[i].mean() as f32
    }

    fn shift_difference(&self, i: usize) -> f32 {
        self.deviations[ i + self.input_length].mean()  as f32
    }

    fn basic_drift(&self, i: usize) -> f32 {
        self.deviations[ i + 3*self.input_length].mean()  as f32
    }

    fn difference(&self, input : &mut [f32], previous : &[f32]) {
         for (x,y) in input.iter_mut().zip(previous){
             *x -= y;
         }
    }

    fn add(&self, input : &mut [f32], previous : &[f32]) {
        for (x,y) in input.iter_mut().zip(previous){
            *x += y;
        }
    }

    fn add_ma(&self, input : &mut [f32]) {
        for i in 0..input.len() {
            input[i] += self.basic_shift(i);
        }
    }

    fn subtract_ma(&self, input : &mut [f32]) {
        for i in 0..input.len() {
            input[i] -=  self.basic_shift(i);
        }
    }

    fn weight(&self, input : &mut[f32]){
        for (x,y) in input.iter_mut().zip(&self.weights){
            *x = *x * (*y);
        }
    }

    fn weight_invert(&self, input : &mut[f32]){
        for (x,y) in input.iter_mut().zip(&self.weights){
            *x = if *y == 0.0 {0.0} else {*x/y};
        }
    }

    fn normalize(&self, input : &mut [f32]){
        for i in 0..input.len() {
                input[i] -= self.basic_shift(i);
                input[i] = input[i] /self.normalized_scale(i);
        }
    }

    fn normalize_invert(&self, input : &mut [f32]){
        for i in 0..input.len() {
            input[i] = input[i]*self.normalized_scale(i);
            input[i] += self.basic_shift(i);
        }
    }

    fn normalize_difference(&self, input : &mut [f32], previous : &[f32]){
        for i in 0..input.len() {
            input[i] -= previous[i];
            input[i] = input[i]/ self.normalized_scale(i);
        }
    }

    fn normalize_difference_invert(&self, input : &mut [f32], previous : &[f32]) {
        for i in 0..input.len() {
            input[i] = input[i]* self.normalized_scale(i);
            input[i] +=  previous[i];
        }
    }

    pub fn transform(&self, input : &[f32], previous : &[f32]) -> Vec<f32> {
        let mut answer: Vec<f32> = Vec::from(input);
        match &self.transform_method {
            NONE => {},
            DIFFERENCE=> {self.difference(&mut answer,previous)},
            SUBTRACT_MA=> {self.subtract_ma(&mut answer)},
            NORMALIZE=> {self.normalize(&mut answer)},
            NORMALIZE_DIFFERENCE => {self.normalize_difference(&mut answer, previous)},
            WEIGHTED => {self.weight(&mut answer)},
        };
        answer
    }

    pub fn invert(&self, input : &[f32], previous : &[f32]) -> Vec<f32> {
        let mut answer: Vec<f32> = Vec::from(input);
        for (x,y) in answer.iter_mut().zip(&self.weights){
            *x = if *y ==0.0 {0.0} else {*x/y};
        }
        match &self.transform_method {
            NONE => {},
            DIFFERENCE=> {self.add(&mut answer,previous)},
            SUBTRACT_MA=> {self.add_ma(&mut answer)},
            NORMALIZE=> {self.normalize_invert(&mut answer)},
            NORMALIZE_DIFFERENCE => {self.normalize_difference_invert(&mut answer,previous)},
            WEIGHTED => {self.weight_invert(&mut answer)},
        };
        answer
    }

    pub fn invert_forecast(&self, forecast :&mut RangeVector<f32>, previous : &[f32]) -> Result<()>{
        let horizon = forecast.values.len() / self.input_length;
        for i in 0..horizon {
            for j in 0..self.input_length {
                let factor = if self.weights[j] == 0.0 { 0.0 } else { 1.0 / self.weights[j] };
                if self.transform_method != NONE {
                    forecast.scale(i * self.input_length + j, factor as f32);
                }

                if self.transform_method == NORMALIZE || self.transform_method == NORMALIZE_DIFFERENCE {
                    forecast.scale(i * self.input_length + j, self.normalized_scale(j));
                }

                forecast.shift(i * self.input_length + j, i as f32 * self.basic_drift(j));

                if self.transform_method == NORMALIZE || self.transform_method == SUBTRACT_MA {
                    forecast.shift(i * self.input_length + j, self.basic_shift(j));
                }
            }
        }
        if self.transform_method == DIFFERENCE || self.transform_method == NORMALIZE_DIFFERENCE {
            forecast.cascaded_add(previous)?;
        }
        Ok(())
    }

    pub fn scale(&self) -> Vec<f32> {
        let mut answer = self.weights.clone();
        if self.transform_method == NORMALIZE || self.transform_method == NORMALIZE_DIFFERENCE {
            for i in 0..self.input_length {
                answer[i] *= self.normalized_scale(i);
            }
        }
        answer
    }

    pub fn shift(&self) -> Vec<f32> {
        let mut answer = vec![0.0;self.input_length];
        if self.transform_method == NORMALIZE || self.transform_method == SUBTRACT_MA {
            for i in 0..self.input_length {
                answer[i] += self.basic_shift(i);
            }
        }
        answer
    }

    pub fn difference_deviations(&self) -> Vec<f32> {
        let mut answer = vec![0.0f32;self.input_length];
        for i in 0..self.input_length {
            answer[i] = self.deviations[ i + 4*self.input_length].mean() as f32;
        }
        answer
    }

}
