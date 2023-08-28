use std::cmp::min;
use rand::Rng;
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;
use crate::common::descriptor::{AnomalyInformation, Descriptor};
use crate::common::deviation::Deviation;
use crate::common::rangevector::RangeVector;
use crate::rcf::{AugmentedRCF, RCF};
use crate::trcf::basictrcf::BasicTRCFBuilder;
use crate::trcf::transformer::WeightedTransformer;
use crate::trcf::types::{ForestMode, ImputationMethod, TransformMethod};
use crate::trcf::types::ForestMode::{STANDARD, STREAMING_IMPUTE, TIME_AUGMENTED};
use crate::trcf::types::ImputationMethod::USE_RCF;
use crate::trcf::types::TransformMethod::{NORMALIZE, NORMALIZE_DIFFERENCE};
use crate::util::check_argument;
use crate::types::Result;

const DEFAULT_START_NORMALIZATION: usize = 10;

const DEFAULT_STOP_NORMALIZATION : usize = usize::MAX;

const DEFAULT_CLIP_FACTOR : f32 = 100.0;

const DEFAULT_NORMALIZATION : bool = false;

pub const DEFAULT_DEVIATION_STATES : usize = 5;

#[repr(C)]
#[derive(Clone)]
pub struct Preprocessor {
    timestamp_deviations : Vec<Deviation>,
    normalize_time : bool,
    weight_time : f32,
    transform_decay : f64,
    previous_timestamps : Vec<u64>,
    internal_timestamp : usize,
    initial_values : Vec<Vec<f32>>,
    initial_timestamps : Vec<u64>,
    start_normalization : usize,
    stop_normalization : usize,
    values_seen : usize,
    default_fill : Vec<f32>,
    use_imputed_fraction : f32,
    number_of_imputed : usize,
    clip_factor : f32,
    shingle_size : usize,
    input_dimensions : usize,
    last_shingled_input: Vec<f32>,
    last_shingled_point: Vec<f32>,
    data_quality : Vec<Deviation>,
    imputation_method : ImputationMethod,
    transform_method : TransformMethod,
    forest_mode : ForestMode,
    transformer : WeightedTransformer
}

impl Preprocessor {

    fn past_initial(&self) -> bool{
        self.values_seen + 1> self.start_normalization ||
            (self.transform_method != NORMALIZE && self.transform_method != NORMALIZE_DIFFERENCE)
    }

    pub fn shingled_point<U :?Sized,Label : Sync + Copy, Attributes: Sync + Copy>(&mut self, _rcf: Option<&Box<U>>, input : &[f32], timestamp: u64) -> Result<Option<Vec<f32>>> where
    U : AugmentedRCF<Label,Attributes> {
        check_argument(input.len() == self.input_dimensions, "incorrect length")?;
        for x in input {
            check_argument(f32::is_finite(*x), " numbers should be finite")?;
        }

        // the shingle will always be created, possibly wih leading 0's
        if self.past_initial() {
            if self.initial_values.len() > 0 {
                // corresponds to external shingling in MultiRCF
                // For anomaly detection it is not relevant when we update the state
                // it can be on seeing the next input (here) or having seen the last input
                // but forecasting for a single time series is better served
                // having seen all available data specially when there are a few values
                self.drain(None as Option<&mut Box<dyn RCF>>)?;
            }
            let input = &self.transform(input,timestamp);
            let mut copy = self.last_shingled_point.clone();
            Self::shift_vector(&mut copy,input);
            Ok(Some(copy))
        } else {
            Ok(None)
        }
    }

    fn shift_vector<T:Copy>(shingle: &mut Vec<T>, point: &[T]) {
        let dimension = shingle.len();
        for i in 0..(dimension-point.len()) {
            shingle[i] = shingle[i + point.len()];
        }
        for i in 0..point.len() {
            shingle[dimension - point.len() + i] = point[i];
        }
    }

    fn update_timestamps(&mut self, timestamp:u64,previous: u64){
        self.timestamp_deviations[0].update(timestamp as f64);
        self.timestamp_deviations[1].update((timestamp - previous) as f64);
        let deviation = self.timestamp_deviations[0].deviation();
        self.timestamp_deviations[2].update(deviation);
        let difference_mean = self.timestamp_deviations[1].mean();
        let difference_deviation = self.timestamp_deviations[1].deviation();
        self.timestamp_deviations[3].update( difference_mean);
        self.timestamp_deviations[4].update(difference_deviation);
    }

    fn update(&mut self, input : &[f32], last_shingled_point: &[f32], timestamp: u64) -> Result<()>{
        let dimension = self.last_shingled_input.len();
        if self.values_seen < self.stop_normalization {
            self.transformer.update(input, &self.last_shingled_input[(dimension - self.input_dimensions)..dimension])?;
            self.update_timestamps(timestamp, self.previous_timestamps[self.shingle_size - 1]);
        }

        Self::shift_vector(&mut self.last_shingled_input,&input);
        for (x,y) in self.last_shingled_point.iter_mut().zip(last_shingled_point) {
            *x = *y;
        }
        Self::shift_vector(&mut self.previous_timestamps, &vec![timestamp]);
        self.internal_timestamp += 1; // will count number of updates
        self.values_seen += 1;
        Ok(())
    }

    pub fn post_process(&mut self, result: &mut Descriptor, point:&[f32],timestamp:u64, _last_descriptor:&Descriptor) -> Result<()> {
        if self.past_initial() {
            if let Some(y) = &result.rcf_point {
                if result.anomaly_grade > 0.0 {
                    let base_dimension = self.input_dimensions;
                    let block = (self.shingle_size as i32 + result.relative_index - 1) as usize;
                    let start: usize = block * base_dimension;
                    let past_values = if result.relative_index == 0 {
                        Vec::from(point)
                    } else {
                        Vec::from(&self.last_shingled_input[(start + base_dimension)..(start + 2 * base_dimension)])
                    };
                    if let Some(x) = &result.expected_rcf_point {
                        let expected_values_list = vec![self.transformer.invert(&x[start..start + base_dimension],
                                                                                &self.last_shingled_input[start..start + base_dimension])];
                        let likelihood_of_values = vec![1.0f32];
                        result.last_anomaly = Some(
                            AnomalyInformation {
                                expected_rcf_point: x.clone(),
                                anomalous_rcf_point: y.clone(),
                                relative_index: result.relative_index,
                                values_seen: result.values_seen,
                                attribution: result.attribution.clone(),
                                score: result.score,
                                grade: result.anomaly_grade,
                                expected_timestamp: timestamp, // changes not implemented
                                relevant_attribution: None,
                                time_attribution: 0.0, // not implemented
                                past_values,
                                past_timestamp: self.previous_timestamps[block],
                                expected_values_list,
                                likelihood_of_values
                            }
                        )
                    }
                }
                self.update(point, y, timestamp)?;
                result.deviations_post = Some(self.difference_deviations());
            }
        } else {
            self.initial_values.push(Vec::from(point));
            self.initial_timestamps.push(timestamp);
            self.values_seen += 1;
            if self.values_seen == self.start_normalization {
                result.deviations_post = Some(self.difference_deviations());
            }
        }
        Ok(())
    }

    pub fn drain<U:?Sized>(&mut self, rcf: Option<&mut Box<U>>) -> Result<()> where U:RCF {
        if self.values_seen == self.start_normalization {
            let mut previous = &self.initial_values[0];
            let mut previous_timestamp = self.initial_timestamps[0];
            for i in 0..self.initial_values.len() {
                self.transformer.update(&self.initial_values[i],&previous)?;
                self.update_timestamps(self.initial_timestamps[i],previous_timestamp);
                previous_timestamp = self.initial_timestamps[i];
                previous = &self.initial_values[i];
            }

            self.previous_timestamps[self.shingle_size - 1] = self.initial_timestamps[0];
            let dimension = self.shingle_size * self.input_dimensions;
            for i in 0..self.input_dimensions {
                self.last_shingled_input[dimension - self.input_dimensions + i] = self.initial_values[0][i];
            }

            match rcf {
                None => {
                    // transformations will work at this point
                    for (x, &y) in self.initial_values.iter().zip(&self.initial_timestamps) {
                        let z = &self.transform(x, y);
                        self.internal_timestamp += 1;
                        Self::shift_vector(&mut self.last_shingled_input, x);
                        Self::shift_vector(&mut self.last_shingled_point, z);
                        Self::shift_vector(&mut self.previous_timestamps, &vec![y]);
                    }
                },
                Some(f) => {
                    for (x, &y) in self.initial_values.iter().zip(&self.initial_timestamps) {
                        let z = &self.transform(x, y);
                        self.internal_timestamp += 1;
                        Self::shift_vector(&mut self.last_shingled_input, x);
                        Self::shift_vector(&mut self.last_shingled_point, z);
                        Self::shift_vector(&mut self.previous_timestamps, &vec![y]);
                        f.update(z, y)?;
                    }
                }
            }
            // block deallocation
            self.initial_timestamps = Vec::new();
            self.initial_values = Vec::new();
        }
        Ok(())
    }

    pub fn invert_extrapolation(&self, mut range_vector: RangeVector<f32>) -> Result<(RangeVector<f32>,Option<RangeVector<f64>>)> {
        if self.forest_mode != TIME_AUGMENTED {
            let dimension = self.input_dimensions * self.shingle_size;
            self.transformer.invert_forecast(&mut range_vector,&self.last_shingled_input[(dimension - self.input_dimensions)..dimension])?;
            return Ok((range_vector,None))
        } else {
            let augmented = self.get_dimension();
            let lookahead = range_vector.upper.len()/augmented;
            let dimension = self.input_dimensions * self.shingle_size;
            let mut sub_range_vector : RangeVector<f32> = RangeVector::<f32>::new(lookahead*dimension);
            let mut time_range_vector : RangeVector<f64> = RangeVector::<f64>::new(lookahead);
            for i in 0..lookahead {
                for j in 0..dimension {
                    sub_range_vector.upper[i * dimension + j] = range_vector.upper[i * augmented + j];
                    sub_range_vector.values[i * dimension + j] = range_vector.values[i * augmented + j];
                    sub_range_vector.lower[i * dimension + j] = range_vector.lower[i * augmented + j];
                }
                time_range_vector.upper[i] = self.invert_time(range_vector.upper[i*augmented + dimension]);
                time_range_vector.values[i] = self.invert_time(range_vector.values[i*augmented + dimension]);
                time_range_vector.lower[i] = self.invert_time(range_vector.lower[i*augmented + dimension]);
                if time_range_vector.upper[i] < time_range_vector.values[i] {
                    time_range_vector.upper[i] = time_range_vector.values[i];
                }
                if time_range_vector.lower[i] > time_range_vector.values[i] {
                    time_range_vector.lower[i] = time_range_vector.values[i];
                }
            }
            self.transformer.invert_forecast(&mut sub_range_vector,&self.last_shingled_input[(dimension - self.input_dimensions)..dimension])?;
            time_range_vector.cascaded_add(&vec![self.previous_timestamps[self.shingle_size-1] as f64])?;
            return Ok((sub_range_vector,Some(time_range_vector)));
        }
    }

    fn invert_time(&self,value:f32) -> f64{
        let factor = if self.weight_time == 0.0 { 0.0 } else { 1.0 / self.weight_time as f64};
        (value as f64)*factor*self.timescale() + self.timedrift()
    }

    pub fn internal_timestamp(&self) -> usize {
        self.internal_timestamp
    }

    pub fn values_seen(&self) -> usize {
        self.values_seen
    }

    pub fn input_dimensions(&self) -> usize {
        self.input_dimensions
    }

    pub fn shingle_size(&self) -> usize {
        self.shingle_size
    }

    pub fn get_dimension(&self) -> usize {
        if self.forest_mode == TIME_AUGMENTED {
            (self.input_dimensions + 1)*self.shingle_size
        } else {
            self.input_dimensions * self.shingle_size
        }
    }

    pub fn transformation_method(&self) -> TransformMethod {
           self.transform_method
    }

    pub fn forest_mode(&self) -> ForestMode {
        self.forest_mode
    }

    pub fn shift(&self) -> Vec<f32> {
        if self.forest_mode != TIME_AUGMENTED {
            self.transformer.shift()
        } else {
            let mut answer = self.transformer.shift();
             answer.push( self.previous_timestamps[self.shingle_size-1] as f32 + self.timedrift() as f32);
            answer
        }
    }

    pub fn scale(&self) -> Vec<f32> {
        if self.forest_mode != TIME_AUGMENTED {
            self.transformer.scale()
        } else {
            let mut answer = self.transformer.scale();
            let factor = if self.weight_time == 0.0 { 0.0 } else { 1.0 / self.weight_time };
            answer.push(factor * (self.timescale()) as f32);
            answer
        }
    }

    pub fn transform(&self,input:&[f32],timestamp: u64) -> Vec<f32> {
        let dimension = self.input_dimensions * self.shingle_size;
        let mut answer = self.transformer.transform
        (input, &self.last_shingled_input[(dimension - self.input_dimensions)..dimension]);

        if self.forest_mode == TIME_AUGMENTED {
            let previous = if self.values_seen > 0 {
                self.previous_timestamps[self.shingle_size - 1]
            } else {
                timestamp
            };
            answer.push((timestamp as f64 - previous as f64 * (self.weight_time as f64) /
                self.timescale()) as f32);
        }
        answer
    }

    pub fn difference_deviations(&self) -> Vec<f32> {
        if self.forest_mode != TIME_AUGMENTED {
            self.transformer.difference_deviations()
        } else {
            let mut answer = self.transformer.difference_deviations();
            answer.push(self.weight_time as f32 * (self.timestamp_deviations[1].deviation() as f32));
            answer
        }
    }

    fn timescale(&self) -> f64 {
        self.timestamp_deviations[4].mean() + 1.0
    }

    fn timedrift(&self) -> f64 {
        self.timestamp_deviations[3].mean()
    }

    pub fn is_ready(&self) -> bool {
        self.internal_timestamp >= self.shingle_size
    }

    pub fn start_normalization(&self) -> usize {
        self.start_normalization
    }

 }


pub struct PreprocessorBuilder {
    normalize_time: bool,
    input_dimensions: usize ,
    transform_decay:Option<f64>,
    weights: Option<Vec<f32>>,
    weight_time: f32,
    imputation_method: ImputationMethod,
    number_of_imputed: usize,
    clip_factor: f32,
    use_imputed_fraction: f32,
    transform_method: TransformMethod,
    forest_mode : ForestMode,
    default_fill: Option<Vec<f32>>,
    shingle_size: usize,
    random_seed: Option<u64>,
    start_normalization: usize,
    stop_normalization : usize
}

impl Default for PreprocessorBuilder {
    fn default() -> Self {
        PreprocessorBuilder{
            normalize_time: DEFAULT_NORMALIZATION,
            input_dimensions: 1,
            transform_decay: Some(0.001),
            weights: None,
            weight_time: 1.0,
            imputation_method: USE_RCF,
            number_of_imputed: 0,
            clip_factor: DEFAULT_CLIP_FACTOR,
            use_imputed_fraction: 0.0,
            transform_method: NORMALIZE,
            forest_mode: STANDARD,
            default_fill: None,
            shingle_size: 8,
            random_seed: None,
            start_normalization: DEFAULT_START_NORMALIZATION,
            stop_normalization: DEFAULT_STOP_NORMALIZATION,
        }
    }
}

impl PreprocessorBuilder {
    pub fn new(input_dimensions: usize, shingle_size: usize) -> Self {
        PreprocessorBuilder { input_dimensions, shingle_size, ..Default::default() }
    }

    pub fn transform_decay(&mut self, transform_decay: f64) -> &mut PreprocessorBuilder {
        self.transform_decay = Some(transform_decay);
        self
    }

    pub fn forest_mode(&mut self, forest_mode: ForestMode) -> &mut PreprocessorBuilder {
        self.forest_mode = forest_mode;
        self
    }

    pub fn transform_method(&mut self, transform_method: TransformMethod) -> &mut PreprocessorBuilder {
        self.transform_method = transform_method;
        self
    }

    pub fn imputation_method(&mut self, imputation_method: ImputationMethod) -> &mut PreprocessorBuilder {
        self.imputation_method = imputation_method;
        self
    }

    pub fn start_normalization(&mut self, start_normalization: usize) -> &mut PreprocessorBuilder {
        self.start_normalization = start_normalization;
        self
    }

    pub fn stop_normalization(&mut self, stop_normalization: usize) -> &mut PreprocessorBuilder {
        self.stop_normalization = stop_normalization;
        self
    }

    pub fn initial_accept_fraction(&mut self, use_imputed_fraction: f32) -> &mut PreprocessorBuilder {
        self.use_imputed_fraction = use_imputed_fraction;
        self
    }

    pub fn default_fill(&mut self, default_fill: &[f32]) -> &mut PreprocessorBuilder {
        self.default_fill = Some(Vec::from(default_fill));
        self
    }

    pub fn random_seed(&mut self, random_seed: u64) -> &mut PreprocessorBuilder {
        self.random_seed = Some(random_seed);
        self
    }

    pub fn weights(&mut self, weights: &[f32]) -> &mut PreprocessorBuilder {
        self.weights = Some(Vec::from(weights));
        self
    }

    pub fn build(&self) -> Result<Preprocessor> {
        check_argument(self.forest_mode != STREAMING_IMPUTE, "not yet supported")?;
        check_argument(self.input_dimensions > 0, "input_dimensions cannot be 0")?;
        check_argument(self.shingle_size > 0, "shingle size cannot be 0")?;
        let transform_decay = self.transform_decay.unwrap_or(0.001);
        let weights = match &self.weights {
            Some(x) => x.clone(),
            _ => vec![1.0; self.input_dimensions],
        };
        check_argument(self.input_dimensions == weights.len(), "incorrect length of weights")?;
        check_argument(transform_decay >=0.0 && transform_decay<=1.0, "transform decay must be in [0,1]")?;
        let mut timestamp_deviations = Vec::new();
        timestamp_deviations.push(Deviation::new(transform_decay)?);
        timestamp_deviations.push(Deviation::new(transform_decay)?);
        for _ in 0..(DEFAULT_DEVIATION_STATES - 2) {
            timestamp_deviations.push(Deviation::new(0.1*transform_decay)?);
        }
        let dimension= self.input_dimensions * self.shingle_size;
        let augmented = dimension + if self.forest_mode == TIME_AUGMENTED {self.shingle_size} else {0};
        check_argument(self.start_normalization <= self.stop_normalization, " cannot stop normalization before starting")?;
        check_argument(self.start_normalization < 2000, "can cause delays, large memory usage")?;
        let random_seed = self.random_seed.unwrap_or(ChaCha20Rng::from_entropy().gen::<u64>());
        let default_fill = match &self.default_fill {
            Some(x) => x.clone(),
            _ => vec![0.0; self.input_dimensions],
        };
        let preprocessor = Preprocessor {
            timestamp_deviations,
            normalize_time: self.normalize_time,
            weight_time: self.weight_time,
            transform_decay,
            previous_timestamps: vec![0;self.shingle_size],
            internal_timestamp: 0,
            initial_values: vec![],
            initial_timestamps: vec![],
            start_normalization: self.start_normalization,
            stop_normalization: self.stop_normalization,
            values_seen: 0,
            default_fill,
            use_imputed_fraction: self.use_imputed_fraction,
            number_of_imputed: 0,
            clip_factor: self.clip_factor,
            shingle_size: self.shingle_size,
            input_dimensions: self.input_dimensions,
            last_shingled_input: vec![0.0;dimension],
            last_shingled_point: vec![0.0;augmented],
            data_quality: vec![],
            imputation_method: self.imputation_method,
            transform_method: self.transform_method,
            forest_mode: self.forest_mode,
            transformer: WeightedTransformer::new(self.transform_method, self.input_dimensions, transform_decay, &weights)?
        };
        Ok(preprocessor)
    }
}