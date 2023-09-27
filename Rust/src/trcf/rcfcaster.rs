use rand::Rng;
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;
use crate::common::descriptor::Descriptor;
use crate::common::rangevector::RangeVector;
use crate::rcf::{RCF, RCFBuilder, RCFOptions};
use crate::trcf::errorhandler::ErrorHandler;
use crate::trcf::predictorcorrector::PredictorCorrector;
use crate::trcf::preprocessor::{Preprocessor, PreprocessorBuilder};
use crate::trcf::types::{Calibration, ForestMode, TransformMethod};
use crate::trcf::types::Calibration::{MINIMAL, NONE};
use crate::trcf::types::ForestMode::{STANDARD, STREAMING_IMPUTE, TIME_AUGMENTED};
use crate::trcf::types::TransformMethod::NORMALIZE;
use crate::types::{Result};
use crate::rcf::RCFOptionsBuilder;
use crate::trcf::basictrcf::{BasicTRCF, core_process, State, TRCFOptions, TRCFOptionsBuilder};
use crate::util::check_argument;

pub const DEFAULT_ERROR_PERCENTILE : f32 = 0.1;

pub const MAX_ERROR_HORIZON : usize = 1024;

pub struct RCFCaster {
    forecast_horizon : usize,
    rcf : Box<dyn RCF + Send + Sync>,
    state: State,
    error_handler : ErrorHandler,
    calibration_method : Calibration,
}

impl RCFCaster {

    pub fn process(&mut self, point: &[f32], timestamp: u64) -> Result<Descriptor>{
        let mut result = core_process(Some(&self.rcf), &mut self.state, point, timestamp)?;
        match result.rcf_point.as_ref() {
            // this path would be taken for all un-normalized transformations
            // relies on internal shingling
            Some(x) => {
                let dimension = x.len();
                let shingle_size = self.state.preprocessor.shingle_size();
                self.rcf.update(&x[(dimension - (dimension / shingle_size))..dimension], timestamp as u64)?;
            },
            _ => { self.state.preprocessor.drain(Some(&mut self.rcf))?; }
        }

        if self.rcf.is_output_ready() {
            self.error_handler.update_actuals(&result.current_input, result.deviations_post.as_ref().expect("should be present"))?;
            self.error_handler.augment_descriptor(&mut result);

            let mut forecast = self.extrapolate(self.forecast_horizon)?;
            self.error_handler.update_forecasts(&mut forecast.0)?;
            result.forecast = Some(forecast.0);
        }
        Ok(result)
    }

    pub fn extrapolate(&self, look_ahead: usize) -> Result<(RangeVector<f32>,Option<RangeVector<f64>>)> {
        let mut a = self.state.preprocessor.invert_extrapolation(self.rcf.extrapolate(look_ahead)?)?;
        self.error_handler.calibrate(self.calibration_method, &mut a.0)?;
        Ok(a)
    }

    pub fn process_sequentially(&mut self, input: &[(&[f32],u64)]) -> Result<Vec<Descriptor>> {
       input.iter().map(|(a, b)| self.process(*a, *b))
            .into_iter().collect()
    }
}


pub struct RCFCasterOptions {
    add_error: bool,
    calibration : Calibration,
    forecast_horizon: usize,
    error_horizon: Option<usize>,
}

pub trait RCFCasterOptionsBuilder: TRCFOptionsBuilder {
    fn get_rcf_caster_options(&mut self) -> &mut RCFCasterOptions;
    fn transform_decay(&mut self, transform_decay: f64) -> &mut Self {
        self.get_trcf_options().transform_decay = Some(transform_decay);
        self
    }
    fn calibration(&mut self, calibration: Calibration) -> &mut Self {
        self.get_rcf_caster_options().calibration = calibration;
        self
    }
    fn forecast_horizon(&mut self, forecast_horizon: usize) -> &mut Self {
        self.get_rcf_caster_options().forecast_horizon = forecast_horizon;
        self
    }
    fn error_horizon(&mut self, error_horizon: usize) -> &mut Self {
        self.get_rcf_caster_options().error_horizon = Some(error_horizon);
        self
    }
}


impl Default for RCFCasterOptions {
    fn default() -> Self {
        RCFCasterOptions{
            add_error: false,
            calibration: MINIMAL,
            forecast_horizon: 10,
            error_horizon: None
        }
    }
}

pub struct RCFCasterBuilder {
    id: u64,
    input_dimensions: usize,
    shingle_size: usize,
    trcf_options: TRCFOptions,
    rcf_options : RCFOptions<u64,u64>,
    rcf_caster_options: RCFCasterOptions,
}

impl RCFOptionsBuilder<u64,u64> for RCFCasterBuilder {
    fn get_rcf_options(&mut self) -> &mut RCFOptions<u64,u64> {
        &mut self.rcf_options
    }
}

impl TRCFOptionsBuilder for RCFCasterBuilder {
    fn get_trcf_options(&mut self) -> &mut TRCFOptions {
        &mut self.trcf_options
    }
}

impl RCFCasterOptionsBuilder for RCFCasterBuilder {
    fn get_rcf_caster_options(&mut self) -> &mut RCFCasterOptions {
        &mut self.rcf_caster_options
    }
}

impl RCFCasterBuilder {
    pub fn new(id : u64,input_dimensions: usize, shingle_size: usize, forecast_horizon: usize) -> Self {
        RCFCasterBuilder {
            id,
            input_dimensions,
            shingle_size,
            trcf_options: Default::default(),
            rcf_options : Default::default(),
            rcf_caster_options: RCFCasterOptions {forecast_horizon, ..Default::default()}
        }
    }

    pub fn build(&self) -> Result<RCFCaster> {
        check_argument(self.trcf_options.forest_mode!= STREAMING_IMPUTE, "not yet supported")?;
        check_argument( self.input_dimensions > 0, "input_dimensions cannot be 0")?;
        check_argument( self.shingle_size > 0, "shingle size cannot be 0")?;
        self.rcf_options.validate()?;
        self.trcf_options.validate(self.input_dimensions)?;
        let output_after = self.rcf_options.output_after.unwrap_or(1 + self.rcf_options.capacity / 4);
        let time_decay = self.rcf_options.time_decay.unwrap_or(0.1/self.rcf_options.capacity as f64);
        let transform_decay = self.trcf_options.transform_decay.unwrap_or(0.1/self.rcf_options.capacity as f64);
        let weights = match &self.trcf_options.weights {
            Some(x) => x.clone(),
            _ => vec![1.0; self.input_dimensions]
        };
        let random_seed = self.rcf_options.random_seed.unwrap_or( ChaCha20Rng::from_entropy().gen::<u64>());
        let rcf = RCFBuilder::<u64,u64>::new(self.input_dimensions,self.shingle_size)
            .tree_capacity(self.rcf_options.capacity).number_of_trees(self.rcf_options.number_of_trees)
            .random_seed(random_seed)
            .parallel_enabled(self.rcf_options.parallel_enabled).time_decay(time_decay)
            .bounding_box_cache_fraction(self.rcf_options.bounding_box_cache_fraction)
            .output_after(output_after)
            .initial_accept_fraction(self.rcf_options.initial_accept_fraction).build_default()?;
        let preprocessor = PreprocessorBuilder::new(self.input_dimensions,self.shingle_size)
            .transform_decay(transform_decay)
            .transform_method(self.trcf_options.transform_method)
            .forest_mode(self.trcf_options.forest_mode)
            .random_seed(random_seed+1)
            .weights(&weights)
            .start_normalization(self.trcf_options.start_normalization)
            .stop_normalization(self.trcf_options.stop_normalization).build()?;
        let predictor_corrector = PredictorCorrector::new(transform_decay,!self.trcf_options.verbose,self.input_dimensions)?;

        let error_horizon= self.rcf_caster_options.error_horizon.unwrap_or(MAX_ERROR_HORIZON);
        check_argument(error_horizon<=MAX_ERROR_HORIZON, "calibration horizon should be smaller")?;
        let error_handler = ErrorHandler::new(self.rcf_caster_options.add_error,self.input_dimensions,self.rcf_caster_options.forecast_horizon,error_horizon,DEFAULT_ERROR_PERCENTILE);
        Ok(RCFCaster{
            forecast_horizon: self.rcf_caster_options.forecast_horizon,
            rcf,
            state: State::new(self.id,1,self.trcf_options.scoring_strategy,predictor_corrector,preprocessor)?,
            error_handler,
            calibration_method: self.rcf_caster_options.calibration
        })
    }
}
