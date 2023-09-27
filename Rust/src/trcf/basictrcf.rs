use std::hash::Hash;
use rand::Rng;
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;
use crate::common::descriptor::Descriptor;
use crate::common::deviation::Deviation;
use crate::common::rangevector::RangeVector;
use crate::rcf::{AugmentedRCF, RCF, RCFBuilder, RCFOptions};
use crate::types::{Result};
use crate::trcf::predictorcorrector::PredictorCorrector;
use crate::trcf::preprocessor::{Preprocessor, PreprocessorBuilder};
use crate::trcf::types::ForestMode::{STANDARD, STREAMING_IMPUTE, TIME_AUGMENTED};
use crate::trcf::types::{ForestMode, ImputationMethod, ScoringStrategy, TransformMethod};
use crate::trcf::types::TransformMethod::{NONE, NORMALIZE};
use crate::util::check_argument;
use crate::rcf::RCFOptionsBuilder;
use crate::trcf::types::ImputationMethod::USE_RCF;
use crate::trcf::types::ScoringStrategy::{DISTANCE, EXPECTED_INVERSE_HEIGHT};

#[repr(C)]
#[derive(Clone)]
pub struct Bandit {
    id: u64,
    current_model: usize,
    switches: usize,
    affirmations: usize,
    interval: (usize, usize),
    stats: Vec<Deviation>,
}

#[repr(C)]
#[derive(Clone)]
pub struct State {
    pub id: u64,
    random_seed : u64,
    pub scoring_strategy: ScoringStrategy,
    pub bandit: Bandit,
    pub predictor_corrector: PredictorCorrector,
    pub preprocessor: Preprocessor,
    pub last_descriptor : Descriptor,
}

impl Bandit{
   pub fn new(id: u64, arms:usize, interval:(usize,usize)) -> Result<Self>{
       Ok(Bandit{
           id,
           switches : 0,
           affirmations : 0,
           current_model: if arms==1 {0} else {arms},
           interval,
           stats: vec![Deviation::new(0.0)?;arms]
       })
   }

    pub fn current_model(&self) -> usize {
        self.current_model
    }

    pub fn is_evaluating(&self,arms:usize, internal_timestamp:usize) -> bool {
        arms == self.current_model || (internal_timestamp >= self.interval.0 && self.interval.1 >= internal_timestamp)
    }

    pub fn update(&mut self, scores:&[f32]) -> Result<()>{
        check_argument(scores.len() == self.stats.len(), "incorrect length")?;
        self.stats.iter_mut().zip(scores).for_each(|(x,&y)|
            {if y>0.0 {
                x.update(y as f64);
            }});
        Ok(())
    }

    // no assumptions are made other than the fact that the scores are +ve and lower is better
    pub fn choose(&mut self, internal_timestamp: usize, shingle_size: usize, random: f32){
        if self.current_model == self.stats.len() || self.interval.1 == internal_timestamp {
            let mut min = self.stats.len();
            let mut min_value = f32::MAX;
            for i in 0..self.stats.len() {
                if !self.stats[i].is_empty() && min_value > self.stats[i].mean() as f32 {
                    min = i;
                    min_value = self.stats[i].mean() as f32;
                    self.stats[i].reset();
                }
            }
            if self.current_model != self.stats.len() {
                if self.current_model != min {
                    self.switches += 1;
                } else {
                    self.affirmations += 1;
                }
            }
            self.current_model = min;
            if self.current_model != self.stats.len() {
                let gap = ((1.0 + random) * self.interval.1 as f32) as usize + 3 * shingle_size;
                self.interval = (self.interval.0 + gap, self.interval.1 + gap);
            }
        }
    }
    pub fn switches(&self) -> usize {
        self.switches
    }

    pub fn affirmations(&self) -> usize {
        self.affirmations
    }
}

impl State {
    pub fn new(id: u64, arms: usize, scoring_strategy: ScoringStrategy, predictor_corrector : PredictorCorrector, preprocessor: Preprocessor) -> Result<Self>{
        let random_seed = ChaCha20Rng::from_entropy().gen::<u64>();
        let last_descriptor = Descriptor::new(id,&Vec::new(),0,preprocessor.forest_mode() == TIME_AUGMENTED,None);
        let base = preprocessor.start_normalization();
        let interval = (base, base + 3 * preprocessor.shingle_size());
        Ok(State{
            id,
            random_seed,
            scoring_strategy,
            bandit: Bandit::new(id,arms,interval)?,
            predictor_corrector,
            preprocessor,
            last_descriptor,
        })
    }

    pub fn random(&mut self) -> f32 {
    let mut rng = ChaCha20Rng::seed_from_u64( self.random_seed);
    self.random_seed = rng.gen::<u64>();
    rng.gen::<f32>()
    }
}

pub struct BasicTRCF {
    rcf : Box<dyn RCF + Send + Sync>,
    state: State
}

pub fn core_process<U :?Sized, Label:Sync + Copy, Attributes: Sync + Copy>(rcf: Option<&Box<U>>, state : &mut State, point: &[f32], timestamp: u64) -> Result<Descriptor>
    where U : AugmentedRCF<Label,Attributes> {
    let mut result = Descriptor::new(state.id,point,timestamp,state.preprocessor.forest_mode() == TIME_AUGMENTED,None);
    result.values_seen = state.preprocessor.values_seen();
    // the check for input length is done in exactly this place
    //along with Nan/finiteness
    match state.preprocessor.shingled_point(rcf, point, timestamp)? {
        Some(x) => {
            result.rcf_point = Some(x);
            result.shift = Some(state.preprocessor.shift());
            result.scale = Some(state.preprocessor.scale());
            result.difference_deviations = Some(state.preprocessor.difference_deviations());
            result.scoring_strategy = state.scoring_strategy;
            result.transform_method = state.preprocessor.transformation_method();
            if state.preprocessor.is_ready() {
                match rcf {
                    Some(y) => state.predictor_corrector.detect_and_modify(&mut result, &state.last_descriptor, state.preprocessor.shingle_size(), y)?,
                    _ =>{}
                }
            }
        },
        None => {}
    }
    state.preprocessor.post_process(&mut result, point, timestamp, &state.last_descriptor)?;
    if result.anomaly_grade > 0.0 {
        state.last_descriptor = result.clone();
    } else {
        state.last_descriptor.values_seen = result.values_seen;
        state.last_descriptor.current_timestamp = result.current_timestamp;
        state.last_descriptor.anomaly_grade = 0.0;
        state.last_descriptor.rcf_point = result.rcf_point.clone();
        state.last_descriptor.score = result.score;
        state.last_descriptor.threshold = result.threshold;
    }
    Ok(result)
}

impl BasicTRCF {

    pub fn process(&mut self, point: &[f32], timestamp: u64) -> Result<Descriptor>{
        let result = core_process(Some(&self.rcf), &mut self.state, point, timestamp)?;
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
        Ok(result)
    }

    pub fn process_sequentially(&mut self, input: &[(&[f32],u64)]) -> Result<Vec<Descriptor>> {
        let answer = input.into_iter().map(|(a, b)| self.process(*a, *b))
           .collect::<Result<Vec<Descriptor>>>()?.into_iter().filter(|x| x.anomaly_grade>0.0).collect();
        Ok(answer)
    }

    pub fn extrapolate(&self, look_ahead: usize) -> Result<(RangeVector<f32>,Option<RangeVector<f64>>)> {
        self.state.preprocessor.invert_extrapolation(self.rcf.extrapolate(look_ahead)?)
    }

}

pub struct TRCFOptions {
    pub(crate) transform_decay: Option<f64>,
    pub(crate) transform_method: TransformMethod,
    pub(crate) forest_mode : ForestMode,
    pub(crate) verbose: bool,
    pub(crate) weights: Option<Vec<f32>>,
    pub(crate) default_fill : Option<Vec<f32>>,
    pub(crate) start_normalization: usize,
    pub(crate) stop_normalization:usize,
    pub(crate) scoring_strategy: ScoringStrategy,
}

pub trait TRCFOptionsBuilder {
    fn get_trcf_options(&mut self) -> &mut TRCFOptions;
    fn transform_decay(&mut self, transform_decay: f64) -> &mut Self {
        self.get_trcf_options().transform_decay = Some(transform_decay);
        self
    }
    fn forest_mode(&mut self, forest_mode: ForestMode) -> &mut Self {
        self.get_trcf_options().forest_mode = forest_mode;
        self
    }
    fn transform_method(&mut self, transform_method: TransformMethod) -> &mut Self {
        self.get_trcf_options().transform_method = transform_method;
        self
    }
    fn start_normalization(&mut self, start_normalization: usize) -> &mut Self {
        self.get_trcf_options().start_normalization = start_normalization;
        self
    }
    fn stop_normalization(&mut self, stop_normalization: usize) -> &mut Self {
        self.get_trcf_options().stop_normalization = stop_normalization;
        self
    }
    fn weights(&mut self, weights: &[f32]) -> &mut Self {
        self.get_trcf_options().weights = Some(Vec::from(weights));
        self
    }
    fn default_fill(&mut self, default_fill: &[f32]) -> &mut Self {
        self.get_trcf_options().default_fill = Some(Vec::from(default_fill));
        self
    }
    fn verbose(&mut self, verbose: bool) -> &mut Self {
        self.get_trcf_options().verbose = verbose;
        self
    }
    fn scoring_strategy(&mut self, scoring_strategy: ScoringStrategy) -> &mut Self {
        self.get_trcf_options().scoring_strategy = scoring_strategy;
        self
    }
}

impl TRCFOptions{
    pub fn validate(&self, input_dimensions: usize) ->Result<()> {
        check_argument(self.transform_decay.unwrap_or(0.0) >= 0.0, "transform decay cannot be negative")?;
        // juct check -- the builder should not be modified in case it is reused
        check_argument(self.weights.as_ref().unwrap_or(&vec![1.0; input_dimensions]).len() == input_dimensions,
                       " incorrect length of weight vector")?;
        check_argument(self.default_fill.as_ref().unwrap_or(&vec![0.0; input_dimensions]).len() == input_dimensions,
                       " incorrect length of default_fill vector")?;
        check_argument(self.start_normalization <= self.stop_normalization, "normalization cannot start cannot be after stopping")?;
        Ok(())
    }
}

impl Default for TRCFOptions {
    fn default() -> Self {
       TRCFOptions {
           transform_decay: None,
           transform_method: NORMALIZE,
           forest_mode: STANDARD,
           verbose: false,
           weights: None,
           default_fill: None,
           start_normalization: 10,
           stop_normalization: usize::MAX,
           scoring_strategy: EXPECTED_INVERSE_HEIGHT
       }
    }
}

pub struct BasicTRCFBuilder {
    input_dimensions: usize,
    shingle_size: usize,
    trcf_options: TRCFOptions,
    rcf_options : RCFOptions<u64,u64>
}

impl BasicTRCFBuilder {
    pub fn new(input_dimensions: usize, shingle_size: usize) -> Self {
        BasicTRCFBuilder {
            input_dimensions,
            shingle_size,
            rcf_options: RCFOptions::default(),
            trcf_options: TRCFOptions::default()
        }
    }

    pub fn build(&self) -> Result<BasicTRCF> {
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
            _ => vec![1.0; self.input_dimensions],
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

        Ok(BasicTRCF {
            rcf,
            state: State::new(0, 1, self.trcf_options.scoring_strategy,predictor_corrector, preprocessor)?
        })
    }
}

impl RCFOptionsBuilder<u64, u64> for BasicTRCFBuilder {
    fn get_rcf_options(&mut self) -> &mut RCFOptions<u64, u64> {
        &mut self.rcf_options
    }
}

impl TRCFOptionsBuilder for BasicTRCFBuilder {
    fn get_trcf_options(&mut self) -> &mut TRCFOptions {
        &mut self.trcf_options
    }
}