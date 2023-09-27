use std::collections::HashMap;
use rand_chacha::ChaCha20Rng;
use rand_core::{RngCore, SeedableRng};
use rayon::prelude::*;
use crate::common::descriptor::Descriptor;
use crate::rcf::{AugmentedRCF, RCF, RCFBuilder, RCFOptions};
use crate::types::{Result};
use crate::trcf::predictorcorrector::PredictorCorrector;
use crate::trcf::preprocessor::{PreprocessorBuilder};
use crate::trcf::types::ForestMode::{STANDARD};
use crate::trcf::types::{ScoringStrategy, TransformMethod};
use crate::util::{check_argument, maxf32, minf32};
use crate::rcf::RCFOptionsBuilder;
use crate::trcf::basictrcf::{core_process, State, TRCFOptions, TRCFOptionsBuilder};
use crate::trcf::types::ScoringStrategy::EXPECTED_INVERSE_HEIGHT;


pub struct MultiTRCF {
    arms : usize,
    rcfs : Vec<Box<dyn AugmentedRCF<(u64,u64),u64> + Send + Sync>>,
    states: HashMap<u64,State>,
    input_dimensions: usize,
    shingle_size: usize,
    transform_decay: f64,
    transform_method: TransformMethod,
    scoring_strategy: ScoringStrategy,
    random_seed: u64,
    probability: f32,
    parallel_enabled: bool,
    selector : fn(&Descriptor) -> bool
}

fn is_anomaly(a:&Descriptor) -> bool {
   a.anomaly_grade>0.0
}

impl MultiTRCF {

    fn select_for_update(&self, _a: u64, _b: usize, _score: f32, probability : f32) -> bool {
        probability < self.probability
    }

    fn core_multi(&self, state:& mut State, point: &[f32],timestamp : u64)
        -> Result<((usize,(u64,u64),Option<Vec<f32>>),Option<Descriptor>)> {
        let internal_timestamp = state.preprocessor.internal_timestamp();
        if self.arms > 1 && state.bandit.is_evaluating(self.arms, internal_timestamp) {
            let shingled_point = state.preprocessor.shingled_point(None as Option<&Box<dyn RCF>>, &point, timestamp)?;
            match shingled_point.as_ref() {
                Some(x) => {
                    let mut scores = vec![-1.0f32; self.arms];

                    for i in 0..self.arms {
                        if self.rcfs[i].is_output_ready() {
                            scores[i] = self.rcfs[i].score(x).unwrap() as f32;
                        }
                    }
                    state.bandit.update(&scores)?;
                    let random = state.random();
                    state.bandit.choose(internal_timestamp, state.preprocessor.shingle_size(), random);
                },
                _ => {}
            }
        }
        let rcf = if state.bandit.current_model() < self.arms {
            Some(&self.rcfs[state.bandit.current_model()])
        } else {
            None
        };
        let t = core_process(rcf, state, point, timestamp).unwrap();
        let probability = state.random();
        let point = if state.preprocessor.is_ready()
            && self.select_for_update(state.id, state.preprocessor.internal_timestamp(), t.score,probability) {
            t.rcf_point.clone()
        } else {
            None
        };
        if (self.selector)(&t) {
            Ok(((state.bandit.current_model(),(state.id,timestamp), point) , Some(t)))
        } else {
            Ok(((state.bandit.current_model(),(state.id,timestamp), point), None))
        }
    }

    pub fn process(&mut self, input: HashMap<u64,(&[f32],u64)>) -> Result<Vec<Descriptor>>{
        let mut join : Vec<(u64,(State,&[f32],u64))> = input.into_iter().map(|e| {
             if !self.states.contains_key(&e.0) {
                let preprocessor = PreprocessorBuilder::new(self.input_dimensions, self.shingle_size)
                    .transform_method(self.transform_method).transform_decay(self.transform_decay)
                    .start_normalization(3 * self.shingle_size).build().unwrap();
                let predictor_corrector = PredictorCorrector::new(self.transform_decay, true, self.input_dimensions)?;
                let state = State::new(e.0,self.arms,self.scoring_strategy,predictor_corrector,preprocessor)?;
                 Ok((e.0,(state,e.1.0,e.1.1)))
            } else {
                Ok((e.0,(self.states.remove(&e.0).unwrap(),e.1.0,e.1.1)))
            }}).collect::<Result<Vec<(u64,(State,&[f32],u64))>>>()?;

        let collection = if self.parallel_enabled {
            join.par_iter_mut().map(|(_x,(state,point,timestamp))| {
                self.core_multi(state,*point,*timestamp)
            }).collect::<Result<Vec<((usize,(u64,u64),Option<Vec<f32>>),Option<Descriptor>)>>>()?
        } else {
            join.iter_mut().map(|(_x,(state,point,timestamp))|  {
                self.core_multi(state,*point,*timestamp)
            }).collect::<Result<Vec<((usize,(u64,u64),Option<Vec<f32>>),Option<Descriptor>)>>>()?
        };

        join.into_iter().for_each(|(id,(state,_,_))| { self.states.insert(id,state);});

        let (updates,proto) : (Vec<(usize,(u64,u64),Option<Vec<f32>>)>,Vec<_>) = collection.into_iter().unzip();

        updates.into_iter().map(|y|  -> Result<()> {
            match y.2.as_ref() {
                Some(x) => {
                    // if there is only one arm then the following will be equivalent -- first branch would
                    // be taken since y.0 == 1 in such a case
                    if y.0 != self.arms {
                        self.rcfs[y.0].update(x, y.1)
                    } else {
                        self.rcfs.iter_mut().map(|z| {
                            z.update(x, y.1)
                        }).collect()
                    }
                },
                None => { Ok(())}
            }}).collect::<Result<Vec<()>>>()?;

        let answer: Vec<Descriptor> = proto.into_iter().filter_map(|x| x).collect();

        Ok(answer)
    }

    pub fn switches(&self) -> usize {
        self.states.iter().map(|x| x.1.bandit.switches()).sum()
    }

    pub fn affirmations(&self) -> usize {
        self.states.iter().map(|x| x.1.bandit.affirmations()).sum()
    }

    pub fn states(&self) -> Vec<State> {
        self.states.iter().map(|x| x.1.clone()).collect::<Vec<State>>()
    }

    pub fn updates(&self) -> Vec<(usize,u64)>{
        self.rcfs.iter().zip(0..self.rcfs.len()).map(|x| (x.1,x.0.entries_seen())).collect()
    }
}

pub struct MultiTRCFBuilder {
    input_dimensions: usize,
    shingle_size: usize,
    arms : usize,
    probability : f32,
    scoring_strategy:ScoringStrategy,
    rcf_options: RCFOptions<(u64,u64),u64>,
    trcf_options: TRCFOptions
}

impl Default for MultiTRCFBuilder {
    // parallel_enabled will apply to MultiTRCF instead of RCF
    fn default() -> Self {
        MultiTRCFBuilder {
            input_dimensions: 1,
            shingle_size: 10,
            arms: 1,
            probability: 0.01,
            scoring_strategy: EXPECTED_INVERSE_HEIGHT,
            rcf_options: RCFOptions { parallel_enabled: true, internal_shingling: false, ..Default::default()},
            trcf_options: Default::default()
        }
    }
}

//just picking time stamp
fn attribute_creator(_a:&[(u64,u64)],label: (u64,u64)) -> Result<u64> {
    Ok(label.1)
}

impl MultiTRCFBuilder {
    pub fn new(input_dimensions:usize, shingle_size: usize, number_of_models: usize, approx_cardinality: usize) -> Self {
        MultiTRCFBuilder{
            input_dimensions,
            shingle_size,
            arms: number_of_models,
            probability : maxf32(0.01,1.0/maxf32(number_of_models as f32, (1+approx_cardinality) as f32)),
            ..Default::default()
        }
    }

    pub fn probability(&mut self, probability:f32) -> &mut Self {
        self.probability = probability;
        self
    }

    pub fn build(&self) -> Result<MultiTRCF>{
        check_argument(self.arms >0," cannot be zero")?;
        check_argument(!self.rcf_options.internal_shingling, "internal shingling is not feasible")?;
        check_argument( self.rcf_options.bounding_box_cache_fraction == 1.0, " bounding box fraction should be 1")?;
        check_argument(self.trcf_options.forest_mode==STANDARD, "forest mode not supported")?;
        let mut random_seed = self.rcf_options.random_seed.unwrap_or(ChaCha20Rng::from_entropy().next_u64());
        let time_decay = self.rcf_options.time_decay.unwrap_or(0.1/self.rcf_options.capacity as f64);
        let output_after = self.rcf_options.output_after.unwrap_or(1 + self.rcf_options.capacity / 4);
        let transform_decay = self.trcf_options.transform_decay.unwrap_or(0.1/self.rcf_options.capacity as f64);
        let mut rcfs= Vec::new();
        for _i in 0..self.arms {
            rcfs.push(RCFBuilder::<(u64,u64),u64>::new(self.input_dimensions, self.shingle_size)
                .tree_capacity(self.rcf_options.capacity)
                .number_of_trees(self.rcf_options.number_of_trees)
                .time_decay(time_decay)
                .store_attributes(self.rcf_options.store_attributes)
                .output_after(output_after)
                .initial_accept_fraction(self.rcf_options.initial_accept_fraction)
                .internal_shingling(false).random_seed(random_seed)
                .build_to_u64(attribute_creator).unwrap());
            random_seed += 1;
        }
        Ok(MultiTRCF {
            arms: self.arms,
            parallel_enabled : self.rcf_options.parallel_enabled,
            rcfs,
            states: Default::default(),
            input_dimensions : self.input_dimensions,
            shingle_size : self.shingle_size,
            transform_decay,
            transform_method: self.trcf_options.transform_method,
            scoring_strategy: self.trcf_options.scoring_strategy,
            random_seed,
            probability: self.probability,
            selector: is_anomaly
        })
    }
}

impl RCFOptionsBuilder<(u64,u64),u64> for MultiTRCFBuilder {
    fn get_rcf_options(&mut self) -> &mut RCFOptions<(u64,u64),u64> {
        &mut self.rcf_options
    }
}

impl TRCFOptionsBuilder for MultiTRCFBuilder {
    fn get_trcf_options(&mut self) -> &mut TRCFOptions {
        &mut self.trcf_options
    }
}