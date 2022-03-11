use rayon::prelude::*;

use crate::{
    pointstore::{PointStore, VectorizedPointStore},
    samplerplustree::SamplerPlusTree,
};
extern crate num;
extern crate rand;
use rand::SeedableRng;
extern crate rand_chacha;
use core::fmt::Debug;

use rand_chacha::ChaCha20Rng;

use crate::{
    rcf::rand::RngCore,
    types::{Location, Max},
};

pub(crate) fn score_seen(x: usize, y: usize) -> f64 {
    1.0 / (x as f64 + f64::log2(1.0 + y as f64))
}
pub(crate) fn score_unseen(x: usize, _y: usize) -> f64 {
    1.0 / (x as f64 + 1.0)
}
pub(crate) fn normalizer(x: f64, y: usize) -> f64 {
    x * f64::log2(1.0 + y as f64)
}
pub(crate) fn damp(x: usize, y: usize) -> f64 {
    1.0 - (x as f64) / (2.0 * y as f64)
}

fn project(point:&[f32],position : &[usize]) -> Vec<f32> {
    let len = position.len();
    let mut answer = vec![0.0f32;len];
    for i in 0..len {
        answer[i] = point[position[i]];
    }
    answer
}

pub trait RCF {
    fn validate_update(&self, point: &[f32]) {
        let expected = if !self.is_internal_shingling_enabled() {
            self.get_dimensions()
        } else {
            self.get_dimensions() / self.get_shingle_size()
        };
        if point.len() != expected {
            println!("The point must be '{}' floats", expected);
            panic!();
        }
    }
    fn update(&mut self, point: &[f32], timestamp: u64);

    fn get_dimensions(&self) -> usize;
    fn get_shingle_size(&self) -> usize;
    fn is_internal_shingling_enabled(&self) -> bool;
    fn get_entries_seen(&self) -> u64;

    fn score(&self, point: &[f32]) -> f64 {
        self.generic_score(point, 0, score_seen, score_unseen, damp, normalizer)
    }

    fn generic_score(
        &self,
        point: &[f32],
        ignore_mass: usize,
        score_seen: fn(usize, usize) -> f64,
        score_unseen: fn(usize, usize) -> f64,
        damp: fn(usize, usize) -> f64,
        normalizer: fn(f64, usize) -> f64,
    ) -> f64;


    fn impute_missing_values(&self, positions: &[usize], point: &[f32]) -> Vec<f32> {
        let mut list = self.conditional_field(positions,point,1.0);
        let mut answer = vec![0.0f32;positions.len()];
        for i in 0..positions.len() {
            list.sort_by(|a, b| a[i].partial_cmp(&b[i]).unwrap());
            answer[i] = list[list.len()/2][i];
        }
        answer
    }

    fn extrapolate(&self, look_ahead : usize) -> Vec<f32>;

    fn conditional_field(&self, positions: &[usize], point: &[f32], centrality: f64) -> Vec<Vec<f32>> {
        self.generic_conditional_field(
            positions,
            point,
            centrality,
            0,
            score_seen,
            score_unseen,
            damp,
            normalizer,
        )
    }
    fn generic_conditional_field(
        &self,
        positions: &[usize],
        point: &[f32],
        centrality: f64,
        ignore_mass: usize,
        score_seen: fn(usize, usize) -> f64,
        score_unseen: fn(usize, usize) -> f64,
        damp: fn(usize, usize) -> f64,
        normalizer: fn(f64, usize) -> f64,
    ) -> Vec<Vec<f32>>;
    fn get_size(&self) -> usize;
    fn get_point_store_size(&self) -> usize;

    // to be extended to match Java version
}

pub struct RCFStruct<C, L, P, N>
where
    C: Location,
    usize: From<C>,
    L: Location,
    usize: From<L>,
    P: Location,
    usize: From<P>,
    N: Location,
    usize: From<N>,
{
    dimensions: usize,
    capacity: usize,
    number_of_trees: usize,
    sampler_plus_trees: Vec<SamplerPlusTree<C, P, N>>,
    time_decay: f64,
    shingle_size: usize,
    entries_seen: u64,
    internal_shingling: bool,
    internal_rotation: bool,
    store_attributes: bool,
    initial_accept_fraction: f64,
    bounding_box_cache_fraction: f64,
    parallel_enabled: bool,
    random_seed: u64,
    point_store: VectorizedPointStore<L>,
}

pub type RCFTiny = RCFStruct<u8, u16, u16, u8>; // sampleSize <= 256 for these and shingleSize * { max { base_dimensions, (number_of_trees + 1) } <= 256
pub type RCFSmall = RCFStruct<u8, usize, u16, u8>; // sampleSize <= 256 and (number_of_trees + 1) <= 256 and dimensions = shingle_size*base_dimensions <= 256
pub type RCFMedium = RCFStruct<u16, usize, usize, u16>; // sampleSize, dimensions <= u16::MAX
pub type RCFLarge = RCFStruct<usize, usize, usize, usize>; // as large as the machine would allow

impl<C, L, P, N> RCFStruct<C, L, P, N>
where
    C: Location,
    usize: From<C>,
    L: Location,
    usize: From<L>,
    P: Location,
    usize: From<P>,
    N: Location,
    usize: From<N>,
    <C as TryFrom<usize>>::Error: Debug,
    <L as TryFrom<usize>>::Error: Debug,
    <P as TryFrom<usize>>::Error: Debug,
    <N as TryFrom<usize>>::Error: Debug,
{
    pub fn new(
        dimensions: usize,
        shingle_size: usize,
        capacity: usize,
        number_of_trees: usize,
        random_seed: u64,
        store_attributes: bool,
        parallel_enabled: bool,
        internal_shingling: bool,
        internal_rotation: bool,
        time_decay: f64,
        initial_accept_fraction: f64,
        bounding_box_cache_fraction: f64,
    ) -> Self {
        let mut point_store_capacity: usize = (capacity * number_of_trees + 1).try_into().unwrap();
        if point_store_capacity < 2 * capacity {
            point_store_capacity = 2 * capacity;
        }
        let initial_capacity = 2 * capacity;
        if shingle_size != 1 && dimensions % shingle_size != 0 {
            println!("Shingle size must divide dimensions.");
            panic!();
        }
        assert!(
            !internal_rotation || internal_shingling,
            " internal shingling required for rotations"
        );
        let mut rng = ChaCha20Rng::seed_from_u64(random_seed);
        let _new_random_seed = rng.next_u64();
        let mut models: Vec<SamplerPlusTree<C, P, N>> = Vec::new();
        let using_transforms = internal_rotation; // other conditions may be added eventually
        for _i in 0..number_of_trees {
            models.push(SamplerPlusTree::<C, P, N>::new(
                dimensions,
                capacity,
                using_transforms,
                rng.next_u64(),
                store_attributes,
                time_decay,
                initial_accept_fraction,
                bounding_box_cache_fraction,
            ));
        }
        RCFStruct {
            random_seed,
            dimensions,
            capacity,
            sampler_plus_trees: models,
            number_of_trees,
            store_attributes,
            shingle_size,
            entries_seen: 0,
            time_decay,
            initial_accept_fraction,
            bounding_box_cache_fraction,
            parallel_enabled,
            point_store: VectorizedPointStore::<L>::new(
                dimensions.into(),
                shingle_size.into(),
                point_store_capacity,
                initial_capacity,
                internal_shingling,
                internal_rotation,
            ),
            internal_shingling,
            internal_rotation,
        }
    }

    pub fn generic_conditional_field_point_list(
        &self,
        positions: &[usize],
        point: &[f32],
        centrality: f64,
        ignore_mass: usize,
        score_seen: fn(usize, usize) -> f64,
        score_unseen: fn(usize, usize) -> f64,
        damp: fn(usize, usize) -> f64,
        normalizer: fn(f64, usize) -> f64,
    ) -> Vec<usize> {
        let new_point = self.point_store.get_shingled_point(point);
        let mut list: Vec<usize> = if self.parallel_enabled {
            self.sampler_plus_trees
                .par_iter()
                .map(|m| {
                    m.conditional_field(
                        &positions,
                        centrality,
                        &new_point,
                        &self.point_store,
                        ignore_mass,
                        score_seen,
                        score_unseen,
                        damp,
                        normalizer,
                    )
                })
                .collect()
        } else {
            self.sampler_plus_trees
                .iter()
                .map(|m| {
                    m.conditional_field(
                        &positions,
                        centrality,
                        &new_point,
                        &self.point_store,
                        ignore_mass,
                        score_seen,
                        score_unseen,
                        damp,
                        normalizer,
                    )
                })
                .collect()
        };
        list.sort();
        list
    }
}

pub fn create_rcf(
    dimensions: usize,
    shingle_size: usize,
    capacity: usize,
    number_of_trees: usize,
    random_seed: u64,
    store_attributes: bool,
    parallel_enabled: bool,
    internal_shingling: bool,
    internal_rotation: bool,
    time_decay: f64,
    initial_accept_fraction: f64,
    bounding_box_cache_fraction: f64,
) -> Box<dyn RCF> {
    if (dimensions < u8::MAX as usize) && (capacity - 1 <= u8::MAX as usize) {
        if capacity * (1 + number_of_trees) * shingle_size <= u16::MAX as usize {
            println!(" choosing RCF_Tiny");
            Box::new(RCFTiny::new(
                dimensions,
                shingle_size,
                capacity,
                number_of_trees,
                random_seed,
                store_attributes,
                parallel_enabled,
                internal_shingling,
                internal_rotation,
                time_decay,
                initial_accept_fraction,
                bounding_box_cache_fraction,
            ))
        } else {
            println!(" choosing RCF_Small");
            Box::new(RCFSmall::new(
                dimensions,
                shingle_size,
                capacity,
                number_of_trees,
                random_seed,
                store_attributes,
                parallel_enabled,
                internal_shingling,
                internal_rotation,
                time_decay,
                initial_accept_fraction,
                bounding_box_cache_fraction,
            ))
        }
    } else if (dimensions < u16::MAX as usize) && (capacity - 1 <= u16::MAX as usize) {
        println!(" choosing medium");
        Box::new(RCFMedium::new(
            dimensions,
            shingle_size,
            capacity,
            number_of_trees,
            random_seed,
            store_attributes,
            parallel_enabled,
            internal_shingling,
            internal_rotation,
            time_decay,
            initial_accept_fraction,
            bounding_box_cache_fraction,
        ))
    } else {
        println!(" choosing large");
        Box::new(RCFLarge::new(
            dimensions,
            shingle_size,
            capacity,
            number_of_trees,
            random_seed,
            store_attributes,
            parallel_enabled,
            internal_shingling,
            internal_rotation,
            time_decay,
            initial_accept_fraction,
            bounding_box_cache_fraction,
        ))
    }
}

impl<C, L, P, N> RCF for RCFStruct<C, L, P, N>
where
    C: Location,
    usize: From<C>,
    L: Location,
    usize: From<L>,
    P: Location,
    usize: From<P>,
    N: Location,
    usize: From<N>,
    <C as TryFrom<usize>>::Error: Debug,
    <L as TryFrom<usize>>::Error: Debug,
    <P as TryFrom<usize>>::Error: Debug,
    <N as TryFrom<usize>>::Error: Debug,
{
    fn update(&mut self, point: &[f32], _timestamp: u64) {
        let point_index = self.point_store.add(&point);
        if point_index != usize::MAX {
            let result: Vec<(usize, usize)> = if self.parallel_enabled {
                self.sampler_plus_trees
                    .par_iter_mut()
                    .map(|m| m.update(point_index, usize::MAX, &self.point_store))
                    .collect()
            } else {
                self.sampler_plus_trees
                    .iter_mut()
                    .map(|m| m.update(point_index, usize::MAX, &self.point_store))
                    .collect()
            };
            self.point_store.adjust_count(&result);
            self.point_store.dec(point_index);
            self.entries_seen += 1;
        }
    }

    fn get_dimensions(&self) -> usize {
        self.dimensions
    }

    fn get_shingle_size(&self) -> usize {
        self.shingle_size
    }

    fn is_internal_shingling_enabled(&self) -> bool {
        self.internal_shingling
    }

    fn get_entries_seen(&self) -> u64 {
        self.entries_seen
    }

    fn generic_score(
        &self,
        point: &[f32],
        ignore_mass: usize,
        score_seen: fn(usize, usize) -> f64,
        score_unseen: fn(usize, usize) -> f64,
        damp: fn(usize, usize) -> f64,
        normalizer: fn(f64, usize) -> f64,
    ) -> f64 {
        assert!(point.len() == self.dimensions || point.len() * self.shingle_size == self.dimensions, "incorrect input length");
        let mut sum = 0.0;
        let new_point = self.point_store.get_shingled_point(point);
        sum = if self.parallel_enabled {
            self.sampler_plus_trees
                .par_iter()
                .map(|m| {
                    m.generic_score(
                        &new_point,
                        &self.point_store,
                        ignore_mass,
                        score_seen,
                        score_unseen,
                        damp,
                        normalizer,
                    )
                })
                .sum()
        } else {
            self.sampler_plus_trees
                .iter()
                .map(|m| {
                    m.generic_score(
                        &new_point,
                        &self.point_store,
                        ignore_mass,
                        score_seen,
                        score_unseen,
                        damp,
                        normalizer,
                    )
                })
                .sum()
        };
        sum / (self.sampler_plus_trees.len() as f64)
    }

    fn generic_conditional_field(
        &self,
        positions: &[usize],
        point: &[f32],
        centrality: f64,
        ignore_mass: usize,
        score_seen: fn(usize, usize) -> f64,
        score_unseen: fn(usize, usize) -> f64,
        damp: fn(usize, usize) -> f64,
        normalizer: fn(f64, usize) -> f64,
    ) -> Vec<Vec<f32>> {
        assert!(point.len() == self.dimensions || point.len() * self.shingle_size == self.dimensions, "incorrect input length");
        if point.len() == self.dimensions {
            self.generic_conditional_field_point_list(positions, point, centrality, ignore_mass, score_seen, score_unseen, damp, normalizer).iter().map( |&i|
                project(&self.point_store.get_copy(i),positions)).collect()
        } else {
            let new_positions = &self.point_store.get_missing_indices(0,positions);
            self.generic_conditional_field_point_list(new_positions, point, centrality, ignore_mass, score_seen, score_unseen, damp, normalizer).iter().map( |&i|
                project(&self.point_store.get_copy(i),new_positions)).collect()
        }
    }

    fn extrapolate(&self, look_ahead: usize) -> Vec<f32> {
        assert!(self.internal_shingling, " look ahead is not meaningful without internal shingling mechanism ");
        assert!(self.shingle_size > 1, " need shingle size > 1 for extrapolation");
        let mut answer = Vec:: new();
        let base = self.dimensions/self.shingle_size;
        let mut fictitious_point = self.point_store.get_shingled_point(&vec![0.0f32;base]);
        for i in 0..look_ahead {
            let missing = self.point_store.get_next_indices(i);
            assert!(missing.len() == base, "incorrect imputation");
            let values = self.impute_missing_values(&missing,&fictitious_point);
            for j in 0..base {
                answer.push(values[j]);
                fictitious_point[missing[j]]=values[j];
            }
        }
        answer
    }


    fn get_size(&self) -> usize {
        let mut sum: usize = 0;
        for model in &self.sampler_plus_trees {
            sum += model.get_size();
        }
        sum + self.point_store.get_size() + std::mem::size_of::<RCFStruct<C, L, P, N>>()
    }

    fn get_point_store_size(&self) -> usize {
        self.point_store.get_size()
    }
}
