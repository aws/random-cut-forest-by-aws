use crate::pointstore::PointStore;
use crate::pointstore::PointStoreView;
use crate::pointstore::PointStoreEdit;
use crate::samplerplustree::SamplerPlusTree;
use rayon::prelude::*;
extern crate num;
use num::Integer;
extern crate rand;
use rand::SeedableRng;
extern crate rand_chacha;
use core::fmt::Debug;
use rand_chacha::ChaCha20Rng;
use crate::rcf::rand::RngCore;
use crate::rcf::rand::Rng;
use crate::sampler::Sampler;


pub trait Max {
	const MAX: Self;
}

impl Max for u8 {
	const MAX: u8 = u8::MAX;
}


impl Max for u16 {
	const MAX: u16 = u16::MAX;
}

impl Max for usize {
	const MAX: usize = usize::MAX;
}

pub trait RCF {
	fn update(&mut self, point: Vec<f32>, timestamp: u64);
	fn score(&self, point: &[f32]) -> f64;
	fn dynamic_score(&self, point: &[f32],ignore_mass: usize, score_seen: fn (usize,usize) -> f64,
					 score_unseen :fn (usize,usize) -> f64, damp : fn (usize,usize) -> f64,
					 normalizer: fn (f64,usize) -> f64 ) -> f64;
	fn get_size(&self) -> usize;
	fn get_entries_seen(&self) -> u64;
	fn get_point_store_size(&self) -> usize;
	// to be extended to match Java version
}

pub type RCFSmall = RCFStruct<u8,u16,u16,u8>;
pub type RCFMedium = RCFStruct<u16,usize,usize,u16>;
pub type RCFLarge = RCFStruct<usize,usize,usize,usize>;

impl RCF for RCFSmall{
	fn update(&mut self, point: Vec<f32>, timestamp: u64) {
		self.update(point,timestamp);
	}

	fn score(&self, point: &[f32]) -> f64 {
		self.score(point)
	}

	fn dynamic_score(&self, point: &[f32],ignore_mass: usize, score_seen: fn (usize,usize) -> f64,
					 score_unseen :fn (usize,usize) -> f64, damp : fn (usize,usize) -> f64,
					 normalizer: fn (f64,usize) -> f64 ) -> f64 {
		self.dynamic_score(point,ignore_mass,score_seen,score_unseen,damp,normalizer)
	}

	fn get_size(&self) -> usize {
		self.get_size()
	}

	fn get_entries_seen(&self) -> u64 {
		self.entries_seen
	}

	fn get_point_store_size(&self) -> usize {
		self.point_store.get_size()
	}
}

impl RCF for RCFMedium{
	fn update(&mut self, point: Vec<f32>, timestamp: u64) {
		self.update(point,timestamp);
	}

	fn score(&self, point: &[f32]) -> f64 {
		self.score(point)
	}

	fn dynamic_score(&self, point: &[f32],ignore_mass: usize, score_seen: fn (usize,usize) -> f64,
					 score_unseen :fn (usize,usize) -> f64, damp : fn (usize,usize) -> f64,
					 normalizer: fn (f64,usize) -> f64 ) -> f64 {
		self.dynamic_score(point,ignore_mass,score_seen,score_unseen,damp,normalizer)
	}

	fn get_size(&self) -> usize {
		self.get_size()
	}

	fn get_entries_seen(&self) -> u64 {
		self.entries_seen
	}

	fn get_point_store_size(&self) -> usize {
		self.point_store.get_size()
	}
}

impl RCF for RCFLarge {
	fn update(&mut self, point: Vec<f32>, timestamp: u64) {
		self.update(point,timestamp);
	}

	fn score(&self, point: &[f32]) -> f64 {
		self.score(point)
	}

	fn dynamic_score(&self, point: &[f32],ignore_mass: usize, score_seen: fn (usize,usize) -> f64,
					 score_unseen :fn (usize,usize) -> f64, damp : fn (usize,usize) -> f64,
					 normalizer: fn (f64,usize) -> f64 ) -> f64 {
		self.dynamic_score(point,ignore_mass,score_seen,score_unseen,damp,normalizer)
	}

	fn get_size(&self) -> usize {
		self.get_size()
	}

	fn get_entries_seen(&self) -> u64 {
		self.entries_seen
	}

	fn get_point_store_size(&self) -> usize {
		self.point_store.get_size()
	}
}

pub(crate) fn score_seen(x: usize, y : usize) -> f64 { 1.0/(x as f64 + f64::log2(1.0 + y as f64)) }
pub(crate) fn score_unseen(x: usize, y : usize) -> f64 { 1.0/(x as f64 + 1.0) }
pub(crate) fn normalizer(x: f64, y: usize) -> f64 { x * f64::log2(1.0 + y as f64)}
pub(crate) fn damp(x: usize, y:usize) -> f64 { 1.0 - (x as f64)/(2.0 * y as f64) }


#[repr(C)]
pub struct RCFStruct<C,L,P,N> {
	dimensions: usize,
	capacity: usize,
	number_of_trees: usize,
	sampler_plus_trees: Vec<SamplerPlusTree<C,P,N>>,
	time_decay: f64,
	shingle_size: usize,
	entries_seen: u64,
	internal_shingling: bool,
	store_attributes: bool,
	initial_accept_fraction: f64,
	bounding_box_cache_fraction: f64,
	random_seed: u64,
	point_store: PointStore<L>,
}


impl<C: Max + Copy, L: Max + Copy + std::cmp::PartialEq, P: Max + Copy + std::cmp::PartialEq, N: Max + Copy>  RCFStruct<C,L,P,N> where
	C: std::convert::TryFrom<usize>, usize: From<C>,
	L: std::convert::TryFrom<usize>, usize: From<L>,
	P: std::convert::TryFrom<usize>, usize: From<P>,
	N: std::convert::TryFrom<usize>, usize: From<N> {
	pub fn new(dimensions: usize, shingle_size: usize,
			   capacity: usize, number_of_trees: usize, random_seed: u64, store_attributes: bool, internal_shingling: bool, time_decay: f64, initial_accept_fraction: f64, bounding_box_cache_fraction: f64)
			   -> Self where
		<C as TryFrom<usize>>::Error: Debug, <L as TryFrom<usize>>::Error: Debug, <P as TryFrom<usize>>::Error: Debug, <N as TryFrom<usize>>::Error: Debug {
		let mut point_store_capacity: usize = (capacity * number_of_trees + 1).try_into().unwrap();
		if point_store_capacity < 2 * capacity {
			point_store_capacity = 2 * capacity;
		}
		let mut initial_capacity = 2 * capacity;
		if shingle_size != 1 && dimensions % shingle_size != 0 {
			println!("Shingle size must divide dimensions.");
			panic!();
		}

		let mut rng = ChaCha20Rng::seed_from_u64(random_seed);
		let new_random_seed = rng.next_u64();
		let mut models: Vec<SamplerPlusTree<C, P, N>> = Vec::new();
		let mut total_fraction = bounding_box_cache_fraction * number_of_trees as f64;
		let mut convexify = bounding_box_cache_fraction > 0.2 && bounding_box_cache_fraction < 0.7;
		for i in 0..number_of_trees {
			let mut amount = bounding_box_cache_fraction;
			if convexify {
				let mut excess = total_fraction - 0.2 * (number_of_trees - i) as f64;
				if excess > 0.5 {
					amount = 0.7;
				} else {
					amount = 0.2 + excess;
				}
			}
			total_fraction -= amount;
			models.push(SamplerPlusTree::<C, P, N>::new(
				dimensions,
				capacity,
				rng.next_u64(),
				store_attributes,
				time_decay,
				initial_accept_fraction, amount));
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
			point_store: PointStore::<L>::new(dimensions.into(), shingle_size.into(), point_store_capacity, initial_capacity, internal_shingling),
			internal_shingling
		}
	}

	pub fn update(&mut self, point: Vec<f32>, timestamp: u64) where
		<C as TryFrom<usize>>::Error: Debug, <L as TryFrom<usize>>::Error: Debug, <P as TryFrom<usize>>::Error: Debug, <N as TryFrom<usize>>::Error: Debug {
		if (point.len() != self.dimensions.into() && !self.internal_shingling)
			|| (point.len() != (self.dimensions / self.shingle_size).into() && self.internal_shingling) {
			println!("The point must be '{}' or  floats long", self.dimensions);
			panic!();
		}


		let point_index = self.point_store.add(&point);
		let result: Vec<(usize, usize)> =  self.sampler_plus_trees.iter_mut().map(|m|
			m.update(point_index, usize::MAX, &self.point_store)).collect();

		for (a, b) in result {
			if a != usize::MAX {
				self.point_store.inc(a);
				if b != usize::MAX {
					self.point_store.dec(b);
				}
			}
		}


		self.point_store.dec(point_index);
		self.entries_seen += 1;
	}

	pub fn get_entries_seen(&self) -> u64 {
		self.entries_seen
	}

	pub fn get_point_store_size(&self) -> usize  {
		self.point_store.get_size()
	}

	pub fn score(&self, point: &[f32]) -> f64 where
		<C as TryFrom<usize>>::Error: Debug, <P as TryFrom<usize>>::Error: Debug, <N as TryFrom<usize>>::Error: Debug {
       self.dynamic_score(point,0,score_seen,score_unseen,damp,normalizer)
	}

	pub fn dynamic_score(&self, point: &[f32], ignore_mass: usize, score_seen: fn (usize,usize) -> f64,
						 score_unseen :fn (usize,usize) -> f64, damp : fn (usize,usize) -> f64,
						 normalizer: fn (f64,usize) -> f64 ) -> f64 where
		<C as TryFrom<usize>>::Error: Debug, <P as TryFrom<usize>>::Error: Debug, <N as TryFrom<usize>>::Error: Debug  {
		let mut sum = 0.0;
		let new_point = self.point_store.get_shingled_point(point);
		sum = self.sampler_plus_trees.iter()
			.map(|m| m.dynamic_score(&new_point, &self.point_store,ignore_mass,score_seen,score_unseen,damp,normalizer)).sum();

		sum / (self.sampler_plus_trees.len() as f64)
	}

	pub fn gereric_dynamic_score(&self, point: &[f32], ignore_mass: usize, score_seen: fn (usize,usize) -> f64,
								 score_unseen :fn (usize,usize) -> f64, damp : fn (usize,usize) -> f64,
								 normalizer: fn (f64,usize) -> f64 ) -> f64 where
		<C as TryFrom<usize>>::Error: Debug, <P as TryFrom<usize>>::Error: Debug, <N as TryFrom<usize>>::Error: Debug
	{
		let mut sum = 0.0;
		let new_point = self.point_store.get_shingled_point(point);
		sum = self.sampler_plus_trees.iter()
			.map(|m| m.generic_dynamic_score(&new_point, &self.point_store,ignore_mass,score_seen,score_unseen,damp,normalizer)).sum();

		sum / (self.sampler_plus_trees.len() as f64)
	}



	pub fn get_size(&self) -> usize {
		let mut sum: usize = 0;
		for model in &self.sampler_plus_trees {
			sum += model.get_size();
		}
		sum + self.point_store.get_size() +std::mem::size_of::<RCFStruct<C,L,P,N>>()
	}
}

