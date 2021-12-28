use crate::pointstore::PointStoreView;
use crate::randomcuttree::RCFTree;
use crate::sampler::Sampler;
 use std::collections::HashSet;
use std::fmt::Debug;
extern crate rand;
use rand::SeedableRng;
extern crate rand_chacha;
use crate::rcf::damp;
use rand_chacha::ChaCha20Rng;
use crate::samplerplustree::rand::RngCore;
use crate::samplerplustree::rand::Rng;
use crate::rcf::Max;

#[repr(C)]
pub struct SamplerPlusTree<C,P,N> {
tree : RCFTree<C,P,N>,
sampler : Sampler<P>,
using_transforms : bool,
time_decay : f64,
entries_seen : usize,
initial_accept_fraction : f64,
random_seed : u64
}

impl<C: Max + Copy, P: Max + Copy + std::cmp::PartialEq, N: Max + Copy> SamplerPlusTree <C,P,N> where
    C: std::convert::TryFrom<usize>, usize: From<C>,
    P: std::convert::TryFrom<usize>, usize: From<P>,
    N: std::convert::TryFrom<usize>, usize: From<N> {
    pub fn new(dimensions: usize,
               capacity: usize, using_transforms: bool, random_seed: u64, store_attributes: bool, time_decay: f64, initial_accept_fraction: f64, bounding_box_cache_fraction: f64) -> Self
    where
    <C as TryFrom<usize>>::Error: Debug, <P as TryFrom<usize>>::Error: Debug, <N as TryFrom<usize>>::Error: Debug {
        let mut rng = ChaCha20Rng::seed_from_u64(random_seed);
        let self_seed = rng.next_u64();

        SamplerPlusTree {
            time_decay,
            initial_accept_fraction,
            using_transforms,
            tree: RCFTree::<C,P,N>::new(dimensions, capacity, using_transforms, bounding_box_cache_fraction, rng.next_u64()),
            sampler: Sampler::new(capacity,  store_attributes),
            entries_seen: 0,
            random_seed: self_seed
        }
    }

    pub fn update(&mut self, point_index: usize, point_attribute: usize, point_store: &dyn PointStoreView) -> (usize, usize)
        where
            <C as TryFrom<usize>>::Error: Debug, <P as TryFrom<usize>>::Error: Debug, <N as TryFrom<usize>>::Error: Debug{
        if (point_index != usize::MAX) {
            let mut initial = false;
            let mut rng = ChaCha20Rng::seed_from_u64(self.random_seed);
            self.random_seed = rng.next_u64();
            let random_number: f64 = rng.gen();
            let weight: f64 = f64::ln(-f64::ln(random_number)) - ((self.entries_seen as f64) * self.time_decay);
            if !self.sampler.is_full() {
                let other_random: f64 = rng.gen();
                let fill_fraction: f64 = self.sampler.get_fill_fraction();
                initial = other_random < self.initial_accept_probability(fill_fraction);
            }
            let accept_state = self.sampler.accept_point(initial, weight as f32, point_index, point_attribute);

            self.entries_seen += 1;
            if accept_state.0 {
                let delete_ref = if accept_state.1 != usize::MAX {
                    self.tree.delete(accept_state.1, accept_state.3, point_store)
                } else {usize::MAX};

                // the tree may choose to return a reference to an existing point
                // whose value is equal to `point`
                let added_ref = self.tree.add(point_index, point_attribute, point_store);

                self.sampler.add_point(added_ref);
                return (added_ref, delete_ref);
            }
        }
        (usize::MAX, usize::MAX)
    }

    fn initial_accept_probability(&self, fill_fraction: f64) -> f64 {
        if fill_fraction < self.initial_accept_fraction {
            return 1.0;
        } else if self.initial_accept_fraction >= 1.0 {
            return 0.0;
        } else {
            return 1.0 - (fill_fraction - self.initial_accept_fraction) / (1.0 - self.initial_accept_fraction);
        }
    }

    pub fn dynamic_score(&self, point: &[f32], point_store: &dyn PointStoreView,ignore_mass: usize,score_seen: fn (usize,usize) -> f64,
                         score_unseen :fn (usize,usize) -> f64, damp : fn (usize,usize) -> f64,
                         normalizer: fn (f64,usize) -> f64 ) -> f64 where
    <C as TryFrom<usize>>::Error: Debug, <P as TryFrom<usize>>::Error: Debug, <N as TryFrom<usize>>::Error: Debug
    {

        self.tree.dynamic_score(point, point_store, ignore_mass, score_seen,score_unseen,damp,normalizer)
    }

    pub fn generic_dynamic_score(&self, point: &[f32], point_store: &dyn PointStoreView,ignore_mass: usize,score_seen: fn (usize,usize) -> f64,
                                 score_unseen :fn (usize,usize) -> f64, damp : fn (usize,usize) -> f64,
                                 normalizer: fn (f64,usize) -> f64 ) -> f64 where
        <C as TryFrom<usize>>::Error: Debug, <P as TryFrom<usize>>::Error: Debug, <N as TryFrom<usize>>::Error: Debug
    {
        self.tree.generic_dynamic_score(point, point_store,ignore_mass,score_seen, score_unseen, damp, normalizer)
    }


    pub fn get_size(&self) -> usize {
        self.tree.get_size() +
            self.sampler.get_size() + std::mem::size_of::<SamplerPlusTree<C,P,N>>()
    }

}

