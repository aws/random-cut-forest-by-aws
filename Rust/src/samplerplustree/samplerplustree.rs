extern crate rand;
extern crate rand_chacha;

use std::fmt::Debug;
use std::hash::Hash;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rand_core::RngCore;

use crate::{
    pointstore::PointStore,
    samplerplustree::{
        nodestore::VectorNodeStore,
        nodeview::UpdatableNodeView,
        randomcuttree::{RCFTree, Traversable},
        sampler::Sampler,
    },
    types::{Location,Result},
    visitor::visitor::{Visitor, VisitorInfo},
};
use crate::samplerplustree::nodestore::NodeStore;

#[repr(C)]
pub struct SamplerPlusTree<C, P, N>
where
    C: Location,
    usize: From<C>,
    P: Location + Eq + Hash,
    usize: From<P>,
    N: Location,
    usize: From<N>
{
    tree: RCFTree<C, P, N>,
    sampler: Sampler<P>,
    using_transforms: bool,
    time_decay: f64,
    entries_seen: usize,
    initial_accept_fraction: f64,
    random_seed: u64,
}

impl<C, P, N> SamplerPlusTree<C, P, N>
where
    C: Location,
    usize: From<C>,
    P: Location + Eq + Hash,
    usize: From<P>,
    N: Location,
    usize: From<N>,
    <C as TryFrom<usize>>::Error: Debug,
    <P as TryFrom<usize>>::Error: Debug,
    <N as TryFrom<usize>>::Error: Debug,
{
    pub fn new(
        dimensions: usize,
        capacity: usize,
        using_transforms: bool,
        random_seed: u64,
        store_attributes: bool,
        store_pointsum: bool,
        propagate_attribute_vectors: bool,
        time_decay: f64,
        initial_accept_fraction: f64,
        bounding_box_cache_fraction: f64,
    ) -> Result<Self> {
        let mut rng = ChaCha20Rng::seed_from_u64(random_seed);
        let self_seed = rng.next_u64();

        Ok(SamplerPlusTree {
            time_decay,
            initial_accept_fraction,
            using_transforms,
            tree: RCFTree::<C, P, N>::new(
                dimensions,
                capacity,
                using_transforms,
                store_attributes,
                store_pointsum,
                propagate_attribute_vectors,
                bounding_box_cache_fraction,
                rng.next_u64(),
            )?,
            sampler: Sampler::new(capacity, store_attributes)?,
            entries_seen: 0,
            random_seed: self_seed,
        })
    }

    pub fn update<Label : Copy + Sync ,Attributes: Copy + Sync+ Hash + Eq + Send, PS: PointStore<Label,Attributes>>(
        &mut self,
        point_index: usize,
        point_attribute: usize,
        point_store: &PS
    ) -> Result<((usize, usize),(usize,usize))> {
        if point_index != usize::MAX {
            let mut initial = false;
            let mut rng = ChaCha20Rng::seed_from_u64(self.random_seed);
            self.random_seed = rng.next_u64();
            let random_number: f64 = rng.gen();
            let weight: f64 =
                f64::ln(-f64::ln(random_number)) - ((self.entries_seen as f64) * self.time_decay);
            if !self.sampler.is_full() {
                let other_random: f64 = rng.gen();
                let fill_fraction: f64 = self.sampler.get_fill_fraction();
                initial = other_random < self.initial_accept_probability(fill_fraction);
            }
            let accept_state =
                self.sampler
                    .accept_point(initial, weight as f32, point_index, point_attribute)?;

            self.entries_seen += 1;
            if accept_state.eviction_occurred {
                let delete_ref = if accept_state.point_index != usize::MAX {
                    (self.tree.delete(
                        accept_state.point_index,
                        accept_state.point_attribute,
                        point_store,
                    )?,accept_state.point_attribute)
                } else {
                    (usize::MAX,usize::MAX)
                };

                // the tree may choose to return a reference to an existing point
                // whose value is equal to `point`
                let added_ref = self.tree.add(point_index, point_attribute, point_store)?;

                self.sampler.add_point(added_ref)?;
                return Ok(((added_ref,point_attribute), delete_ref));
            }
        }
        Ok(((usize::MAX, usize::MAX),(usize::MAX,usize::MAX)))
    }

    fn initial_accept_probability(&self, fill_fraction: f64) -> f64 {
        return if fill_fraction < self.initial_accept_fraction {
            1.0
        } else if self.initial_accept_fraction >= 1.0 {
            0.0
        } else {
            1.0 - (fill_fraction - self.initial_accept_fraction)
                / (1.0 - self.initial_accept_fraction)
        };
    }

    pub fn simple_traversal<NodeView, V, R, PS,Label,Attributes>(
        &self,
        point: &[f32],
        point_store: &PS,
        parameters: &[usize],
        visitor_info: &VisitorInfo,
        visitor_factory: fn(usize, &[usize], &VisitorInfo) -> V,
        default: &R,
    ) -> Result<R>
    where
        NodeView: UpdatableNodeView<Label,Attributes>,
        V: Visitor<NodeView, R>,
        Label : Sync + Copy,
        Attributes: Copy + Sync+ Hash + Eq + Send,
        PS: PointStore<Label,Attributes>,
        R: Clone,
    {
        self.tree.traverse(
            point,
            parameters,
            visitor_factory,
            visitor_info,
            point_store,
            default,
        )
    }

    pub fn conditional_field<Label : Copy + Sync ,Attributes: Copy + Sync+ Hash + Eq + Send, PS>(
        &self,
        positions: &[usize],
        centrality: f64,
        point: &[f32],
        point_store: &PS,
        visitor_info: &VisitorInfo,
    ) -> Result<(f64, usize, f64)>
    where
        PS: PointStore<Label,Attributes>
    {
        self.tree.conditional_field(
            positions,
            point,
            point_store,
            centrality,
            self.random_seed,
            visitor_info,
        )
    }

    pub fn get_size(&self) -> usize {
        self.tree.get_size()
            + self.sampler.get_size()
            + std::mem::size_of::<SamplerPlusTree<C, P, N>>()
    }
}
