
use std::collections::HashSet;
use std::fmt::Debug;
use crate::rcf::Max;
#[repr(C)]
pub struct Sampler <P>{
capacity: usize,
weights : Vec<f32>,
points : Vec<P>,
store_attributes: bool,
point_attributes : Vec<P>,
current_size : usize,
accepted_state : (f32, usize, usize)
}

impl<P: Max + Copy + std::cmp::PartialEq>  Sampler<P> where
    P: std::convert::TryFrom<usize>, usize: From<P> {
    pub fn new(capacity: usize,  store_attributes: bool) -> Self {
        let attrib_vec: Vec<P> = if store_attributes { vec![P::MAX; capacity] }
                                 else { Vec::new() };
        Sampler {
            store_attributes,
            capacity,
            weights: vec![0.0; capacity],
            points: vec![P::MAX; capacity],
            point_attributes: attrib_vec,
            accepted_state: (0.0, usize::MAX, usize::MAX),
            current_size: 0
        }
    }

    pub fn get_references(&self) -> &[P] {
        &self.points[0..self.current_size]
    }

    fn swap_down(&mut self, start_index: usize, validate: bool) {
        let mut current: usize = start_index;
        while 2 * current + 1 < self.current_size {
            let mut max_index: usize = 2 * current + 1;
            if 2 * current + 2 < self.current_size && self.weights[2 * current + 2] > self.weights[max_index] {
                max_index = 2 * current + 2;
            }
            if self.weights[max_index] > self.weights[current] {
                if validate {
                    println!("the heap property is not satisfied at index '{}' ", current);
                    panic!();
                }
                self.swap_weights(current, max_index);
                current = max_index;
            } else {
                break;
            }
        }
    }

    pub fn reheap(&mut self, validate: bool) {
        for i in ((self.current_size + 1) / 2)..=0 {
            self.swap_down(i, validate);
        }
    }

    fn swap_weights(&mut self, a: usize, b: usize) {
        if self.points[a] == P::MAX || self.points[b] == P::MAX {
            panic!();
        }

        let tmp: P = self.points[a];
        self.points[a] = self.points[b];
        self.points[b] = tmp;

        let tmp_weight: f32 = self.weights[a];
        self.weights[a] = self.weights[b];
        self.weights[b] = tmp_weight;

        if self.store_attributes {
            let tmp_attrib: P = self.point_attributes[a];
            self.point_attributes[a] = self.point_attributes[b];
            self.point_attributes[b] = tmp_attrib;
        }
    }

    pub fn add_point(&mut self, point_index: usize) where <P as TryFrom<usize>>::Error: Debug {
        if point_index != usize::MAX {
            assert!(self.current_size < self.capacity.into(), "sampler full");
            assert!(self.accepted_state.1 != usize::MAX,
                    "this method should only be called after a successful call to accept_sample(long)");

            self.weights[self.current_size] = self.accepted_state.0;
            self.points[self.current_size] = point_index.try_into().unwrap();
            // note, not self.accepted_state.1, even though we want that to not
            // P::MAX This corresponds to the change in the index value via
            // duplicates in the trees
            if self.store_attributes {
                self.point_attributes[self.current_size] = if self.accepted_state.2 != usize::MAX
                { self.accepted_state.2.try_into().unwrap() } else { P::MAX }
            };

            let mut current = self.current_size;
            self.current_size += 1;

            while current > 0 {
                let tmp = (current - 1) / 2;
                if self.weights[tmp] < self.weights[current] {
                    self.swap_weights(current, tmp);
                    current = tmp;
                } else {
                    break;
                }
            }
            // resetting the state
            self.accepted_state = (0.0, usize::MAX, usize::MAX);
        }
    }

    pub fn accept_point(&mut self, initial: bool, weight: f32, point_index: usize, attribute: usize) -> (bool, usize, f32, usize) {
        if initial || (weight < self.weights[0]) {
            self.accepted_state = (weight, point_index, attribute);
            //println!("adding {}", weight);
            let mut return_val = (true, usize::MAX, weight, usize::MAX);
            if !initial {
                let partial = self.evict_max();
                //println!(" evicing {} weight {} ",partial.0,partial.1);
                return_val = (true, partial.0, partial.1, partial.2);
            }
            return return_val;
        }
        (false, usize::MAX, weight, usize::MAX)
    }

    /**
     * evicts the maximum weight point from the sampler. can be used repeatedly to
     * change the size of the sampler and associated tree
     */

    pub fn evict_max(&mut self) -> (usize, f32, usize) {
        let evicted_attribute_index: usize = if self.store_attributes { self.point_attributes[0].into() } else { usize::MAX };

        let evicted_point = (self.points[0].into(), self.weights[0], evicted_attribute_index);
        self.current_size -= 1;
        let current: usize = self.current_size.into();
        self.weights[0] = self.weights[current];
        self.points[0] = self.points[current];
        if self.store_attributes {
            self.point_attributes[0] = self.point_attributes[current];
        }
        self.swap_down(0, false);

        evicted_point
    }

    pub fn is_full(&self) -> bool {
        self.current_size == self.capacity
    }

    pub fn get_fill_fraction(&self) -> f64 {
        if self.current_size == self.capacity {
            return 1.0
        }
        let fill_fraction: f64 = self.current_size as f64 / self.capacity as f64;
        fill_fraction
    }

    pub fn get_size(&self) -> usize {
        (self.weights.len()) * std::mem::size_of::<f32>() +
            (self.points.len()) * std::mem::size_of::<P>() +
            std::mem::size_of::<Sampler<P>>()
    }
}

