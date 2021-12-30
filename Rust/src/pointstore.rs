use crate::rcf::Max;
use std::fmt::Debug;
extern crate num;
use num::Integer;
use std::convert::TryFrom;
use std::collections::HashMap;
use std::collections::HashSet;
use crate::intervalstoremanager::IntervalStoreManager;

#[repr(C)]
pub struct PointStore<L> {
    internal_shingling : bool,
    internal_rotation : bool,
    last_known_shingle: Vec<f32>,
    dimensions: usize,
    shingle_size: usize,
    capacity: usize,
    store: Vec<f32>,
    reference_count: Vec<u8>,
    location: Vec<L>,
    next_sequence_index: usize,
    start_free_region: usize,
    index_manager: IntervalStoreManager<usize>,
    hash_reference_counts: HashMap<usize, usize>,
    entries_seen: i32
}

pub trait PointStoreView {
    fn get_shingled_point(&self, point: &[f32]) -> Vec<f32>;
    fn get_size(&self) -> usize;
    fn get_missing_values(&self, values: &[usize]) -> Vec<usize>;
    fn get_copy(&self, index:usize) -> Vec<f32>;
    fn is_equal(&self, point : &[f32], index : usize) -> bool;
    fn get_reference_and_offset(&self, index: usize) -> (&[f32],usize);
}

pub trait PointStoreEdit {
    fn add(&mut self, point: &[f32]) -> usize;
    fn inc(&mut self, index: usize);
    fn dec(&mut self, index: usize);
    fn adjust_count(&mut self, result: &[(usize,usize)]);
    fn compact(&mut self);
}


impl<L : Copy + Max + std::cmp::PartialEq>  PointStore<L> {
    pub fn new(dimensions: usize,
               shingle_size: usize,
               capacity: usize,
               initial_capacity: usize,
               internal_shingling: bool,
               internal_rotation: bool) -> Self {
        PointStore {
            internal_shingling,
            internal_rotation,
            dimensions,
            shingle_size,
            capacity,
            store: vec![0.0; initial_capacity * dimensions],
            location: vec![L::MAX; initial_capacity],
            reference_count: vec![0; initial_capacity],
            start_free_region: 0,
            index_manager: IntervalStoreManager::<usize>::new(initial_capacity),
            last_known_shingle: vec![0.0; dimensions],
            hash_reference_counts: HashMap::new(),
            entries_seen: 0,
            next_sequence_index : 0
        }
    }

    fn ready_to_copy(&self, point: &[f32]) -> bool {
        let mut answer: bool = self.shingle_size > 1;
        let base = self.dimensions / self.shingle_size;
        let mut index: usize = self.start_free_region;
        let extra = self.dimensions - base;
        if answer && index > extra {
            index -= extra;
            for i in 0..(extra) {
                answer = answer && (self.store[index + i] == point[i]);
            }
        } else {
            answer = false;
        }
        answer
    }

}


impl<L :Copy + Max + std::cmp::PartialEq> PointStoreView for PointStore<L>
    where L: std::convert::TryFrom<usize>, usize: From<L> {

    fn get_shingled_point(&self, point: &[f32]) -> Vec<f32> {
        let mut new_point = vec![0.0; self.dimensions];
        if self.internal_shingling {
            let base = self.dimensions / self.shingle_size;
            if point.len() != base {
                println!("The point must be '{}' floats long", self.dimensions);
                panic!();
            }
            if (!self.internal_rotation) {
                for i in 0..(self.dimensions - base) {
                    new_point[i] = self.last_known_shingle[i + base];
                }
                for i in 0..base {
                    new_point[self.dimensions - base + i] = point[i];
                }
            } else {
                for i in 0..(self.dimensions){
                    new_point[i] = self.last_known_shingle[i];
                }
                let offset = (self.next_sequence_index * base) % self.dimensions;
                for i in  0..base {
                    new_point[offset + i] = point[i];
                }
            }
            return new_point;
        }
        if point.len() != self.dimensions {
            println!("The point must be '{}' floats long", self.dimensions);
            panic!();
        }
        for i in 0..self.dimensions {
            new_point[i] = point[i];
        }
        new_point
    }

    fn get_reference_and_offset(&self, index: usize) -> (&[f32],usize) {
        let base = self.dimensions / self.shingle_size;
        if self.reference_count[index] == 0 {
            println!(" Index '{}' not in use", index);
            panic!();
        }
        let locn: usize = self.location[index].try_into().unwrap(); // because of u32
        let adj_locn = locn * base;
        let offset = if (!self.internal_rotation) { 0 } else { adj_locn % self.dimensions };
        (&self.store[adj_locn..(adj_locn + self.dimensions)], offset)
    }

    fn get_copy(&self, index: usize) -> Vec<f32>{
        let mut new_point = vec![0.0;self.dimensions];
        let (reference, offset) = self.get_reference_and_offset(index);
        if (self.internal_rotation) {
            for i in 0..self.dimensions {
                new_point[(i + offset) % self.dimensions] = reference[i];
            }
        } else {
            for i in 0..self.dimensions {
                new_point[i] = reference[i];
            }
        }
        new_point
    }

    fn get_missing_values(&self, values:&[usize]) -> Vec<usize>{
        if !self.internal_shingling {
            return Vec::from(values);
        }
        let mut answer = Vec::new();
        let base = self.dimensions / self.shingle_size;
        for i in 0..values.len() {
          assert!(values[i]< base);
            if self.internal_rotation {
              answer.push((self.next_sequence_index * base) % self.dimensions + values[i]);
            } else {
                answer.push(self.dimensions - base + values[i]);
            }
        }
        answer
    }

    fn is_equal(&self, point: &[f32], index: usize) -> bool{
        let (reference, offset) = self.get_reference_and_offset(index);
        if (self.internal_rotation) {
            for i in 0..self.dimensions {
                if (point[(i + offset) % self.dimensions] != reference[i]) {
                    return false;
                }
            }
            return true;
        } else {
            return point.eq(reference);
        }
    }

    fn get_size(&self) -> usize {
        self.store.len() * std::mem::size_of::<f32>()
            + self.location.len() * std::mem::size_of::<L>()
            + self.reference_count.len() * std::mem::size_of::<u8>()
            + self.index_manager.get_size() + std::mem::size_of::<PointStore<L>>()
    }
}



impl<L: Max + Copy + std::cmp::PartialEq> PointStoreEdit for PointStore<L>
    where L: std::convert::TryFrom<usize>, usize: From<L>, <L as TryFrom<usize>>::Error: Debug {

    fn add(&mut self, point: &[f32]) -> usize where <L as TryFrom<usize>>::Error: Debug {
        let base = self.dimensions / self.shingle_size;
        self.next_sequence_index += 1;
        if self.internal_shingling {
            if point.len() != base {
                println!("The point must be '{}' floats long", self.dimensions);
                panic!();
            }
            for i in 0..(self.dimensions - base) {
                self.last_known_shingle[i] = self.last_known_shingle[i + base];
            }
            for i in 0..base {
                self.last_known_shingle[self.dimensions - base + i] = point[i];
            }
            if (self.next_sequence_index < self.shingle_size){
                return usize::MAX;
            }
        } else if point.len() != self.dimensions {
            println!("The point must be '{}' floats long", self.dimensions);
            panic!();
        }


        if self.dimensions + self.start_free_region > self.store.len() {
            self.compact();
            if self.dimensions + self.start_free_region > self.store.len() {
                let mut new_size = self.store.len() + self.store.len() / 5;
                if new_size > self.capacity * self.dimensions {
                    new_size = self.capacity * self.dimensions;
                }
                self.store.resize(new_size, 0.0);
            }
        }

        if self.index_manager.is_empty() {
            assert!(self.reference_count.len() == self.location.len());
            let mut new_size = self.location.len() + self.location.len() / 5;
            if new_size > self.capacity {
                new_size = self.capacity;
            }
            self.location.resize(new_size, L::MAX);
            self.reference_count.resize(new_size, 0);
            self.index_manager.change_capacity(new_size);
        }
        let position: usize = self.index_manager.get();
        assert!(self.reference_count[position] == 0);
        self.reference_count[position] = 1;
        let mut new_point: &[f32] = if self.internal_shingling { &self.last_known_shingle }
        else { &point };

        if self.ready_to_copy(&new_point) {
            let base = self.dimensions / self.shingle_size;
            let mut index: usize = self.start_free_region;
            let extra = self.dimensions - base;
            let idx_value: usize = (index - extra) / base;
            self.location[position] = idx_value.try_into().unwrap();
            for i in 0..base {
                self.store[index] = new_point[extra + i];
                index += 1;
            }
            self.start_free_region += base;
        } else {
            let mut index: usize = self.start_free_region;
            let idx_value: usize = index / base;
            self.location[position] = idx_value.try_into().unwrap();
            for i in 0..self.dimensions {
                self.store[index] = new_point[i];
                index += 1;
            }
            self.start_free_region += self.dimensions;
        }
        position
    }

    fn inc(&mut self, index: usize) {
        if self.reference_count[index] == 0 {
            println!(" Index '{}' not in use", index);
            panic!();
        } else if self.reference_count[index] == u8::MAX {
            if let Some(a) = self.hash_reference_counts.remove(&index) {
                self.hash_reference_counts.insert(index, a + 1);
            } else {
                self.hash_reference_counts.insert(index, 1);
            }
        } else {
            self.reference_count[index] += 1;
        }
    }

    fn dec(&mut self, index: usize) {
        if self.reference_count[index] == 0 {
            println!(" Index '{}' not in use", index);
            panic!();
        } else if let Some(a) = self.hash_reference_counts.remove(&index) {
            if a > 1 {
                self.hash_reference_counts.insert(index, a - 1);
            }
        } else {
            self.reference_count[index] -= 1;
        }

        if self.reference_count[index] == 0 {
            self.index_manager.release(index);
            self.location[index] = L::MAX;
        }
    }

    fn adjust_count(&mut self, result:&[(usize,usize)]) {
        for (insert,delete) in result {
            if *insert != usize::MAX {
                self.inc(*insert);
                if *delete != usize::MAX {
                    self.dec(*delete);
                }
            }
        }
    }

    fn compact(&mut self) where <L as TryFrom<usize>>::Error: Debug {
        let base = self.dimensions / self.shingle_size;
        let mut reverse_reference: Vec<(usize, usize)> = Vec::new();
        for i in 0..self.location.len() {
            if self.location[i] != L::MAX {
                reverse_reference.push((self.location[i].try_into().unwrap(), i));
            }
        }
        reverse_reference.sort();
        let mut fresh_start: usize = 0;
        let mut j_static: usize = 0;
        let mut j_dynamic: usize = 0;
        let end: usize = reverse_reference.len();
        while j_static < end {
            let mut block_start: usize = reverse_reference[j_static].0;
            block_start = block_start * base;
            let mut block_end: usize = block_start + self.dimensions;
            let initial = if (self.internal_rotation) {
                (self.dimensions - fresh_start + block_start) % self.dimensions
            } else {0};

            let mut k = j_static + 1;
            j_dynamic = j_static + 1;
            while k < end {
                let new_locn: usize = reverse_reference[k].0;
                let new_elem: usize = base * new_locn;
                if block_end >= new_elem {
                    k += 1;
                    j_dynamic += 1;
                    if block_end < new_elem + self.dimensions {
                        block_end = new_elem + self.dimensions;
                    }
                } else {
                    k = end;
                }
            }

            // aligning the boundaries
            for i in 0..initial {
                self.store[fresh_start] = 0.0;
                fresh_start += 1;
            }

            for i in block_start..block_end {
                self.store[fresh_start] = self.store[i];
                assert!(!self.internal_rotation || fresh_start % self.dimensions == i % self.dimensions);
                if j_static < end {
                    let locn: usize = reverse_reference[j_static].0;
                    if i == base * locn {
                        let new_idx: usize = reverse_reference[j_static].1;
                        self.location[new_idx] = (fresh_start / base).try_into().unwrap();
                        j_static += 1;
                    }
                }
                fresh_start += 1;
            }

            if j_static != j_dynamic {
                println!("There is discepancy in indices between '{}' versus '{}'", j_static, j_dynamic);
                panic!();
            }
        }
        self.start_free_region = fresh_start.try_into().unwrap();
    }


}

