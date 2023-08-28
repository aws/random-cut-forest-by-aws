extern crate num;
use std::{collections::HashMap, convert::TryFrom, fmt::Debug};
use std::hash::Hash;
use std::ptr::hash;
use crate::types::{Result};
use crate::{common::intervalstoremanager::IntervalStoreManager, types::Location};
use crate::errors::RCFError;
use crate::util::check_argument;

pub const MAX_ATTRIBUTES: usize = 10;

pub trait PointStore<Label,Attributes> where Label: Copy + Sync, Attributes: Copy + Sync + Hash + Eq + Send {
    fn shingled_point(&self, point: &[f32]) -> Result<Vec<f32>>;
    fn size(&self) -> usize;
    fn missing_indices(&self, look_ahead: usize, values: &[usize]) -> Result<Vec<usize>>;
    fn next_indices(&self, look_ahead: usize) -> Result<Vec<usize>>;
    fn copy(&self, index: usize) -> Result<Vec<f32>>;
    fn is_equal(&self, point: &[f32], index: usize) -> Result<bool>;
    fn reference_and_offset(&self, index: usize) -> Result<(&[f32], usize)>;
    fn entries_seen(&self) -> u64;
    fn add(&mut self, point: &[f32], label:Label) -> Result<(usize,usize,Option<Vec<f32>>)>;
    fn inc(&mut self, index: usize,attribute_index: usize) -> Result<()>;
    fn dec(&mut self, index: usize,attribute_index: usize) -> Result<()>;
    fn adjust_count(&mut self, result: &[((usize, usize),(usize,usize))]) -> Result<()>;
    fn compact(&mut self) -> Result<()>;
    fn label(&self, index: usize) -> Result<Label>;
    fn attribute(&self, index: usize) -> Result<Attributes>;
    fn point_sum(&self, list:&[(usize,usize)]) -> Result<Vec<f32>>;
    fn attribute_vec(&self, index: usize) -> Result<Vec<f32>>;
}

#[repr(C)]
pub struct VectorizedPointStore<L,Label,Attributes>
where
    L: Location,
    Label: Copy + Sync,
    Attributes: Copy + Sync + Hash + Eq + Send,
{
    internal_shingling: bool,
    internal_rotation: bool,
    store_labels: bool,
    store_attributes: bool,
    propagate_attributes: bool,
    last_known_shingle: Vec<f32>,
    dimensions: usize,
    shingle_size: usize,
    capacity: usize,
    store: Vec<f32>,
    labels: HashMap<usize,(Label,usize)>,
    attribute_reverse_map: HashMap<Attributes,usize>,
    attributes: Vec<Attributes>,
    label_shingle: Vec<Label>,
    reference_count: Vec<u8>,
    label_count: Vec<u8>,
    attribute_count: Vec<u8>,
    location: Vec<L>,
    next_sequence_index: usize,
    start_free_region: usize,
    index_manager: IntervalStoreManager<usize>,
    label_manager: IntervalStoreManager<usize>,
    attribute_manager: IntervalStoreManager<usize>,
    hash_reference_counts: HashMap<usize, usize>,
    hash_label_counts: HashMap<usize, usize>,
    hash_attribute_counts: HashMap<usize, usize>,
    entries_seen: u64,
    attribute_creator: fn(_label_shingle: &[Label], _current_label : Label) -> Result<Attributes>,
    attribute_to_vec: Option<fn( _attribute :&Attributes) -> Result<Vec<f32>>>
}

impl<L,Label,Attributes> VectorizedPointStore<L,Label,Attributes>
where
    L: Location,
    usize: From<L>,
    Label : Copy + Sync + Send,
    Attributes: Copy + Sync + Hash + Eq + Send,
    <L as TryFrom<usize>>::Error: Debug,
{
    pub fn new(
        dimensions: usize,
        shingle_size: usize,
        capacity: usize,
        initial_capacity: usize,
        internal_shingling: bool,
        internal_rotation: bool,
        store_attributes: bool,
        propagate_attributes: bool,
        attribute_creator: fn(_label_shingle: &[Label], _current_label : Label) -> Result<Attributes>,
        attribute_to_vec: Option<fn( _attribute :&Attributes) -> Result<Vec<f32>>>
    ) -> Result<Self> {
        Ok(VectorizedPointStore {
            internal_shingling,
            internal_rotation,
            store_labels: store_attributes,
            store_attributes,
            dimensions,
            shingle_size,
            capacity,
            store: vec![0.0; initial_capacity * dimensions],
            labels: HashMap::new(),
            attribute_reverse_map: HashMap::new(),
            location: vec![L::MAX; initial_capacity],
            reference_count: vec![0; initial_capacity],
            start_free_region: 0,
            index_manager: IntervalStoreManager::<usize>::new(initial_capacity),
            label_manager: IntervalStoreManager::<usize>::new(capacity),
            attribute_manager: IntervalStoreManager::<usize>::new(capacity),
            last_known_shingle: vec![0.0; dimensions],
            hash_reference_counts: HashMap::new(),
            hash_label_counts: HashMap::new(),
            hash_attribute_counts: HashMap::new(),
            entries_seen: 0,
            next_sequence_index: 0,
            attribute_count: Vec::new(),
            label_count: Vec::new(),
            label_shingle: Vec::new(),
            attribute_creator,
            attribute_to_vec,
            propagate_attributes,
            attributes: Vec::new()
        })
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

    fn inc_helper(index : usize, reference_counts: &mut [u8], hashmap: &mut HashMap<usize,usize>) -> Result<()>{
        check_argument(index < reference_counts.len(), "incorrect range of index at insert")?;
        if reference_counts[index] == u8::MAX {
            if let Some(a) = hashmap.remove(&index) {
                hashmap.insert(index, a + 1);
            } else {
                hashmap.insert(index, 1);
            }
        } else {
            reference_counts[index] += 1;
        };
        Ok(())
    }

    fn dec_helper(index : usize, reference_counts: &mut [u8], hashmap: &mut HashMap<usize,usize>) -> Result<()>{
        check_argument(index < reference_counts.len(), "incorrect range of index at delete")?;
        check_argument(reference_counts[index] != 0, "index not in use for delete")?;

        if let Some(a) = hashmap.remove(&index) {
            if a > 1 {
                hashmap.insert(index, a - 1);
            }
        } else {
            reference_counts[index] -= 1;
        }
        Ok(())
    }
}

impl<L,Label,Attributes> PointStore<Label,Attributes> for VectorizedPointStore<L,Label,Attributes>
where
    L: Location,
    usize: From<L>,
    Label: Copy + Sync + Send,
    Attributes : Copy + Sync + Eq + Hash + Send,
    <L as TryFrom<usize>>::Error: Debug,
{
    fn shingled_point(&self, point: &[f32]) -> Result<Vec<f32>> {
        let mut new_point = vec![0.0; self.dimensions];
        let base = self.dimensions / self.shingle_size;
        if point.len() == base && self.shingle_size > 1 {
            check_argument(self.internal_shingling, "expecting input corresponding to internal shingling")?;
            if !self.internal_rotation {
                for i in 0..(self.dimensions - base) {
                    new_point[i] = self.last_known_shingle[i + base];
                }
                for i in 0..base {
                    new_point[self.dimensions - base + i] = point[i];
                }
            } else {
                for i in 0..(self.dimensions) {
                    new_point[i] = self.last_known_shingle[i];
                }
                let offset = (self.next_sequence_index * base) % self.dimensions;
                for i in 0..base {
                    new_point[offset + i] = point[i];
                }
            }
            return Ok(new_point);
        } else {
            check_argument(point.len() == self.dimensions, " expecting externally shingled input")?;
        }
        for i in 0..self.dimensions {
            new_point[i] = point[i];
        }
        Ok(new_point)
    }

    fn size(&self) -> usize {
        self.store.len() * std::mem::size_of::<f32>()
            + self.location.len() * std::mem::size_of::<L>()
            + self.reference_count.len() * std::mem::size_of::<u8>()
            + self.index_manager.get_size()
            + std::mem::size_of::<VectorizedPointStore<L,Label,Attributes>>()
    }

    fn missing_indices(&self, look_ahead: usize, values: &[usize]) -> Result<Vec<usize>> {
        if !self.internal_shingling {
            for x in values {
                check_argument(*x<self.dimensions, "incorrect input")?;
            }
            return Ok(Vec::from(values));
        }
        let mut answer = Vec::new();
        let base = self.dimensions / self.shingle_size;
        for i in 0..values.len() {
            check_argument(values[i] < base, "incorrect input")?;
            if self.internal_rotation {
                answer.push(
                    ((self.next_sequence_index + look_ahead) * base + values[i]) % self.dimensions,
                );
            } else {
                answer.push(self.dimensions - base + values[i]);
            }
        }
        Ok(answer)
    }

    fn next_indices(&self, look_ahead: usize) -> Result<Vec<usize>> {
        let base = self.dimensions / self.shingle_size;
        let vec: Vec<usize> = (0..base).collect();
        self.missing_indices(look_ahead, &vec)
    }

    fn copy(&self, index: usize) -> Result<Vec<f32>> {
        let mut new_point = vec![0.0; self.dimensions];
        let (reference, offset) = self.reference_and_offset(index)?;
        if self.internal_rotation {
            for i in 0..self.dimensions {
                new_point[(i + offset) % self.dimensions] = reference[i];
            }
        } else {
            for i in 0..self.dimensions {
                new_point[i] = reference[i];
            }
        }
        Ok(new_point)
    }

    fn is_equal(&self, point: &[f32], index: usize) -> Result<bool> {
        let (reference, offset) = self.reference_and_offset(index)?;
        if self.internal_rotation {
            for i in 0..self.dimensions {
                if point[(i + offset) % self.dimensions] != reference[i] {
                    return Ok(false);
                }
            }
            return Ok(true);
        } else {
            return Ok(point.eq(reference));
        }
    }

    fn reference_and_offset(&self, index: usize) -> Result<(&[f32], usize)> {
        let base = self.dimensions / self.shingle_size;
        check_argument(self.reference_count[index] != 0 , "index not in use")?;

        let locn : usize = self.location[index].try_into().expect("corrupt state");
        let adj_locn = locn * base;
        let offset = if !self.internal_rotation {
            0
        } else {
            adj_locn % self.dimensions
        };
        Ok((&self.store[adj_locn..(adj_locn + self.dimensions)], offset))
    }

    fn entries_seen(&self) -> u64 {
        self.entries_seen
    }

    fn add(&mut self, point: &[f32], label:Label) -> Result<(usize,usize,Option<Vec<f32>>)> {
        let base = self.dimensions / self.shingle_size;
        self.next_sequence_index += 1;

        if self.internal_shingling {
            check_argument(point.len() == base, "incorrect length")?;
            for i in 0..(self.dimensions - base) {
                self.last_known_shingle[i] = self.last_known_shingle[i + base];
            }
            for i in 0..base {
                self.last_known_shingle[self.dimensions - base + i] = point[i];
            }
            if self.store_labels {
                if self.next_sequence_index <= self.shingle_size {
                    self.label_shingle.push(label);
                } else {
                    for i in 0..self.shingle_size - 1 {
                        self.label_shingle[i] = self.label_shingle[i + 1];
                    }
                    self.label_shingle[self.shingle_size - 1] = label;
                }
            }
            if self.next_sequence_index < self.shingle_size {
                return Ok((usize::MAX,usize::MAX,None));
            }
        } else {
            check_argument(point.len() == self.dimensions, "incorrect lengths")?;
        }

        let mut attrib_vec = None;
        let attrib_pos = if self.store_attributes {
            let new_attribute = (self.attribute_creator)(&self.label_shingle, label)?;
            let a_pos = *self.attribute_reverse_map.get(&new_attribute).unwrap_or(&self.attributes.len());
            let b_pos = if a_pos >= self.attributes.len() {
                let y = self.attribute_manager.get()?;
                self.attribute_reverse_map.insert(new_attribute, y);
                y
            } else {
                a_pos
            };
            if b_pos == self.attribute_count.len() {
                    self.attributes.push(new_attribute);
                    self.attribute_count.push(1);
            } else {
                self.attributes[b_pos] = new_attribute;
                Self::inc_helper(b_pos,&mut self.attribute_count,&mut self.hash_attribute_counts)?;
            };
            b_pos
        } else {usize::MAX};

        let label_pos = if self.store_labels {
            let y = self.label_manager.get()?;
            if y >= self.labels.len() {
                check_argument(y == self.labels.len(), " incorrect behavior in labels")?;
                self.labels.insert(y,(label,attrib_pos));
                self.label_count.push(1);
            } else {
                check_argument(self.label_count[y] == 0, " incorrect state in label management")?;
                self.labels.insert(y,(label,attrib_pos));
                self.label_count[y] = 1;
            };
            y
        } else {usize::MAX};


        if self.dimensions + self.start_free_region > self.store.len() {
            self.compact()?;
            if self.dimensions + self.start_free_region > self.store.len() {
                let mut new_size = self.store.len() + self.store.len() / 5;
                if new_size > self.capacity * self.dimensions {
                    new_size = self.capacity * self.dimensions;
                }
                self.store.resize(new_size, 0.0);
            }
        }

        if self.index_manager.is_empty() {
            check_argument(self.reference_count.len() == self.location.len(), "incorrect state")?;
            let mut new_size = self.location.len() + self.location.len() / 5;
            if new_size > self.capacity {
                new_size = self.capacity;
            }
            self.location.resize(new_size, L::MAX);
            self.reference_count.resize(new_size, 0);
            self.index_manager.change_capacity(new_size);
        }
        let position: usize = self.index_manager.get()?;
        check_argument(self.reference_count[position] == 0, "incorrect state")?;
        self.reference_count[position] = 1;
        let new_point: &[f32] = if self.internal_shingling {
            &self.last_known_shingle
        } else {
            &point
        };

        if self.ready_to_copy(&new_point) {
            let base = self.dimensions / self.shingle_size;
            let mut index: usize = self.start_free_region;
            let extra = self.dimensions - base;
            let idx_value: usize = (index - extra) / base;
            self.location[position] = idx_value.try_into().expect("incorrect range");
            for i in 0..base {
                self.store[index] = new_point[extra + i];
                index += 1;
            }
            self.start_free_region += base;
        } else {
            let mut index: usize = self.start_free_region;
            let idx_value: usize = index / base;
            self.location[position] = idx_value.try_into().expect("range error");
            for i in 0..self.dimensions {
                self.store[index] = new_point[i];
                index += 1;
            }
            self.start_free_region += self.dimensions;
        }
        if self.store_labels {
            Ok((position, label_pos, attrib_vec))
        } else {
            Ok((position,attrib_pos,attrib_vec))
        }
    }

    fn inc(&mut self, index: usize, secondary_index: usize) -> Result<()>{
        Self::inc_helper(index,&mut self.reference_count,&mut self.hash_reference_counts)?;
        if self.store_labels {
            let attrib_index = (*self.labels.get(&secondary_index).expect("not found")).1;
            if attrib_index != usize::MAX {
                Self::inc_helper(attrib_index,&mut self.attribute_count, &mut self.hash_attribute_counts)?;
            }
            Self::inc_helper(secondary_index, &mut self.label_count, &mut self.hash_label_counts)?;
        } else if self.store_attributes {
            Self::inc_helper(secondary_index,&mut self.attribute_count, &mut self.hash_attribute_counts)?;
        }
        Ok(())
    }

    fn dec(&mut self, index: usize,secondary_index: usize) -> Result<()> {
        Self::dec_helper(index,&mut self.reference_count,&mut self.hash_reference_counts)?;
        if self.reference_count[index] == 0 {
            self.index_manager.release(index)?;
            self.location[index] = L::MAX;
        }
        if self.store_labels {
            let attribute_index = (*self.labels.get(&secondary_index).expect("not found")).1;
            if attribute_index != usize::MAX {
                Self::dec_helper(attribute_index,&mut self.attribute_count, &mut self.hash_attribute_counts)?;
                if self.attribute_count[attribute_index] == 0 {
                    self.attribute_manager.release(attribute_index)?;
                }
            }
            Self::dec_helper(secondary_index, &mut self.label_count, &mut self.hash_label_counts)?;
            if self.label_count[secondary_index] == 0 {
                self.label_manager.release(secondary_index)?;
            }
        } else {
            if secondary_index !=usize::MAX {
                Self::dec_helper(secondary_index,&mut self.attribute_count, &mut self.hash_attribute_counts)?;
                let x = self.attribute_reverse_map.remove(&self.attributes[secondary_index]).expect(" error in secondary");
                check_argument(x == secondary_index, "attribute index accounting is incorrect")?;
                if self.attribute_count[secondary_index] == 0 {
                    self.attribute_manager.release(secondary_index)?;
                }
            }
        }
        Ok(())
    }

    fn adjust_count(&mut self, result: &[((usize, usize),(usize,usize))]) -> Result<()> {
        for (insert, delete) in result {
            if (*insert).0 != usize::MAX {
                self.inc((*insert).0,(*insert).1)?;
                if (*delete).0 != usize::MAX {
                    self.dec((*delete).0,(*delete).1)?;
                }
            }
        }
        Ok(())
    }

    fn compact(&mut self) -> Result<()>{
        let base = self.dimensions / self.shingle_size;
        let mut reverse_reference: Vec<(usize, usize)> = Vec::new();
        for i in 0..self.location.len() {
            if self.location[i] != L::MAX {
                reverse_reference.push((self.location[i].try_into().expect("range error"), i));
            }
        }
        reverse_reference.sort();
        let mut fresh_start: usize = 0;
        let mut j_static: usize = 0;
        let mut j_dynamic: usize;
        let end: usize = reverse_reference.len();
        while j_static < end {
            let mut block_start: usize = reverse_reference[j_static].0;
            block_start = block_start * base;
            let mut block_end: usize = block_start + self.dimensions;
            let initial = if self.internal_rotation {
                (self.dimensions - fresh_start + block_start) % self.dimensions
            } else {
                0
            };

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
            for _i in 0..initial {
                self.store[fresh_start] = 0.0;
                fresh_start += 1;
            }

            for i in block_start..block_end {
                self.store[fresh_start] = self.store[i];
                check_argument(!self.internal_rotation || fresh_start % self.dimensions == i % self.dimensions, "corrupt state in compaction")?;
                if j_static < end {
                    let locn: usize = reverse_reference[j_static].0;
                    if i == base * locn {
                        let new_idx: usize = reverse_reference[j_static].1;
                        self.location[new_idx] = (fresh_start / base).try_into().expect("range error");
                        j_static += 1;
                    }
                }
                fresh_start += 1;
            }

            check_argument(j_static == j_dynamic, "There is discrepancy in indices")?;
        }
        self.start_free_region = fresh_start.try_into().expect("range error");
        Ok(())
    }

    fn label(&self, index: usize) -> Result<Label> {
        check_argument(self.store_labels && index < self.labels.len() && self.label_count[index] != 0, " cannot access the label")?;
        let (label,attribute_index) = *self.labels.get(&index).expect("unexpected index");
        Ok(label)
    }

    fn attribute(&self, index: usize) -> Result<Attributes> {
        check_argument(self.store_attributes, " attributes not stored")?;
        if self.store_labels {
            let y = *self.labels.get(&index).expect("label not in use");
            check_argument(y.1<self.attributes.len()," not in use")?;
            Ok(self.attributes[y.1])
        } else {
            check_argument(index<self.attributes.len()," not in use")?;
            Ok(self.attributes[index])
        }
    }

    fn point_sum(&self, list:&[(usize,usize)]) -> Result<Vec<f32>> {
        let mut answer = vec![0.0; self.dimensions];
        for (a,b) in list {
            let (point, offset) = self.reference_and_offset(*a)?;
            for (x,&y) in answer.iter_mut().zip(point) {
                *x += y * (*b) as f32;
            }
        }
        Ok(answer)
    }

    fn attribute_vec(&self, index: usize) -> Result<Vec<f32>> {
        check_argument(self.store_attributes, " need to store attributes first")?;
        let y = if self.store_labels {
            let x = *self.labels.get(&index).expect("label not in use");
            x.1
        } else {
            index
        };
        if let Some(function) = self.attribute_to_vec {
            check_argument(index < self.attributes.len(), " out of range")?;
            (function) (&self.attributes[index])
        } else {
            let mut answer = vec![0.0f32; MAX_ATTRIBUTES];
            answer[y % MAX_ATTRIBUTES] = 1.0;
            Ok(answer)
        }
    }
}
