use std::{collections::HashMap, fmt::Debug, mem};

use crate::{
    common::{divector::DiVector, intervalstoremanager::IntervalStoreManager},
    pointstore::PointStore,
    samplerplustree::{boundingbox::BoundingBox, cut::Cut},
    types::Location,
};

///
/// capacity is the number of leaves in the tree
/// this is the (per tree) samplesize in RCF
/// in the encoding below, the leaves are point_index + capacity
/// the value capacity - 1 stands for null
/// the values 0..(capacity-2) corresponds to the internal nodes; note that a regular binary tree
/// where each node has 0 or 2 children, has (capacity - 1) internal nodes
///
/// the nodestore does not need to save the parent information; it is saved if the bounding box cache is
/// more than 0.
///
/// Note that the mass of each node (in use) is at least 1. Subtracting 1 from each node implicitly
/// makes the values between [0..(capacity-1)] which is very convenient for 2^8 and 2^16.
///

#[repr(C)]
pub struct VectorNodeStore<C, P, N>
where
    C: Location,
    usize: From<C>,
    P: Location,
    usize: From<P>,
    N: Location,
    usize: From<N>,
{
    capacity: usize,
    dimensions: usize,
    using_transforms: bool,
    project_to_tree: fn(Vec<f32>) -> Vec<f32>,
    bounding_box_cache_fraction: f64,
    parent_index: Vec<N>,
    mass: Vec<N>,
    pub left_index: Vec<P>,
    pub right_index: Vec<P>,
    pub cut_dimension: Vec<C>,
    pub cut_value: Vec<f32>,
    bounding_box_data: Vec<f32>,
    range_sum_data: Vec<f64>,
    hash_mass_leaves: HashMap<usize, usize>,
    pub internal_node_manager: IntervalStoreManager<usize>,
}

const SWITCH_THRESHOLD: f64 = 0.5;

pub trait NodeStore {
    fn get_mass(&self, index: usize) -> usize;
    fn get_box(&self, index: usize, point_store: &dyn PointStore) -> BoundingBox;
    fn get_probability_of_cut(
        &self,
        index: usize,
        point: &[f32],
        point_store: &dyn PointStore,
    ) -> f64;
    fn get_probability_of_cut_with_missing_coordinates(
        &self,
        index: usize,
        point: &[f32],
        missing_coordinates: &[bool],
        point_store: &dyn PointStore,
    ) -> f64;
    fn modify_in_place_probability_of_cut_di_vector(
        &self,
        index: usize,
        point: &[f32],
        point_store: &dyn PointStore,
        di_vector: &mut DiVector,
    );
    fn modify_in_place_probability_of_cut_di_vector_with_missing_coordinates(
        &self,
        index: usize,
        point: &[f32],
        missing_coordinates: &[bool],
        point_store: &dyn PointStore,
        di_vector: &mut DiVector,
    );
    fn grow_node_box_pair(
        &self,
        first: &mut BoundingBox,
        second: &mut BoundingBox,
        point_store: &dyn PointStore,
        node: usize,
        sibling: usize,
    );
    fn grow_node_box(
        &self,
        bounding_box: &mut BoundingBox,
        point_store: &dyn PointStore,
        node: usize,
        sibling: usize,
    );
    fn get_sibling(&self, node: usize, parent: usize) -> usize;
    fn get_leaf_point_index(&self, index: usize) -> usize;
    fn get_cut_dimension(&self, index: usize) -> usize;
    fn get_left_index(&self, index: usize) -> usize;
    fn get_right_index(&self, index: usize) -> usize;
    fn get_cut_value(&self, index: usize) -> f32;
    fn is_leaf(&self, index: usize) -> bool;
    fn is_left_of(&self, index: usize, point: &[f32]) -> bool;
    fn use_path_for_box(&self) -> bool;
    fn get_distribution(&self, index: usize) -> (usize, f32, usize, usize);
    fn get_cut_and_children(&self, index: usize) -> (usize, f32, usize, usize);
    fn set_path(&self, answer: &mut Vec<(usize, usize)>, root: usize, point: &[f32]);
    fn null_node(&self) -> usize;
}

impl<C, P, N> VectorNodeStore<C, P, N>
where
    C: Location,
    usize: From<C>,
    P: Location,
    usize: From<P>,
    N: Location,
    usize: From<N>,
    <C as TryFrom<usize>>::Error: Debug,
    <P as TryFrom<usize>>::Error: Debug,
    <N as TryFrom<usize>>::Error: Debug,
{
    pub fn new(
        capacity: usize,
        dimensions: usize,
        using_transforms: bool,
        project_to_tree: fn(Vec<f32>) -> Vec<f32>,
        bounding_box_cache_fraction: f64,
    ) -> Self {
        if capacity - 1 > N::MAX.into() {
            println!(
                " invalid parameter, increase size of N to represent {}",
                capacity
            );
            panic!();
        }
        let cache_limit: usize = (bounding_box_cache_fraction * capacity as f64) as usize;
        let null_node = Self::null_value(capacity);
        VectorNodeStore {
            capacity,
            dimensions,
            using_transforms,
            project_to_tree,
            bounding_box_cache_fraction,
            left_index: vec![null_node.try_into().unwrap(); capacity - 1],
            right_index: vec![null_node.try_into().unwrap(); capacity - 1],
            mass: vec![0.try_into().unwrap(); capacity - 1],
            parent_index: if bounding_box_cache_fraction > 0.0 {
                vec![null_node.try_into().unwrap(); capacity - 1]
            } else {
                Vec::new()
            },
            cut_value: vec![0.0; capacity - 1],
            cut_dimension: vec![C::MAX; capacity - 1],
            bounding_box_data: vec![0.0; dimensions * 2 * cache_limit],
            range_sum_data: vec![0.0; cache_limit],
            hash_mass_leaves: HashMap::new(),
            internal_node_manager: IntervalStoreManager::<usize>::new(capacity - 1),
        }
    }

    /// 0 is indicative of null given unsigned representation
    /// otherwise index X uses slot X-1

    fn translate(&self, index: usize) -> usize {
        if index != self.null_node() && self.range_sum_data.len() <= index {
            usize::MAX
        } else {
            index
        }
    }

    fn copy_box_to_data(&mut self, index: usize, bounding_box: &BoundingBox) {
        let idx: usize = self.translate(index);
        if idx != usize::MAX {
            let base: usize = 2 * idx * self.dimensions;
            let mid = base + self.dimensions;
            let minarray = &mut self.bounding_box_data[base..mid];
            for (x, y) in minarray.iter_mut().zip((*bounding_box).get_min_values()) {
                *x = *y;
            }
            let maxarray = &mut self.bounding_box_data[mid..mid + self.dimensions];
            for (x, y) in maxarray.iter_mut().zip((*bounding_box).get_max_values()) {
                *x = *y;
            }
            self.range_sum_data[idx] = (*bounding_box).get_range_sum();
        }
    }

    fn check_contains_and_add_point(&mut self, index: usize, point: &[f32]) -> bool {
        let idx: usize = self.translate(index);
        if idx != usize::MAX {
            let base = 2 * idx * self.dimensions;
            let mid = base + self.dimensions;
            let minarray = &mut self.bounding_box_data[base..mid];
            for (x, y) in minarray.iter_mut().zip(point) {
                *x = if (*x) > (*y) { *y } else { *x };
            }

            let maxarray = &mut self.bounding_box_data[mid..mid + self.dimensions];
            for (x, y) in maxarray.iter_mut().zip(point) {
                *x = if *x < *y { *y } else { *x };
            }

            let newminarray = &self.bounding_box_data[base..mid];
            let newmaxarray = &self.bounding_box_data[mid..mid + self.dimensions];
            let newsum: f64 = newminarray
                .iter()
                .zip(newmaxarray)
                .map(|(x, y)| (y - x) as f64)
                .sum();
            let answer = self.range_sum_data[idx] == newsum;
            self.range_sum_data[idx] = newsum;
            return answer;
        }
        false
    }

    fn check_strictly_contains(&mut self, index: usize, point: &[f32]) -> bool {
        let idx: usize = self.translate(index);
        if idx != usize::MAX {
            let base = 2 * idx * self.dimensions;
            let mid = base + self.dimensions;
            let minarray = &self.bounding_box_data[base..mid];
            let maxarray = &self.bounding_box_data[mid..mid + self.dimensions];
            let not_inside = minarray
                .iter()
                .zip(point)
                .zip(maxarray)
                .any(|((x, y), z)| x >= y || y >= z);
            return !not_inside;
        }
        false
    }

    fn get_box_from_data(&self, idx: usize) -> BoundingBox {
        let dimensions = self.dimensions;
        let base = 2 * idx * dimensions;
        return BoundingBox::new(
            &self.bounding_box_data[base..base + dimensions],
            &self.bounding_box_data[base + dimensions..base + 2 * dimensions],
        );
    }

    pub fn reconstruct_box(&self, index: usize, point_store: &dyn PointStore) -> BoundingBox {
        let mut mutated_bounding_box = self.get_box(self.left_index[index].into(), point_store);
        self.grow_node_box(
            &mut mutated_bounding_box,
            point_store,
            index,
            self.right_index[index].into(),
        );
        mutated_bounding_box
    }

    pub fn check_contains_and_rebuild_box(
        &mut self,
        index: usize,
        point: &[f32],
        point_store: &dyn PointStore,
    ) -> bool {
        let idx = self.translate(index);
        if idx != usize::MAX {
            if !self.check_strictly_contains(index, point) {
                let mutated_bounding_box = self.reconstruct_box(index, point_store);
                self.copy_box_to_data(index, &mutated_bounding_box);
                return false;
            }
            true
        } else {
            false
        }
    }

    pub fn add_node(
        &mut self,
        parent_index: usize,
        point: &[f32],
        child: usize,
        point_index: usize,
        cut: Cut,
        saved_box: &BoundingBox,
    ) -> usize {
        let index: usize = self.internal_node_manager.get().into();
        self.cut_value[index] = cut.value;
        self.cut_dimension[index] = cut.dimension.try_into().unwrap();
        if point[cut.dimension] <= cut.value {
            self.left_index[index] = self.leaf_index(point_index).try_into().unwrap();
            self.right_index[index] = child.try_into().unwrap();
        } else {
            self.left_index[index] = child.try_into().unwrap();
            self.right_index[index] = self.leaf_index(point_index).try_into().unwrap();
        }

        self.mass[index] = (self.get_mass(child)).try_into().unwrap();
        // Not adding 1 to the above (new leaf) since all mass is represented as mass- 1
        if self.bounding_box_cache_fraction > 0.0 {
            self.copy_box_to_data(index, saved_box);
            self.check_contains_and_add_point(index, point);

            self.parent_index[index] = parent_index.try_into().unwrap();
            if !self.is_leaf(child) {
                self.parent_index[child] = index.try_into().unwrap();
            }
        }

        if parent_index != self.null_node() {
            self.replace_node(parent_index, child, index);
        }
        index
    }

    pub fn leaf_index(&self, point_index: usize) -> usize {
        point_index + self.capacity
    }

    pub fn set_root(&mut self, index: usize)
    where
        <N as TryFrom<usize>>::Error: Debug,
    {
        if !self.is_leaf(index) && self.bounding_box_cache_fraction > 0.0 {
            self.parent_index[index] = 0.try_into().unwrap();
        }
    }

    // capacity is the number of leaves
    pub fn increase_leaf_mass(&mut self, index: usize) {
        if index >= self.capacity {
            let y = index - self.capacity;
            if let Some(a) = self.hash_mass_leaves.remove(&y) {
                self.hash_mass_leaves.insert(y, a + 1);
            } else {
                self.hash_mass_leaves.insert(y, 1);
            }
        } else {
            panic!(" incorrect call with a non-leaf index");
        }
    }

    pub fn decrease_leaf_mass(&mut self, index: usize) -> usize {
        let y = index - self.capacity;
        return if let Some(a) = self.hash_mass_leaves.remove(&y) {
            if a > 1 {
                self.hash_mass_leaves.insert(y, a - 1);
                a
            } else {
                1 //default
            }
        } else {
            0
        };
    }

    pub fn manage_ancestors_add(
        &mut self,
        path: &mut Vec<(usize, usize)>,
        point: &[f32],
        _point_store: &dyn PointStore,
        box_resolved: bool,
    ) {
        let mut resolved = box_resolved;
        while path.len() != 0 {
            let index = path.pop().unwrap().0;
            let val: usize = self.mass[index].into();
            self.mass[index] = (val + 1).try_into().unwrap();
            if self.bounding_box_cache_fraction > 0.0 && !resolved {
                resolved = self.check_contains_and_add_point(index.into(), point);
            }
        }
    }

    pub fn manage_ancestors_delete(
        &mut self,
        path: &mut Vec<(usize, usize)>,
        point: &[f32],
        point_store: &dyn PointStore,
        box_resolved: bool,
    ) {
        let mut resolved = box_resolved;
        while path.len() != 0 {
            let index = path.pop().unwrap().0;
            let val: usize = self.mass[index].into();
            self.mass[index] = (val - 1).try_into().unwrap();
            if self.bounding_box_cache_fraction > 0.0 && !resolved {
                resolved = self.check_contains_and_rebuild_box(index, point, point_store);
            }
        }
    }

    pub fn delete_internal_node(&mut self, index: usize) {
        let null_node = self.null_node();

        self.left_index[index] = null_node.try_into().unwrap();
        self.right_index[index] = null_node.try_into().unwrap();
        self.mass[index] = 0.try_into().unwrap();
        if self.bounding_box_cache_fraction > 0.0 {
            self.parent_index[index] = null_node.try_into().unwrap(); // null
        }
        self.cut_dimension[index] = C::MAX;
        self.cut_value[index] = 0.0;
        self.internal_node_manager.release(index);
    }

    pub fn get_point_index(&self, index: usize) -> usize {
        assert!(self.is_leaf(index));
        index - self.capacity
    }

    pub fn get_cut_value(&self, index: usize) -> f32 {
        self.cut_value[index]
    }

    pub fn get_cut_dimension(&self, index: usize) -> usize {
        self.cut_dimension[index].into()
    }

    pub fn check_left(
        &self,
        index: usize,
        dim: usize,
        value: f32,
        point_store: &dyn PointStore,
    ) -> bool {
        if self.is_leaf(index) {
            let point = (self.project_to_tree)(point_store.get_copy(self.get_point_index(index)));
            return point[dim] < value;
        }
        self.check_left(self.get_left_index(index), dim, value, point_store)
            && self.check_left(self.get_right_index(index), dim, value, point_store)
    }

    pub fn check_right(
        &self,
        index: usize,
        dim: usize,
        value: f32,
        point_store: &dyn PointStore,
    ) -> bool {
        if self.is_leaf(index) {
            let point = (self.project_to_tree)(point_store.get_copy(self.get_point_index(index)));
            return point[dim] >= value;
        }
        self.check_right(self.get_left_index(index), dim, value, point_store)
            && self.check_right(self.get_right_index(index), dim, value, point_store)
    }

    pub fn replace_node(&mut self, grand_parent: usize, parent: usize, node: usize) {
        if parent == self.left_index[grand_parent].into() {
            self.left_index[grand_parent] = node.try_into().unwrap();
        } else {
            self.right_index[grand_parent] = node.try_into().unwrap();
        }
        if !self.is_leaf(node) && self.bounding_box_cache_fraction > 0.0 {
            self.parent_index[node] = grand_parent.try_into().unwrap();
        }
    }

    pub fn get_size(&self, _dimensions: usize) -> usize {
        (self.internal_node_manager.get_size() + self.left_index.len() + self.right_index.len())
            * std::mem::size_of::<P>()
            + (self.parent_index.len() + self.mass.len()) * std::mem::size_of::<N>()
            + (self.cut_dimension.len()) * std::mem::size_of::<C>()
            + (self.cut_value.len()) * mem::size_of::<f32>()
            + (self.bounding_box_data.len() + 2 * self.range_sum_data.len()) * mem::size_of::<f32>()
            + std::mem::size_of::<VectorNodeStore<C, P, N>>()
    }

    fn null_value(capacity: usize) -> usize {
        capacity - 1
    }

    fn is_internal(&self, index: usize) -> bool {
        index != self.null_node() && index < self.capacity
    }
}

impl<C, P, N> NodeStore for VectorNodeStore<C, P, N>
where
    C: Location,
    usize: From<C>,
    P: Location,
    usize: From<P>,
    N: Location,
    usize: From<N>,
    <C as TryFrom<usize>>::Error: Debug,
    <P as TryFrom<usize>>::Error: Debug,
    <N as TryFrom<usize>>::Error: Debug,
{
    fn get_mass(&self, index: usize) -> usize {
        if self.is_leaf(index) {
            let y = index - self.capacity;
            return if let Some(a) = self.hash_mass_leaves.get(&y) {
                (*a).into()
            } else {
                1
            };
        }
        let base: usize = self.mass[index].into();
        base + 1
    }

    fn get_box(&self, index: usize, point_store: &dyn PointStore) -> BoundingBox {
        if self.is_leaf(index) {
            return if self.using_transforms {
                let point =
                    &(self.project_to_tree)(point_store.get_copy(self.get_point_index(index)));
                BoundingBox::new(point, point)
            } else {
                let point = point_store
                    .get_reference_and_offset(self.get_point_index(index))
                    .0;
                BoundingBox::new(point, point)
            };
        } else {
            let idx: usize = self.translate(index);
            if idx != usize::MAX {
                return self.get_box_from_data(idx);
            }
            let mut mutated_bounding_box = self.get_box(self.left_index[index].into(), point_store);
            self.grow_node_box(
                &mut mutated_bounding_box,
                point_store,
                index,
                self.right_index[index].into(),
            );
            return mutated_bounding_box;
        }
    }

    fn get_probability_of_cut(
        &self,
        index: usize,
        point: &[f32],
        point_store: &dyn PointStore,
    ) -> f64 {
        let node_idx: usize = self.translate(index);
        if node_idx != usize::MAX {
            let base = 2 * node_idx * self.dimensions;
            let mid = base + self.dimensions;
            let minarray = &self.bounding_box_data[base..mid];
            let maxarray = &self.bounding_box_data[mid..mid + self.dimensions];
            let minsum: f32 = minarray
                .iter()
                .zip(point)
                .map(|(&x, &y)| if x - y > 0.0 { x - y } else { 0.0 })
                .sum();
            let maxsum: f32 = point
                .iter()
                .zip(maxarray)
                .map(|(&x, &y)| if x - y > 0.0 { x - y } else { 0.0 })
                .sum();
            let sum = maxsum + minsum;

            if sum == 0.0 {
                return 0.0;
            }
            sum as f64 / (self.range_sum_data[node_idx] + sum as f64)
        } else {
            let bounding_box = self.get_box(index, point_store);
            bounding_box.probability_of_cut(point)
        }
    }
    fn get_probability_of_cut_with_missing_coordinates(
        &self,
        index: usize,
        point: &[f32],
        missing_coordinates: &[bool],
        point_store: &dyn PointStore,
    ) -> f64 {
        let node_idx: usize = self.translate(index);
        if node_idx != usize::MAX {
            let base = 2 * node_idx * self.dimensions;
            let mid = base + self.dimensions;
            let minarray = &self.bounding_box_data[base..mid];
            let maxarray = &self.bounding_box_data[mid..mid + self.dimensions];
            let minsum: f32 = minarray
                .iter()
                .zip(point)
                .zip(missing_coordinates)
                .map(|((&x, &y), &b)| if !b && x - y > 0.0 { x - y } else { 0.0 })
                .sum();
            let maxsum: f32 = point
                .iter()
                .zip(maxarray)
                .zip(missing_coordinates)
                .map(|((&x, &y), &b)| if !b && x - y > 0.0 { x - y } else { 0.0 })
                .sum();
            let sum = maxsum + minsum;

            if sum == 0.0 {
                return 0.0;
            }
            sum as f64 / (self.range_sum_data[node_idx] + sum as f64)
        } else {
            let bounding_box = self.get_box(index, point_store);
            bounding_box.probability_of_cut_with_missing_coordinates(point, missing_coordinates)
        }
    }
    fn modify_in_place_probability_of_cut_di_vector(
        &self,
        index: usize,
        point: &[f32],
        point_store: &dyn PointStore,
        di_vector: &mut DiVector,
    ) {
        assert!(di_vector.high.len() == point.len());
        let node_idx: usize = self.translate(index);
        if node_idx != usize::MAX {
            let base = 2 * node_idx * self.dimensions;
            let mid = base + self.dimensions;
            let minsum: f64 = di_vector
                .low
                .iter_mut()
                .zip(&self.bounding_box_data[base..mid])
                .zip(point)
                .map(|((x, &y), &z)| {
                    if y - z > 0.0 {
                        *x = (y - z) as f64;
                        *x
                    } else {
                        *x = 0.0;
                        *x
                    }
                })
                .sum();
            let maxsum: f64 = di_vector
                .high
                .iter_mut()
                .zip(point)
                .zip(&self.bounding_box_data[mid..mid + self.dimensions])
                .map(|((x, &y), &z)| {
                    if y - z > 0.0 {
                        *x = (y - z) as f64;
                        *x
                    } else {
                        *x = 0.0;
                        *x
                    }
                })
                .sum();
            let sum = maxsum + minsum;
            if sum > 0.0 {
                di_vector.scale(1.0 / (self.range_sum_data[node_idx] + sum));
            }
        } else {
            let bounding_box = self.get_box(index, point_store);
            di_vector.assign_as_probability_of_cut(&bounding_box, point);
        };
    }

    fn modify_in_place_probability_of_cut_di_vector_with_missing_coordinates(
        &self,
        index: usize,
        point: &[f32],
        missing_coordinates: &[bool],
        point_store: &dyn PointStore,
        di_vector: &mut DiVector,
    ) {
        assert!(di_vector.high.len() == point.len());
        let node_idx: usize = self.translate(index);
        if node_idx != usize::MAX {
            let base = 2 * node_idx * self.dimensions;
            let mid = base + self.dimensions;
            let minsum: f64 = di_vector
                .low
                .iter_mut()
                .zip(&self.bounding_box_data[base..mid])
                .zip(point)
                .zip(missing_coordinates)
                .map(|(((x, &y), &z), &b)| {
                    if !b && y - z > 0.0 {
                        *x = (y - z) as f64;
                        *x
                    } else {
                        *x = 0.0;
                        *x
                    }
                })
                .sum();
            let maxsum: f64 = di_vector
                .high
                .iter_mut()
                .zip(point)
                .zip(&self.bounding_box_data[mid..mid + self.dimensions])
                .zip(missing_coordinates)
                .map(|(((x, &y), &z), &b)| {
                    if !b && y - z > 0.0 {
                        *x = (y - z) as f64;
                        *x
                    } else {
                        *x = 0.0;
                        *x
                    }
                })
                .sum();
            let sum = maxsum + minsum;
            if sum > 0.0 {
                di_vector.scale(1.0 / (self.range_sum_data[node_idx] + sum));
            }
        } else {
            let bounding_box = self.get_box(index, point_store);
            di_vector.assign_as_probability_of_cut_with_missing_coordinates(
                &bounding_box,
                point,
                missing_coordinates,
            );
        };
    }

    fn grow_node_box_pair(
        &self,
        first: &mut BoundingBox,
        second: &mut BoundingBox,
        point_store: &dyn PointStore,
        _node: usize,
        sibling: usize,
    ) {
        if self.is_leaf(sibling) {
            if self.using_transforms {
                let point =
                    &(self.project_to_tree)(point_store.get_copy(self.get_point_index(sibling)));
                (*first).check_contains_and_add_point(point);
                (*second).check_contains_and_add_point(point);
            } else {
                let point = point_store
                    .get_reference_and_offset(self.get_point_index(sibling))
                    .0;
                (*first).check_contains_and_add_point(point);
                (*second).check_contains_and_add_point(point);
            }
        } else {
            let idx: usize = self.translate(sibling);
            if idx != usize::MAX {
                let dimensions = self.dimensions;
                let base = 2 * idx * dimensions;
                (*first)
                    .check_contains_and_add_point(&self.bounding_box_data[base..base + dimensions]);
                (*second)
                    .check_contains_and_add_point(&self.bounding_box_data[base..base + dimensions]);
                (*first).check_contains_and_add_point(
                    &self.bounding_box_data[base + dimensions..base + 2 * dimensions],
                );
                (*second).check_contains_and_add_point(
                    &self.bounding_box_data[base + dimensions..base + 2 * dimensions],
                );
            } else {
                self.grow_node_box_pair(
                    first,
                    second,
                    point_store,
                    sibling,
                    self.get_left_index(sibling),
                );
                self.grow_node_box_pair(
                    first,
                    second,
                    point_store,
                    sibling,
                    self.get_right_index(sibling),
                );
            }
        }
    }

    fn grow_node_box(
        &self,
        bounding_box: &mut BoundingBox,
        point_store: &dyn PointStore,
        _node: usize,
        sibling: usize,
    ) {
        if self.is_leaf(sibling) {
            if self.using_transforms {
                let point =
                    &(self.project_to_tree)(point_store.get_copy(self.get_point_index(sibling)));
                (*bounding_box).check_contains_and_add_point(point);
            } else {
                let point = point_store
                    .get_reference_and_offset(self.get_point_index(sibling))
                    .0;
                (*bounding_box).check_contains_and_add_point(point);
            }
        } else {
            let idx: usize = self.translate(sibling.into());
            if idx != usize::MAX {
                let dimensions = self.dimensions;
                let base = 2 * idx * dimensions;
                (*bounding_box)
                    .check_contains_and_add_point(&self.bounding_box_data[base..base + dimensions]);
                (*bounding_box).check_contains_and_add_point(
                    &self.bounding_box_data[base + dimensions..base + 2 * dimensions],
                );
            } else {
                self.grow_node_box(
                    bounding_box,
                    point_store,
                    sibling,
                    self.get_left_index(sibling),
                );
                self.grow_node_box(
                    bounding_box,
                    point_store,
                    sibling,
                    self.get_right_index(sibling),
                );
            }
        }
    }

    fn get_sibling(&self, node: usize, parent: usize) -> usize {
        let mut sibling = self.left_index[parent].into();
        if node == sibling {
            sibling = self.right_index[parent].into();
        }
        sibling
    }

    fn get_leaf_point_index(&self, index: usize) -> usize {
        if !self.is_leaf(index) {
            println!(" not at a leaf");
            panic!();
        }
        self.get_point_index(index)
    }

    fn get_cut_dimension(&self, index: usize) -> usize {
        self.cut_dimension[index].into()
    }

    fn get_left_index(&self, index: usize) -> usize {
        self.left_index[index].try_into().unwrap()
    }

    fn get_right_index(&self, index: usize) -> usize {
        self.right_index[index].try_into().unwrap()
    }

    fn get_cut_value(&self, index: usize) -> f32 {
        self.cut_value[index]
    }

    fn is_leaf(&self, index: usize) -> bool {
        index != self.null_node() && index >= self.capacity
    }

    fn is_left_of(&self, index: usize, point: &[f32]) -> bool {
        let dim_idx: usize = self.cut_dimension[index].try_into().unwrap();
        point[dim_idx] <= self.cut_value[index]
    }

    fn use_path_for_box(&self) -> bool {
        self.bounding_box_cache_fraction < SWITCH_THRESHOLD
    }

    fn get_distribution(&self, index: usize) -> (usize, f32, usize, usize) {
        (
            self.cut_dimension[index].into(),
            self.cut_value[index],
            self.get_mass(self.get_left_index(index)),
            self.get_mass(self.get_right_index(index)),
        )
    }

    fn get_cut_and_children(&self, index: usize) -> (usize, f32, usize, usize) {
        if self.is_internal(index) {
            (
                self.cut_dimension[index].into(),
                self.cut_value[index],
                self.left_index[index].into(),
                self.right_index[index].into(),
            )
        } else {
            (usize::MAX, f32::MAX, usize::MAX, usize::MAX)
        }
    }

    fn set_path(&self, answer: &mut Vec<(usize, usize)>, root: usize, point: &[f32]) {
        let mut node = root;
        answer.push((root, self.null_node()));
        while !self.is_leaf(node) {
            if self.is_left_of(node, point) {
                answer.push((self.left_index[node].into(), self.right_index[node].into()));
                node = self.left_index[node].into();
            } else {
                answer.push((self.right_index[node].into(), self.left_index[node].into()));
                node = self.right_index[node].into();
            }
        }
    }

    fn null_node(&self) -> usize {
        Self::null_value(self.capacity)
    }
}
