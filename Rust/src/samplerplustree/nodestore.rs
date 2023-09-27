use std::{collections::HashMap, fmt::Debug, mem};
use std::hash::Hash;
use crate::types::Result;

use crate::{
    common::{divector::DiVector, intervalstoremanager::IntervalStoreManager},
    pointstore::PointStore,
    samplerplustree::{boundingbox::BoundingBox, cut::Cut},
    types::Location,
};
use crate::errors::RCFError;
use crate::util::check_argument;

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
    P: Location + Eq + Hash + Send,
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
    internal_node_manager: IntervalStoreManager<usize>,
    store_attributes: bool,
    propagate_attributes: bool,
    store_pointsum: bool,
    pointsum : Vec<f32>,
    attributes: HashMap<P,HashMap<P,N>>,
    propagated_attributes: HashMap<N,Vec<f32>>
}

const SWITCH_THRESHOLD: f64 = 0.5;

pub trait NodeStore<Label: Sync + Copy, Attributes: Sync + Copy+ Hash + Eq + Send> : BasicStore + BoxStore<Label,Attributes> {}

pub trait BasicStore {
    fn mass(&self, index: usize) -> usize;
    fn sibling(&self, node: usize, parent: usize) -> usize;
    fn leaf_point_index(&self, index: usize) -> Result<usize>;
    fn cut_dimension(&self, index: usize) -> usize;
    fn left_index(&self, index: usize) -> usize;
    fn right_index(&self, index: usize) -> usize;
    fn cut_value(&self, index: usize) -> f32;
    fn is_leaf(&self, index: usize) -> bool;
    fn is_left_of(&self, index: usize, point: &[f32]) -> bool;
    fn use_path_for_box(&self) -> bool;
    fn distribution(&self, index: usize) -> (usize, f32, usize, usize);
    fn cut_and_children(&self, index: usize) -> (usize, f32, usize, usize);
    fn set_path(&self, answer: &mut Vec<(usize, usize)>, root: usize, point: &[f32]);
    fn null_node(&self) -> usize;
    fn attribute_at_leaf(&self,point_index: usize) -> Result<Vec<(usize,usize)>>;
}

pub trait BoxStore<Label: Sync + Copy, Attributes: Sync + Copy + Hash + Eq + Send> : BasicStore {
    fn attribut_vec<PS: PointStore<Label, Attributes>>(&self, index: usize, point_store: &PS) -> Result<Vec<f32>>;
    fn recompute_attribute_vec<PS: PointStore<Label, Attributes>>(&mut self, index: usize, point_store: &PS) -> Result<()>;
    fn pointsum<PS: PointStore<Label, Attributes>>(&self, index: usize, point_store: &PS) -> Result<Vec<f32>>;
    fn recompute_pointsum<PS: PointStore<Label, Attributes>>(&mut self, index: usize, point_store: &PS) -> Result<()>;
    fn manage_ancestors_add<PS: PointStore<Label,Attributes>>(
        &mut self,
        path: &mut Vec<(usize, usize)>,
        point: &[f32],
        _point_store: &PS,
        box_resolved: bool,
    ) -> Result<()>;
    fn manage_ancestors_delete<PS: PointStore<Label, Attributes>>(
        &mut self,
        path: &mut Vec<(usize, usize)>,
        point: &[f32],
        point_store: &PS,
        box_resolved: bool,
    ) -> Result<()>;
    fn reconstruct_box<PS: PointStore<Label, Attributes>>(&self, index: usize, point_store: &PS) -> Result<BoundingBox>;
    fn check_contains_and_rebuild_box<PS: PointStore<Label, Attributes>>(
        &mut self,
        index: usize,
        point: &[f32],
        point_store: &PS,
    ) -> Result<bool>;
    fn bounding_box<PS: PointStore<Label,Attributes>>(&self, index: usize, point_store: &PS) -> Result<BoundingBox>;
    fn probability_of_cut<PS: PointStore<Label,Attributes>>(
        &self,
        index: usize,
        point: &[f32],
        point_store: &PS,
    ) -> Result<f64>;
    fn probability_of_cut_with_missing_coordinates<PS: PointStore<Label,Attributes>>(
        &self,
        index: usize,
        point: &[f32],
        missing_coordinates: &[bool],
        point_store: &PS,
    ) -> Result<f64>;
    fn modify_in_place_probability_of_cut_di_vector<PS: PointStore<Label,Attributes>>(
        &self,
        index: usize,
        point: &[f32],
        point_store: &PS,
        di_vector: &mut DiVector,
    ) -> Result<()>;
    fn modify_in_place_probability_of_cut_di_vector_with_missing_coordinates<PS: PointStore<Label,Attributes>>(
        &self,
        index: usize,
        point: &[f32],
        missing_coordinates: &[bool],
        point_store: &PS,
        di_vector: &mut DiVector,
    ) -> Result<()>;
    fn grow_node_box_pair<PS: PointStore<Label,Attributes>>(
        &self,
        first: &mut BoundingBox,
        second: &mut BoundingBox,
        point_store: &PS,
        node: usize,
        sibling: usize,
    ) -> Result<()>;
    fn grow_node_box<PS: PointStore<Label,Attributes>>(
        &self,
        bounding_box: &mut BoundingBox,
        point_store: &PS,
        node: usize,
        sibling: usize,
    ) -> Result<()>;
    fn check_left<PS: PointStore<Label,Attributes>>(
        &self,
        index: usize,
        dim: usize,
        value: f32,
        point_store: &PS
    ) -> Result<bool>;
    fn check_right<PS: PointStore<Label,Attributes>>(
        &self,
        index: usize,
        dim: usize,
        value: f32,
        point_store: &PS
    ) -> Result<bool>;
}

impl<C, P, N> VectorNodeStore<C, P, N>
where
    C: Location,
    usize: From<C>,
    P: Location+ Eq + Hash + Send,
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
        store_attributes: bool,
        store_pointsum: bool,
        propagate_attributes: bool,
        project_to_tree: fn(Vec<f32>) -> Vec<f32>,
        bounding_box_cache_fraction: f64,
    ) -> Result<Self> {
        check_argument( capacity - 1 <= N::MAX.into() ,
                " invalid parameter, increase size of N to represent {}")?;
        let cache_limit: usize = (bounding_box_cache_fraction * capacity as f64) as usize;
        let null_node = Self::null_value(capacity);
        let pointsum = if store_pointsum {
            vec![0.0f32;(capacity - 1)*dimensions]
        } else {
            Vec::new()
        };
        Ok(VectorNodeStore {
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
            attributes: HashMap::new(),
            store_attributes,
            propagate_attributes: propagate_attributes,
            store_pointsum,
            pointsum,
            propagated_attributes : HashMap::new()
        })
    }

    /// 0 is indicative of null given unsigned representation
    /// otherwise index X uses slot X-1
    pub fn invalidate_pointsum(&mut self,index: usize) -> Result<()>{
        check_argument(self.store_pointsum, "incorrct invocation")?;
        for x in self.pointsum[(index*self.dimensions).. ((index+1)*self.dimensions)].iter_mut() {
            *x=0.0;
        };
        Ok(())
    }

    pub fn add_attrib_at_leaf(&mut self,point_index: usize, point_attribute: usize) -> Result<()>{
        if self.store_attributes {
            let p: P = point_index.try_into().unwrap();
            let v: P = point_attribute.try_into().unwrap();
            if let Some(x) = self.attributes.get_mut(&p) {
                let a: usize = if let Some(y) = x.remove(&v) {
                    usize::from(y) + 1
                } else {
                    1
                };
                x.insert(v, a.try_into().unwrap());
            } else {
                let mut x = HashMap::new();
                let a: usize = 1;
                x.insert(v, a.try_into().unwrap());
                self.attributes.insert(p, x);
            };
        }
        Ok(())
    }

    pub fn del_attrib_at_leaf(&mut self,point_index: usize, point_attribute: usize) -> Result<()> {
        if self.store_attributes {
            let p: P = point_index.try_into().unwrap();
            let v: P = point_attribute.try_into().unwrap();
            if let Some(x) = self.attributes.get_mut(&p) {
                if let Some(y) = x.remove(&v) {
                    check_argument(usize::from(y) > 0, " error")?;
                    if usize::from(y) > 1 {
                        x.insert(v, (usize::from(y) - 1).try_into().unwrap());
                    }
                    return Ok(());
                }
            }
            return Err(RCFError::InvalidArgument { msg: "element should be present" });
        }
        Ok(())
    }

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

    fn box_from_data(&self, idx: usize) -> BoundingBox {
        let dimensions = self.dimensions;
        let base = 2 * idx * dimensions;
        return BoundingBox::new(
            &self.bounding_box_data[base..base + dimensions],
            &self.bounding_box_data[base + dimensions..base + 2 * dimensions],
        ).unwrap();
    }

    pub fn add_node(
        &mut self,
        parent_index: usize,
        point: &[f32],
        child: usize,
        point_index: usize,
        cut: Cut,
        saved_box: &BoundingBox,
    ) -> Result<usize> {
        let index = self.internal_node_manager.get()?.into();
        self.cut_value[index] = cut.value;
        self.cut_dimension[index] = cut.dimension.try_into().unwrap();
        if point[cut.dimension] <= cut.value {
            self.left_index[index] = self.leaf_index(point_index).try_into().unwrap();
            self.right_index[index] = child.try_into().unwrap();
        } else {
            self.left_index[index] = child.try_into().unwrap();
            self.right_index[index] = self.leaf_index(point_index).try_into().unwrap();
        }

        self.mass[index] = (self.mass(child)).try_into().unwrap();
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
        Ok(index)
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
    pub fn increase_leaf_mass(&mut self, index: usize) -> Result<()> {
        if index >= self.capacity {
            let y = index - self.capacity;
            if let Some(a) = self.hash_mass_leaves.remove(&y) {
                self.hash_mass_leaves.insert(y, a + 1);
            } else {
                self.hash_mass_leaves.insert(y, 1);
            }
            return Ok(());
        }
        Err(RCFError::InvalidArgument { msg: " incorrect call with a non-leaf index" })
    }

    pub fn decrease_leaf_mass(&mut self, index: usize) -> Result<usize> {
        check_argument(self.is_leaf(index), "incorrect leaf index")?;
        let y = index - self.capacity;
        if let Some(a) = self.hash_mass_leaves.remove(&y) {
            if a > 1 {
                self.hash_mass_leaves.insert(y, a - 1);
                Ok(a)
            } else {
                Ok(1) //default
            }
        } else {
            Ok(0)
        }
    }


    pub fn delete_internal_node(&mut self, index: usize) -> Result<()>{
        let null_node = self.null_node();

        self.left_index[index] = null_node.try_into().unwrap();
        self.right_index[index] = null_node.try_into().unwrap();
        self.mass[index] = 0.try_into().unwrap();
        if self.bounding_box_cache_fraction > 0.0 {
            self.parent_index[index] = null_node.try_into().unwrap(); // null
        }
        self.cut_dimension[index] = C::MAX;
        self.cut_value[index] = 0.0;
        if self.propagate_attributes {
            self.propagated_attributes.remove(&index.try_into().expect("incorrect state"));
        }
        self.internal_node_manager.release(index)
    }

    pub fn cut_value(&self, index: usize) -> f32 {
        self.cut_value[index]
    }

    pub fn cut_dimension(&self, index: usize) -> usize {
        self.cut_dimension[index].into()
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

    pub fn size(&self, _dimensions: usize) -> usize {
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


impl<C, P, N,Label,Attributes> BoxStore<Label,Attributes> for VectorNodeStore<C, P, N>
    where
        C: Location,
        usize: From<C>,
        P: Location+ Eq + Hash + Send,
        usize: From<P>,
        N: Location,
        usize: From<N>,
        <C as TryFrom<usize>>::Error: Debug,
        <P as TryFrom<usize>>::Error: Debug,
        <N as TryFrom<usize>>::Error: Debug,
        Label : Sync + Copy,
        Attributes: Sync + Copy+ Hash + Eq + Send,
{
    fn attribut_vec<PS: PointStore<Label, Attributes>>(&self, index: usize, point_store: &PS) -> Result<Vec<f32>>{
        check_argument(self.propagate_attributes, " enable propagation of vectors")?;
        if self.is_leaf(index) {
            let list = self.attributes.get(&(self.leaf_point_index(index)?.try_into().unwrap())).expect("incorrect state");
            check_argument(list.len() > 0, "cannot be 0")?;
            let veclist = list.iter().map(|(&x, &y)| {
                let weight : usize = y.into();
                let mut vec = point_store.attribute_vec(x.into())?;
                for z in vec.iter_mut() {
                    *z *= weight as f32;
                };
                Ok(vec)
            }).collect::<Result<Vec<Vec<f32>>>>()?;
            let mut answer =  veclist[0].clone();
            for i in 1..list.len() {
                for (x,y) in answer.iter_mut().zip(&veclist[i]) {
                    *x += *y;
                }
            }
            Ok(answer)
        } else {
            Ok(self.propagated_attributes.get(&index.try_into().unwrap()).expect("incorrect state").clone())
        }
    }

    fn recompute_attribute_vec<PS: PointStore<Label, Attributes>>(&mut self, index: usize, point_store: &PS) -> Result<()>{
        check_argument(!self.is_leaf(index), "incorrect invocation")?;
        let mut left = self.attribut_vec(self.left_index[index].into(),point_store)?;
        let right = self.attribut_vec(self.right_index[index].into(),point_store)?;
        for (x,y) in left.iter_mut().zip(right){
            *x += y;
        }
        self.propagated_attributes.insert(index.try_into().expect("incorrect state"),left);
        Ok(())
    }

    fn pointsum<PS: PointStore<Label, Attributes>>(&self, index: usize, point_store: &PS) -> Result<Vec<f32>>{
        check_argument(self.store_pointsum, " enable store_pointsum")?;
        if self.is_leaf(index) {
            let mut point = point_store.copy(self.leaf_point_index(index)?)?;
            let mass = self.mass(index);
            for x in point.iter_mut() {
                *x *= mass as f32;
            }
            Ok(point)
        } else {
            Ok(Vec::from(&self.pointsum[(index * self.dimensions)..((index + 1) * self.dimensions)]))
        }
    }

    fn recompute_pointsum<PS: PointStore<Label, Attributes>>(&mut self, index: usize, point_store: &PS) -> Result<()>{
        check_argument(!self.is_leaf(index), "incorrect invocation")?;
        let left = self.pointsum(self.left_index[index].into(),point_store)?;
        let right = self.pointsum(self.right_index[index].into(),point_store)?;
        for ((x,y),z) in self.pointsum[(index * self.dimensions)..((index + 1) * self.dimensions)].iter_mut().zip(left).zip(right) {
            *x = y + z;
        }
        Ok(())
    }

    fn manage_ancestors_add<PS: PointStore<Label, Attributes>>(
        &mut self,
        path: &mut Vec<(usize, usize)>,
        point: &[f32],
        point_store: &PS,
        box_resolved: bool,
    ) -> Result<()>{
        let mut resolved = box_resolved;
        while path.len() != 0 {
            let index = path.pop().unwrap().0;
            let val: usize = self.mass[index].into();
            self.mass[index] = (val + 1).try_into().unwrap();
            if self.store_pointsum {
                self.recompute_pointsum(index,point_store)?;
            }
            if self.propagate_attributes {
                self.recompute_attribute_vec(index,point_store)?;
            }
            if self.bounding_box_cache_fraction > 0.0 && !resolved {
                resolved = self.check_contains_and_add_point(index.into(), point);
            }
        }
        Ok(())
    }

    fn manage_ancestors_delete<PS: PointStore<Label, Attributes>>(
        &mut self,
        path: &mut Vec<(usize, usize)>,
        point: &[f32],
        point_store: &PS,
        box_resolved: bool,
    ) -> Result<()>{
        let mut resolved = box_resolved;
        while path.len() != 0 {
            let index = path.pop().unwrap().0;
            let val: usize = self.mass[index].into();
            self.mass[index] = (val - 1).try_into().unwrap();
            if self.store_pointsum {
                self.recompute_pointsum(index,point_store)?;
            }
            if self.propagate_attributes {
                self.recompute_attribute_vec(index,point_store)?;
            }
            if self.bounding_box_cache_fraction > 0.0 && !resolved {
                resolved = self.check_contains_and_rebuild_box(index, point, point_store)?;
            }
        }
        Ok(())
    }

    fn reconstruct_box<PS: PointStore<Label, Attributes>>(&self, index: usize, point_store: &PS) -> Result<BoundingBox> {
        let mut mutated_bounding_box = self.bounding_box(self.left_index[index].into(), point_store)?;
        self.grow_node_box(
            &mut mutated_bounding_box,
            point_store,
            index,
            self.right_index[index].into(),
        )?;
        Ok(mutated_bounding_box)
    }

    fn check_contains_and_rebuild_box<PS: PointStore<Label, Attributes>>(
        &mut self,
        index: usize,
        point: &[f32],
        point_store: &PS,
    ) -> Result<bool> {
        let idx = self.translate(index);
        if idx != usize::MAX {
            if !self.check_strictly_contains(index, point) {
                let mutated_bounding_box = self.reconstruct_box(index, point_store)?;
                self.copy_box_to_data(index, &mutated_bounding_box);
                return Ok(false);
            }
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn bounding_box<PS: PointStore<Label, Attributes>>(&self, index: usize, point_store: &PS) -> Result<BoundingBox> {
        if self.is_leaf(index) {
            return if self.using_transforms {
                let point =
                    &(self.project_to_tree)(point_store.copy(self.leaf_point_index(index)?)?);
                BoundingBox::new(point, point)
            } else {
                let point = point_store
                    .reference_and_offset(self.leaf_point_index(index)?)?
                    .0;
                BoundingBox::new(point, point)
            };
        } else {
            let idx: usize = self.translate(index);
            if idx != usize::MAX {
                return Ok(self.box_from_data(idx));
            }
            let mut mutated_bounding_box = self.bounding_box(self.left_index[index].into(), point_store)?;
            self.grow_node_box(
                &mut mutated_bounding_box,
                point_store,
                index,
                self.right_index[index].into(),
            )?;
            return Ok(mutated_bounding_box);
        }
    }

    fn probability_of_cut<PS: PointStore<Label, Attributes>>(
        &self,
        index: usize,
        point: &[f32],
        point_store: &PS,
    ) -> Result<f64> {
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
                return Ok(0.0);
            }
            Ok(sum as f64 / (self.range_sum_data[node_idx] + sum as f64))
        } else {
            let bounding_box = self.bounding_box(index, point_store)?;
            Ok(bounding_box.probability_of_cut(point))
        }
    }
    fn probability_of_cut_with_missing_coordinates<PS: PointStore<Label, Attributes>>(
        &self,
        index: usize,
        point: &[f32],
        missing_coordinates: &[bool],
        point_store: &PS,
    ) -> Result<f64> {
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
                return Ok(0.0);
            }
            Ok(sum as f64 / (self.range_sum_data[node_idx] + sum as f64))
        } else {
            let bounding_box = self.bounding_box(index, point_store)?;
            Ok(bounding_box.probability_of_cut_with_missing_coordinates(point, missing_coordinates))
        }
    }
    fn modify_in_place_probability_of_cut_di_vector<PS: PointStore<Label, Attributes>>(
        &self,
        index: usize,
        point: &[f32],
        point_store: &PS,
        di_vector: &mut DiVector,
    ) -> Result<()>{
        check_argument(di_vector.high.len() == point.len(), " incorrect dimensions of bounding box")?;
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
            let bounding_box = self.bounding_box(index, point_store)?;
            di_vector.assign_as_probability_of_cut(&bounding_box, point);
        };
        Ok(())
    }

    fn modify_in_place_probability_of_cut_di_vector_with_missing_coordinates<PS: PointStore<Label, Attributes>>(
        &self,
        index: usize,
        point: &[f32],
        missing_coordinates: &[bool],
        point_store: &PS,
        di_vector: &mut DiVector,
    ) -> Result<()>{
        check_argument(di_vector.high.len() == point.len(), " incorrect dimensions of bounding box")?;
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
            let bounding_box = self.bounding_box(index, point_store)?;
            di_vector.assign_as_probability_of_cut_with_missing_coordinates(
                &bounding_box,
                point,
                missing_coordinates,
            );
        };
        Ok(())
    }

    fn grow_node_box_pair<PS: PointStore<Label, Attributes>>(
        &self,
        first: &mut BoundingBox,
        second: &mut BoundingBox,
        point_store: &PS,
        _node: usize,
        sibling: usize,
    ) -> Result<()>{
        if self.is_leaf(sibling) {
            if self.using_transforms {
                let point =
                    &(self.project_to_tree)(point_store.copy(self.leaf_point_index(sibling)?)?);
                (*first).check_contains_and_add_point(point);
                (*second).check_contains_and_add_point(point);
            } else {
                let point = point_store
                    .reference_and_offset(self.leaf_point_index(sibling)?)?
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
                    self.left_index(sibling),
                )?;
                self.grow_node_box_pair(
                    first,
                    second,
                    point_store,
                    sibling,
                    self.right_index(sibling),
                )?;
            }
        }
        Ok(())
    }

    fn grow_node_box<PS: PointStore<Label, Attributes>>(
        &self,
        bounding_box: &mut BoundingBox,
        point_store: &PS,
        _node: usize,
        sibling: usize,
    ) -> Result<()>{
        if self.is_leaf(sibling) {
            if self.using_transforms {
                let point =
                    &(self.project_to_tree)(point_store.copy(self.leaf_point_index(sibling)?)?);
                (*bounding_box).check_contains_and_add_point(point);
            } else {
                let point = point_store
                    .reference_and_offset(self.leaf_point_index(sibling)?)?
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
                    self.left_index(sibling),
                )?;
                self.grow_node_box(
                    bounding_box,
                    point_store,
                    sibling,
                    self.right_index(sibling),
                )?;
            }
        }
        Ok(())
    }
    fn check_left<PS: PointStore<Label,Attributes>>(
        &self,
        index: usize,
        dim: usize,
        value: f32,
        point_store: &PS
    ) -> Result<bool> {
        if self.is_leaf(index) {
            let point = (self.project_to_tree)(point_store.copy(self.leaf_point_index(index)?)?);
            return Ok(point[dim] < value);
        }
        // both are left -- we want both to be less than value
        Ok(self.check_left(self.left_index(index), dim, value, point_store)?
            && self.check_left(self.right_index(index), dim, value, point_store)?)
    }

    fn check_right<PS: PointStore<Label,Attributes>>(
        &self,
        index: usize,
        dim: usize,
        value: f32,
        point_store: &PS
    ) -> Result<bool> {
        if self.is_leaf(index) {
            let point = (self.project_to_tree)(point_store.copy(self.leaf_point_index(index)?)?);
            return Ok(point[dim] >= value);
        }
        // both are right -- we want the subtree to be greater or equal value
        Ok(self.check_right(self.left_index(index), dim, value, point_store)?
            && self.check_right(self.right_index(index), dim, value, point_store)?)
    }
}

impl<C, P, N> BasicStore for VectorNodeStore<C, P, N>
        where
            C: Location,
            usize: From<C>,
            P: Location + Eq + Hash + Send,
            usize: From<P>,
            N: Location,
            usize: From<N>,
            <C as TryFrom<usize>>::Error: Debug,
            <P as TryFrom<usize>>::Error: Debug,
            <N as TryFrom<usize>>::Error: Debug,
{
    fn mass(&self, index: usize) -> usize {
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

    fn leaf_point_index(&self, index: usize) -> Result<usize> {
        check_argument(self.is_leaf(index), " not a leaf index")?;
        Ok(index - self.capacity)
    }

    fn sibling(&self, node: usize, parent: usize) -> usize {
        let mut sibling = self.left_index[parent].into();
        if node == sibling {
            sibling = self.right_index[parent].into();
        }
        sibling
    }


    fn cut_dimension(&self, index: usize) -> usize {
        self.cut_dimension[index].into()
    }

    fn left_index(&self, index: usize) -> usize {
        self.left_index[index].try_into().unwrap()
    }

    fn right_index(&self, index: usize) -> usize {
        self.right_index[index].try_into().unwrap()
    }

    fn cut_value(&self, index: usize) -> f32 {
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

    fn distribution(&self, index: usize) -> (usize, f32, usize, usize) {
        (
            self.cut_dimension[index].into(),
            self.cut_value[index],
            self.mass(self.left_index(index)),
            self.mass(self.right_index(index)),
        )
    }

    fn cut_and_children(&self, index: usize) -> (usize, f32, usize, usize) {
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

    fn attribute_at_leaf(&self,point_index: usize) -> Result<Vec<(usize,usize)>> {
        self.attributes.get(&point_index.try_into().expect("out of range")).expect("should be present").iter()
            .map(|(&x,&y)| {
                Ok((x.into(),y.into()))
            }).collect()
    }
}

impl<C, P, N,Label,Attributes> NodeStore<Label,Attributes> for VectorNodeStore<C, P, N>
    where
        C: Location,
        usize: From<C>,
        P: Location+ Eq + Hash + Send,
        usize: From<P>,
        N: Location,
        usize: From<N>,
        <C as TryFrom<usize>>::Error: Debug,
        <P as TryFrom<usize>>::Error: Debug,
        <N as TryFrom<usize>>::Error: Debug,
        Label : Sync + Copy,
        Attributes: Sync + Copy+ Hash + Eq + Send,
{}
