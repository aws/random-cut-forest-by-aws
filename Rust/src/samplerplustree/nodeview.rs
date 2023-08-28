use std::hash::Hash;
use crate::{
    common::divector::DiVector,
    pointstore::PointStore,
    samplerplustree::{boundingbox::BoundingBox, nodestore::{BasicStore, NodeStore}},
    visitor::visitor::VisitorInfo,
    types::Result
};


pub trait UpdatableNodeView <Label: Copy + Sync, Attributes : Copy + Sync+ Hash + Eq + Send>{
    fn create<NS : NodeStore<Label,Attributes>>(root: usize, node_store: &NS) -> Self;
    fn update_at_leaf<PS: PointStore<Label,Attributes>, NS: NodeStore<Label,Attributes>>(
        &mut self,
        point: &[f32],
        index: usize,
        node_store: &NS,
        point_store: &PS,
        visitor_info: &VisitorInfo,
    ) -> Result<()>;
    fn update_from_node_traversing_down< PS: PointStore<Label,Attributes>, NS: NodeStore<Label,Attributes>>(
        &mut self,
        point: &[f32],
        index: usize,
        node_store: &NS,
        point_store: &PS,
        visitor_info: &VisitorInfo,
    ) -> Result<()>;
    fn update_from_node_traversing_up< PS: PointStore<Label,Attributes>, NS: NodeStore<Label,Attributes>>(
        &mut self,
        point: &[f32],
        index: usize,
        node_store: &NS,
        point_store: &PS,
        visitor_info: &VisitorInfo,
    ) -> Result<()>;
    fn current_node(&self) -> usize;
    fn set_use_shadow_box< PS: PointStore<Label,Attributes>, NS: NodeStore<Label,Attributes>>(&mut self, node_store: &NS, point_store: &PS) -> Result<()>;
}

pub trait UpdatableMultiNodeView<Label: Copy + Sync, Attributes : Copy + Sync+ Hash + Eq + Send> : UpdatableNodeView<Label, Attributes> {
    fn create<NS: NodeStore<Label,Attributes>>(root: usize, node_store: &NS) -> Self;
    fn set_trigger_traversing_down<PS: PointStore<Label,Attributes>, NS: NodeStore<Label,Attributes>>(
        &mut self,
        point: &[f32],
        parent: usize,
        node_store: &NS,
        point_store: &PS,
        visitor_info: &VisitorInfo,
    );
    fn update_view_to_parent_with_missing_coordinates<PS: PointStore<Label,Attributes>, NS: NodeStore<Label,Attributes>>(
        &mut self,
        parent: usize,
        point: &[f32],
        missing_coordinates: &[bool],
        node_store: &NS,
        point_store: &PS,
        visitor_info: &VisitorInfo,
    ) -> Result<()>;
    fn set_current_node(&mut self, index: usize);
    fn bounding_box(&self) -> Option<BoundingBox>;
    fn merge_paths<PS: PointStore<Label,Attributes>, NS: NodeStore<Label,Attributes>>(
        &mut self,
        parent: usize,
        saved_box: Option<BoundingBox>,
        point: &[f32],
        missing_coordinates: &[bool],
        node_store: &NS,
        point_store: &PS,
    ) -> Result<()>;
}

#[repr(C)]
pub struct SmallNodeView {
    current_node: usize,
    probability_of_cut: f64,
    shadow_box_probability_of_cut: f64,
    mass: usize,
    depth: usize,
    leaf_index: usize,
    leaf_duplicate: bool,
    use_shadow_box: bool,
    current_box: Option<BoundingBox>,
    shadow_box: Option<BoundingBox>,
}

impl SmallNodeView {
    pub fn probability_of_cut(&self) -> f64 {
        self.probability_of_cut
    }
    pub fn shadow_box_probability_of_cut(&self) -> f64 {
        self.shadow_box_probability_of_cut
    }
    pub fn mass(&self) -> usize {
        self.mass
    }
    pub fn depth(&self) -> usize {
        self.depth
    }
    pub fn leaf_index(&self) -> usize {
        self.leaf_index
    }
    pub fn is_duplicate(&self) -> bool {
        self.leaf_duplicate
    }
    pub fn new<Label: Copy + Sync, Attributes : Copy + Sync+ Hash + Eq + Send, NS: NodeStore<Label,Attributes>>(root: usize, _node_store: &NS) -> Self {
        SmallNodeView {
            current_node: root,
            probability_of_cut: f64::MAX, // not feasible; but that is the point!
            shadow_box_probability_of_cut: f64::MAX,
            mass: 0,
            depth: 0,
            leaf_index: usize::MAX,
            leaf_duplicate: false,
            use_shadow_box: false,
            current_box: None,
            shadow_box: None,
        }
    }
}

impl<Label: Copy + Sync, Attributes : Copy + Sync+ Hash + Eq + Send> UpdatableNodeView<Label,Attributes> for SmallNodeView {

    fn create<NS: NodeStore<Label,Attributes>>(root: usize, _node_store: &NS) -> Self {
        SmallNodeView::new(root, _node_store)
    }

    fn update_at_leaf< PS: PointStore<Label,Attributes>, NS: NodeStore<Label,Attributes>>(
        &mut self,
        point: &[f32],
        index: usize,
        node_store: &NS,
        point_store: &PS,
        _visitor_info: &VisitorInfo,
    ) ->Result<()>{
        self.leaf_index = node_store.leaf_point_index(index)?;
        self.mass = node_store.mass(index);
        self.probability_of_cut = if point_store.is_equal(point, self.leaf_index)? {
            self.leaf_duplicate = true;
            0.0
        } else {
            self.leaf_duplicate = false;
            1.0f64
        };
        if node_store.use_path_for_box() {
            self.current_box = Some(node_store.bounding_box(self.current_node, point_store)?);
        }
        Ok(())
    }

    fn update_from_node_traversing_down< PS: PointStore<Label,Attributes>, NS: NodeStore<Label,Attributes>>(
        &mut self,
        point: &[f32],
        _index: usize,
        node_store: &NS,
        _point_store: &PS,
        _visitor_info: &VisitorInfo,
    ) -> Result<()>{
        if node_store.is_left_of(self.current_node, point) {
            self.current_node = node_store.left_index(self.current_node);
        } else {
            self.current_node = node_store.right_index(self.current_node);
        }
        self.depth += 1;
        Ok(())
    }

    fn update_from_node_traversing_up<PS: PointStore<Label,Attributes>, NS: NodeStore<Label,Attributes>>(
        &mut self,
        point: &[f32],
        index: usize,
        node_store: &NS,
        point_store: &PS,
        _visitor_info: &VisitorInfo,
    ) -> Result<()>{
        self.probability_of_cut = match &mut self.current_box {
            Some(x) => {
                let sibling = node_store.sibling(self.current_node, index);
                if self.use_shadow_box {
                    let z = node_store.bounding_box(sibling, point_store)?;
                    x.add_box(&z);
                    match &mut self.shadow_box {
                        Some(y) => y.add_box(&z),
                        None => self.shadow_box = Some(z),
                    }
                    self.shadow_box_probability_of_cut =
                        self.shadow_box.as_ref().unwrap().probability_of_cut(point);
                } else {
                    node_store.grow_node_box(x, point_store, index, sibling)?;
                };
                x.probability_of_cut(point)
            }
            None => node_store.probability_of_cut(index, point, point_store)?,
        };
        self.current_node = index;
        self.mass = node_store.mass(index);
        self.depth -= 1;
        Ok(())
    }

    fn current_node(&self) -> usize {
        self.current_node
    }

    fn set_use_shadow_box< PS: PointStore<Label,Attributes>, NS: NodeStore<Label,Attributes>>(&mut self, node_store: &NS, point_store: &PS) -> Result<()>{
        self.use_shadow_box = true;
        // we will maintain a current box since we haver to maintain a shadow box in any case
        // the shadow box is not set; it can only be set at the next level
        // when update_from_node_up() is invoked
        // we will maintain the invariant that *if the shadow is present then the current box also is present*
        self.current_box = Some(node_store.bounding_box(self.current_node, point_store)?);
        Ok(())
    }
}

#[repr(C)]
pub struct MediumNodeView {
    current_node: usize,
    sibling: usize,
    probability_of_cut: f64,
    shadow_box_probablity_of_cut: f64,
    mass: usize,
    depth: usize,
    leaf_index: usize,
    leaf_duplicate: bool,
    use_shadow_box: bool,
    current_box: Option<BoundingBox>,
    shadow_box: Option<BoundingBox>,
    cut_dimension: usize,
    cut_value: f32,
    point_at_leaf: Vec<f32>,
}

impl MediumNodeView {
    pub fn probability_of_cut(&self) -> f64 {
        self.probability_of_cut
    }
    pub fn shadow_box_probability_of_cut(&self) -> f64 {
        self.shadow_box_probablity_of_cut
    }
    pub fn mass(&self) -> usize {
        self.mass
    }
    pub fn depth(&self) -> usize {
        self.depth
    }
    pub fn leaf_index(&self) -> usize {
        self.leaf_index
    }
    pub fn is_duplicate(&self) -> bool {
        self.leaf_duplicate
    }
    pub fn cut_dimension(&self) -> usize {
        self.cut_dimension
    }
    pub fn cut_value(&self) -> f32 {
        self.cut_value
    }
    pub fn leaf_point(&self) -> Vec<f32> {
        self.point_at_leaf.clone()
    }
    pub fn new<Label: Copy + Sync, Attributes : Copy + Sync>(root: usize, cut_dimension : usize, cut_value: f32, mass: usize ) -> Self {
        Self {
            current_node: root,
            sibling: usize::MAX,
            probability_of_cut: f64::MAX,
            shadow_box_probablity_of_cut: f64::MAX,
            mass,
            depth: 0,
            leaf_index: usize::MAX,
            leaf_duplicate: false,
            use_shadow_box: false,
            current_box: None,
            shadow_box: None,
            cut_dimension,
            cut_value,
            point_at_leaf: Vec::new(),
        }
    }
}

impl<Label: Copy + Sync, Attributes : Copy + Sync+ Hash + Eq + Send> UpdatableNodeView<Label,Attributes> for MediumNodeView {
    fn create<NS: NodeStore<Label,Attributes>>(root: usize, node_store: &NS) -> Self {
        let (cut_dimension, cut_value, _left_child, _right_child) =
            node_store.cut_and_children(root);
        let mass = node_store.mass(root);
        Self {
            current_node: root,
            sibling: usize::MAX,
            probability_of_cut: f64::MAX,
            shadow_box_probablity_of_cut: f64::MAX,
            mass,
            depth: 0,
            leaf_index: usize::MAX,
            leaf_duplicate: false,
            use_shadow_box: false,
            current_box: None,
            shadow_box: None,
            cut_dimension,
            cut_value,
            point_at_leaf: Vec::new(),
        }
    }

    fn update_at_leaf<PS: PointStore<Label,Attributes>, NS: NodeStore<Label,Attributes>>(
        &mut self,
        point: &[f32],
        index: usize,
        node_store: &NS,
        point_store: &PS,
        _visitor_info: &VisitorInfo,
    ) -> Result<()>{
        self.leaf_index = node_store.leaf_point_index(index)?;
        self.point_at_leaf = point_store.copy(self.leaf_index)?;
        self.mass = node_store.mass(index);
        self.probability_of_cut = if self.point_at_leaf.eq(point) {
            self.leaf_duplicate = true;
            0.0
        } else {
            self.leaf_duplicate = false;
            1.0f64
        };
        if node_store.use_path_for_box() {
            self.current_box = Some(BoundingBox::new(&self.point_at_leaf, &self.point_at_leaf)?);
        }
        Ok(())
    }

    fn update_from_node_traversing_down<PS: PointStore<Label,Attributes>, NS: NodeStore<Label,Attributes>>(
        &mut self,
        point: &[f32],
        _index: usize,
        node_store: &NS,
        _point_store: &PS,
        _visitor_info: &VisitorInfo,
    ) -> Result<()>{
        if node_store.is_left_of(self.current_node, point) {
            self.current_node = node_store.left_index(self.current_node);
        } else {
            self.current_node = node_store.right_index(self.current_node);
        }
        self.depth += 1;
        Ok(())
    }

    fn update_from_node_traversing_up<PS: PointStore<Label,Attributes>, NS: NodeStore<Label,Attributes>>(
        &mut self,
        point: &[f32],
        index: usize,
        node_store: &NS,
        point_store: &PS,
        _visitor_info: &VisitorInfo,
    ) -> Result<()>{
        self.probability_of_cut = match &mut self.current_box {
            Some(x) => {
                self.sibling = node_store.sibling(self.current_node, index);
                if self.use_shadow_box {
                    let z = node_store.bounding_box(self.sibling, point_store)?;
                    x.add_box(&z);
                    match &mut self.shadow_box {
                        Some(y) => y.add_box(&z),
                        None => self.shadow_box = Some(z),
                    }
                    self.shadow_box_probablity_of_cut =
                        self.shadow_box.as_ref().unwrap().probability_of_cut(point);
                } else {
                    node_store.grow_node_box(x, point_store, index, self.sibling)?;
                };
                x.probability_of_cut(point)
            }
            None => node_store.probability_of_cut(index, point, point_store)?,
        };
        self.current_node = index;
        let (cut_dimension, cut_value, _left_child, _right_child) =
            node_store.cut_and_children(self.current_node);
        self.cut_dimension = cut_dimension;
        self.cut_value = cut_value;
        self.mass = node_store.mass(index);
        self.depth -= 1;
        Ok(())
    }

    fn current_node(&self) -> usize {
        self.current_node
    }

    fn set_use_shadow_box<PS: PointStore<Label,Attributes>, NS: NodeStore<Label,Attributes>>(&mut self, node_store: &NS, point_store: &PS) -> Result<()>{
        self.use_shadow_box = true;
        // we will maintain a current box since we haver to maintain a shadow box in any case
        // the shadow box is not set; it can only be set at the next level
        // when update_from_node_up() is invoked
        // we will maintain the invariant that *if the shadow is present then the current box also is present*
        self.current_box = Some(node_store.bounding_box(self.current_node, point_store)?);
        Ok(())
    }
}

impl<Label: Copy + Sync, Attributes : Copy + Sync+ Hash + Eq + Send> UpdatableMultiNodeView<Label,Attributes> for MediumNodeView {
    fn create<NS: NodeStore<Label,Attributes>>(root: usize, node_store: &NS) -> Self {
        let (cut_dimension,cut_value,_left_child,_right_child) = node_store.cut_and_children(root);
        let mass = node_store.mass(root);
        MediumNodeView::new::<Label,Attributes>(root, cut_dimension,cut_value,mass)
    }

    fn set_trigger_traversing_down< PS: PointStore<Label,Attributes>, NS: NodeStore<Label,Attributes>>(
        &mut self,
        _point: &[f32],
        _parent: usize,
        node_store: &NS,
        _point_store: &PS,
        _visitor_info: &VisitorInfo,
    ) {
        let (cut_dimension, cut_value, _left_child, _right_child) =
            node_store.cut_and_children(self.current_node);
        self.cut_dimension = cut_dimension;
        self.cut_value = cut_value;
    }

    fn update_view_to_parent_with_missing_coordinates< PS: PointStore<Label,Attributes>, NS: NodeStore<Label,Attributes>>(
        &mut self,
        parent: usize,
        point: &[f32],
        missing_coordinates: &[bool],
        node_store: &NS,
        point_store: &PS,
        _visitor_info: &VisitorInfo,
    ) -> Result<()>{
        if node_store.use_path_for_box() {
            let sibling = node_store.sibling(self.current_node, parent);
            node_store.grow_node_box(
                self.current_box.as_mut().unwrap(),
                point_store,
                parent,
                sibling,
            )?;
            self.probability_of_cut = self
                .current_box
                .as_ref()
                .unwrap()
                .probability_of_cut_with_missing_coordinates(point, missing_coordinates);
        } else {
            self.probability_of_cut = node_store.probability_of_cut_with_missing_coordinates(
                parent,
                point,
                missing_coordinates,
                point_store,
            )?;
        }
        self.current_node = parent;
        Ok(())
    }

    fn set_current_node(&mut self, index: usize) {
        self.current_node = index;
    }

    fn bounding_box(&self) -> Option<BoundingBox> {
        match &self.current_box {
            Some(x) => Some(x.clone()),
            None => None,
        }
    }

    fn merge_paths<PS: PointStore<Label,Attributes>, NS: NodeStore<Label,Attributes>>(
        &mut self,
        parent: usize,
        saved_box: Option<BoundingBox>,
        point: &[f32],
        missing_coordinates: &[bool],
        node_store: &NS,
        point_store: &PS,
    ) -> Result<()>{
        if node_store.use_path_for_box() {
            // both boxes, current and saved, should be present as invariant
            self.current_box
                .as_mut()
                .unwrap()
                .add_box(saved_box.as_ref().unwrap());
            self.probability_of_cut = self
                .current_box
                .as_ref()
                .unwrap()
                .probability_of_cut_with_missing_coordinates(point, missing_coordinates);
        } else {
            self.probability_of_cut = node_store.probability_of_cut_with_missing_coordinates(
                parent,
                point,
                missing_coordinates,
                point_store,
            )?;
        }
        self.current_node = parent;
        Ok(())
    }
}

#[repr(C)]
pub struct LargeNodeView {
    current_node: usize,
    sibling: usize,
    probability_of_cut: f64,
    shadow_box_probablity_of_cut: f64,
    mass: usize,
    depth: usize,
    leaf_index: usize,
    leaf_duplicate: bool,
    use_shadow_box: bool,
    current_box: Option<BoundingBox>,
    shadow_box: Option<BoundingBox>,
    cut_dimension: usize,
    cut_value: f32,
    left_child: usize,
    right_child: usize,
    point_at_leaf: Vec<f32>,
}

impl LargeNodeView {
    pub fn probability_of_cut(&self) -> f64 {
        self.probability_of_cut
    }
    pub fn shadow_box_probability_of_cut(&self) -> f64 {
        self.shadow_box_probablity_of_cut
    }
    pub fn mass(&self) -> usize {
        self.mass
    }
    pub fn depth(&self) -> usize {
        self.depth
    }
    pub fn leaf_index(&self) -> usize {
        self.leaf_index
    }
    pub fn is_duplicate(&self) -> bool {
        self.leaf_duplicate
    }
    pub fn cut_dimension(&self) -> usize {
        self.cut_dimension
    }
    pub fn cut_value(&self) -> f32 {
        self.cut_value
    }
    pub fn leaf_point(&self) -> Vec<f32> {
        self.point_at_leaf.clone()
    }
    pub fn bounding_box(&self) -> Option<BoundingBox> {
        match &self.current_box {
            Some(x) => Some(x.clone()),
            None => None,
        }
    }

    pub fn shadow_box(&self) -> Option<BoundingBox> {
        match &self.shadow_box {
            Some(x) => Some(x.clone()),
            None => None,
        }
    }
    pub fn assign_probability_of_cut(&self, di_vector: &mut DiVector, point: &[f32]) {
        di_vector.assign_as_probability_of_cut(self.current_box.as_ref().unwrap(), point)
    }
    pub fn assign_probability_of_cut_shadow_box(&self, di_vector: &mut DiVector, point: &[f32]) {
        assert!(self.use_shadow_box, "shadow box not in use");
        di_vector.assign_as_probability_of_cut(self.shadow_box.as_ref().unwrap(), point)
    }

    pub fn new<Label: Copy + Sync, Attributes : Copy + Sync>(root: usize, cut_dimension : usize, cut_value : f32,
                                                             left_child : usize, right_child: usize, mass: usize) -> Self {
        Self {
            current_node: root,
            sibling: usize::MAX,
            probability_of_cut: f64::MAX,
            shadow_box_probablity_of_cut: f64::MAX,
            mass,
            depth: 0,
            leaf_index: usize::MAX,
            leaf_duplicate: false,
            use_shadow_box: false,
            current_box: None,
            shadow_box: None,
            cut_dimension,
            cut_value,
            left_child,
            right_child,
            point_at_leaf: Vec::new(),
        }
    }
}

impl<Label: Copy + Sync, Attributes : Copy + Sync+ Hash + Eq + Send> UpdatableNodeView<Label,Attributes> for LargeNodeView {
    fn create<NS: NodeStore<Label,Attributes>>(root: usize, node_store: &NS) -> Self {
        let (cut_dimension,cut_value,left_child,right_child) = node_store.cut_and_children(root);
        let mass = node_store.mass(root);
        LargeNodeView::new::<Label,Attributes>(root, cut_dimension,cut_value,left_child,right_child,mass)
    }

    fn update_at_leaf< PS: PointStore<Label,Attributes>, NS: NodeStore<Label,Attributes>>(
        &mut self,
        point: &[f32],
        index: usize,
        node_store: &NS,
        point_store: &PS,
        _visitor_info: &VisitorInfo,
    ) -> Result<()>{
        self.leaf_index = node_store.leaf_point_index(index)?;
        self.point_at_leaf = point_store.copy(self.leaf_index)?;
        self.mass = node_store.mass(index);
        self.probability_of_cut = if self.point_at_leaf.eq(point) {
            self.leaf_duplicate = true;
            0.0
        } else {
            self.leaf_duplicate = false;
            1.0f64
        };
        self.current_box = Some(BoundingBox::new(&self.point_at_leaf, &self.point_at_leaf)?);
        Ok(())
    }

    fn update_from_node_traversing_down<PS: PointStore<Label,Attributes>, NS: NodeStore<Label,Attributes>>(
        &mut self,
        point: &[f32],
        _index: usize,
        node_store: &NS,
        _point_store: &PS,
        _visitor_info: &VisitorInfo,
    ) -> Result<()>{
        if point[self.cut_dimension] <= self.cut_value {
            self.current_node = self.left_child;
        } else {
            self.current_node = self.right_child;
        }
        let (cut_dimension, cut_value, left_child, right_child) =
            node_store.cut_and_children(self.current_node);
        self.cut_dimension = cut_dimension;
        self.cut_value = cut_value;
        self.left_child = left_child;
        self.right_child = right_child;
        self.depth += 1;
        self.mass = node_store.mass(self.current_node);
        Ok(())
    }

    fn update_from_node_traversing_up< PS: PointStore<Label,Attributes>, NS: NodeStore<Label,Attributes>>(
        &mut self,
        point: &[f32],
        index: usize,
        node_store: &NS,
        point_store: &PS,
        _visitor_info: &VisitorInfo,
    ) -> Result<()>{
        self.sibling = node_store.sibling(self.current_node, index);
        if self.use_shadow_box {
            let z = node_store.bounding_box(self.sibling, point_store)?;
            self.current_box.as_mut().unwrap().add_box(&z);
            match &mut self.shadow_box {
                Some(y) => y.add_box(&z),
                None => self.shadow_box = Some(z),
            }
            self.shadow_box_probablity_of_cut =
                self.shadow_box.as_ref().unwrap().probability_of_cut(point);
        } else {
            node_store.grow_node_box(
                self.current_box.as_mut().unwrap(),
                point_store,
                index,
                self.sibling,
            )?;
        };
        self.probability_of_cut = self.current_box.as_ref().unwrap().probability_of_cut(point);

        self.current_node = index;
        let (cut_dimension, cut_value, left_child, right_child) =
            node_store.cut_and_children(self.current_node);
        self.cut_dimension = cut_dimension;
        self.cut_value = cut_value;
        self.left_child = left_child;
        self.right_child = right_child;
        self.mass = node_store.mass(index);
        self.depth -= 1;
        Ok(())
    }

    fn current_node(&self) -> usize {
        self.current_node
    }

    fn set_use_shadow_box< PS: PointStore<Label,Attributes>, NS: NodeStore<Label,Attributes>>(&mut self, _node_store: &NS, _point_store: &PS) -> Result<()>{
        self.use_shadow_box = true;
        // note that the current box is always maintained
        Ok(())
    }
}

impl<Label: Copy + Sync, Attributes : Copy + Sync+ Hash + Eq + Send,> UpdatableMultiNodeView<Label,Attributes> for LargeNodeView {
    fn create<NS: NodeStore<Label,Attributes>>(root: usize, node_store: &NS) -> Self {
        let (cut_dimension,cut_value,left_child,right_child) = node_store.cut_and_children(root);
        let mass = node_store.mass(root);
        LargeNodeView::new::<Label,Attributes>(root, cut_dimension,cut_value,left_child,right_child,mass)
    }

    fn set_trigger_traversing_down< PS: PointStore<Label,Attributes>, NS: NodeStore<Label,Attributes>>(
        &mut self,
        _point: &[f32],
        _parent: usize,
        node_store: &NS,
        _point_store: &PS,
        _visitor_info: &VisitorInfo,
    ) {
        let (cut_dimension, cut_value, _left_child, _right_child) =
            node_store.cut_and_children(self.current_node);
        self.cut_dimension = cut_dimension;
        self.cut_value = cut_value;
    }

    fn update_view_to_parent_with_missing_coordinates<PS: PointStore<Label,Attributes>, NS: NodeStore<Label,Attributes>>(
        &mut self,
        parent: usize,
        point: &[f32],
        missing_coordinates: &[bool],
        node_store: &NS,
        point_store: &PS,
        _visitor_info: &VisitorInfo,
    ) -> Result<()> {
        if node_store.use_path_for_box() {
            let sibling = node_store.sibling(self.current_node, parent);
            node_store.grow_node_box(
                self.current_box.as_mut().unwrap(),
                point_store,
                parent,
                sibling,
            )?;
            self.probability_of_cut = self
                .current_box
                .as_ref()
                .unwrap()
                .probability_of_cut_with_missing_coordinates(point, missing_coordinates);
        } else {
            self.probability_of_cut = node_store.probability_of_cut_with_missing_coordinates(
                parent,
                point,
                missing_coordinates,
                point_store,
            )?;
        }
        self.current_node = parent;
        Ok(())
    }

    fn set_current_node(&mut self, index: usize) {
        self.current_node = index;
    }

    fn bounding_box(&self) -> Option<BoundingBox> {
        match &self.current_box {
            Some(x) => Some(x.clone()),
            None => None,
        }
    }

    fn merge_paths< PS: PointStore<Label,Attributes>, NS: NodeStore<Label,Attributes>>(
        &mut self,
        parent: usize,
        saved_box: Option<BoundingBox>,
        point: &[f32],
        missing_coordinates: &[bool],
        node_store: &NS,
        point_store: &PS,
    ) -> Result <()>{
        if node_store.use_path_for_box() {
            // both boxes, current and saved, should be present as invariant
            self.current_box
                .as_mut()
                .unwrap()
                .add_box(saved_box.as_ref().unwrap());
            self.probability_of_cut = self
                .current_box
                .as_ref()
                .unwrap()
                .probability_of_cut_with_missing_coordinates(point, missing_coordinates);
        } else {
            self.probability_of_cut = node_store.probability_of_cut_with_missing_coordinates(
                parent,
                point,
                missing_coordinates,
                point_store,
            )?;
        }
        self.current_node = parent;
        Ok(())
    }
}
