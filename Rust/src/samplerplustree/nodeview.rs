use crate::common::divector::DiVector;
use crate::pointstore::PointStore;
use crate::samplerplustree::boundingbox::BoundingBox;
use crate::samplerplustree::nodestore::NodeStore;
use crate::visitor::visitor::VisitorInfo;

pub trait UpdatableNodeView<NS, PS>
    where
        NS: NodeStore,
        PS: PointStore,
{
    fn create(root:usize, node_store:&NS) -> Self;
    fn update_at_leaf(&mut self, point: &[f32], index: usize, node_store:&NS, point_store: &PS, visitor_info: &VisitorInfo);
    fn update_from_node_traversing_down(&mut self, point: &[f32], index: usize, node_store: &NS, point_store: &PS, visitor_info: &VisitorInfo);
    fn update_from_node_traversing_up(&mut self, point: &[f32], index: usize, node_store: &NS, point_store: &PS, visitor_info: &VisitorInfo);
    fn get_current_node(&self) -> usize;
    fn set_use_shadow_box(&mut self, node_store:&NS, point_store: &PS);
}

pub trait UpdatableMultiNodeView<NS, PS> : UpdatableNodeView<NS, PS>
    where
        NS: NodeStore,
        PS: PointStore,
{
    fn create(root:usize, node_store:&NS) -> Self;
    fn set_trigger_traversing_down(&mut self,point :&[f32],parent : usize, node_store : &NS, point_store : &PS,visitor_info : &VisitorInfo);
    fn update_view_to_parent_with_missing_coordinates(&mut self,
    parent : usize,
    point : &[f32],
    missing_coordinates : &[bool],
    node_store :&NS,
    point_store : &PS,
    visitor_info : &VisitorInfo
    );
    fn set_current_node(&mut self, index:usize);
    fn get_bounding_box(&self) -> Option<BoundingBox>;
    fn merge_paths(&mut self, parent : usize,saved_box: Option<BoundingBox>,point:&[f32],missing_coordinates:&[bool],node_store:&NS,point_store :&PS);
}



#[repr(C)]
pub struct SmallNodeView {
    current_node : usize,
    probability_of_cut: f64,
    shadow_box_probablity_of_cut : f64,
    mass: usize,
    depth : usize,
    leaf_index: usize,
    leaf_duplicate: bool,
    use_shadow_box: bool,
    current_box : Option<BoundingBox>,
    shadow_box : Option<BoundingBox>
}

impl SmallNodeView {
    pub fn get_probability_of_cut(&self) -> f64 {self.probability_of_cut}
    pub fn get_shadow_box_probability_of_cut(&self) -> f64 {self.shadow_box_probablity_of_cut}
    pub fn get_mass(&self) -> usize {self.mass}
    pub fn get_depth(&self) -> usize {self.depth}
    pub fn get_leaf_index(&self) -> usize {self.leaf_index}
    pub fn is_duplicate(&self) -> bool {self.leaf_duplicate}
    pub fn new<NS:NodeStore>(root : usize, node_store : &NS) -> Self {
        SmallNodeView {
            current_node : root,
            probability_of_cut: f64::MAX, // not feasible; but that is the point!
            shadow_box_probablity_of_cut: f64::MAX,
            mass: 0,
            depth: 0,
            leaf_index: usize::MAX,
            leaf_duplicate: false,
            use_shadow_box : false,
            current_box : None,
            shadow_box   : None
        }
    }
}


impl<NS, PS> UpdatableNodeView<NS, PS> for SmallNodeView
    where
        NS: NodeStore,
        PS: PointStore,
{
    fn create(root: usize, _node_store: &NS) -> Self {
        SmallNodeView::new(root,_node_store)
    }

    fn update_at_leaf(&mut self, point: &[f32], index: usize, node_store: &NS, point_store: &PS, visito_info:&VisitorInfo) {
        self.leaf_index = node_store.get_leaf_point_index(index);
        self.mass = node_store.get_mass(index);
        self.probability_of_cut = if point_store.is_equal(point, self.leaf_index) {
            self.leaf_duplicate = true;
            0.0
        } else {
            self.leaf_duplicate = false;
            1.0f64
        };
        if node_store.use_path_for_box() {
            self.current_box = Some(node_store.get_box(self.current_node,point_store));
        }
    }

    fn update_from_node_traversing_down(&mut self, point: &[f32], index: usize, node_store: &NS, point_store: &PS,visitor_info:&VisitorInfo) {
        if node_store.is_left_of(self.current_node,point) {
            self.current_node = node_store.get_left_index(self.current_node);
        }  else {
            self.current_node = node_store.get_right_index(self.current_node);
        }
        self.depth += 1;
    }

    fn update_from_node_traversing_up(&mut self, point: &[f32], index: usize, node_store: &NS, point_store: &PS,visitor_info:&VisitorInfo) {
        self.probability_of_cut = match &mut self.current_box{
            Some( x) => {
                let sibling = node_store.get_sibling(self.current_node,index);
                if self.use_shadow_box {
                    let z = node_store.get_box(sibling, point_store);
                    x.add_box(&z);
                    match &mut self.shadow_box {
                        Some(y) => { y.add_box(&z)},
                        None => { self.shadow_box = Some(z)},
                    }
                    self.shadow_box_probablity_of_cut = self.shadow_box.as_ref().unwrap().probability_of_cut(point);
                } else {
                    node_store.grow_node_box(x, point_store, index, sibling);
                };
                x.probability_of_cut(point)
            }
            None => node_store.get_probability_of_cut(index, point, point_store)
        };
        self.current_node = index;
        self.mass = node_store.get_mass(index);
        self.depth -= 1;
    }

    fn get_current_node(&self) -> usize {
        self.current_node
    }

    fn set_use_shadow_box(&mut self, node_store: &NS, point_store:&PS) {
        self.use_shadow_box = true;
        // we will maintain a current box since we haver to maintain a shadow box in any case
        // the shadow box is not set; it can only be set at the next level
        // when update_from_node_up() is invoked
        // we will maintain the invariant that *if the shadow is present then the current box also is present*
        self.current_box=Some(node_store.get_box(self.current_node,point_store));
    }
}

#[repr(C)]
pub struct MediumNodeView {
    current_node : usize,
    sibling: usize,
    probability_of_cut: f64,
    shadow_box_probablity_of_cut : f64,
    mass: usize,
    depth : usize,
    leaf_index: usize,
    leaf_duplicate: bool,
    use_shadow_box: bool,
    current_box : Option<BoundingBox>,
    shadow_box : Option<BoundingBox>,
    cut_dimension: usize,
    cut_value : f32,
    point_at_leaf : Vec<f32>
}

impl MediumNodeView {
    pub fn get_probability_of_cut(&self) -> f64 {self.probability_of_cut}
    pub fn get_shadow_box_probability_of_cut(&self) -> f64 {self.shadow_box_probablity_of_cut}
    pub fn get_mass(&self) -> usize {self.mass}
    pub fn get_depth(&self) -> usize {self.depth}
    pub fn get_leaf_index(&self) -> usize {self.leaf_index}
    pub fn is_duplicate(&self) -> bool {self.leaf_duplicate}
    pub fn get_cut_dimension(&self) -> usize {self.cut_dimension}
    pub fn get_cut_value(&self) -> f32 {self.cut_value}
    pub fn get_leaf_point(&self) -> Vec<f32> {self.point_at_leaf.clone()}
    pub fn new<NS:NodeStore>(root : usize, node_store : &NS) -> Self {
        let (cut_dimension,cut_value,left_child,right_child)
            = node_store.get_cut_and_children(root);
        let mass = node_store.get_mass(root);
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
            point_at_leaf: Vec::new()
        }
    }
}

impl<NS, PS> UpdatableNodeView<NS, PS> for MediumNodeView
    where
        NS: NodeStore,
        PS: PointStore,
{
    fn create(root: usize, node_store : &NS) -> Self {
        MediumNodeView::new(root,node_store)
    }

    fn update_at_leaf(&mut self, point: &[f32], index: usize, node_store: &NS, point_store: &PS, visitor_info:&VisitorInfo) {
        self.leaf_index = node_store.get_leaf_point_index(index);
        self.point_at_leaf = point_store.get_copy(self.leaf_index);
        self.mass = node_store.get_mass(index);
        self.probability_of_cut = if self.point_at_leaf.eq(point) {
            self.leaf_duplicate = true;
            0.0
        } else {
            self.leaf_duplicate = false;
            1.0f64
        };
        if node_store.use_path_for_box() {
            self.current_box = Some(BoundingBox::new(&self.point_at_leaf,&self.point_at_leaf));
        }
    }

    fn update_from_node_traversing_down(&mut self, point: &[f32], index: usize, node_store: &NS, point_store: &PS,visitor_info:&VisitorInfo) {
        if node_store.is_left_of(self.current_node,point) {
            self.current_node = node_store.get_left_index(self.current_node);
        }  else {
            self.current_node = node_store.get_right_index(self.current_node);
        }
        self.depth += 1;
    }

    fn update_from_node_traversing_up(&mut self, point: &[f32], index: usize, node_store: &NS, point_store: &PS,visitor_info:&VisitorInfo) {
        self.probability_of_cut = match &mut self.current_box{
            Some( x) => {
                self.sibling = node_store.get_sibling(self.current_node,index);
                if self.use_shadow_box {
                    let z = node_store.get_box(self.sibling, point_store);
                    x.add_box(&z);
                    match &mut self.shadow_box {
                        Some(y) => { y.add_box(&z)},
                        None => { self.shadow_box = Some(z)},
                    }
                    self.shadow_box_probablity_of_cut = self.shadow_box.as_ref().unwrap().probability_of_cut(point);
                } else {
                    node_store.grow_node_box(x, point_store, index, self.sibling);
                };
                x.probability_of_cut(point)
            }
            None => node_store.get_probability_of_cut(index, point, point_store)
        };
        self.current_node = index;
        let (cut_dimension,cut_value,left_child,right_child) = node_store.get_cut_and_children(self.current_node);
        self.cut_dimension = cut_dimension;
        self.cut_value = cut_value;
        self.mass = node_store.get_mass(index);
        self.depth -= 1;
    }


    fn get_current_node(&self) -> usize {
        self.current_node
    }

    fn set_use_shadow_box(&mut self, node_store: &NS, point_store:&PS) {
        self.use_shadow_box = true;
        // we will maintain a current box since we haver to maintain a shadow box in any case
        // the shadow box is not set; it can only be set at the next level
        // when update_from_node_up() is invoked
        // we will maintain the invariant that *if the shadow is present then the current box also is present*
        self.current_box=Some(node_store.get_box(self.current_node,point_store));
    }
}

impl<NS,PS> UpdatableMultiNodeView<NS, PS> for MediumNodeView
    where
        NS: NodeStore,
        PS: PointStore,
{
    fn create(root: usize, node_store : &NS) -> Self {
        MediumNodeView::new(root,node_store)
    }

    fn set_trigger_traversing_down(&mut self, point: &[f32], parent: usize, node_store: &NS, point_store: &PS, visitor_info : &VisitorInfo) {
        let (cut_dimension,cut_value,left_child,right_child) = node_store.get_cut_and_children(self.current_node);
        self.cut_dimension = cut_dimension;
        self.cut_value = cut_value;
    }

    fn update_view_to_parent_with_missing_coordinates(&mut self, parent: usize, point: &[f32], missing_coordinates: &[bool], node_store: &NS, point_store: &PS, visitor_info: &VisitorInfo) {
        if node_store.use_path_for_box(){
            let sibling = node_store.get_sibling(self.current_node,parent);
            node_store.grow_node_box(self.current_box.as_mut().unwrap(), point_store,parent,sibling);
            self.probability_of_cut = self.current_box.as_ref().unwrap().probability_of_cut_with_missing_coordinates(point,missing_coordinates);
        } else {
            self.probability_of_cut = node_store.get_probability_of_cut_with_missing_coordinates(parent,point,missing_coordinates,point_store);
        }
        self.current_node = parent;
    }

    fn set_current_node(&mut self, index: usize) {
        self.current_node = index;
    }

    fn get_bounding_box(&self) -> Option<BoundingBox> {
        match &self.current_box {
            Some(x) => Some(x.copy()),
            None => None,
        }
    }

    fn merge_paths(&mut self, parent: usize, saved_box: Option<BoundingBox>, point: &[f32], missing_coordinates: &[bool], node_store: &NS, point_store: &PS) {
        if node_store.use_path_for_box(){
            // both boxes, current and saved, should be present as invariant
            self.current_box.as_mut().unwrap().add_box(saved_box.as_ref().unwrap());
            self.probability_of_cut = self.current_box.as_ref().unwrap().probability_of_cut_with_missing_coordinates(point,missing_coordinates);
        } else {
            self.probability_of_cut = node_store.get_probability_of_cut_with_missing_coordinates(parent,point,missing_coordinates,point_store);
        }
        self.current_node = parent;
    }
}

#[repr(C)]
pub struct LargeNodeView {
    current_node : usize,
    sibling: usize,
    probability_of_cut: f64,
    shadow_box_probablity_of_cut : f64,
    mass: usize,
    depth : usize,
    leaf_index: usize,
    leaf_duplicate: bool,
    use_shadow_box: bool,
    current_box : Option<BoundingBox>,
    shadow_box : Option<BoundingBox>,
    cut_dimension: usize,
    cut_value : f32,
    left_child: usize,
    right_child: usize,
    point_at_leaf : Vec<f32>
}

impl LargeNodeView {
    pub fn get_probability_of_cut(&self) -> f64 {self.probability_of_cut}
    pub fn get_shadow_box_probability_of_cut(&self) -> f64 {self.shadow_box_probablity_of_cut}
    pub fn get_mass(&self) -> usize {self.mass}
    pub fn get_depth(&self) -> usize {self.depth}
    pub fn get_leaf_index(&self) -> usize {self.leaf_index}
    pub fn is_duplicate(&self) -> bool {self.leaf_duplicate}
    pub fn get_cut_dimension(&self) -> usize {self.cut_dimension}
    pub fn get_cut_value(&self) -> f32 {self.cut_value}
    pub fn get_leaf_point(&self) -> Vec<f32> {self.point_at_leaf.clone()}
    pub fn get_bounding_box(&self) -> BoundingBox {self.current_box.as_ref().unwrap().copy()}
    pub fn assign_probability_of_cut(&self,di_vector:&mut DiVector, point : &[f32]) {
        di_vector.assign_as_probability_of_cut(self.current_box.as_ref().unwrap(),point)
    }
    pub fn assign_probability_of_cut_shadow_box(&self,di_vector:&mut DiVector, point : &[f32]) {
        assert!(self.use_shadow_box, "shadow box not in use");
        di_vector.assign_as_probability_of_cut(self.shadow_box.as_ref().unwrap(),point)
    }
    pub fn new<NS: NodeStore>(root:usize,node_store:&NS) -> Self {
        let (cut_dimension, cut_value, left_child, right_child)
            = node_store.get_cut_and_children(root);
        let mass = node_store.get_mass(root);
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
            point_at_leaf: Vec::new()
        }
    }
}

impl<NS, PS> UpdatableNodeView<NS, PS> for LargeNodeView
    where
        NS: NodeStore,
        PS: PointStore,
{
    fn create(root: usize, node_store : &NS) -> Self {
        LargeNodeView::new(root,node_store)
    }

    fn update_at_leaf(&mut self, point: &[f32], index: usize, node_store: &NS, point_store: &PS, visitor_info:&VisitorInfo) {
        self.leaf_index = node_store.get_leaf_point_index(index);
        self.point_at_leaf = point_store.get_copy(self.leaf_index);
        self.mass = node_store.get_mass(index);
        self.probability_of_cut = if self.point_at_leaf.eq(point) {
            self.leaf_duplicate = true;
            0.0
        } else {
            self.leaf_duplicate = false;
            1.0f64
        };
        self.current_box = Some(BoundingBox::new(&self.point_at_leaf,&self.point_at_leaf));
    }

    fn update_from_node_traversing_down(&mut self, point: &[f32], index: usize, node_store: &NS, point_store: &PS,visitor_info:&VisitorInfo) {
        if point[self.cut_dimension] <= self.cut_value {
            self.current_node = self.left_child;
        }  else {
            self.current_node = self.right_child;
        }
        let (cut_dimension,cut_value,left_child,right_child) = node_store.get_cut_and_children(self.current_node);
        self.cut_dimension = cut_dimension;
        self.cut_value = cut_value;
        self.left_child = left_child;
        self.right_child = right_child;
        self.depth += 1;
        self.mass =node_store.get_mass(self.current_node);
    }

    fn update_from_node_traversing_up(&mut self, point: &[f32], index: usize, node_store: &NS, point_store: &PS,visitor_info:&VisitorInfo) {
        self.sibling = node_store.get_sibling(self.current_node, index);
        if self.use_shadow_box {
            let z = node_store.get_box(self.sibling, point_store);
            self.current_box.as_mut().unwrap().add_box(&z);
            match &mut self.shadow_box {
                Some(y) => { y.add_box(&z) },
                None => { self.shadow_box = Some(z) },
            }
            self.shadow_box_probablity_of_cut = self.shadow_box.as_ref().unwrap().probability_of_cut(point);
        } else {
            node_store.grow_node_box(self.current_box.as_mut().unwrap(), point_store, index, self.sibling);
        };
        self.probability_of_cut = self.current_box.as_ref().unwrap().probability_of_cut(point);

        self.current_node = index;
        let (cut_dimension, cut_value, left_child, right_child) = node_store.get_cut_and_children(self.current_node);
        self.cut_dimension = cut_dimension;
        self.cut_value = cut_value;
        self.left_child = left_child;
        self.right_child = right_child;
        self.mass = node_store.get_mass(index);
        self.depth -= 1;
    }

    fn get_current_node(&self) -> usize {
        self.current_node
    }

    fn set_use_shadow_box(&mut self, node_store: &NS, point_store:&PS) {
        self.use_shadow_box = true;
        // note that the current box is always maintained
    }
}

impl<NS,PS> UpdatableMultiNodeView<NS, PS> for LargeNodeView
    where
        NS: NodeStore,
        PS: PointStore,
{
    fn create(root: usize, node_store : &NS) -> Self {
        LargeNodeView::new(root,node_store)
    }

    fn set_trigger_traversing_down(&mut self, point: &[f32], parent: usize, node_store: &NS, point_store: &PS, visitor_info : &VisitorInfo) {
        let (cut_dimension,cut_value,left_child,right_child) = node_store.get_cut_and_children(self.current_node);
        self.cut_dimension = cut_dimension;
        self.cut_value = cut_value;
    }

    fn update_view_to_parent_with_missing_coordinates(&mut self, parent: usize, point: &[f32], missing_coordinates: &[bool], node_store: &NS, point_store: &PS, visitor_info: &VisitorInfo) {
        if node_store.use_path_for_box(){
            let sibling = node_store.get_sibling(self.current_node,parent);
            node_store.grow_node_box(self.current_box.as_mut().unwrap(), point_store,parent,sibling);
            self.probability_of_cut = self.current_box.as_ref().unwrap().probability_of_cut_with_missing_coordinates(point,missing_coordinates);
        } else {
            self.probability_of_cut = node_store.get_probability_of_cut_with_missing_coordinates(parent,point,missing_coordinates,point_store);
        }
        self.current_node = parent;
    }

    fn set_current_node(&mut self, index: usize) {
        self.current_node = index;
    }

    fn get_bounding_box(&self) -> Option<BoundingBox> {
        match &self.current_box {
            Some(x) => Some(x.copy()),
            None => None,
        }
    }

    fn merge_paths(&mut self, parent: usize, saved_box: Option<BoundingBox>, point: &[f32], missing_coordinates: &[bool], node_store: &NS, point_store: &PS) {
        if node_store.use_path_for_box(){
            // both boxes, current and saved, should be present as invariant
            self.current_box.as_mut().unwrap().add_box(saved_box.as_ref().unwrap());
            self.probability_of_cut = self.current_box.as_ref().unwrap().probability_of_cut_with_missing_coordinates(point,missing_coordinates);
        } else {
            self.probability_of_cut = node_store.get_probability_of_cut_with_missing_coordinates(parent,point,missing_coordinates,point_store);
        }
        self.current_node = parent;
    }
}
