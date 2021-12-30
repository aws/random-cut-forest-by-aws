
use std::io::empty;
use crate::boundingbox::BoundingBox;
use crate::newnodestore::NewNodeStore;
use crate::newnodestore::NodeStoreView;
use crate::pointstore::PointStoreView;
use crate::rcf::Max;
use crate::visitor::{MultiVisitorDescriptor, StreamingMultiVisitor, Visitor, VisitorDescriptor};

pub struct AbstractNodeView{
    current_node_offset: usize,
    current_box : Option<BoundingBox>,
    left_box : Option<BoundingBox>,
    right_box : Option<BoundingBox>,
    direction_is_left  : bool,
    shadow_box : Option<BoundingBox>,
    leaf_index: usize,
    leaf_duplicate: bool,
    use_point_copy_for_accept: bool,
    leaf_point : Vec<f32>,
    cut_value : f32,
    maintain_shadow_box_for_accept : bool,
    use_cuts_for_accept : bool,
    use_cuts_for_trigger : bool,
    use_box_for_accept : bool,
    use_box_for_trigger : bool,
    use_child_boxes_for_accept : bool,
    use_child_boxes_for_trigger : bool,
    use_mass_distribution_for_accept: bool,
    use_mass_distribution_for_trigger: bool,
    update_node_view_between_triggers : bool,
    probability_of_cut: f64,
    cut_dimension : usize,
    shadow_box_set: bool,
    dimensions : usize,
    mass : usize,
    left_mass : usize,
    right_mass : usize,
    depth : usize
}

impl AbstractNodeView {
    pub fn new(dimensions: usize, root: usize, node_store_needs_box : bool, visitor_descriptor: VisitorDescriptor, multivisitor_descriptor: MultiVisitorDescriptor) -> Self {
        AbstractNodeView {
            current_node_offset: root,
            dimensions,
            use_point_copy_for_accept : visitor_descriptor.use_point_copy_for_accept,
            use_box_for_accept : visitor_descriptor.use_box_for_accept || node_store_needs_box,
            use_child_boxes_for_accept: visitor_descriptor.use_child_boxes_for_accept,
            use_cuts_for_accept : visitor_descriptor.use_cuts_for_accept,
            use_mass_distribution_for_accept: visitor_descriptor.use_mass_distribution_for_accept,
            maintain_shadow_box_for_accept : visitor_descriptor.maintain_shadow_box_for_accept,
            shadow_box_set: false,
            current_box: Option::None,
            left_box: Option::None,
            right_box: Option::None,
            direction_is_left : false,
            shadow_box: Option::None,
            depth : 0,
            mass : 0,
            left_mass : 0,
            right_mass : 0,
            probability_of_cut : 1.0,
            cut_value :0.0,
            leaf_index : usize::MAX,
            leaf_point : Vec::new(),
            cut_dimension: usize::MAX,
            leaf_duplicate: false,
            use_cuts_for_trigger : multivisitor_descriptor.use_cuts_for_trigger,
            use_box_for_trigger : multivisitor_descriptor.use_box_for_trigger,
            use_child_boxes_for_trigger : multivisitor_descriptor.use_child_boxes_for_trigger,
            use_mass_distribution_for_trigger : multivisitor_descriptor.use_mass_distribution_for_trigger,
            update_node_view_between_triggers : multivisitor_descriptor.trigger_manipulation_needs_node_view_accept_fields
        }
    }

    pub fn update_to_child(&mut self, node : usize){
        self.depth += 1;
        self.current_node_offset = node;
    }

    pub fn update_shadow_box(&mut self, other_box : &BoundingBox ){
        if self.maintain_shadow_box_for_accept {
            match &mut self.shadow_box{
                Some(y) => {
                    assert!(self.shadow_box_set);
                    y.add_box_and_check_absorbs(other_box);
                },
                None => {
                    assert!(!self.shadow_box_set);
                    self.shadow_box_set = true;
                    self.shadow_box = Some(other_box.copy());
                }
            }
        }
    }

    pub fn leaf_view(&mut self, point: &[f32], node_store: &dyn NodeStoreView, point_store: &dyn PointStoreView){
        self.leaf_index = node_store.get_leaf_point_index(self.current_node_offset);
        if self.use_point_copy_for_accept {
            self.leaf_point = point_store.get_copy(self.leaf_index);
            self.leaf_duplicate = self.leaf_point.eq(point);
        } else {
            self.leaf_duplicate = point_store.is_equal(point,self.leaf_index);
        }
        if self.use_box_for_accept {
            self.leaf_point = point_store.get_copy(self.leaf_index);
            self.current_box = Some(BoundingBox::new(&self.leaf_point, &self.leaf_point));
        }
        self.mass = node_store.get_mass(self.current_node_offset);
    }

    pub fn update_view(&mut self, node: usize, point: &[f32], node_store: &dyn NodeStoreView, point_store: &dyn PointStoreView) {
        let past_node = self.current_node_offset;
        let left_child = node_store.get_left_index(node);
        self.direction_is_left = past_node == left_child;

        if self.use_box_for_accept {
            let mut x = self.current_box.as_mut().unwrap();
            if self.use_child_boxes_for_accept || self.maintain_shadow_box_for_accept { // move current box
                if self.use_child_boxes_for_accept {
                    if self.direction_is_left {
                        self.left_box = Some(x.copy());
                    } else {
                        self.right_box = Some(x.copy());
                    }
                }
                let other_box = if self.direction_is_left {
                    node_store.get_box(node_store.get_right_index(node), point_store)
                } else {
                    node_store.get_box(left_child, point_store)
                }; // get the other box
                x.add_box_and_check_absorbs(&other_box); // update current box
                self.update_shadow_box(&other_box); // update shadow box; if needed
                if self.use_child_boxes_for_accept { // store other box
                    if self.direction_is_left {
                        self.right_box = Some(other_box);
                    } else {
                        self.left_box = Some(other_box);
                    }
                }
            } else {
                node_store.grow_node_box(&mut x, point_store, node, node_store.get_sibling(past_node, node));
            }
        } else {
            if self.use_child_boxes_for_accept { // boxes have to be created
                self.left_box = Some(node_store.get_box(left_child, point_store));
                self.right_box = Some(node_store.get_box(node_store.get_right_index(node), point_store));
            }
            if self.maintain_shadow_box_for_accept { // some duplicate effort to save borrowing hassles
                let other_box = if self.direction_is_left {
                    node_store.get_box(node_store.get_right_index(node), point_store)
                } else {
                    node_store.get_box(left_child, point_store)
                };
                match &mut self.shadow_box {
                    Some(y) => {
                        assert!(self.shadow_box_set);
                        y.add_box_and_check_absorbs(&other_box);
                    },
                    None => {
                        self.shadow_box_set = true;
                        self.shadow_box = Some(other_box);
                    },
                }
            }
        }

        self.depth -= 1;
        self.current_node_offset = node;
        self.mass = node_store.get_mass(self.current_node_offset);
        if !self.use_box_for_accept {
            self.probability_of_cut = node_store.get_probability_of_cut(node, point, point_store);
        } else {
            match &self.current_box {
                Some(x) => self.probability_of_cut = x.probability_of_cut(point),
                None => panic!(),
            };
        }
        if self.use_cuts_for_accept {
            self.cut_value = node_store.get_cut_value(self.current_node_offset);
            self.cut_dimension = node_store.get_cut_dimension(self.current_node_offset);
        }
        if self.use_mass_distribution_for_accept {
            self.left_mass = node_store.get_mass(left_child);
            self.right_mass = node_store.get_mass(node_store.get_mass(node_store.get_right_index(node)));
        }
    }


    pub fn update_before_trigger(&mut self, point : &[f32], node_store: &dyn NodeStoreView, point_store: &dyn PointStoreView){
        if self.use_cuts_for_trigger {
            self.cut_value = node_store.get_cut_value(self.current_node_offset);
            self.cut_dimension = node_store.get_cut_dimension(self.current_node_offset);
        }
        if self.use_box_for_trigger {
            self.current_box = Some(node_store.get_box(self.current_node_offset,point_store));
        }
    }

    pub fn update_view_for_trigger(&mut self, node: usize, point: &[f32], node_store: &dyn NodeStoreView, point_store: &dyn PointStoreView) {
        self.current_node_offset = node;
        self.shadow_box_set = false; // shadow is meaningliess
        self.shadow_box = Option::None;
        if self.use_child_boxes_for_trigger {
            self.left_box = Some(node_store.get_box(node_store.get_left_index(self.current_node_offset), point_store));
            self.right_box = Some(node_store.get_box(node_store.get_right_index(self.current_node_offset), point_store));
        }
        if self.use_box_for_trigger {
            self.current_box = Some(node_store.get_box(self.current_node_offset, point_store));
        }
        self.mass = node_store.get_mass(self.current_node_offset);
        if self.use_cuts_for_trigger {
            self.cut_value = node_store.get_cut_value(self.current_node_offset);
            self.cut_dimension = node_store.get_cut_dimension(self.current_node_offset);
        }
    }

    pub fn traverse<T>(&mut self, visitor: &mut dyn Visitor<T>, point:&[f32], point_store : &dyn PointStoreView, node_store : &dyn NodeStoreView){
        if node_store.is_leaf(self.current_node_offset) {
            self.leaf_view(point,node_store,point_store);
            visitor.accept_leaf(point,self);
        } else {
            let saved = self.current_node_offset;
            if node_store.is_left_of(self.current_node_offset,point) {
                self.update_to_child(node_store.get_left_index(self.current_node_offset));
            } else {
                self.update_to_child(node_store.get_right_index(self.current_node_offset));
            }
            self.traverse(visitor,point,point_store,node_store);
            if !visitor.has_converged() {
                self.update_view(saved,point,node_store, point_store);
                visitor.accept(point, self);
            }
        }
    }

    pub fn traverse_multi<T,Q>(&mut self, visitor: &mut dyn StreamingMultiVisitor<T,Q>, point:&[f32], point_store : &dyn PointStoreView, node_store : &dyn NodeStoreView){

        if node_store.is_leaf(self.current_node_offset) {
            self.leaf_view(point,node_store,point_store);
            visitor.accept_leaf(point,self);
        } else {
            let saved = self.current_node_offset;
            self.update_before_trigger(point,node_store,point_store);
            if visitor.trigger(point,self) {
                if self.update_node_view_between_triggers{
                    self.update_view_for_trigger(saved, point,node_store,point_store);
                }
                visitor.init_trigger(point,self);
                self.update_to_child(node_store.get_left_index(self.current_node_offset));
                self.traverse_multi(visitor,point,point_store,node_store);
                if self.update_node_view_between_triggers {
                    self.update_view_for_trigger(saved,point,node_store,point_store);
                }
                visitor.half_trigger(point,self);
                self.update_to_child(node_store.get_right_index(saved));
                self.traverse_multi(visitor,point,point_store,node_store);
                if self.update_node_view_between_triggers {
                    self.update_view_for_trigger(saved,point,node_store,point_store);
                }
                visitor.close_trigger(point,self);
            } else if node_store.is_left_of(self.current_node_offset,point) {
                self.update_to_child(node_store.get_left_index(self.current_node_offset));
                self.traverse_multi(visitor,point,point_store,node_store);
            } else {
                self.update_to_child(node_store.get_right_index(self.current_node_offset));
                self.traverse_multi(visitor,point,point_store,node_store);
            }
            if !visitor.has_converged() {
                self.update_view(saved,point,node_store, point_store);
                visitor.accept(point, self);
            }
        }

    }
}

pub trait NodeView {
    fn get_mass(&self) -> usize;
    fn get_bounding_box(&self) -> BoundingBox;
    fn get_probability_of_cut(&self, point: &[f32]) -> f64;
    fn get_depth(&self) -> usize;
    fn get_probability_of_cut_vector(&self,point: &[f32]) -> Vec<f32>;
    fn get_left_box(&self) -> BoundingBox;
    fn get_right_box(&self) -> BoundingBox;
    fn get_shadow_box(&self) -> BoundingBox;
    fn leaf_equals(&self) -> bool;
    fn get_leaf_point(&self) -> &[f32];
    fn get_leaf_index(&self) -> usize;
    fn get_cut_dimension(&self) -> usize;
    fn get_cut_value(&self) -> f32;
}

impl NodeView for AbstractNodeView {
    fn get_mass(&self) -> usize {
        self.mass
    }

    fn get_bounding_box(&self) -> BoundingBox {
        assert!(self.use_box_for_accept||self.use_box_for_trigger);
        match &self.current_box {
            Some(x) => x.copy(),
            None => panic!()
        }
        //self.current_box.copy()
    }

    fn get_probability_of_cut(&self, point: &[f32]) -> f64{
        self.probability_of_cut
    }

    fn get_depth(&self) -> usize {
        self.depth
    }

    fn get_probability_of_cut_vector(&self,point: &[f32]) -> Vec<f32> {
        assert!(self.use_box_for_accept||self.use_box_for_trigger);
        match &self.current_box {
            Some(x) => x.probability_of_cut_di_vector(point),
            None => panic!()
        }
        //self.current_box.probability_of_cut_di_vector(point)
    }

    fn get_left_box(&self) -> BoundingBox {
        assert!(self.use_child_boxes_for_trigger);
        match &self.left_box {
            Some(x) => x.copy(),
            None => panic!()
        }
    }

    fn get_right_box(&self) -> BoundingBox {
        assert!(self.use_child_boxes_for_trigger);
        match &self.right_box {
            Some(x) => x.copy(),
            None => panic!()
        }
    }

    fn get_shadow_box(&self) -> BoundingBox {
        assert!(self.maintain_shadow_box_for_accept);
        match &self.shadow_box {
            Some(x) => x.copy(),
            None => panic!()
        }
    }

    fn leaf_equals(&self) -> bool {
        self.leaf_duplicate
    }

    fn get_leaf_point(&self) -> &[f32] {
        assert!(self.use_point_copy_for_accept);
        &self.leaf_point
    }

    fn get_leaf_index(&self) -> usize {
        self.leaf_index
    }

    fn get_cut_dimension(&self) -> usize {
        assert!(self.use_cuts_for_accept || self.use_cuts_for_trigger);
        self.cut_dimension
    }

    fn get_cut_value(&self) -> f32 {
        assert!(self.use_cuts_for_accept || self.use_cuts_for_trigger);
        self.cut_value
    }
}




