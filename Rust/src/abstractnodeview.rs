
use std::io::empty;
use crate::boundingbox::BoundingBox;
use crate::newnodestore::NewNodeStore;
use crate::newnodestore::NodeStoreView;
use crate::pointstore::PointStoreView;
use crate::rcf::Max;
use crate::visitor::Visitor;
use crate::multivisitor::StreamingMultiVisitor;

pub struct AbstractNodeView{
    current_node_offset: usize,
    current_box : BoundingBox,
    sibling_box : BoundingBox,
    shadow_box : BoundingBox,
    leaf_point : Vec<f32>,
    cut_value : f32,
    use_current_box: bool,
    probability_of_cut: f64,
    cut_dimension : usize,
    shadow_box_set: bool,
    dimensions : usize,
    mass : usize,
    depth : usize
}

impl AbstractNodeView {
    pub fn new(dimensions: usize, root: usize) -> Self {
        let a = vec![0.0 as f32;dimensions];
        let b = vec![0.0 as f32;dimensions];
        AbstractNodeView {
            current_node_offset: root,
            dimensions,
            shadow_box_set: false,
            use_current_box : false,
            current_box: BoundingBox::new(&vec![0.0 as f32;dimensions],&vec![0.0 as f32;dimensions]),
            sibling_box: BoundingBox::new(&vec![0.0 as f32;dimensions],&vec![0.0 as f32;dimensions]),
            shadow_box: BoundingBox::new(&vec![0.0 as f32;dimensions],&vec![0.0 as f32;dimensions]),
            depth : 0,
            mass : 0,
            probability_of_cut : 1.0,
            cut_value :0.0,
            leaf_point : vec![0.0 as f32;dimensions],
            cut_dimension: usize::MAX
        }
    }

    pub fn update_to_child(&mut self, node : usize){
        self.depth += 1;
        self.current_node_offset = node;
    }

    pub fn update_view(&mut self, node: usize, shadow_boxes: bool, point: &[f32], node_store: &dyn NodeStoreView, point_store: &dyn PointStoreView){
        if shadow_boxes {
            let past_node = self.current_node_offset;
            let sibling = node_store.get_sibling(past_node,node);
            self.sibling_box.copy_from(&node_store.get_box(sibling, point_store));
            if self.use_current_box {
                self.current_box.add_box_and_check_absorbs(&self.sibling_box);
            }
            if (self.shadow_box_set){
                self.shadow_box.add_box_and_check_absorbs(&self.sibling_box);
            } else {
                self.shadow_box.copy_from(&self.sibling_box);
                self.shadow_box_set = true;
            }
        } else if self.use_current_box {
            node_store.grow_node_box(&mut self.current_box, point_store, node, node_store.get_sibling(self.current_node_offset, node));
        }

        self.depth -= 1;
        self.current_node_offset = node;
        self.mass = node_store.get_mass(self.current_node_offset);
        if (!self.use_current_box){
            self.probability_of_cut = node_store.get_probability_of_cut(node,point,point_store);
        } else {
            self.probability_of_cut = self.current_box.probability_of_cut(point);
        }
        self.cut_value = node_store.get_cut_value(self.current_node_offset);
        self.cut_dimension = node_store.get_cut_dimension(self.current_node_offset);
    }

    pub fn traverse<T>(&mut self, visitor: &mut dyn Visitor<T>, point:&[f32], point_store : &dyn PointStoreView, node_store : &dyn NodeStoreView){
        if node_store.is_leaf(self.current_node_offset) {
            self.leaf_point = point_store.get_copy(node_store.get_leaf_point_index(self.current_node_offset));
            self.use_current_box = visitor.accept_needs_box() || node_store.use_path_for_box();
            if (self.use_current_box) {
                self.current_box.copy_from(&BoundingBox::new(&self.leaf_point, &self.leaf_point));
            }
            self.mass = node_store.get_mass(self.current_node_offset);
            self.use_current_box = visitor.accept_needs_box();
            visitor.accept_leaf(point,self);
        } else {
            let saved = self.current_node_offset;
            if node_store.is_left_of(self.current_node_offset,point) {
                self.update_to_child(node_store.get_left_index(self.current_node_offset));
            } else {
                self.update_to_child(node_store.get_right_index(self.current_node_offset));
            }
            self.traverse(visitor,point,point_store,node_store);
            if (!visitor.has_converged()) {
                self.update_view(saved,visitor.use_shadow_box(),point,node_store, point_store);
                visitor.accept(point, self);
            }
        }
    }

    pub fn traverse_multi<T,Q>(&mut self, visitor: &mut dyn StreamingMultiVisitor<T,Q>, point:&[f32], point_store : &dyn PointStoreView, node_store : &dyn NodeStoreView){
        if node_store.is_leaf(self.current_node_offset) {
            self.leaf_point = point_store.get_copy(node_store.get_leaf_point_index(self.current_node_offset));
            self.use_current_box = visitor.accept_needs_box() || node_store.use_path_for_box();
            if (self.use_current_box) {
                self.current_box.copy_from(&BoundingBox::new(&self.leaf_point, &self.leaf_point));
            }
            self.mass = node_store.get_mass(self.current_node_offset);
            visitor.accept_leaf(point,self);
        } else {
            let saved = self.current_node_offset;
            self.update_view(self.current_node_offset,visitor.use_shadow_box(),point,node_store,point_store);
            if visitor.trigger_needs_box() {
                self.current_box = node_store.get_box(self.current_node_offset,point_store);
            }
            if visitor.trigger() {
                visitor.init_trigger();
                self.update_to_child(node_store.get_left_index(self.current_node_offset));
                self.traverse_multi(visitor,point,point_store,node_store);
                self.depth += 1;
                visitor.half_trigger();
                self.update_to_child(node_store.get_right_index(self.current_node_offset));
                self.traverse_multi(visitor,point,point_store,node_store);
                visitor.close_trigger();
            } else if node_store.is_left_of(self.current_node_offset,point) {
                self.update_to_child(node_store.get_left_index(self.current_node_offset));
                self.traverse_multi(visitor,point,point_store,node_store);
            } else {
                self.update_to_child(node_store.get_right_index(self.current_node_offset));
                self.traverse_multi(visitor,point,point_store,node_store);
            }
            if (!visitor.has_converged()) {
                self.update_view(saved,visitor.use_shadow_box(),point,node_store, point_store);
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
    fn get_sibling_box(&self) -> BoundingBox;
    fn get_shadow_box(&self) -> BoundingBox;
    fn get_leaf_point(&self) -> &[f32];
    fn get_cut_dimension(&self) -> usize;
    fn get_cut_value(&self) -> f32;
}

impl NodeView for AbstractNodeView {
    fn get_mass(&self) -> usize {
        self.mass
    }

    fn get_bounding_box(&self) -> BoundingBox {
        self.current_box.copy()
    }

    fn get_probability_of_cut(&self, point: &[f32]) -> f64{
        self.probability_of_cut
    }

    fn get_depth(&self) -> usize {
        self.depth
    }

    fn get_probability_of_cut_vector(&self,point: &[f32]) -> Vec<f32> {
        self.current_box.probability_of_cut_di_vector(point)
    }

    fn get_sibling_box(&self) -> BoundingBox {
        self.sibling_box.copy()
    }

    fn get_shadow_box(&self) -> BoundingBox {
        self.shadow_box.copy()
    }

    fn get_leaf_point(&self) -> &[f32] {
        &self.leaf_point
    }

    fn get_cut_dimension(&self) -> usize {
        self.cut_dimension
    }

    fn get_cut_value(&self) -> f32 {
        self.cut_value
    }
}




