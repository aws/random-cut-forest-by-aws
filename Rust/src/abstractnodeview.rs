
use std::io::empty;
use crate::boundingbox::BoundingBox;
use crate::newnodestore::NewNodeStore;
use crate::newnodestore::NodeStoreView;
use crate::pointstore::PointStoreView;
use crate::rcf::Max;
use crate::visitor::Visitor;

pub struct AbstractNodeView{
    current_node_offset: usize,
    current_box : BoundingBox,
    sibling_box : BoundingBox,
    shadow_box : BoundingBox,
    leaf_point : Vec<f32>,
    cut_value : f32,
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
            current_box: BoundingBox::new(&vec![0.0 as f32;dimensions],&vec![0.0 as f32;dimensions]),
            sibling_box: BoundingBox::new(&vec![0.0 as f32;dimensions],&vec![0.0 as f32;dimensions]),
            shadow_box: BoundingBox::new(&vec![0.0 as f32;dimensions],&vec![0.0 as f32;dimensions]),
            depth : 0,
            mass : 0,
            cut_value :0.0,
            leaf_point : vec![0.0 as f32;dimensions],
            cut_dimension: usize::MAX
        }
    }
    pub fn set_depth(&mut self, depth : usize){
        self.depth = depth;
    }

    pub fn traverse(&mut self, visitor: &mut dyn Visitor, point:&[f32], point_store : &dyn PointStoreView, node_store : &dyn NodeStoreView){
        if node_store.is_leaf(self.current_node_offset) {
            self.leaf_point = Vec::from(point_store.get(node_store.get_leaf_point_index(self.current_node_offset)));
            self.current_box.copy_from(&BoundingBox::new(&self.leaf_point, &self.leaf_point));
            self.mass = node_store.get_mass(self.current_node_offset);
            visitor.accept_leaf(point,self);
        } else {

            let saved = self.current_node_offset;
            self.depth += 1;
            if node_store.is_left_of(self.current_node_offset,point) {
                self.current_node_offset = node_store.get_left_index(self.current_node_offset);
                self.traverse(visitor,point,point_store,node_store);
            } else {
                self.current_node_offset = node_store.get_right_index(self.current_node_offset);
                self.traverse(visitor,point,point_store,node_store);
            }
            if (!visitor.has_converged()) {
                self.depth -= 1;
                self.current_node_offset = saved;
                self.sibling_box.copy_from(&node_store.get_sibling_box(saved, point, point_store));
                self.current_box.add_box_and_check_absorbs(&self.sibling_box);
                if (self.shadow_box_set){
                    self.shadow_box.add_box_and_check_absorbs(&self.sibling_box);
                } else {
                    self.shadow_box.copy_from(&self.sibling_box);
                    self.shadow_box_set = true;
                }
                self.mass = node_store.get_mass(self.current_node_offset);
                self.cut_value = node_store.get_cut_value(self.current_node_offset);
                self.cut_dimension = node_store.get_cut_dimension(self.current_node_offset);
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
    fn get_sibling_box(&self, point: &[f32]) -> BoundingBox;
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
        self.current_box.probability_of_cut(point)
    }

    fn get_depth(&self) -> usize {
        self.depth
    }

    fn get_probability_of_cut_vector(&self,point: &[f32]) -> Vec<f32> {
        self.current_box.probability_of_cut_di_vector(point)
    }

    fn get_sibling_box(&self,point: &[f32]) -> BoundingBox {
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




