use crate::pointstore::PointStore;
use crate::samplerplustree::boundingbox::BoundingBox;
use crate::samplerplustree::nodestore::NodeStore;
use crate::common::divector::DiVector;
use crate::visitor::visitor::{LeafPointVisitor, MissingCoordinatesMultiLeafPointVisitor, MissingCoordinatesMultiVisitor, SimpleVisitor, UniqueMultiVisitor, Visitor};

pub trait MinimalNodeView {
    fn get_mass(&self) -> usize;
    fn probability_of_cut_on_path(&self) -> f64;
    fn get_depth(&self) -> usize;
    fn get_shadow_box_probability_of_cut(&self) -> f64;
    fn leaf_equals(&self) -> bool;
    fn get_leaf_index(&self) -> usize;
    fn get_cut_dimension(&self) -> usize;
    fn get_cut_value(&self) -> f32;
}

pub trait LeafPointNodeView : MinimalNodeView{
    fn get_leaf_point(&self) -> &[f32];
}

pub trait NodeView:LeafPointNodeView {
    fn get_bounding_box(&self) -> BoundingBox;
    fn get_probability_of_cut(&self, point: &[f32]) -> f64;
    fn modify_in_place_probability_of_cut_di_vector(&self, point: &[f32], di_vector :&mut DiVector);
    fn get_left_box(&self) -> BoundingBox {
        panic!("not implemented in basic view");
    }
    fn get_right_box(&self) -> BoundingBox {
        panic!("not implemented in basic view");
    }
    fn get_shadow_box(&self) -> BoundingBox;
}

pub trait AllowTraversal<Q> {
    fn accept_leaf(&mut self, point : &[f32], view : Q);
    fn accept(&mut self, point : &[f32], view : Q);
}

#[repr(C)]
pub struct BasicNodeView {
    current_node: usize,
    sibling: usize,
    current_box: Option<BoundingBox>,
    shadow_box: Option<BoundingBox>,
    leaf_index: usize,
    leaf_duplicate: bool,
    use_point_copy_for_accept: bool,
    leaf_point: Option<Vec<f32>>,
    cut_value: f32,
    maintain_shadow_box_for_accept: bool,
    use_box_for_accept: bool,
    probability_of_cut: f64,
    shadow_box_probability_of_cut : f64,
    cut_dimension: usize,
    shadow_box_set: bool,
    dimensions: usize,
    left_child: usize,
    right_child: usize,
    mass: usize,
    depth: usize,
}

impl BasicNodeView {
    pub fn new(
        dimensions: usize,
        root: usize,
        use_box_for_accept: bool,
        use_point_copy_for_accept: bool
    ) -> Self {
        BasicNodeView {
            current_node: root,
            sibling: 0,
            dimensions,
            left_child: 0,
            use_point_copy_for_accept,
            use_box_for_accept,
            maintain_shadow_box_for_accept : false,
            shadow_box_set: false,
            current_box: Option::None,
            shadow_box: Option::None,
            depth: 0,
            mass: 0,
            probability_of_cut: 1.0,
            shadow_box_probability_of_cut : 0.0,
            cut_value: 0.0,
            leaf_index: usize::MAX,
            leaf_point: Option::None,
            cut_dimension: usize::MAX,
            leaf_duplicate: false,
            right_child: 0,
        }
    }

    pub fn set_leaf_view(
        &mut self,
        point: &[f32],
        point_store: &dyn PointStore,
        node_store: &dyn NodeStore,
    ) {
        self.leaf_index = node_store.get_leaf_point_index(self.current_node);

        if self.use_point_copy_for_accept {
            let leaf_copy = point_store.get_copy(self.leaf_index);
            self.leaf_duplicate = leaf_copy.eq(point);
            self.leaf_point = Some(leaf_copy);
        }
        if self.use_box_for_accept {
            let leaf_box = node_store.get_box(self.current_node, point_store);
            self.leaf_duplicate = point.eq(leaf_box.get_min_values());
            self.current_box = Some(leaf_box);
        } else {
            self.leaf_duplicate = point_store.is_equal(point, self.leaf_index);
        }

        self.mass = node_store.get_mass(self.current_node);
    }

    pub fn update_view_to_child(&mut self, point: &[f32]) {
        self.depth += 1;
        self.current_node = if point[self.cut_dimension] <= self.cut_value {
            self.left_child
        } else {
            self.right_child
        };
    }

    pub fn update_view_for_path(&mut self, node_store: &dyn NodeStore) {
        let (a, b, c, d) = node_store.get_cut_and_children(self.current_node);
        self.cut_dimension = a;
        self.cut_value = b;
        self.left_child = c;
        self.right_child = d;
    }

    pub fn update_view_to_parent_with_missing(
        &mut self,
        parent: usize,
        point: &[f32],
        missing_coordinates : &[bool],
        point_store: &dyn PointStore,
        node_store: &dyn NodeStore,
    ) {
        let past_node = self.current_node;
        self.current_node = parent;
        self.update_view_for_path(node_store);
        assert!(past_node == self.left_child || past_node == self.right_child);
        let sibling = if past_node == self.left_child {
            self.right_child
        } else {
            self.left_child
        };

        if self.use_box_for_accept {
            let mut x = self.current_box.as_mut().unwrap();
            node_store.grow_node_box(
                &mut x,
                point_store,
                parent,
                node_store.get_sibling(past_node, parent),
            );
            self.probability_of_cut = x.probability_of_cut_with_missing_coordinates(point,missing_coordinates);
        } else {
                self.probability_of_cut =
                    node_store.get_probability_of_cut_with_missing_coordinates(parent, point, missing_coordinates, point_store);
        }

        self.depth -= 1;
        self.mass = node_store.get_mass(self.current_node);
    }



    pub fn update_view_to_parent(
        &mut self,
        parent: usize,
        point: &[f32],
        point_store: &dyn PointStore,
        node_store: &dyn NodeStore,
    ) {
        let past_node = self.current_node;
        self.current_node = parent;
        self.update_view_for_path(node_store);
        assert!(past_node == self.left_child || past_node == self.right_child);
        let sibling = if past_node == self.left_child {
            self.right_child
        } else {
            self.left_child
        };

        if self.maintain_shadow_box_for_accept {
            if !self.shadow_box_set {
                self.shadow_box = Some(node_store.get_box(sibling, point_store));
            } else {
                node_store.grow_node_box(
                    self.shadow_box.as_mut().unwrap(),
                    point_store,
                    parent,
                    sibling,
                );
                self.shadow_box_probability_of_cut=self.shadow_box.as_ref().unwrap().probability_of_cut(point);
            }
            if self.use_box_for_accept {
                let x = self.current_box.as_mut().unwrap();
                let y = self.shadow_box.as_mut().unwrap();
                x.check_contains_and_add_point(y.get_min_values());
                x.check_contains_and_add_point(y.get_max_values());
                self.probability_of_cut = x.probability_of_cut(point);
            } else {
                    self.probability_of_cut =
                        node_store.get_probability_of_cut(parent, point, point_store);
            }
        } else {
            if self.use_box_for_accept {
                let mut x = self.current_box.as_mut().unwrap();
                node_store.grow_node_box(
                    &mut x,
                    point_store,
                    parent,
                    node_store.get_sibling(past_node, parent),
                );
                self.probability_of_cut = x.probability_of_cut(point);
            } else {
                    self.probability_of_cut =
                        node_store.get_probability_of_cut(parent, point, point_store);

            }
        }

        self.depth -= 1;
        self.mass = node_store.get_mass(self.current_node);
    }

    pub fn traverse_simple<T>(
        &mut self,
        visitor: &mut dyn SimpleVisitor<T>,
        point: &[f32],
        point_store: &dyn PointStore,
        node_store: &dyn NodeStore,
    ) {
        if node_store.is_leaf(self.current_node) {
            self.set_leaf_view(point, point_store, node_store);
            visitor.accept_leaf(point, self);
            self.maintain_shadow_box_for_accept = visitor.use_shadow_box();
        } else {
            let saved = self.current_node;
            self.update_view_for_path(node_store);
            self.update_view_to_child(point);
            self.traverse_simple(visitor, point, point_store, node_store);
            if !visitor.has_converged() {
                self.update_view_to_parent(saved, point, point_store, node_store);
                visitor.accept(point, self);
            }
        }
    }

    pub fn traverse_leaf_point<T>(
        &mut self,
        visitor: &mut dyn LeafPointVisitor<T>,
        point: &[f32],
        point_store: &dyn PointStore,
        node_store: &dyn NodeStore,
    ) {
        if node_store.is_leaf(self.current_node) {
            self.set_leaf_view(point, point_store, node_store);
            visitor.accept_leaf(point, self);
            self.maintain_shadow_box_for_accept = visitor.use_shadow_box();
        } else {
            let saved = self.current_node;
            self.update_view_for_path(node_store);
            self.update_view_to_child(point);
            self.traverse_leaf_point(visitor, point, point_store, node_store);
            if !visitor.has_converged() {
                self.update_view_to_parent(saved, point, point_store, node_store);
                visitor.accept(point, self);
            }
        }
    }

    pub fn traverse<T>(
        &mut self,
        visitor: &mut dyn Visitor<T>,
        point: &[f32],
        point_store: &dyn PointStore,
        node_store: &dyn NodeStore,
    ) {
        if node_store.is_leaf(self.current_node) {
            self.set_leaf_view(point, point_store, node_store);
            visitor.accept_leaf(point, self);
            self.maintain_shadow_box_for_accept = visitor.use_shadow_box();
        } else {
            let saved = self.current_node;
            self.update_view_for_path(node_store);
            self.update_view_to_child(point);
            self.traverse(visitor, point, point_store, node_store);
            if !visitor.has_converged() {
                self.update_view_to_parent(saved, point, point_store, node_store);
                visitor.accept(point, self);
            }
        }
    }


    pub fn traverse_multi_with_missing_coordinates<T>(
        &mut self,
        visitor: &mut dyn MissingCoordinatesMultiLeafPointVisitor<T>,
        point: &[f32],
        missing_coordinates: &[bool],
        point_store: &dyn PointStore,
        node_store: &dyn NodeStore,
    ) {
        if node_store.is_leaf(self.current_node) {
            self.set_leaf_view(point, point_store, node_store);
            visitor.accept_leaf(point, self);
        } else {
            let parent = self.current_node;
            self.update_view_for_path(node_store);
            if missing_coordinates[self.cut_dimension] {
                let right = self.right_child;
                self.current_node = self.left_child;
                self.traverse_multi_with_missing_coordinates(visitor, point, missing_coordinates,point_store, node_store);
                let saved_box = if self.use_box_for_accept {
                    Some(self.current_box.as_ref().unwrap().copy())
                } else {
                    Option::None
                };
                self.current_node = right;
                self.traverse_multi_with_missing_coordinates(visitor, point,missing_coordinates, point_store, node_store);
                visitor.combine_branches(point, self);
                if !visitor.has_converged() {
                    self.current_node = parent;
                    if self.use_box_for_accept {
                        let x = self.current_box.as_mut().unwrap();
                        x.check_contains_and_add_point(
                            saved_box.as_ref().unwrap().get_min_values(),
                        );
                        x.check_contains_and_add_point(
                            saved_box.as_ref().unwrap().get_max_values(),
                        );
                        self.probability_of_cut = x.probability_of_cut_with_missing_coordinates(point,missing_coordinates);
                    } else {
                        self.probability_of_cut = node_store.get_probability_of_cut_with_missing_coordinates(
                            self.current_node,
                            point,
                            missing_coordinates,
                            point_store,
                        );
                    }
                    self.update_view_for_path(node_store);
                }
            } else {
                self.update_view_to_child(point);
                self.traverse_multi_with_missing_coordinates(visitor, point, missing_coordinates,point_store, node_store);
                if !visitor.has_converged() {
                    self.update_view_to_parent_with_missing(
                        parent,
                        point,
                        missing_coordinates,
                        point_store,
                        node_store,
                    );
                }
            }
            if !visitor.has_converged() {
                visitor.accept(point, self);
            }
        }
    }


    pub fn traverse_unique_multi<T>(
        &mut self,
        visitor: &mut dyn UniqueMultiVisitor<T>,
        point: &[f32],
        point_store: &dyn PointStore,
        node_store: &dyn NodeStore,
    ) {
        if node_store.is_leaf(self.current_node) {
            self.set_leaf_view(point, point_store, node_store);
            visitor.accept_leaf(point, self);
        } else {
            let parent = self.current_node;
            self.update_view_for_path(node_store);
            if visitor.trigger(point, self) {
                let right = self.right_child;
                self.current_node = self.left_child;
                self.traverse_unique_multi(visitor, point, point_store, node_store);
                let saved_box = if self.use_box_for_accept {
                    Some(self.current_box.as_ref().unwrap().copy())
                } else {
                    Option::None
                };
                self.current_node = right;
                self.traverse_unique_multi(visitor, point, point_store, node_store);
                visitor.combine_branches(point, self);
                if !visitor.has_converged() {
                    self.current_node = parent;
                    if self.use_box_for_accept {
                        let x = self.current_box.as_mut().unwrap();
                        x.check_contains_and_add_point(
                            saved_box.as_ref().unwrap().get_min_values(),
                        );
                        x.check_contains_and_add_point(
                            saved_box.as_ref().unwrap().get_max_values(),
                        );
                        self.probability_of_cut = x.probability_of_cut(visitor.unique_answer());
                    } else {
                        self.probability_of_cut = node_store.get_probability_of_cut(
                            self.current_node,
                            visitor.unique_answer(),
                            point_store,
                        );
                    }
                    self.update_view_for_path(node_store);
                }
            } else {
                self.update_view_to_child(point);
                self.traverse_unique_multi(visitor, point, point_store, node_store);
                if !visitor.has_converged() {
                    self.update_view_to_parent(
                        parent,
                        visitor.unique_answer(),
                        point_store,
                        node_store,
                    );
                }
            }
            if !visitor.has_converged() {
                visitor.accept(point, self);
            }
        }
    }
}

impl MinimalNodeView for BasicNodeView {
    fn get_mass(&self) -> usize {
        self.mass
    }


    fn probability_of_cut_on_path(&self) -> f64 {
        self.probability_of_cut
    }


    fn get_depth(&self) -> usize {
        self.depth
    }


    fn get_shadow_box_probability_of_cut(&self) -> f64 {
        assert!(self.maintain_shadow_box_for_accept);
        self.shadow_box_probability_of_cut
    }

    fn leaf_equals(&self) -> bool {
        self.leaf_duplicate
    }



    fn get_leaf_index(&self) -> usize {
        self.leaf_index
    }

    fn get_cut_dimension(&self) -> usize {
        self.cut_dimension
    }

    fn get_cut_value(&self) -> f32 {
        self.cut_value
    }
}

impl LeafPointNodeView for BasicNodeView {
    fn get_leaf_point(&self) -> &[f32] {
        self.leaf_point.as_ref().unwrap()
    }
}

impl NodeView for BasicNodeView {
    fn get_shadow_box(&self) -> BoundingBox {
        assert!(self.maintain_shadow_box_for_accept);
        match &self.shadow_box {
            Some(x) => x.copy(),
            None => panic!(),
        }
    }


    /// not required in any of the current visitors -- but can be
    fn modify_in_place_probability_of_cut_di_vector(&self, point: &[f32], di_vector : &mut DiVector) {
        assert!(self.use_box_for_accept);
        match &self.current_box {
            Some(x) => di_vector.assign_as_probability_of_cut(x,point),
            None => panic!(),
        }
    }

    fn get_bounding_box(&self) -> BoundingBox {
        assert!(self.use_box_for_accept);
        match &self.current_box {
            Some(x) => x.copy(),
            None => panic!(),
        }
    }

    fn get_probability_of_cut(&self,point: &[f32]) -> f64 {
        assert!(self.use_box_for_accept);
        match &self.current_box {
            Some(x) => x.probability_of_cut(point),
            None => panic!(),
        }
    }
}