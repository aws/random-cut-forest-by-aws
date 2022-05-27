use std::fmt::Debug;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rand_core::RngCore;

use crate::{
    pointstore::PointStore,
    samplerplustree::{
        boundingbox::BoundingBox,
        cut::Cut,
        nodestore::{NodeStore, VectorNodeStore},
        nodeview::{MediumNodeView, UpdatableMultiNodeView, UpdatableNodeView},
    },
    types::Location,
    visitor::{
        imputevisitor::ImputeVisitor,
        visitor::{SimpleMultiVisitor, Visitor, VisitorInfo},
    },
};

extern crate rand;
extern crate rand_chacha;

#[repr(C)]
pub struct RCFTree<C, P, N>
where
    C: Location,
    usize: From<C>,
    P: Location,
    usize: From<P>,
    N: Location,
    usize: From<N>,
{
    dimensions: usize,
    capacity: usize,
    node_store: VectorNodeStore<C, P, N>,
    random_seed: u64,
    root: usize,
    tree_mass: usize,
    using_transforms: bool,
}

impl<C, P, N> RCFTree<C, P, N>
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
        dimensions: usize,
        capacity: usize,
        using_transforms: bool,
        bounding_box_cache_fraction: f64,
        random_seed: u64,
    ) -> Self {
        let project_to_tree: fn(Vec<f32>) -> Vec<f32> = { |x| x };
        let node_store = VectorNodeStore::<C, P, N>::new(
            capacity,
            dimensions,
            using_transforms,
            project_to_tree,
            bounding_box_cache_fraction,
        );
        let root = node_store.null_node();
        RCFTree {
            dimensions,
            capacity,
            using_transforms,
            random_seed,
            node_store,
            root,
            tree_mass: 0,
        }
    }

    pub fn add(
        &mut self,
        point_index: usize,
        _point_attribute: usize,
        point_store: &dyn PointStore,
    ) -> usize {
        if self.root == self.node_store.null_node() {
            self.root = self.node_store.leaf_index(point_index);
            self.tree_mass = 1;
            point_index
        } else {
            let point = &point_store.get_copy(point_index);
            let mut path_to_root = Vec::new();
            self.node_store
                .set_path(&mut path_to_root, self.root, point);
            let (mut node, mut sibling) = path_to_root.pop().unwrap();

            let leaf_point_index = self.node_store.get_point_index(node);
            let old_point = &point_store.get_copy(leaf_point_index);

            self.tree_mass += 1;
            if point.eq(old_point) {
                self.node_store.increase_leaf_mass(node);
                self.node_store
                    .manage_ancestors_add(&mut path_to_root, point, point_store, true);
                return leaf_point_index;
            } else {
                let mut saved_parent = if path_to_root.len() != 0 {
                    path_to_root.last().unwrap().0
                } else {
                    self.node_store.null_node()
                };
                let mut saved_node = node;
                let mut current_box = BoundingBox::new(old_point, old_point);
                let mut saved_box = current_box.copy();
                let mut parent_path: Vec<(usize, usize)> = Vec::new();
                let mut rng = ChaCha20Rng::seed_from_u64(self.random_seed);
                self.random_seed = rng.next_u64();

                let mut parent = saved_parent;
                let mut saved_cut = Cut::new(usize::MAX, 0.0);
                /* the loop has the execute once */
                loop {
                    let factor: f64 = rng.gen();
                    let (new_cut, separation) =
                        Cut::random_cut_and_separation(&current_box, factor, point);
                    if separation {
                        saved_cut = new_cut;
                        saved_parent = parent;
                        saved_node = node;
                        saved_box = current_box.copy();
                        parent_path.clear();
                    } else {
                        parent_path.push((node, sibling));
                    }
                    assert!(saved_cut.dimension != usize::MAX);

                    if parent == self.node_store.null_node() {
                        break;
                    } else {
                        self.node_store.grow_node_box(
                            &mut current_box,
                            point_store,
                            parent,
                            sibling,
                        );
                        let (a, b) = path_to_root.pop().unwrap();
                        node = a;
                        sibling = b;
                        parent = if path_to_root.len() != 0 {
                            path_to_root.last().unwrap().0
                        } else {
                            self.node_store.null_node()
                        };
                    }
                }

                if saved_parent != self.node_store.null_node() {
                    while !parent_path.is_empty() {
                        path_to_root.push(parent_path.pop().unwrap());
                    }
                    assert!(path_to_root.last().unwrap().0 == saved_parent);
                } else {
                    assert!(path_to_root.len() == 0);
                }
                let merged_node = self.node_store.add_node(
                    saved_parent,
                    point,
                    saved_node,
                    point_index,
                    saved_cut,
                    &saved_box,
                );

                if saved_parent != self.node_store.null_node() {
                    self.node_store.manage_ancestors_add(
                        &mut path_to_root,
                        point,
                        point_store,
                        false,
                    );
                } else {
                    self.root = merged_node;
                }
            }
            point_index
        }
    }

    pub fn delete(
        &mut self,
        point_index: usize,
        _point_attribute: usize,
        point_store: &dyn PointStore,
    ) -> usize {
        if self.root == self.node_store.null_node() {
            println!(" deleting from an empty tree");
            panic!();
        }
        self.tree_mass = self.tree_mass - 1;
        let point = &point_store.get_copy(point_index);
        let mut leaf_path = Vec::new();
        self.node_store.set_path(&mut leaf_path, self.root, point);
        let (leaf_node, leaf_saved_sibling) = leaf_path.pop().unwrap();

        let leaf_point_index = self.node_store.get_point_index(leaf_node);

        if leaf_point_index != point_index {
            if !point_store.is_equal(point, leaf_point_index) {
                println!(
                    " deleting wrong node; looking for {} found {}",
                    point_index, leaf_point_index
                );
                let old_point = point_store.get_copy(leaf_point_index);
                for j in 0..self.dimensions.into() {
                    println!("want {} found {}", point[j], old_point[j]);
                }
                panic!();
            }
        }

        if self.node_store.decrease_leaf_mass(leaf_node) == 0 {
            if leaf_path.len() == 0 {
                self.root = self.node_store.null_node();
            } else {
                let (parent, _sibling) = leaf_path.pop().unwrap();
                let grand_parent = if leaf_path.len() == 0 {
                    self.node_store.null_node()
                } else {
                    leaf_path.last().unwrap().0
                };

                if grand_parent == self.node_store.null_node() {
                    self.root = leaf_saved_sibling;
                    self.node_store.set_root(self.root);
                } else {
                    self.node_store
                        .replace_node(grand_parent, parent, leaf_saved_sibling);
                    self.node_store.manage_ancestors_delete(
                        &mut leaf_path,
                        point,
                        point_store,
                        false,
                    );
                }

                self.node_store.delete_internal_node(parent);
            }
        } else {
            self.node_store
                .manage_ancestors_delete(&mut leaf_path, point, point_store, true);
        }
        leaf_point_index
    }

    pub fn conditional_field<PS: PointStore>(
        &self,
        missing: &[usize],
        point: &[f32],
        point_store: &PS,
        centrality: f64,
        seed: u64,
        visitor_info: &VisitorInfo,
    ) -> (f64, usize, f64) {
        if self.root == self.node_store.null_node() {
            return (0.0, usize::MAX, 0.0);
        }
        let mut visitor = ImputeVisitor::new(missing, centrality, self.tree_mass, seed);
        let mut node_view = MediumNodeView::new(self.root, &self.node_store);
        let mut missing_coordinates = vec![false; self.dimensions];
        for i in missing.iter() {
            missing_coordinates[*i] = true;
        }
        self.traverse_multi_with_missing_coordinates(
            &mut node_view,
            &mut visitor,
            visitor_info,
            point,
            &missing_coordinates,
            point_store,
        );
        visitor.result(&visitor_info)
    }

    pub fn traverse_multi_with_missing_coordinates<V, NodeView, PS, R>(
        &self,
        node_view: &mut NodeView,
        visitor: &mut V,
        visitor_info: &VisitorInfo,
        point: &[f32],
        missing_coordinates: &[bool],
        point_store: &PS,
    ) where
        V: SimpleMultiVisitor<NodeView, R>,
        NodeView: UpdatableMultiNodeView<VectorNodeStore<C, P, N>, PS>,
        PS: PointStore,
    {
        let node = node_view.get_current_node();
        if self.node_store.is_leaf(node) {
            node_view.update_at_leaf(point, node, &self.node_store, &point_store, &visitor_info);
            visitor.accept_leaf(point, visitor_info, node_view);
        } else {
            let parent = node;
            node_view.set_trigger_traversing_down(
                point,
                parent,
                &self.node_store,
                point_store,
                visitor_info,
            );
            if missing_coordinates[self.node_store.get_cut_dimension(parent)] {
                let right = self.node_store.get_left_index(parent);
                let left = self.node_store.get_right_index(parent);
                node_view.set_current_node(left);
                self.traverse_multi_with_missing_coordinates(
                    node_view,
                    visitor,
                    visitor_info,
                    point,
                    missing_coordinates,
                    point_store,
                );
                let saved_box = node_view.get_bounding_box();
                node_view.set_current_node(right);
                self.traverse_multi_with_missing_coordinates(
                    node_view,
                    visitor,
                    visitor_info,
                    point,
                    missing_coordinates,
                    point_store,
                );
                visitor.combine_branches(point, &node_view, visitor_info);
                if !visitor.is_converged() {
                    node_view.merge_paths(
                        parent,
                        saved_box,
                        point,
                        missing_coordinates,
                        &self.node_store,
                        point_store,
                    );
                }
            } else {
                node_view.update_from_node_traversing_down(
                    point,
                    parent,
                    &self.node_store,
                    point_store,
                    &visitor_info,
                );
                self.traverse_multi_with_missing_coordinates(
                    node_view,
                    visitor,
                    visitor_info,
                    point,
                    missing_coordinates,
                    point_store,
                );
                if !visitor.is_converged() {
                    node_view.update_view_to_parent_with_missing_coordinates(
                        parent,
                        point,
                        missing_coordinates,
                        &self.node_store,
                        point_store,
                        &visitor_info,
                    );
                }
            }
            if !visitor.is_converged() {
                visitor.accept(point, visitor_info, node_view);
            }
        }
    }

    pub fn get_size(&self) -> usize {
        self.node_store.get_size(self.dimensions.into()) + std::mem::size_of::<RCFTree<C, P, N>>()
    }

    fn traverse_recursive<R, PS, NodeView, V>(
        &self,
        point: &[f32],
        node_view: &mut NodeView,
        visitor: &mut V,
        visitor_info: &VisitorInfo,
        point_store: &PS,
    ) where
        PS: PointStore,
        V: Visitor<NodeView, R>,
        R: Clone,
        NodeView: UpdatableNodeView<VectorNodeStore<C, P, N>, PS>,
    {
        let current_node = node_view.get_current_node();
        if self.node_store.is_leaf(current_node) {
            node_view.update_at_leaf(
                point,
                current_node,
                &self.node_store,
                &point_store,
                &visitor_info,
            );
            visitor.accept_leaf(point, visitor_info, &node_view);
            if visitor.use_shadow_box() {
                node_view.set_use_shadow_box(&self.node_store, point_store);
            }
        } else {
            node_view.update_from_node_traversing_down(
                point,
                current_node,
                &self.node_store,
                point_store,
                visitor_info,
            );
            self.traverse_recursive(point, node_view, visitor, visitor_info, point_store);
            if !visitor.is_converged() {
                node_view.update_from_node_traversing_up(
                    point,
                    current_node,
                    &self.node_store,
                    &point_store,
                    &visitor_info,
                );
                visitor.accept(point, visitor_info, &node_view);
            }
        }
    }
}

pub trait Traversable<NodeView, N, PS, V, R>
where
    N: Location,
    <N as TryFrom<usize>>::Error: Debug,
    usize: From<N>,
    PS: PointStore,
    V: Visitor<NodeView, R>,
{
    fn traverse(
        &self,
        point: &[f32],
        parameters: &[usize],
        visitor_factory: fn(usize, &[usize], &VisitorInfo) -> V,
        visitor_info: &VisitorInfo,
        point_store: &PS,
        default: &R,
    ) -> R;
}

impl<C, P, N, PS, NodeView, V, R> Traversable<NodeView, N, PS, V, R> for RCFTree<C, P, N>
where
    C: Location,
    <C as TryFrom<usize>>::Error: Debug,
    usize: From<C>,
    P: Location,
    <P as TryFrom<usize>>::Error: Debug,
    usize: From<P>,
    N: Location,
    <N as TryFrom<usize>>::Error: Debug,
    usize: From<N>,
    PS: PointStore,
    NodeView: UpdatableNodeView<VectorNodeStore<C, P, N>, PS>,
    V: Visitor<NodeView, R>,
    R: Clone,
{
    fn traverse(
        &self,
        point: &[f32],
        parameters: &[usize],
        visitor_factory: fn(usize, &[usize], &VisitorInfo) -> V,
        visitor_info: &VisitorInfo,
        point_store: &PS,
        default: &R,
    ) -> R {
        if self.root == self.node_store.null_node() {
            return default.clone();
        }
        let mut visitor = visitor_factory(self.tree_mass, parameters, &visitor_info);
        let mut node_view = NodeView::create(self.root, &self.node_store);
        self.traverse_recursive(
            point,
            &mut node_view,
            &mut visitor,
            &visitor_info,
            &point_store,
        );
        visitor.result(visitor_info)
    }
}
