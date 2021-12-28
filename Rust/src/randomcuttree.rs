use std::fmt::Debug;
use std::mem::size_of;
use crate::newnodestore::{NewNodeStore, NodeStoreView};
use crate::pointstore::PointStore;
use crate::boundingbox::BoundingBox;
use crate::cut::Cut;
use crate::rcf::{damp, Max};
extern crate rand;
use rand::SeedableRng;
extern crate rand_chacha;
use crate::rcf::score_seen;
use crate::rcf::score_unseen;
use crate::rcf::normalizer;
use crate::pointstore::PointStoreView;
use rand_chacha::ChaCha20Rng;
use crate::abstractnodeview::AbstractNodeView;
use crate::randomcuttree::rand::RngCore;
use crate::randomcuttree::rand::Rng;
use crate::scalarscorevisitor::ScalarScoreVisitor;
use crate::visitor::Visitor;

pub type StoreInUse = NewNodeStore<u8,u16,u8>;

#[repr(C)]
pub struct RCFTree<C,P,N> {
	dimensions: usize,
	capacity: usize,
	node_store: NewNodeStore<C,P,N>,
	random_seed: u64,
	root: usize,
	tree_mass: usize,
	using_transforms: bool
}

impl<C: Max + Copy, P: Max + Copy, N: Max + Copy> RCFTree<C,P,N> where
	C: std::convert::TryFrom<usize>, usize: From<C>,
	P: std::convert::TryFrom<usize>, usize: From<P>,
	N: std::convert::TryFrom<usize>, usize: From<N>,
{
	pub fn new(dimensions: usize,
			   capacity: usize, using_transforms: bool, bounding_box_cache_fraction: f64, random_seed: u64) -> Self
		where <C as TryFrom<usize>>::Error: Debug, <P as TryFrom<usize>>::Error: Debug, <N as TryFrom<usize>>::Error: Debug{
		let project_to_tree: fn(Vec<f32>) -> Vec<f32> = {|x| x};
		RCFTree {
			dimensions,
			capacity,
			using_transforms,
			random_seed,
			node_store: NewNodeStore::<C,P,N>::new(capacity, dimensions, using_transforms, project_to_tree, bounding_box_cache_fraction),
			root: 0,
			tree_mass: 0
		}
	}

	pub fn add(&mut self, point_index: usize, point_attribute: usize, point_store: &dyn PointStoreView) -> usize
		where <C as TryFrom<usize>>::Error: Debug, <P as TryFrom<usize>>::Error: Debug,  <N as TryFrom<usize>>::Error: Debug {
		if self.root == 0 {
			self.root = self.node_store.add_leaf(0, point_index, 1);
			self.tree_mass = 1;
			point_index
		} else {
			let point = &point_store.get_copy(point_index);
			let mut leaf_path = self.node_store.get_path(self.root, point);
			let (leaf_node, leaf_saved_sibling) = leaf_path.pop().unwrap();
			let mut sibling = leaf_saved_sibling;
			let leaf_point_index = self.node_store.get_point_index(leaf_node);
			let old_point = &point_store.get_copy(leaf_point_index);
			let mut saved_parent = 0;
			if leaf_path.len() != 0 {
				saved_parent = leaf_path.last().unwrap().0;
			}

			self.tree_mass += 1;
			if point.eq(old_point) {
				self.node_store.increase_leaf_mass(leaf_node);
				if saved_parent != 0 {
					self.node_store.manage_ancestors_add(leaf_path,point,point_store);
				}
				return leaf_point_index
			} else {
				let mut node = leaf_node;
				let mut saved_node = node;
				let mut parent = saved_parent;
				let mut saved_mass = self.node_store.get_mass(leaf_node);
				let mut saved_cut_value: f32 = 0.0;
				let mut current_box = BoundingBox::new(old_point, old_point);
				let mut saved_box = current_box.copy();
				let mut saved_dim: usize = usize::MAX; // deliberate, to be converted later
				let mut parent_path : Vec<(usize,usize)> = Vec::new();
				let mut rng = ChaCha20Rng::seed_from_u64(self.random_seed);
				self.random_seed = rng.next_u64();
				loop {
					let factor: f64 = rng.gen();
					let (new_cut, separation, inside_box) = current_box.get_cut_and_separation(factor, point,false);
					if separation {
						saved_cut_value = new_cut.get_value();
						saved_dim = new_cut.get_dimension();
						saved_parent = parent;
						saved_node = node;
						saved_box = current_box.copy();
						parent_path.clear();
					} else {
						parent_path.push((node,sibling));
					}

					if (saved_dim == usize::MAX) {
						println!("cut failed ");
						panic!()
					}


					if parent == 0 {
						break;
					} else {
						self.node_store.grow_node_box(&mut current_box, point_store, parent, sibling);
						let (a, b) = leaf_path.pop().unwrap();
						node = a;
						sibling = b;
						parent = if leaf_path.len() != 0 { leaf_path.last().unwrap().0 } else { 0 };
					}
				}
				let new_leaf_node = self.node_store.add_leaf(0, point_index, 1);
				let mut merged_node: usize;
				let saved_mass = self.node_store.get_mass(saved_node);

				if point[saved_dim] <= saved_cut_value {
					merged_node = self.node_store.add_node(saved_parent, new_leaf_node, saved_node, saved_dim.try_into().unwrap(), saved_cut_value, saved_mass + 1);
			        //self.node_store.check_right(saved_node,saved_dim,saved_cut_value,point_store);
				} else {
					merged_node = self.node_store.add_node(saved_parent, saved_node, new_leaf_node, saved_dim.try_into().unwrap(), saved_cut_value, saved_mass + 1);
					//self.node_store.check_left(saved_node,saved_dim,saved_cut_value,point_store);
				}

				saved_box.check_contains_and_add_point(point);
				self.node_store.add_box(merged_node, &saved_box);

				if saved_parent != 0 {
					// add the new node
					self.node_store.replace_node(saved_parent, saved_node, merged_node);

					while(!parent_path.is_empty()){
						leaf_path.push(parent_path.pop().unwrap());
					}
					// fix bounding boxes and mass
					self.node_store.manage_ancestors_add(leaf_path, point, point_store);
				} else {
					self.root = merged_node;
				}
			}
			point_index
		}
	}

	pub fn delete(&mut self, point_index: usize, point_attribute: usize, point_store: &dyn PointStoreView) -> usize
	 where <C as TryFrom<usize>>::Error: Debug, <P as TryFrom<usize>>::Error: Debug,  <N as TryFrom<usize>>::Error: Debug
	{
		if self.root == 0 {
			println!(" deleting from an empty tree");
			panic!();
		}
		self.tree_mass = self.tree_mass - 1;
		let point = &point_store.get_copy(point_index);
		let mut leaf_path = self.node_store.get_path(self.root, point);
		let (leaf_node, leaf_saved_sibling) = leaf_path.pop().unwrap();

		let leaf_point_index = self.node_store.get_point_index(leaf_node);

		if (leaf_point_index != point_index) {
			if !point_store.is_equal(point,leaf_point_index) {
				println!(" deleting wrong node; looking for {} found {}", point_index, leaf_point_index);
				let old_point =point_store.get_copy(leaf_point_index);
				for j in 0..self.dimensions.into() {
					println!("want {} found {}", point[j], old_point[j]);
				}
				panic!();
			}
		}

		if self.node_store.decrease_leaf_mass(leaf_node) == 0 {
			if leaf_path.len() == 0 {
				self.root = 0;
			} else {
				let (parent, sibling) = leaf_path.pop().unwrap();
				let grand_parent = if leaf_path.len() == 0 { 0} else { leaf_path.last().unwrap().0};

				if grand_parent == 0 {
					self.root = leaf_saved_sibling;
					self.node_store.set_root(self.root);
				} else {
					self.node_store.replace_node(grand_parent, parent, leaf_saved_sibling);
					self.node_store.manage_ancestors_delete(leaf_path, point, point_store);
				}
				self.node_store.delete_internal_node(parent);
			}
		}
		leaf_point_index
	}

	pub fn dynamic_score(&self, point: &[f32], point_store: &dyn PointStoreView, ignore_mass: usize, score_seen: fn (usize,usize) -> f64,
						 score_unseen :fn (usize,usize) -> f64, damp : fn (usize,usize) -> f64,
						 normalizer: fn (f64,usize) -> f64) -> f64
		where <C as TryFrom<usize>>::Error: Debug, <P as TryFrom<usize>>::Error: Debug,  <N as TryFrom<usize>>::Error: Debug {
		if self.root == 0 {
			return 0.0;
		}
		self.node_store.dynamic_score(self.root, ignore_mass,self.tree_mass, point, point_store,score_seen,score_unseen,damp,normalizer)
	}

	pub fn generic_dynamic_score(&self, point: &[f32], point_store: &dyn PointStoreView, ignore_mass: usize, score_seen: fn (usize,usize) -> f64,
								 score_unseen :fn (usize,usize) -> f64, damp : fn (usize,usize) -> f64,
								 normalizer: fn (f64,usize) -> f64) -> f64 {
		if self.root == 0 {
			return 0.0;
		}

		let mut visitor = ScalarScoreVisitor::new(self.tree_mass,ignore_mass,damp,score_seen,score_unseen,normalizer);
		let mut node_view = AbstractNodeView::new(self.dimensions,self.root);
		node_view.traverse(&mut visitor,point,point_store,&self.node_store);
		normalizer(visitor.get_result(),self.tree_mass)
	}


	pub fn get_size(&self) -> usize {
		self.node_store.get_size(self.dimensions.into()) + std::mem::size_of::<RCFTree<C,P,N>>()
	}
}

