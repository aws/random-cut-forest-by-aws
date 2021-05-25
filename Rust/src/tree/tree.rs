extern crate num_traits;
use num_traits::Float;

extern crate rand;
use rand::SeedableRng;

extern crate rand_chacha;
use rand_chacha::ChaCha8Rng;

use std::cell::{Ref, RefMut, RefCell};
use std::iter::Sum;
use std::rc::Rc;

use crate::algorithm::Visitor;
use crate::store::{PointStore, NodeStore};
use crate::tree::{Cut, Node};

/// Random cut tree data structure on nodes and points.
///
/// A random cut tree contains leaf nodes and internal nodes.
/// [`Leaf`](crate::Leaf) nodes live at the leaves of the tree and mainly
/// represent a data point in the tree. [`Internal`](`crate::Internal`) nodes
/// mainly contain a [`BoundingBox`](`crate::BoundingBox`) on all of the points
/// in its subtree as well as a random [`Cut`] on that bounding box.
///
/// To store these nodes and points, a random cut tree contains a node store
/// and a point store. The [`NodeStore`] contains all of the nodes of the tree
/// of type `Leaf` or `Internal`. The [`PointStore`] contains the data points or
/// vectors stored in the leaves of the tree. In some cases it is sufficient for
/// a `Tree` to uniquely own its point store. In other cases, such as the case
/// where we want share points across multiple trees in a forest, this tree's
/// point store may live outside of this tree.
///
/// # Examples
///
/// ```
/// use random_cut_forest::Tree;
/// use random_cut_forest::tree::{AddResult, DeleteResult};
///
/// // create a new random cut tree with its own point store
/// let mut tree: Tree<f32> = Tree::new();
///
/// // add some (two-dimensional) data points to the tree
/// let result = tree.add_point(vec![0.0, 0.0]);
/// tree.add_point(vec![1.0, 2.0]);
/// tree.add_point(vec![0.2, 1.5]);
/// ```
pub struct Tree<T> {
    point_store: Rc<RefCell<PointStore<T>>>,
    node_store: NodeStore<T>,
    root_node: Option<usize>,
    rng: ChaCha8Rng,
}


impl<T> Tree<T>
    where T: Float + Sum
{

    /// Create a new `Tree` with a shared point store.
    ///
    /// Given a reference counted (shared) point store, create a new tree that
    /// gets points from this point store.
    ///
    /// # Examples
    ///
    /// ```
    /// extern crate slab;
    ///
    /// // create a shared point store
    /// use slab::Slab;
    /// use std::cell::RefCell;
    /// use std::rc::Rc;
    /// let point_store = Rc::new(RefCell::new(Slab::new()));
    ///
    /// use random_cut_forest::Tree;
    /// let tree: Tree<f32> = Tree::new_with_point_store(point_store);
    /// ```
    pub fn new_with_point_store(point_store: Rc<RefCell<PointStore<T>>>) -> Self {
        Tree {
            point_store: point_store.clone(),
            node_store: NodeStore::new(),
            root_node: None,
            rng: ChaCha8Rng::from_entropy(),
        }
    }

    /// Create a new `Tree`.
    ///
    /// The created tree will have its own private point store.
    ///
    /// # Examples
    ///
    /// ```
    /// use random_cut_forest::Tree;
    ///
    /// let tree: Tree<f32> = Tree::new();
    /// ```
    pub fn new() -> Self {
        Tree::new_with_point_store(Rc::new(RefCell::new(PointStore::new())))
    }

    /// Re-initializes the tree's random number generator with a seed.
    ///
    /// On construction, the tree's random number generator is initialized
    /// using the host system's random number generator. This function
    /// reconstructs a random number generator from a specified seed.
    ///
    /// Random cut trees use the [`ChaCha8Rng`][cha] random number generator.
    /// It has fast initialization, high throughput and relatively small memory
    /// footprint.
    ///
    /// # Examples
    ///
    /// ```
    /// use random_cut_forest::Tree;
    ///
    /// let mut tree: Tree<f32> = Tree::new();
    /// tree.seed(42);
    /// ```
    ///
    /// [cha]: https://rust-random.github.io/rand/rand_chacha/struct.ChaCha8Rng.html
    pub fn seed(&mut self, seed: u64) {
        self.rng = ChaCha8Rng::seed_from_u64(seed);
    }

    /// Return the number of points in the tree's point store.
    ///
    /// It is important to note that if this is a shared point store then this
    /// function returns the total number of points in the point store. This
    /// number is not directly related to the number of points used by this
    /// particular tree.
    ///
    /// # Examples
    ///
    /// ```
    /// use random_cut_forest::Tree;
    ///
    /// // create a tree with a private point store
    /// let mut tree: Tree<f32> = Tree::new();
    /// assert_eq!(tree.num_points(), 0);
    ///
    /// tree.add_point(vec![0.0, 0.0]);
    /// assert_eq!(tree.num_points(), 1);
    /// ```
    pub fn num_points(&self) -> usize {
        let store = self.point_store.borrow();
        store.len()
    }

    /// Returns the mass of the tree.
    ///
    /// The mass of the tree is equal to the sum of the masses of the nodes.
    /// This quantity is tracked during point addition and deletion such that
    /// the mass of the root node is always equal to the mass of the tree.
    ///
    /// # Examples
    ///
    /// ```
    /// use random_cut_forest::Tree;
    ///
    /// let mut tree: Tree<f32> = Tree::new();
    /// assert_eq!(tree.mass(), 0);
    ///
    /// tree.add_point(vec![0.0, 0.0]);
    /// assert_eq!(tree.mass(), 1);
    ///
    /// tree.add_point(vec![1.0, 1.0]);
    /// assert_eq!(tree.mass(), 2);
    ///
    /// tree.add_point(vec![0.0, 0.0]);
    /// assert_eq!(tree.mass(), 3);
    /// ```
    pub fn mass(&self) -> u32 {
        match self.root_node {
            None => 0,
            Some(key) => self.node_store.get(key).unwrap().mass(),
        }
    }

    /// Returns an iterator on nodes and depths.
    ///
    /// Given a query point, a random cut tree iteration begins at the root node
    /// of the tree and returns a branch to the leaf node which is approximately
    /// closest to a query point in the L1-norm. (Relative to the random cuts
    /// chosen in the tree.)
    ///
    /// See [`NodeTraverser`] for more information.
    ///
    ///
    /// # Examples
    ///
    /// ```
    /// use random_cut_forest::{Node, Tree};
    ///
    /// let mut tree: Tree<f32> = Tree::new();
    /// tree.seed(0);  // seed this test to fix a random cut
    /// tree.add_point(vec![0.0, 1.0]);
    ///
    /// // check that we recover the only node in the tree
    /// let query = vec![0.1, 0.9];
    /// let nodes: Vec<&Node<f32>> = tree.iter(&query).collect();
    /// assert_eq!(nodes.len(), 1);
    ///
    /// // after adding a second point the traversal should contain the
    /// // point closest to the query in the L1 norm
    /// tree.add_point(vec![-1.0, -2.0]);
    /// let nodes: Vec<&Node<f32>> = tree.iter(&query).collect();
    /// assert_eq!(nodes.len(), 2);
    ///
    /// // a traversal implements iter, so we can use it in a loop
    /// for node in tree.iter(&query) {
    ///     println!("mass = {}", node.mass());
    /// }
    /// ```
    pub fn iter<'a>(&'a self, point: &'a Vec<T>) -> NodeTraverser<'a, T> {
        NodeTraverser::new(self, point)
    }

    /// Apply a visitor to a tree traversal with a query point.
    ///
    /// Given a query point and a visitor, we apply the visitor to the nodes
    /// of the query point iteration as given by [`iter()`](Self::iter).
    ///
    /// See the [`Visitor`] trait for more information.
    ///
    /// # Examples
    ///
    /// ```
    /// use random_cut_forest::Tree;
    /// use random_cut_forest::algorithm::AnomalyScoreVisitor;
    /// ```
    ///
    pub fn traverse<'a, U, V>(
        &'a self, point: &'a Vec<T>,
        visitor: &mut V,
    ) -> U where V: Visitor<T, Output=U>
    {
        let nodes: Vec<&Node<T>> = self.iter(point).collect();
        for (depth, node) in nodes.iter().enumerate().rev() {
            let depth: T = T::from(depth).unwrap();
            match node {
                Node::Leaf(leaf) => visitor.accept_leaf(leaf, depth),
                Node::Internal(node) => visitor.accept(node, depth),
            }
        }
        visitor.get_result()
    }

    // =========================================================================
    // Helper Functions
    //
    // Mostly used by tree_point_addition and tree_point_deletion.
    // =========================================================================
    #[inline(always)]
    pub fn root_node(&self) -> Option<usize> { self.root_node }

    #[inline(always)]
    pub fn set_root_node(&mut self, root_key: Option<usize>) {
        self.root_node = root_key;
    }

    #[inline(always)]
    pub fn borrow_point_store(&self) -> Ref<PointStore<T>> { self.point_store.borrow() }

    #[inline(always)]
    pub fn borrow_mut_point_store(&self) -> RefMut<PointStore<T>> { self.point_store.borrow_mut() }

    #[inline(always)]
    pub fn node_store(&self) -> &NodeStore<T> { &self.node_store }

    #[inline(always)]
    pub fn node_store_mut(&mut self) -> &mut NodeStore<T> { &mut self.node_store }

    #[inline(always)]
    pub fn rng_mut(&mut self) -> &mut ChaCha8Rng { &mut self.rng }

    #[inline(always)]
    pub fn get_node(&self, node_key: usize) -> &Node<T> {
        self.node_store().get(node_key).unwrap()
    }

    #[inline(always)]
    pub fn get_node_mut(&mut self, node_key: usize) -> &mut Node<T> {
        self.node_store_mut().get_mut(node_key).unwrap()
    }

    #[inline(always)]
    pub fn get_parent(&self, node_key: usize) -> Option<usize> {
        self.get_node(node_key).parent()
    }
}


/// A type for traversing nodes from root to the nearest leaf.
///
/// Given an input data point/vector, this type traces the path from the root
/// node of a tree to the leaf node nearest to the input. Returned by
/// [`Tree::traverse`].
///
pub struct NodeTraverser<'a, T> {
    tree: &'a Tree<T>,
    point: &'a Vec<T>,
    current_node_key: Option<usize>,
}

impl<'a, T> NodeTraverser<'a, T>
    where T: Float + Sum
{

    /// Create a new node traverser from a tree and a query point.
    ///
    /// # Examples
    ///
    /// ```
    /// use random_cut_forest::Tree;
    /// use random_cut_forest::tree::NodeTraverser;
    ///
    /// let mut tree = Tree::new();
    /// let point = vec![0.0, 0.0];
    ///
    /// // create a new node traverser from a tree and a query point
    /// let mut node_traverser = NodeTraverser::new(&tree, &point);
    /// ```
    pub fn new(tree: &'a Tree<T>, point: &'a Vec<T>) -> Self {
        NodeTraverser {
            tree: tree,
            point: point,
            current_node_key: tree.root_node(),
        }
    }

    /// Return the key of the next node in a traversal.
    fn next_node_key(&mut self, node: &Node<T>) -> Option<usize> {
        match node {
            Node::Leaf(_) => None,
            Node::Internal(node) => {
                if Cut::is_left_of(self.point, node.cut()) {
                    Some(node.left())
                } else {
                    Some(node.right())
                }
            }
        }
    }
}

impl<'a, T> Iterator for NodeTraverser<'a, T>
    where T: Float + Sum
{
    type Item = &'a Node<T>;

    fn next(&mut self) -> Option<&'a Node<T>> {
        match self.current_node_key {
            Some(node_key) => {
                let node = self.tree.node_store().get(node_key).unwrap();
                self.current_node_key = self.next_node_key(node);
                Some(node)
            },
            None => None
        }
    }
}


#[cfg(test)]
mod tests {
    use rand::Rng;
    use rand_distr::StandardNormal;

    use crate::tree::{AddResult, DeleteResult, Node};
    use super::*;

    fn generate_random_normal(dimension: usize, num_points: usize) -> Vec<Vec<f32>> {
        let mut points: Vec<Vec<f32>> = Vec::with_capacity(num_points);
        let mut rng = rand::thread_rng();
        for _ in 0..num_points {
            let mut point: Vec<f32> = Vec::with_capacity(dimension);
            for _ in 0..dimension {
                point.push(rng.sample(StandardNormal));
            }
            points.push(point);
        }
        points
    }

    #[test]
    fn test_traversal() {
        let mut tree: Tree<f32> = Tree::new();
        let query = vec![10.0, 10.0, 10.0, 10.0];
        let nodes: Vec<&Node<f32>> = tree.iter(&query).collect();
        assert_eq!(nodes.len(), 0);

        // add a bunch of N(0,1) points to the tree including the query point
        for point in generate_random_normal(4, 32) {
            tree.add_point(point);
        }
        tree.add_point(query.clone());

        // traverse the tree. the leaf node should contain the query point
        for node in tree.iter(&query) {
            match node {
                Node::Internal(_) => (),
                Node::Leaf(n) => {
                    let store = tree.borrow_point_store();
                    let point = store.get(n.point()).unwrap();
                    assert_eq!(*point, query);
                }
            }
        }
    }

    #[test]
    fn test_store_sizes() {
        let mut tree: Tree<f32> = Tree::new();
        assert_eq!(tree.num_points(), 0);
        assert_eq!(tree.node_store().len(), 0);

        tree.add_point(vec![-1.0, -1.0]);
        assert_eq!(tree.num_points(), 1);
        assert_eq!(tree.node_store().len(), 1);

        // add points
        let points = vec![vec![0.0, 0.0], vec![1.0, 1.0], vec![2.0, 2.0], vec![3.0, 3.0]];
        for (i, point) in points.iter().enumerate() {
            let result = tree.add_point(point.clone());
            assert!(std::matches!(result, AddResult::AddedPoint {..}));
            assert_eq!(tree.num_points(), i + 2);
            assert_eq!(tree.node_store().len(), 2*i + 3);
        }

        // re-add points
        let num_points = tree.num_points();
        for point in points.iter()  {
            let result = tree.add_point(point.clone());
            assert!(std::matches!(result, AddResult::MassIncreased {..}));
            assert_eq!(tree.num_points(), num_points);
            assert_eq!(tree.node_store().len(), 2*num_points - 1);
        }

        // delete points: check mass decrease
        for point in points.iter() {
            let result = tree.delete_point(point);
            assert!(std::matches!(result, DeleteResult::MassDecreased {..}));
            assert_eq!(tree.num_points(), num_points);
            assert_eq!(tree.node_store().len(), 2*num_points - 1);
        }

        // delete points: check deleted
        for (i, point) in points.iter().enumerate() {
            let result = tree.delete_point(point);
            assert!(std::matches!(result, DeleteResult::DeletedPoint {..}));
            assert_eq!(tree.num_points(), num_points - (i + 1));
            assert_eq!(tree.node_store().len(), 2*(num_points - (i+1)) - 1);
        }
    }

    /// Traverses the tree to check if the node masses are consistent
    fn check_node_masses<T: Float + Sum>(tree: &Tree<T>, node_idx: usize) -> u32
    {
        let node = tree.node_store().get(node_idx).unwrap();
        let mass = node.mass();
        match node {
            Node::Internal(internal) => {
                let left_mass = check_node_masses(tree, internal.left());
                let right_mass = check_node_masses(tree, internal.right());
                assert_eq!(mass, left_mass + right_mass);
                mass
            }
            Node::Leaf(_) => {
                mass
            }
        }
    }

    #[test]
    fn test_tree_mass() {
        let mut tree: Tree<f32> = Tree::new();
        assert_eq!(tree.mass(), 0);

        // add points and check
        let points = vec![vec![0.0, 0.0], vec![1.0, 1.0], vec![2.0, 2.0], vec![3.0, 3.0]];
        for (i, point) in points.iter().enumerate() {
            tree.add_point(point.clone());
            assert_eq!(tree.mass(), i as u32 + 1);
            check_node_masses(&tree, tree.root_node().unwrap());
        }

        // delete points and check
        let mass = tree.mass();
        for (i, point) in points.iter().enumerate() {
            tree.delete_point(point);
            assert_eq!(tree.mass(), mass - i as u32 - 1);
            if let Some(root_key) = tree.root_node() {
                check_node_masses(&tree, root_key);
            }
        }
   }
}