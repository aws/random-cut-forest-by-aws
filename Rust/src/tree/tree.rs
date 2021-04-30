extern crate rand;
use rand::SeedableRng;

extern crate rand_chacha;
use rand_chacha::ChaCha8Rng;

use std::cell::{Ref, RefMut, RefCell};
use std::rc::Rc;

use crate::RCFFloat;
use crate::store::{PointStore, NodeStore};
use crate::tree::{BoundingBox, Cut, Internal, Node};


/// The result of a point addition operation.
/// 
/// The `AddedPoint` result contains the key of the point that was added to the
/// tree. The `MassIncreased` result contains the key of the point whose mass
/// was increased.
pub enum AddResult {
    AddedPoint(usize),
    MassIncreased(usize),
}

/// Description of the result of a point deletion operation on a tree by a 
/// `PointDeleter`.
/// 
/// This enum has the following possible values:
/// * `EmptyTree` - the deletion was performed on an empty tree
/// * `PointNotFound` - the point could not be found in the tree
/// * `DeletedPoint(usize)` - the point with the given key in the tree's point
///    store was deleted from the tree
/// * `MassDecreased(idx)` - the point with the given key in the tree's point
///    store had its mass reduced
pub enum DeleteResult {
    EmptyTree,
    PointNotFound,
    DeletedPoint(usize),
    MassDecreased(usize),
}

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


impl<T: RCFFloat> Tree<T> {

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
    /// Given a query point, a random cut tree traversal begins at the root node
    /// of the tree and returns a branch to the leaf node which is approximately
    /// closest to a query point in the L1-norm. (Relative to the random cuts
    /// chosen in the tree.)
    ///
    /// See [`NodeTraverser`] for more information.
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
    /// let nodes: Vec<&Node<f32>> = tree.traverse(&query).collect();
    /// assert_eq!(nodes.len(), 1);
    ///
    /// // after adding a second point the traversal should contain the
    /// // point closest to the query in the L1 norm
    /// tree.add_point(vec![-1.0, -2.0]);
    /// let nodes: Vec<&Node<f32>> = tree.traverse(&query).collect();
    /// assert_eq!(nodes.len(), 2);
    ///
    /// // a traversal implements iter, so we can use it in a loop
    /// for node in tree.traverse(&query) {
    ///     println!("mass = {}", node.mass());
    /// }
    /// ```
    pub fn traverse<'a>(&'a self, point: &'a Vec<T>) -> NodeTraverser<'a, T> {
        NodeTraverser::new(self, point)
    }

    // ########################################################################
    // Point Addition
    // ########################################################################

    /// Add a point to the tree.
    ///
    /// The input point is added to the tree's point store. The location in the
    /// tree is randomly determined based on a choice of random cut. An
    /// [`AddResult`] is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use random_cut_forest::Tree;
    /// use random_cut_forest::tree::AddResult;
    ///
    /// let mut tree: Tree<f32> = Tree::new();
    ///
    /// let result = tree.add_point(vec![0.0, 0.0]);
    /// assert!(std::matches!(result, AddResult::AddedPoint {..} ));
    ///
    /// let result = tree.add_point(vec![0.0, 0.0]);
    /// assert!(std::matches!(result, AddResult::MassIncreased {..} ));
    /// ```
    pub fn add_point(&mut self, point: Vec<T>) -> AddResult {
        match self.root_node() {
            Some(root_key) => self.add_point_by_node(point, root_key),
            None => {
                let point_key = {
                    let mut store = self.borrow_mut_point_store();
                    store.insert(point)
                };
                let node = Node::new_leaf(point_key);
                let node_key = self.node_store_mut().insert(node);
                self.set_root_node(Some(node_key));
                AddResult::AddedPoint(point_key)
            }
        }
    }

    /// Main recursive point addition algorithm givena new point and a current
    /// node.
    /// 
    /// Steps of the point addition algorithm:
    /// 
    /// 1. Check if the current node is leaf representing the same point as the
    ///    one being inserted. If so, increase the mass of this leaf node and
    ///    return.
    /// 2. Compute the bounding box made by the merging of the new point with 
    ///    the current node and create a random cut on this bounding box. If
    ///    the cut separates the point from the original contents of the current
    ///    node then create a new leaf node at this level. See 
    ///    `insert_new_leaf()` for more information.
    /// 3. Otherwise, recurse to the left or right of the tree depending on the
    ///    location of the point relative to the proposed random cut.
    /// 4. When traversing back up the tree via recursion callback we update the
    ///    bounding boxes along the way using the merged boxes computed on the
    ///    way down.
    /// 
    fn add_point_by_node(&mut self, point: Vec<T>, node_key: usize) -> AddResult {
        // 1. this check will in-place increase the mass of the current node if
        // it is a leaf node. Is there a monadic way to do this?
        if self.increased_mass_at_node(&point, node_key) {
            if let Node::Leaf(leaf) = self.get_node(node_key) {
                return AddResult::MassIncreased(leaf.point());
            } else {
                panic!("Inconsistent node: expected leaf when increasing point mass")
            }
        }

        // 2. Shortcut this step if the new point is inside the existing
        // bounding box. We need to go deeper to find separation.
        let merged_box = self.merge_node_with_point(node_key, &point);
        if !self.point_inside_node(&point, node_key) {
            let cut = Cut::new_random_cut(&merged_box, self.rng_mut()).unwrap();
            let (min, max) = self.range_on_cut_dimension(node_key, &cut);
            if (cut.value() < min) || (max <= cut.value()) {
                let new_point_key = self.insert_new_leaf(
                    point, node_key, merged_box, cut, min);
                return AddResult::AddedPoint(new_point_key);
            }
        }

        // 3. The new point is contained in the current node's bounding box or
        // the proposed cut did not separate the point from said bounding box.
        // Determine if the new point is to the left or right of the *original*
        // cut at this node and recurse in the appropriate direction.
        let (cut, left, right) = match self.get_node(node_key) {
            Node::Internal(n) => (n.cut(), n.left(), n.right()),
            Node::Leaf(_) => panic!("Inconsistent node: unexpected leaf")
        };
        let result = match Cut::is_left_of(&point, cut) {
            true => self.add_point_by_node(point, left),
            false => self.add_point_by_node(point, right),
        };

        // 4. update the bounding boxes with the merged boxes, as well as the
        // masses, when traversing back up the tree
        if let Node::Internal(node) = self.get_node_mut(node_key) {
            node.set_bounding_box(merged_box);
            node.increment_mass();
        }
        result
    }

    /// If the current node is a leaf *and* its point is equal to that of the
    /// input point then increase the mass of this leaf and return `true`.
    /// Otherwise, return `false`.
    /// 
    /// TODO: this function is a bit strange because it will in-place increase
    /// the mass but won't do anything (and return `false`) otherwise. This
    /// information is then used in `add_point_at_node()` to determine if we
    /// should return. There must be a better way to do this using `Result<>` or
    /// something.
    fn increased_mass_at_node(&mut self, point: &Vec<T>, node_key: usize) -> bool {
        let leaf_with_same_point = match self.get_node(node_key) {
            Node::Internal(_) => false,
            Node::Leaf(leaf) => {
                // TODO - are there easier ways to access the point store?
                let store = self.borrow_point_store();
                let leaf_point = store.get(leaf.point()).unwrap();
                *leaf_point == *point
            }
        };

        if leaf_with_same_point {
            self.get_node_mut(node_key).increment_mass();
        }

        leaf_with_same_point
    }

    /// Returns a bounding box formed by the merging of the input point with
    /// the contents of the given node.
    /// 
    /// If the node is a leaf then the bounding box is just formed by these two
    /// points. If the node is internal then we return a new bounding box formed
    /// by the merging of the bounding box at this node with the new point.
    fn merge_node_with_point(
        &self,
        node_key: usize,
        point: &Vec<T>,
    ) -> BoundingBox<T> {
        match self.get_node(node_key) {
            Node::Leaf(leaf) => {
                let store = self.borrow_point_store();
                let leaf_point = store.get(leaf.point()).unwrap();
                let bounding_box = BoundingBox::new_from_point(leaf_point);
                return BoundingBox::merged_box_with_point(
                    &bounding_box, point);
            },
            Node::Internal(internal) => {
                return BoundingBox::merged_box_with_point(
                    internal.bounding_box(), point);
            }
        }
    }

    /// Given a proposed cut, return the range of values on the bounding box at
    /// the current node in the dimension of the cut.
    /// 
    /// For example, suppose the bounding box is two-dimensional with min values
    /// `(0, 1)` and max values `(2, 4)`. If the input cut is along dimension 
    /// 0 then the output range is `(0, 2)`. Otherwise, if the cut is along 
    /// dimension 1 then the output range is `(1, 4)`.
    fn range_on_cut_dimension(&self, node_key: usize, cut: &Cut<T>) -> (T, T) {
        let dim = cut.dimension();
        match self.get_node(node_key) {
            Node::Leaf(leaf) => {
                let store = self.borrow_point_store();
                let leaf_point = store.get(leaf.point()).unwrap();
                (leaf_point[dim], leaf_point[dim])
            },
            Node::Internal(internal) => {
                let min_values = internal.bounding_box().min_values();
                let max_values = internal.bounding_box().max_values();
                (min_values[dim], max_values[dim])
            }
        }
    }

    /// Insert a new leaf node into the tree containing the input point.
    /// 
    /// When this function is called we are at a node in the tree where the 
    /// merged box (between this node and the new point) has a proposed cut
    /// that separates the point from original bounding box at this node. 
    /// 
    /// Our current tree state is:
    ///     
    /// ```text
    ///       A        N = current node
    ///      / \       A = parent
    ///     S   N      S = sibling of N
    ///        / \
    /// ```
    /// 
    /// This needs to be transformed to:
    /// 
    /// ```text
    ///       A        N = current node
    ///      / \       A = parent
    ///     S   B      S = (former) sibling of N
    ///        / \     B = new "merged node" formed from the merged bounding box
    ///       N   P    P = new leaf node
    ///      / \
    /// ```
    /// 
    /// Note that the parent node and all nodes above the newly created node `B`
    /// will need to have their bounding boxes and masses updated. This is done
    /// in the reverse recursion in `add_point_at_node()`. 
    /// 
    /// Returns the key of the newly inserted point in the tree's point store.
    fn insert_new_leaf(
        &mut self,
        point: Vec<T>,
        node_key: usize,
        merged_box: BoundingBox<T>, 
        proposed_cut: Cut<T>,
        min: T,
    ) -> usize {
        let parent_key = self.get_parent(node_key);

        // P: new leaf node.
        let new_point_key = self.insert_point(point);
        let new_leaf = Node::new_leaf(new_point_key);
        let new_leaf_key = self.insert_node(new_leaf);

        // B: new merged node. update parent to node's parent
        let (left, right) = if min > proposed_cut.value() {
            (new_leaf_key, node_key) 
        } else {
            (node_key, new_leaf_key)
        };
        let mut merged_node = Node::new_internal(left, right, merged_box, proposed_cut);
        if let Some(key) = parent_key { merged_node.set_parent(Some(key)); }
        let merged_node_key = self.insert_node(merged_node);

        // update parent-child relationships with new merged node.
        self.get_node_mut(node_key).set_parent(Some(merged_node_key));
        self.get_node_mut(new_leaf_key).set_parent(Some(merged_node_key));

        // update parent-child relationship with the parent node. if the parent
        // node is `None` then the merged node becomes the new tree root
        if let Some(parent_key) = parent_key {
            if let Node::Internal(parent) = self.get_node_mut(parent_key) {
                if parent.left() == node_key {
                    parent.set_left(merged_node_key);
                } else if parent.right() == node_key {
                    parent.set_right(merged_node_key);
                } else {
                    panic!("Inconsistent node: broken parent-child relationship");
                }
            } else {
                panic!("Inconsistent node: parent should not be a leaf node ")
            }
        } else {
            self.set_root_node(Some(merged_node_key));
        }

        // finally, set the mass of the merged node to the sum of the masses
        // of its children: the original node N and the new point P
        let node_mass = self.get_node(node_key).mass();
        self.get_node_mut(merged_node_key).set_mass(node_mass + 1);
        return new_point_key;
    }

    // ########################################################################
    // Point Deletion
    // ########################################################################

    /// Delete a point from the tree.
    ///
    /// The input point is removed from the tree's point store. A
    /// [`DeleteResult`] is returned, providing information on which point was
    /// deleted or if the point was not actually contained by the tree.
    ///
    /// # Examples
    ///
    /// ```
    /// use random_cut_forest::Tree;
    /// use random_cut_forest::tree::DeleteResult;
    ///
    /// let mut tree: Tree<f32> = Tree::new();
    /// tree.add_point(vec![0.0, 0.0]);
    /// tree.add_point(vec![0.0, 0.0]);
    ///
    /// // deleting a point with mass greater than one
    /// let result = tree.delete_point(&vec![0.0, 0.0]);
    /// assert!(std::matches!(result, DeleteResult::MassDecreased {..} ));
    ///
    /// // deleting a point that doesn't live in the tree
    /// let result = tree.delete_point(&vec![42.0, 123.0]);
    /// assert!(std::matches!(result, DeleteResult::PointNotFound));
    ///
    /// // deleting a point with mass equal to one
    /// let result = tree.delete_point(&vec![0.0, 0.0]);
    /// assert!(std::matches!(result, DeleteResult::DeletedPoint {..} ));
    ///
    /// // deleting a point from an empty tree
    /// let result = tree.delete_point(&vec![0.0, 0.0]);
    /// assert!(std::matches!(result, DeleteResult::EmptyTree));
    /// ```
    pub fn delete_point(&mut self, point: &Vec<T>) -> DeleteResult {
        match self.root_node() {
            None => DeleteResult::EmptyTree,
            Some(node_key) => self.delete_point_by_node(point, node_key),
        }
    }

    fn delete_point_by_node(
        &mut self,
        point: &Vec<T>,
        node_key: usize,
    ) -> DeleteResult {
        match self.get_node(node_key) {
            Node::Internal(_) => self.visit_internal(point, node_key),
            Node::Leaf(_) => self.visit_leaf(point, node_key),
        }
    }
    
    /// Deletion algorithm at an internal node.
    /// 
    /// We search for the point to delete by following cuts down the tree. That
    /// is, at a particular internal node we check if the input point to delete
    /// is on the left or right of the node's cut and select one of the node's
    /// children, accordingly.
    /// 
    /// One a leaf node is encountered and the leaf node contains the point that
    /// is to be deleted then we recurse back up, updating internal node
    /// bounding boxes as necessary.
    fn visit_internal(&mut self, point: &Vec<T>, node_key: usize) -> DeleteResult {
        // 1. Determine which side of the current node's cut lies the point and
        // recurse in that direction.
        let next_node_key = {
            if let Node::Internal(node) = self.get_node(node_key) {
                if Cut::is_left_of(point, node.cut()) {
                    node.left()
                } else {
                    node.right()
                }
            } else { panic!("Inconsistent node: expected non-leaf node"); }
        };

        // Recurse. Skip updating bounding boxes if a point was not deleted
        // or if the mass of a point was decreased. Skip if the current node
        // was deleted
        let result = self.delete_point_by_node(point, next_node_key);
        match result {
            DeleteResult::EmptyTree => return result,
            DeleteResult::PointNotFound => return result,
            DeleteResult::MassDecreased(_) => return result,
            DeleteResult::DeletedPoint(_) => {
                if !self.contains_node(node_key) {
                    return result;
                }
            },
        }

        // 2. As we traverse back up the tree, rebuild the node's bounding box.
        // This is done simply by merging the bounding boxes of the left and
        // right nodes. Also decrement mass since we deleted a point.
        let merged_box = {
            if let Node::Internal(node) = self.get_node(node_key) {
                let left = self.get_node(node.left());
                let right = self.get_node(node.right());
                self.merged_box_from_nodes(&left, &right)
            } else { panic!("Inconsistent node: expected non-leaf node"); }
        };
        if let Node::Internal(node) = self.get_node_mut(node_key) {
            node.set_bounding_box(merged_box);
            node.decrement_mass();
        } else { panic!("Inconsistent node: expected non-leaf node"); }
        
        result
    }

    /// Returns the bounding box formed by the merging of point or bounding 
    /// boxes of two different nodes.
    fn merged_box_from_nodes(&self, left: &Node<T>, right: &Node<T>) -> BoundingBox<T> {
        let left_bbox = self.node_bounding_box(left);
        let right_bbox = self.node_bounding_box(right);
        BoundingBox::merged_box_with_box(&left_bbox, &right_bbox)
    }

    /// Returns the bounding box at a node.
    /// 
    /// If the node internal then it just returns a copy of that node's bounding
    /// box. If the node is a leaf then it returns the zero-size bounding
    /// box made of that leaf's point.
    fn node_bounding_box(&self, node: &Node<T>) -> BoundingBox<T> {
        match node {
            Node::Internal(node) => {
                let min_values = node.bounding_box().min_values();
                let max_values = node.bounding_box().max_values();
                BoundingBox::new(min_values, max_values)
            },
            Node::Leaf(node) => {
                let store = self.borrow_point_store();
                let point = store.get(node.point()).unwrap();
                BoundingBox::new(point, point)
            }
        }
    }

    /// Deletion algorithm at a leaf node.
    /// 
    /// If the point to delete is not equal to the point at this leaf then we 
    /// we return a no-op result. If the point has mass greater than one then
    /// we simply decrease mass.
    /// 
    /// In the general case we have reached a leaf node `P` in the following
    /// diagram:
    /// 
    /// ```text
    ///     A
    ///    / \     P = current leaf node with point P
    ///   N   B    N = previously encountered node
    ///  / \       S = P's sibling
    /// P   S
    /// ```
    /// 
    /// What we want to do is delete this node along with its parent and replace
    /// with the current node's sibling:
    /// 
    /// ```text
    ///   A
    ///  / \
    /// S   B
    /// ```
    /// 
    /// This amounts to deleting these poitns and nodes from their respective
    /// stores and "rewiring" the parent-child relationshpis of the remaining
    /// nodes.
    /// 
    fn visit_leaf(&mut self, point: &Vec<T>, leaf_key: usize) -> DeleteResult {
        // Handle several edge cases: (1) the leaf node is not equal to the
        // input point, (2) 
        if !self.leaf_matches_point(point, leaf_key) {
            return DeleteResult::PointNotFound;
        } else if let Some(point_key) = self.decremented_leaf_mass(leaf_key) {
            return DeleteResult::MassDecreased(point_key);
        } else if let Some(point_key) = self.handle_only_node_case(leaf_key) {
            return DeleteResult::DeletedPoint(point_key);
        }
        
        // Set the parent-child relationship between the sibling and grandparent
        //
        // Two cases: if a grandparent to this leaf exists then perform the 
        // rewriring described above. Otherwise, the rewriring reduced to the
        // sibling node becoming the new root node
        let parent_key = self.get_parent(leaf_key).unwrap();
        let sibling_key = self.sibling_of(leaf_key).unwrap();
        if let Some(grandparent_key) = self.get_parent(parent_key) {
            self.get_node_mut(sibling_key).set_parent(Some(grandparent_key));
            if let Node::Internal(grandparent) = self.get_node_mut(grandparent_key) {
                if grandparent.left() == parent_key {
                    grandparent.set_left(sibling_key);
                } else if grandparent.right() == parent_key {
                    grandparent.set_right(sibling_key);
                } else { panic!("Inconsistent parent-grandparent relationsion") }
            } else { panic!("Inconsistent node: grandparent should be internal"); }
        } else {
            self.set_root_node(Some(sibling_key));
            self.get_node_mut(sibling_key).set_parent(None);
        }

        // finally, delete the current leaf node, its point from the point
        // store, and the parent node
        let point_key = match self.get_node(leaf_key) {
            Node::Leaf(leaf) => {
                let point_key = leaf.point();
                let mut store = self.borrow_mut_point_store();
                store.remove(point_key);
                point_key
            },
            Node::Internal(_) => panic!("Inconsistent node: expected leaf")
        };
        self.remove_node(leaf_key);
        self.remove_node(parent_key);

        DeleteResult::DeletedPoint(point_key)
    }

    /// Returns true if the leaf node's point is equal to the given point.
    fn leaf_matches_point(&self, point: &Vec<T>, node_key: usize) -> bool {
        if let Node::Leaf(leaf) = self.get_node(node_key) {
            let store = self.borrow_point_store();
            let leaf_point = store.get(leaf.point()).unwrap();
            *point == *leaf_point
        } else { panic!("Inconsistent node: expected leaf") }
    }

    /// Checks if the leaf node has a point with mass greater than one. If so, 
    /// returns `Some(key)` where `key` is the key of the point in the point
    /// store. Otherwise, returns `None`
    fn decremented_leaf_mass(&mut self, leaf_key: usize) -> Option<usize> {
        if let Node::Leaf(leaf) = self.get_node_mut(leaf_key) {
            if leaf.mass() > 1 {
                leaf.decrement_mass();
                Some(leaf.point())
            } else {
                None
            }
        } else { panic!("Inconsistent node: expected leaf") }
    }

    /// Handle the case when this node is the only node in the tree. Returns
    /// `Some(key)` if this is indeed the case where `key` is the key of the 
    /// point in the point store.
    /// 
    /// Note that we've already handled the case where the node has mass
    /// greater than one.
    fn handle_only_node_case(&mut self, leaf_key: usize) -> Option<usize> {
        if let Node::Leaf(leaf) = self.get_node(leaf_key) {
            if leaf.parent().is_none() {
                let point_key = {
                    let point_key = leaf.point();
                    let mut store = self.borrow_mut_point_store();
                    store.remove(point_key);
                    point_key
                };
                self.remove_node(leaf_key);
                self.set_root_node(None);
                return Some(point_key);
            }
        }
        None
    }

    /// Returns the node key of the sibling of the input node.
    /// 
    /// If the sibling doesn't exist, which should only happen in the case when
    /// the input node key is the root node, then returns `None`.
    fn sibling_of(&self, node_key: usize) -> Option<usize> {
        if let Some(parent_key) = self.get_node(node_key).parent() {
            let parent: &Internal<T> = match self.get_node(parent_key) {
                Node::Internal(node) => node,
                Node::Leaf(_) => panic!("Inconsistent node: parents cannot be leaves"),
            };

            if parent.left() == node_key {
                return Some(parent.right());
            } else if parent.right() == node_key {
                return Some(parent.left());
            } else {
                panic!("Inconsistent node: parent does not have node as a child");
            }
        }
        None
    }
    
    // ########################################################################
    // Point Addition and Deletion Helper Functions
    // ########################################################################

    #[inline(always)]
    fn insert_point(&mut self, point: Vec<T>) -> usize {
        let mut store = self.borrow_mut_point_store();
        store.insert(point)
    }

    #[inline(always)]
    fn insert_node(&mut self, node: Node<T>) -> usize {
        self.node_store_mut().insert(node)
    }

    #[inline(always)]
    fn get_node(&self, node_key: usize) -> &Node<T> {
        self.node_store().get(node_key).unwrap()
    }

    #[inline(always)]
    fn get_node_mut(&mut self, node_key: usize) -> &mut Node<T> {
        self.node_store_mut().get_mut(node_key).unwrap()
    }

    #[inline(always)]
    fn remove_node(&mut self, node_key: usize) {
        self.node_store_mut().remove(node_key);
    }

    #[inline(always)]
    fn contains_node(&self, node_key: usize) -> bool {
        self.node_store().contains(node_key)
    }

    #[inline(always)]
    fn get_parent(&self, node_key: usize) -> Option<usize> {
        self.get_node(node_key).parent()
    }

    #[inline(always)]
    fn point_inside_node(&self, point: &Vec<T>, node_key: usize) -> bool {
        match self.get_node(node_key) {
            Node::Leaf(_) => false,
            Node::Internal(internal) => internal
                .bounding_box()
                .contains_point(point),
        }
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

impl<'a, T: RCFFloat> NodeTraverser<'a, T> {

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

impl<'a, T: RCFFloat> Iterator for NodeTraverser<'a, T> {
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

    use crate::tree::Node;
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
        let nodes: Vec<&Node<f32>> = tree.traverse(&query).collect();
        assert_eq!(nodes.len(), 0);

        // add a bunch of N(0,1) points to the tree including the query point
        for point in generate_random_normal(4, 32) {
            tree.add_point(point);
        }
        tree.add_point(query.clone());

        // traverse the tree. the leaf node should contain the query point
        for node in tree.traverse(&query) {
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
    fn check_node_masses<T: RCFFloat>(tree: &Tree<T>, node_idx: usize) -> u32
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