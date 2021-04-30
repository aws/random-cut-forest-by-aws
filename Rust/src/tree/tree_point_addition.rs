use crate::RCFFloat;
use crate::tree::{BoundingBox, Cut, Node, Tree};

/// The result of a point addition operation.
/// 
/// The `AddedPoint` result contains the key of the point that was added to the
/// tree. The `MassIncreased` result contains the key of the point whose mass
/// was increased.
pub enum AddResult {
    AddedPoint(usize),
    MassIncreased(usize),
}

impl<T: RCFFloat> Tree<T> {

    #[inline(always)]
    fn point_inside_node(&self, point: &Vec<T>, node_key: usize) -> bool {
        match self.get_node(node_key) {
            Node::Leaf(_) => false,
            Node::Internal(internal) => internal
                .bounding_box()
                .contains_point(point),
        }
    }

    #[inline(always)]
    fn insert_point(&mut self, point: Vec<T>) -> usize {
        let mut store = self.borrow_mut_point_store();
        store.insert(point)
    }

    #[inline(always)]
    fn insert_node(&mut self, node: Node<T>) -> usize {
        self.node_store_mut().insert(node)
    }

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

    /// Main recursive point addition algorithm given a new point and a current
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
}