use crate::RCFFloat;
use super::{Tree, Node, Cut, BoundingBox, Internal};

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


// /// A mechanism for deleting points from trees.
pub struct PointDeleter<'a, T> {
    tree: &'a mut Tree<T>,
}

impl<'a, T: RCFFloat> PointDeleter<'a, T> {
    pub fn new(tree: &'a mut Tree<T>) -> Self {
        PointDeleter { tree: tree }
    }


    /// Delete a point from the tree.__rust_force_expr!
    /// 
    /// Calls the recursive function `delete_point_by_node()` using the root
    /// node of the tree as input.
    pub fn delete_point(&mut self, point: &Vec<T>) -> DeleteResult {
        match self.tree.root_node() {
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
                let store = self.tree.borrow_point_store();
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
            self.tree.set_root_node(Some(sibling_key));
            self.get_node_mut(sibling_key).set_parent(None);
        }

        // finally, delete the current leaf node, its point from the point
        // store, and the parent node
        let point_key = match self.get_node(leaf_key) {
            Node::Leaf(leaf) => {
                let point_key = leaf.point();
                let mut store = self.tree.borrow_mut_point_store();
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
            let store = self.tree.borrow_point_store();
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
                    let mut store = self.tree.borrow_mut_point_store();
                    store.remove(point_key);
                    point_key
                };
                self.remove_node(leaf_key);
                self.tree.set_root_node(None);
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

    #[inline(always)]
    fn get_node(&self, node_key: usize) -> &Node<T> {
        self.tree.node_store().get(node_key).unwrap()
    }

    #[inline(always)]
    fn get_node_mut(&mut self, node_key: usize) -> &mut Node<T> {
        self.tree.node_store_mut().get_mut(node_key).unwrap()
    }

    #[inline(always)]
    fn remove_node(&mut self, node_key: usize) {
        self.tree.node_store_mut().remove(node_key);
    }

    #[inline(always)]
    fn contains_node(&self, node_key: usize) -> bool {
        self.tree.node_store().contains(node_key)
    }

    #[inline(always)]
    fn get_parent(&self, node_key: usize) -> Option<usize> {
        self.get_node(node_key).parent()
    }
}