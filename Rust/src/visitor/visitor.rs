use crate::{Internal, Leaf};

/// Visitors on random cut trees.
///
/// Many random cut forest scoring algorithms use a visitor design pattern on
/// the nodes of a tree. The nodes typically run from the leaf closest to the
/// input query point along the path to the root node such as in the function
/// [`Tree::traverse()`](crate::tree::Tree::traverse).
///
pub trait Visitor<T> {

    /// The output type of the visitor.
    type Output;

    /// Update on a leaf node in the node traversal.
    ///
    /// This function is called on the leaf node nearest to an input query
    /// point. The leaf node is typically the first time the visitor is called
    /// but this is dependent on the algorithm using the visitor.
    ///
    /// The depth of the leaf is also provided as an input argument.
    fn accept_leaf(&mut self, node: &Leaf, depth: T);

    /// Update on an internal node in the node traversal.
    ///
    /// Every node other than the starting leaf node is an internal node.
    /// Internal nodes are defined by a bounding box on the leaf points
    /// contained in its subtree as well as a choice of random cut.
    ///
    /// The depth of the node is also provided as an input argument.
    fn accept(&mut self, node: &Internal<T>, depth: T);

    /// Returns the result of this visitor.
    ///
    /// Called at the end of traversal on a given tree. The type of the output
    /// is given by the associated type of this trait, [`Output`](Self::Output).
    fn get_result(&self) -> Self::Output;
}