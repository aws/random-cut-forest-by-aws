//! Submodule containing types and components of a random cut tree.
//!
mod bounding_box;
pub use bounding_box::BoundingBox;

mod cut;
pub use cut::Cut;

mod node;
pub use node::{Internal, Leaf, Node};

mod tree_point_addition;
pub use tree_point_addition::AddResult;

mod tree_point_deletion;
pub use tree_point_deletion::DeleteResult;

mod tree;
pub use tree::{NodeIterator, Tree};
