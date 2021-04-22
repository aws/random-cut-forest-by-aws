//! Submodule containing types and components of a random cut tree.
//! 
mod bounding_box;
pub use bounding_box::BoundingBox;

mod cut;
pub use cut::Cut;

mod node;
pub use node::{Internal, Leaf, Node};