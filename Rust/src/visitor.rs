
use std::io::empty;
use crate::abstractnodeview::NodeView;
use crate::boundingbox::BoundingBox;
use crate::newnodestore::NewNodeStore;
use crate::newnodestore::NodeStoreView;
use crate::pointstore::PointStoreView;
use crate::rcf::Max;
pub trait Visitor{
    fn accept_leaf(&mut self, point : &[f32],node_view : &mut dyn NodeView);
    fn accept(&mut self, point : &[f32],node_view : &mut dyn NodeView);
    fn get_score(&self) -> f64;
    fn has_converged(&self) -> bool;
}
