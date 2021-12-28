
use std::io::empty;
use crate::abstractnodeview::NodeView;
use crate::boundingbox::BoundingBox;
use crate::newnodestore::NewNodeStore;
use crate::newnodestore::NodeStoreView;
use crate::pointstore::PointStoreView;
use crate::rcf::Max;
pub trait Visitor<T>{
    fn accept_leaf(&mut self, point : &[f32],node_view : &mut dyn NodeView);
    fn accept(&mut self, point : &[f32],node_view : &mut dyn NodeView);
    fn get_result(&self) -> T;
    fn has_converged(&self) -> bool;
    fn use_shadow_box(&self) -> bool;
    fn accept_needs_box(&self) -> bool;
}
