
use std::io::empty;
use crate::visitor::Visitor;
use crate::abstractnodeview::NodeView;
use crate::boundingbox::BoundingBox;
use crate::newnodestore::NewNodeStore;
use crate::newnodestore::NodeStoreView;
use crate::pointstore::PointStoreView;
use crate::rcf::Max;
pub trait StreamingMultiVisitor<T,Q> : Visitor<T> {
    fn get_arguments(&self) -> Q;
    fn trigger(&self) -> bool;
    fn trigger_needs_box(&self) -> bool;
    fn init_trigger(&mut self);
    fn half_trigger(&mut self);
    fn close_trigger(&mut self);
}
