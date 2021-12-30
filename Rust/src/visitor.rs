
use std::io::empty;
use crate::abstractnodeview::NodeView;
use crate::boundingbox::BoundingBox;
use crate::newnodestore::NewNodeStore;
use crate::newnodestore::NodeStoreView;
use crate::pointstore::PointStoreView;
use crate::rcf::Max;
pub trait Visitor<T> {
    fn accept_leaf(&mut self, point : &[f32],node_view : &dyn NodeView);
    fn accept(&mut self, point : &[f32],node_view : &dyn NodeView);
    fn get_result(&self) -> T;
    fn has_converged(&self) -> bool;
    fn descriptor(&self) -> VisitorDescriptor;
    fn multivisitor_descriptor(&self) -> MultiVisitorDescriptor { // by default a visitor is not a multivisitor
        MultiVisitorDescriptor{
            use_box_for_trigger: false,
            use_child_boxes_for_trigger: false,
            use_mass_distribution_for_trigger: false,
            use_cuts_for_trigger: false,
            trigger_manipulation_needs_node_view_accept_fields: false
        }
    }
}

pub trait StreamingMultiVisitor<T,Q> : Visitor<T> {
    fn get_arguments(&self) -> Q;
    fn trigger(&self,point : &[f32],node_view: &dyn NodeView) -> bool;
    fn init_trigger(&mut self,point : &[f32],node_view: &dyn NodeView);
    fn half_trigger(&mut self,point : &[f32],node_view: &dyn NodeView);
    fn close_trigger(&mut self,point : &[f32],node_view: &dyn NodeView);
}

pub struct VisitorDescriptor {
    pub(crate) use_point_copy_for_accept :bool,
    pub(crate) use_box_for_accept : bool,
    pub(crate) use_child_boxes_for_accept: bool,
    pub(crate) use_mass_distribution_for_accept : bool,
    pub(crate) use_cuts_for_accept : bool,
    pub(crate) maintain_shadow_box_for_accept: bool
}

pub struct MultiVisitorDescriptor {
    pub(crate) use_box_for_trigger : bool,
    pub(crate) use_child_boxes_for_trigger: bool,
    pub(crate) use_mass_distribution_for_trigger : bool,
    pub(crate) use_cuts_for_trigger : bool,
    pub(crate) trigger_manipulation_needs_node_view_accept_fields: bool
}