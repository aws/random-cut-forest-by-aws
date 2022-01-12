use crate::nodeview::NodeView;

pub trait Visitor<T> {
    fn accept_leaf(&mut self, point: &[f32], node_view: &dyn NodeView);
    fn accept(&mut self, point: &[f32], node_view: &dyn NodeView);
    fn get_result(&self) -> T;
    fn has_converged(&self) -> bool;
    fn descriptor(&self) -> VisitorDescriptor {
        // by default a visitor is not a multivisitor
        VisitorDescriptor {
            use_point_copy_for_accept: false,
            use_box_for_accept: false,
            use_child_boxes_for_accept: false,
            use_mass_distribution_for_accept: false,
            maintain_shadow_box_for_accept: false,
            use_box_for_trigger: false,
            use_child_boxes_for_trigger: false,
            use_child_mass_distribution_for_trigger: false,
            trigger_manipulation_needs_node_view_accept_fields: false,
        }
    }
}

pub trait UniqueMultiVisitor<T, Q>: Visitor<T> {
    fn get_arguments(&self) -> Q;
    fn trigger(&self, point: &[f32], node_view: &dyn NodeView) -> bool;
    fn combine_branches(&mut self, point: &[f32], node_view: &dyn NodeView);
    fn unique_answer(&self) -> &[f32];
}

pub trait StreamingMultiVisitor<T, Q>: UniqueMultiVisitor<T, Q> {
    fn initialize_branch_split(&mut self, point: &[f32], node_view: &dyn NodeView);
    fn second_branch(&mut self, point: &[f32], node_view: &dyn NodeView);
}

pub struct VisitorDescriptor {
    pub(crate) use_point_copy_for_accept: bool,
    pub(crate) use_box_for_accept: bool,
    pub(crate) use_child_boxes_for_accept: bool,
    pub(crate) use_mass_distribution_for_accept: bool,
    pub(crate) maintain_shadow_box_for_accept: bool,
    pub(crate) use_box_for_trigger: bool,
    pub(crate) use_child_boxes_for_trigger: bool,
    pub(crate) use_child_mass_distribution_for_trigger: bool,
    pub(crate) trigger_manipulation_needs_node_view_accept_fields: bool,
}
