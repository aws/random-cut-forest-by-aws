use crate::{
    common::{directionaldensity::InterpolationMeasure},
    samplerplustree::nodeview::LargeNodeView,
    visitor::visitor::{Visitor, VisitorInfo},
};

#[repr(C)]
pub struct InterpolationVisitor {
    converged: bool,
    leaf_index: usize,
    score: f64,
    tree_mass: usize,
    hit_duplicate: bool,
    use_shadow_box: bool,
    interpolation_measure: InterpolationMeasure,
}

impl InterpolationVisitor {
    pub fn new(tree_mass: usize, dimension: usize, visitor_info: &VisitorInfo) -> Self {
        InterpolationVisitor {
            tree_mass,
            leaf_index: usize::MAX,
            converged: false,
            score: 0.0,
            hit_duplicate: false,
            use_shadow_box: false,
            interpolation_measure: InterpolationMeasure::empty(dimension, tree_mass as f32),
        }
    }

    pub fn create_visitor(
        tree_mass: usize,
        parameters: &[usize],
        visitor_info: &VisitorInfo,
    ) -> InterpolationVisitor {
        let dimension = parameters[0];
        InterpolationVisitor::new(tree_mass, dimension, visitor_info)
    }
}

impl Visitor<LargeNodeView, InterpolationMeasure> for InterpolationVisitor {
    fn accept_leaf(
        &mut self,
        point: &[f32],
        visitor_info: &VisitorInfo,
        node_view: &LargeNodeView,
    ) {
        let mass = node_view.get_mass();
        let leaf_point = node_view.get_leaf_point();
        self.leaf_index = node_view.get_leaf_index();
        if mass > visitor_info.ignore_mass {
            if node_view.is_duplicate() {
                self.score = (visitor_info.damp)(mass, self.tree_mass)
                    * (visitor_info.score_seen)(node_view.get_depth(), mass);
                self.hit_duplicate = true;
                self.use_shadow_box = true;
            } else {
                let t = (visitor_info.score_unseen)(node_view.get_depth(), mass);
                self.score = t;
                let bounding_box = node_view.bounding_box();
                self.interpolation_measure.update(point, &bounding_box, t);
            }
        } else {
            self.score = (visitor_info.score_unseen)(node_view.get_depth(), mass);
            self.use_shadow_box = true;
        }
    }

    fn accept(&mut self, point: &[f32], visitor_info: &VisitorInfo, node_view: &LargeNodeView) {
        if !self.converged {
            let bounding_box = if !self.use_shadow_box {
                node_view.bounding_box()
            } else {
                node_view.shadow_box()
            };
            let new_value =
                (visitor_info.score_unseen)(node_view.get_depth(), node_view.get_mass());
            let prob = self
                .interpolation_measure
                .update(point, &bounding_box, new_value);
            if prob == 0.0 {
                self.converged = true;
            } else {
                if !self.hit_duplicate {
                    self.score = (1.0 - prob) * self.score + prob * new_value;
                }
            }
        }
    }

    fn result(&self, visitor_info: &VisitorInfo) -> InterpolationMeasure {
        let t = (visitor_info.normalizer)(self.score, self.tree_mass);
        let mut answer = self.interpolation_measure.clone();
        answer.measure.normalize(t);
        answer
    }

    fn is_converged(&self) -> bool {
        self.converged
    }

    fn use_shadow_box(&self) -> bool {
        self.use_shadow_box
    }
}
