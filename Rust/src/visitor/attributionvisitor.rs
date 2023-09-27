use num::abs;

use crate::{
    common::divector::DiVector,
    samplerplustree::nodeview::LargeNodeView,
    visitor::visitor::{Visitor, VisitorInfo},
    types::Result,
};

#[repr(C)]
pub struct AttributionVisitor {
    converged: bool,
    leaf_index: usize,
    score: f64,
    tree_mass: usize,
    hit_duplicate: bool,
    use_shadow_box: bool,
    attribution: DiVector,
    probability: DiVector,
}

impl AttributionVisitor {
    pub fn new(tree_mass: usize, dimension: usize, _visitor_info: &VisitorInfo) -> Self {
        AttributionVisitor {
            tree_mass,
            leaf_index: usize::MAX,
            converged: false,
            score: 0.0,
            hit_duplicate: false,
            use_shadow_box: false,
            attribution: DiVector::empty(dimension),
            probability: DiVector::empty(dimension),
        }
    }

    pub fn create_visitor(
        tree_mass: usize,
        parameters: &[usize],
        visitor_info: &VisitorInfo,
    ) -> AttributionVisitor {
        let dimension = parameters[0];
        AttributionVisitor::new(tree_mass, dimension, visitor_info)
    }
}

impl Visitor<LargeNodeView, DiVector> for AttributionVisitor {
    fn accept_leaf(
        &mut self,
        point: &[f32],
        visitor_info: &VisitorInfo,
        node_view: &LargeNodeView,
    ) -> Result<()> {
        let mass = node_view.mass();
        self.leaf_index = node_view.leaf_index();
        if mass > visitor_info.ignore_mass {
            if node_view.is_duplicate() {
                self.score = (visitor_info.damp)(mass, self.tree_mass)
                    * (visitor_info.score_seen)(node_view.depth(), mass);
                self.hit_duplicate = true;
                self.use_shadow_box = true;
            } else {
                self.score = (visitor_info.score_unseen)(node_view.depth(), mass);
                node_view.assign_probability_of_cut(&mut self.probability, point);
                assert!(abs(self.probability.total() - 1.0) < 1e-6);
                self.attribution.add_from(&self.probability, self.score);
            }
        } else {
            self.score = (visitor_info.score_unseen)(node_view.depth(), mass);
            self.use_shadow_box = true;
        }
        Ok(())
    }

    fn accept(&mut self, point: &[f32], visitor_info: &VisitorInfo, node_view: &LargeNodeView) -> Result<()>{
        if !self.converged {
            if !self.use_shadow_box {
                node_view.assign_probability_of_cut(&mut self.probability, point);
            } else {
                node_view.assign_probability_of_cut_shadow_box(&mut self.probability, point);
            };
            let prob = self.probability.total();
            if prob == 0.0 {
                self.converged = true;
            } else {
                let new_value =
                    (visitor_info.score_unseen)(node_view.depth(), node_view.mass());
                if !self.hit_duplicate {
                    self.score = (1.0 - prob) * self.score + prob * new_value;
                }
                self.attribution.scale(1.0 - prob);
                self.attribution.add_from(&self.probability, new_value);
            }
        }
        Ok(())
    }

    fn result(&self, visitor_info: &VisitorInfo) -> Result<DiVector> {
        let t = (visitor_info.normalizer)(self.score, self.tree_mass);
        let mut answer = self.attribution.clone();
        answer.normalize(t);
        Ok(answer)
    }

    fn is_converged(&self) -> Result<bool> {
        Ok(self.converged)
    }

    fn use_shadow_box(&self) -> bool {
        self.use_shadow_box
    }
}
