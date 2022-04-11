use num::abs;
use crate::samplerplustree::nodeview::NodeView;
use crate::common::divector::DiVector;
use crate::visitor::visitor::{Visitor, VisitorInfo, VisitorResult};


#[repr(C)]
pub struct AttributionVisitor {
    damp: fn(usize, usize) -> f64,
    score_seen: fn(usize, usize) -> f64,
    score_unseen: fn(usize, usize) -> f64,
    normalizer: fn(f64, usize) -> f64,
    ignore_mass: usize,
    converged: bool,
    leaf_index: usize,
    score: f64,
    tree_mass: usize,
    hit_duplicate: bool,
    use_shadow_box: bool,
    attribution : DiVector,
    probability : DiVector
}

impl AttributionVisitor {
    pub fn new(
        tree_mass: usize,
        dimension : usize,
        visitor_info : &VisitorInfo
    ) -> Self {
        AttributionVisitor {
            tree_mass,
            ignore_mass: visitor_info.ignore_mass,
            damp : visitor_info.damp,
            score_seen : visitor_info.score_seen,
            score_unseen : visitor_info.score_unseen,
            normalizer : visitor_info.normalizer,
            leaf_index: usize::MAX,
            converged: false,
            score: 0.0,
            hit_duplicate : false,
            use_shadow_box : false,
            attribution : DiVector::empty(dimension),
            probability : DiVector::empty(dimension)
        }
    }

    pub fn create_visitor(
        tree_mass: usize,
        parameters: &[usize],
        visitor_info : &VisitorInfo
    ) -> Box<dyn Visitor<DiVector>> {
        let dimension = parameters[0];
        Box::new(AttributionVisitor::new(tree_mass,dimension,visitor_info))
    }
}

impl VisitorResult<DiVector> for AttributionVisitor {
    fn get_result(&self) -> DiVector {
        let t = (self.normalizer)(self.score,self.tree_mass);
        let mut answer = self.attribution.clone();
        answer.normalize(t);
        answer
    }

    fn has_converged(&self) -> bool {
        self.converged
    }

    fn use_shadow_box(&self) -> bool {
        self.use_shadow_box
    }

}

impl Visitor<DiVector> for AttributionVisitor {
    fn accept_leaf(&mut self, point: &[f32], node_view: &dyn NodeView) {
        let mass = node_view.get_mass();
        self.leaf_index = node_view.get_leaf_index();
        if mass > self.ignore_mass {
            if node_view.leaf_equals() {
                self.score =
                    (self.damp)(mass, self.tree_mass) * (self.score_seen)(node_view.get_depth(), mass);
                self.hit_duplicate = true;
                self.use_shadow_box = true;
            } else {
                self.score = (self.score_unseen)(node_view.get_depth(), mass);
                node_view.modify_in_place_probability_of_cut_di_vector(point,&mut self.probability);
                assert!(abs(self.probability.total() - 1.0) < 1e-6);
                self.attribution.add_from(&self.probability,self.score);
            }
        } else {
            self.score = (self.score_unseen)(node_view.get_depth(), mass);
            self.use_shadow_box = true;
        }
    }

    fn accept(&mut self, point: &[f32], node_view: &dyn NodeView) {
        if !self.converged {
            if !self.use_shadow_box {
                node_view.modify_in_place_probability_of_cut_di_vector(point,&mut self.probability);
            } else {
                self.probability.assign_as_probability_of_cut(&node_view.get_shadow_box(),point);
            };
            let prob = self.probability.total();
            if prob == 0.0 {
                self.converged = true;
            } else {
                let new_value = (self.score_unseen)(node_view.get_depth(), node_view.get_mass());
                if !self.hit_duplicate {
                    self.score = (1.0 - prob) * self.score + prob * new_value;
                }
                self.attribution.scale(1.0 - prob);
                self.attribution.add_from(&self.probability, new_value)
            }
        }
    }


}
