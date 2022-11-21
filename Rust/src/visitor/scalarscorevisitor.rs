use crate::{
    samplerplustree::nodeview::SmallNodeView,
    visitor::visitor::{Visitor, VisitorInfo},
};

#[repr(C)]
pub struct ScalarScoreVisitor {
    converged: bool,
    leaf_index: usize,
    score: f64,
    tree_mass: usize,
    use_shadow_box: bool,
}

impl ScalarScoreVisitor {
    pub fn default(tree_mass: usize, _parameters: &[usize], _visitor_info: &VisitorInfo) -> Self {
        ScalarScoreVisitor {
            tree_mass,
            leaf_index: usize::MAX,
            converged: false,
            score: 0.0,
            use_shadow_box: false,
        }
    }
}

impl Visitor<SmallNodeView, f64> for ScalarScoreVisitor {
    fn accept_leaf(
        &mut self,
        _point: &[f32],
        visitor_info: &VisitorInfo,
        node_view: &SmallNodeView,
    ) {
        let mass = node_view.get_mass();
        self.leaf_index = node_view.get_leaf_index();
        if mass > visitor_info.ignore_mass {
            if node_view.is_duplicate() {
                self.score = (visitor_info.damp)(mass, self.tree_mass)
                    * (visitor_info.score_seen)(node_view.get_depth(), mass);
                self.converged = true;
            } else {
                self.score = (visitor_info.score_unseen)(node_view.get_depth(), mass);
            }
        } else {
            self.score = (visitor_info.score_unseen)(node_view.get_depth(), mass);
            self.use_shadow_box = true;
        }
    }

    fn accept(&mut self, point: &[f32], visitor_info: &VisitorInfo, node_view: &SmallNodeView) {
        if !self.converged {
            let prob = if !self.use_shadow_box {
                node_view.get_probability_of_cut()
            } else {
                node_view.get_shadow_box_probability_of_cut()
            };
            if prob == 0.0 {
                self.converged = true;
            } else {
                self.score = (1.0 - prob) * self.score
                    + prob
                        * (visitor_info.score_unseen)(node_view.get_depth(), node_view.get_mass());
            }
        }
    }

    fn result(&self, visitor_info: &VisitorInfo) -> f64 {
        (visitor_info.normalizer)(self.score, self.tree_mass)
    }

    fn is_converged(&self) -> bool {
        self.converged
    }

    fn use_shadow_box(&self) -> bool {
        self.use_shadow_box
    }
}
