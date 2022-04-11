use crate::samplerplustree::nodeview::{MinimalNodeView, NodeView};
use crate::visitor::visitor::{SimpleVisitor, Visitor, VisitorInfo, VisitorResult};

#[repr(C)]
pub struct ScalarScoreVisitor {
    damp: fn(usize, usize) -> f64,
    score_seen: fn(usize, usize) -> f64,
    score_unseen: fn(usize, usize) -> f64,
    normalizer: fn(f64, usize) -> f64,
    ignore_mass: usize,
    converged: bool,
    leaf_index: usize,
    score: f64,
    tree_mass: usize,
    use_shadow_box: bool
}

impl ScalarScoreVisitor {
    pub fn new(
        tree_mass: usize,
        visitor_info : &VisitorInfo
    ) -> Self {
        ScalarScoreVisitor {
            tree_mass,
            ignore_mass : visitor_info.ignore_mass,
            damp : visitor_info.damp,
            score_seen : visitor_info.score_seen,
            score_unseen : visitor_info.score_unseen,
            normalizer : visitor_info.normalizer,
            leaf_index: usize::MAX,
            converged: false,
            score: 0.0,
            use_shadow_box : false
        }
    }

    pub fn create_visitor(
        tree_mass: usize,
        _not_used: &[usize],
        visitor_info : &VisitorInfo
    ) -> Box<dyn SimpleVisitor<f64>> {
        Box::new(ScalarScoreVisitor::new(tree_mass,visitor_info))
    }
}

impl SimpleVisitor<f64> for ScalarScoreVisitor {
    fn accept_leaf(&mut self, _point: &[f32], node_view: &dyn MinimalNodeView) {
        let mass = node_view.get_mass();
        self.leaf_index = node_view.get_leaf_index();
        if mass > self.ignore_mass {
            if node_view.leaf_equals() {
                self.score =
                    (self.damp)(mass, self.tree_mass) * (self.score_seen)(node_view.get_depth(), mass);
                self.converged = true;
            } else {
                self.score = (self.score_unseen)(node_view.get_depth(), mass);
            }
        } else {
            self.score = (self.score_unseen)(node_view.get_depth(), mass);
            self.use_shadow_box = true;
        }
    }

    fn accept(&mut self, point: &[f32], node_view: &dyn MinimalNodeView) {
        if !self.converged {
            let prob = if !self.use_shadow_box {
                node_view.probability_of_cut_on_path()
            } else {
                node_view.get_shadow_box_probability_of_cut()
            };
            if prob == 0.0 {
                self.converged = true;
            } else {
                self.score = (1.0 - prob) * self.score
                    + prob * (self.score_unseen)(node_view.get_depth(), node_view.get_mass());
            }
        }
    }
}

impl VisitorResult<f64> for ScalarScoreVisitor {
    fn get_result(&self) -> f64 {
        (self.normalizer)(self.score,self.tree_mass)
    }

    fn has_converged(&self) -> bool {
        self.converged
    }

    fn use_shadow_box(&self) -> bool {
        self.use_shadow_box
    }
}
