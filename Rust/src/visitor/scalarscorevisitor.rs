use crate::{
    samplerplustree::nodeview::SmallNodeView,
    visitor::visitor::{Visitor, VisitorInfo},
    types::Result,
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
    ) ->Result<()> {
        let mass = node_view.mass();
        self.leaf_index = node_view.leaf_index();
        if mass > visitor_info.ignore_mass {
            if node_view.is_duplicate() {
                self.score = (visitor_info.damp)(mass, self.tree_mass)
                    * (visitor_info.score_seen)(node_view.depth(), mass);
                self.converged = true;
            } else {
                self.score = (visitor_info.score_unseen)(node_view.depth(), mass);
            }
        } else {
            self.score = (visitor_info.score_unseen)(node_view.depth(), mass);
            self.use_shadow_box = true;
        }
        Ok(())
    }

    fn accept(&mut self, _point: &[f32], visitor_info: &VisitorInfo, node_view: &SmallNodeView) -> Result<()>{
        if !self.converged {
            let prob = if !self.use_shadow_box {
                node_view.probability_of_cut()
            } else {
                node_view.shadow_box_probability_of_cut()
            };
            if prob == 0.0 {
                self.converged = true;
            } else {
                self.score = (1.0 - prob) * self.score
                    + prob
                        * (visitor_info.score_unseen)(node_view.depth(), node_view.mass());
            }
        }
        Ok(())
    }

    fn result(&self, visitor_info: &VisitorInfo) -> Result<f64> {
        Ok((visitor_info.normalizer)(self.score, self.tree_mass))
    }

    fn is_converged(&self) -> Result<bool> {
        Ok(self.converged)
    }

    fn use_shadow_box(&self) -> bool {
        self.use_shadow_box
    }
}
