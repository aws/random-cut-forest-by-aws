
use std::io::empty;
use crate::abstractnodeview::NodeView;
use crate::boundingbox::BoundingBox;
use crate::newnodestore::NewNodeStore;
use crate::newnodestore::NodeStoreView;
use crate::pointstore::PointStoreView;
use crate::rcf::Max;
use crate::visitor::Visitor;

pub struct ScalarScoreVisitor {
    damp: fn(usize,usize) -> f64,
    score_seen : fn(usize,usize) -> f64,
    score_unseen : fn(usize,usize) -> f64,
    normalizer : fn(f64,usize) -> f64,
    ignore_mass : usize,
    converged : bool,
    score : f64,
    tree_mass : usize
}

impl ScalarScoreVisitor {
    pub fn new(tree_mass: usize, ignore_mass : usize, damp: fn(usize,usize) -> f64, score_seen : fn(usize,usize) -> f64, score_unseen : fn(usize,usize) -> f64,
               normalizer : fn(f64,usize) -> f64) -> Self {
        ScalarScoreVisitor{
            tree_mass,
            ignore_mass,
            damp,
            score_seen,
            score_unseen,
            normalizer,
            converged : false,
            score : 0.0
        }
    }
}

impl Visitor<f64> for ScalarScoreVisitor{
    fn accept_leaf(&mut self, point: &[f32],node_view: &mut dyn NodeView) {
        let mass =node_view.get_mass();
        if point.eq(node_view.get_leaf_point()) && mass > self.ignore_mass {
            self.score = (self.damp)(mass,self.tree_mass)*(self.score_seen)(node_view.get_depth(),mass);
            self.converged = true;
        } else {
            self.score = (self.score_unseen)(node_view.get_depth(),mass);
        }
    }

    fn accept(&mut self, point : &[f32], node_view: &mut dyn NodeView) {
        if (!self.converged) {
            let prob = if (self.ignore_mass == 0 ) { node_view.get_probability_of_cut(point)}
                  else { node_view.get_shadow_box().probability_of_cut(point)};
            if (prob == 0.0) {
                self.converged = true;
            } else {
                self.score = (1.0-prob)*self.score + prob * (self.score_unseen)(node_view.get_depth(),node_view.get_mass());
            }
        }
    }

    fn get_result(&self) -> f64 {
        self.score
    }

    fn has_converged(&self) -> bool {
        self.converged
    }

    fn use_shadow_box(&self) -> bool {
        self.ignore_mass > 0
    }

    fn accept_needs_box(&self) -> bool {
        self.use_shadow_box()
    }
}



