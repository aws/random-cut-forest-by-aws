use num::abs;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

use crate::{
    samplerplustree::nodeview::MediumNodeView,
    visitor::visitor::{SimpleMultiVisitor, Visitor, VisitorInfo},
    types::Result
};
use crate::util::check_argument;

#[repr(C)]
pub struct ImputeVisitor {
    centrality: f64,
    tree_mass: usize,
    rng: ChaCha20Rng,
    missing: Vec<usize>,
    stack: Vec<ImputeVisitorStackElement>,
    use_shadow_box: bool,
}

#[repr(C)]
struct ImputeVisitorStackElement {
    converged: bool,
    score: f64,
    random: f32,
    index: usize,
    distance: f64,
}

impl ImputeVisitor {
    pub fn new(missing: &[usize], centrality: f64, tree_mass: usize, seed: u64) -> Self {
        ImputeVisitor {
            tree_mass,
            centrality,
            rng: ChaCha20Rng::seed_from_u64(seed),
            missing: Vec::from(missing),
            stack: Vec::new(),
            use_shadow_box: false,
        }
    }

    pub fn create_nbr_visitor(
        tree_mass: usize,
        parameters: &[usize],
        _visitor_info: &VisitorInfo,
    ) -> Self {
        let percentile = if parameters.len() > 0 {
            parameters[0]
        } else {
            50
        };
        let seed = if parameters.len() > 1 {
            parameters[1]
        } else {
            0
        };
        let centrality = if percentile < 5 || percentile > 95 {
            0.0
        } else {
            1.0 - abs(1.0 - percentile as f64 / 50.0)
        };
        ImputeVisitor::new(&Vec::new(), centrality, tree_mass, seed as u64)
    }

    /// the following function allows the score to vary between the score used in
    /// anomaly detection and fully random sample based on the parameter centrality
    /// these two cases correspond to centrality = 1 and centrality = 0 respectively

    fn adjusted_score(&self, e: &ImputeVisitorStackElement, visitor_info: &VisitorInfo) -> f64 {
        self.centrality * (visitor_info.normalizer)(e.score, self.tree_mass)
            + (1.0 - self.centrality) * e.random as f64
    }
}

impl Visitor<MediumNodeView, (f64, usize, f64)> for ImputeVisitor {
    fn accept_leaf(
        &mut self,
        point: &[f32],
        visitor_info: &VisitorInfo,
        node_view: &MediumNodeView,
    ) -> Result<()>{
        let mass = node_view.mass();
        let leaf_point = node_view.leaf_point();
        let mut new_point = Vec::from(point);
        for i in self.missing.iter() {
            new_point[*i] = leaf_point[*i];
        }

        let mut converged = false;
        let score: f64;
        if mass > visitor_info.ignore_mass || self.missing.len() != 0 {
            if node_view.is_duplicate() {
                score = (visitor_info.damp)(mass, self.tree_mass)
                    * (visitor_info.score_seen)(node_view.depth(), mass);
                converged = true;
            } else {
                score = (visitor_info.score_unseen)(node_view.depth(), mass);
            }
        } else {
            // shadow box is undefined for missing values
            // for not missing values, this block corresponds to exact same evaluation
            // in score and attribution visitor
            // note that multi-visitors ignore the shadow box anyways
            score = (visitor_info.score_unseen)(node_view.depth(), mass);
            self.use_shadow_box = true;
        }
        let dist = (visitor_info.distance)(&new_point, &leaf_point);
        self.stack.push(ImputeVisitorStackElement {
            converged,
            score,
            index: node_view.leaf_index(),
            random: self.rng.gen::<f32>(),
            distance: dist,
        });
        Ok(())
    }

    fn accept(&mut self, _point: &[f32], visitor_info: &VisitorInfo, node_view: &MediumNodeView) -> Result<()>{
        let mut top_of_stack = self.stack.pop().expect(" stack cannot be empty");
        if !top_of_stack.converged {
            let prob = if !self.use_shadow_box {
                // note that this probability ignores any missing coordinates
                // which would be accurate since the value used is inside the box
                node_view.probability_of_cut()
            } else {
                node_view.shadow_box_probability_of_cut()
            };
            if prob == 0.0 {
                top_of_stack.converged = true;
            } else {
                let new_score = (1.0 - prob) * top_of_stack.score
                    + prob
                        * (visitor_info.score_unseen)(node_view.depth(), node_view.mass());
                top_of_stack.converged = false;
                top_of_stack.score = new_score;
            }
            self.stack.push(top_of_stack);
        }
        Ok(())
    }

    fn result(&self, visitor_info: &VisitorInfo) -> Result<(f64, usize, f64)> {
        check_argument(self.stack.len() == 1,
            "incorrect state, stack length should be 1"
        )?;
        let top_of_stack = self.stack.last().expect("should be length 1");
        let t = (visitor_info.normalizer)(top_of_stack.score, self.tree_mass);
        Ok((t, top_of_stack.index, top_of_stack.distance))
    }

    fn is_converged(&self) -> Result<bool> {
        Ok(self.stack.len() != 0 && self.stack.last().expect("cannot be empty").converged)
    }

    fn use_shadow_box(&self) -> bool {
        self.use_shadow_box
    }
}

impl SimpleMultiVisitor<MediumNodeView, (f64, usize, f64)> for ImputeVisitor {
    fn combine_branches(
        &mut self,
        _point: &[f32],
        _node_view: &MediumNodeView,
        visitor_info: &VisitorInfo,
    ) -> Result<()>{
        let mut top_of_stack = self.stack.pop().expect("has to be size 1, corrupt state");
        let mut next_of_stack = self.stack.pop().expect("has to be size 2, corrupt state");

        if self.adjusted_score(&top_of_stack, &visitor_info)
            < self.adjusted_score(&next_of_stack, &visitor_info)
        {
            top_of_stack.converged = top_of_stack.converged || next_of_stack.converged;
            self.stack.push(top_of_stack);
        } else {
            next_of_stack.converged = top_of_stack.converged || next_of_stack.converged;
            self.stack.push(next_of_stack);
        }
        Ok(())
    }
}
