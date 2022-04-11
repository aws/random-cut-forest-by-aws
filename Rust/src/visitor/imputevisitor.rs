use num::abs;
use rand::{random, Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use crate::samplerplustree::nodeview::{LeafPointNodeView, NodeView};
use crate::visitor::visitor::{LeafPointVisitor, MissingCoordinatesMultiLeafPointVisitor, VisitorInfo, VisitorResult};


#[repr(C)]
pub struct ImputeVisitor {
    ignore_mass: usize,
    score_seen: fn(usize, usize) -> f64,
    score_unseen: fn(usize, usize) -> f64,
    damp: fn(usize, usize) -> f64,
    normalizer: fn(f64, usize) -> f64,
    distance: fn(&[f32],&[f32]) -> f64,
    centrality: f64,
    tree_mass: usize,
    rng : ChaCha20Rng,
    missing: Vec<usize>,
    stack: Vec<ImputeVisitorStackElement>,
    use_shadow_box: bool
}

#[repr(C)]
struct ImputeVisitorStackElement{
    converged: bool,
    score : f64,
    random : f32,
    index : usize,
    distance : f64
}

impl ImputeVisitor {
    pub fn new(
        missing: &[usize],
        centrality: f64,
        tree_mass: usize,
        seed : u64,
        visitor_info : &VisitorInfo
    ) -> Self {

        ImputeVisitor {
            tree_mass,
            centrality,
            rng : ChaCha20Rng::seed_from_u64(seed),
            ignore_mass : visitor_info.ignore_mass,
            score_seen : visitor_info.score_seen,
            score_unseen : visitor_info.score_unseen,
            damp : visitor_info.damp,
            normalizer : visitor_info.normalizer,
            distance : visitor_info.distance,
            missing: Vec::from(missing),
            stack : Vec::new(),
            use_shadow_box : false
        }
    }

    pub fn create_nbr_visitor(
        tree_mass: usize,
        parameters: &[usize],
        visitor_info : &VisitorInfo
    ) -> Box<dyn LeafPointVisitor<(f64,usize,f64)>>{
        let percentile = parameters[0];
        let centrality = if percentile<5 || percentile > 95 {0.0} else {1.0 - abs(1.0 - percentile as f64/50.0)};
         Box::new(ImputeVisitor::new(&Vec::new(), centrality, tree_mass, 0, &visitor_info))
    }

    /// the following function allows the score to vary between the score used in
    /// anomaly detection and fully random sample based on the parameter centrality
    /// these two cases correspond to centrality = 1 and centrality = 0 respectively

    fn adjusted_score(&self, e : &ImputeVisitorStackElement)->f64{
           self.centrality * (self.normalizer) (e.score,self.tree_mass) +
               (1.0 - self.centrality) * e.random as f64
    }
}

impl VisitorResult<(f64,usize,f64)> for ImputeVisitor {

    fn get_result(&self) -> (f64,usize,f64) {
        assert_eq!(self.stack.len(), 1, "incorrect state");
        let top_of_stack = self.stack.last().unwrap();
        let t = (self.normalizer)(top_of_stack.score,self.tree_mass);
        (t,top_of_stack.index,top_of_stack.distance)
    }

    fn has_converged(&self) -> bool {
        self.stack.last().unwrap().converged
    }

    fn use_shadow_box(&self) -> bool{
        self.use_shadow_box
    }
}

impl LeafPointVisitor<(f64,usize,f64)> for ImputeVisitor {
    fn accept_leaf(&mut self, point: &[f32], node_view: &dyn LeafPointNodeView) {
        let mass = node_view.get_mass();
        let leaf_point = node_view.get_leaf_point();
        let mut new_point = Vec::from(point);
        for i in self.missing.iter() {
            new_point[*i] = leaf_point[*i];
        }

        let mut converged = false;
        let score: f64;
        if mass > self.ignore_mass || self.missing.len() != 0 {
            if node_view.leaf_equals() {
                score =
                    (self.damp)(mass, self.tree_mass) * (self.score_seen)(node_view.get_depth(), mass);
                converged = true;
            } else {
                score = (self.score_unseen)(node_view.get_depth(), mass);
            }
        } else {
            // shadow box is undefined for missing values
            // for not missing values, this block corresponds to exact same evaluation
            // in score and attribution visitor
            // note that multi-visitors ignore the shadow box anyways
            score = (self.score_unseen)(node_view.get_depth(), mass);
            self.use_shadow_box = true;
        }

        let dist = (self.distance)(&new_point, &leaf_point);
        self.stack
            .push(ImputeVisitorStackElement {
                converged,
                score,
                index: node_view.get_leaf_index(),
                random: self.rng.gen::<f32>(),
                distance: dist
            });
    }

    fn accept(&mut self, _point: &[f32], node_view: &dyn LeafPointNodeView) {
        assert!(self.stack.len() > 0, " there should have been an accept_leaf call which would have created a non-null stack");
        let mut top_of_stack = self.stack.pop().unwrap();
        if !top_of_stack.converged {
            let prob = if !self.use_shadow_box {
                node_view.probability_of_cut_on_path()
            } else {
                node_view.get_shadow_box_probability_of_cut()
            };
            if prob == 0.0 {
                top_of_stack.converged = true;
            } else {
                let new_score = (1.0 - prob) * top_of_stack.score
                    + prob * (self.score_unseen)(node_view.get_depth(), node_view.get_mass());
                top_of_stack.converged = false;
                top_of_stack.score = new_score;
            }
            self.stack.push(top_of_stack);
        }
    }
}

impl MissingCoordinatesMultiLeafPointVisitor<(f64,usize,f64)> for ImputeVisitor {

    fn combine_branches(&mut self, _point: &[f32], _node_view: &dyn LeafPointNodeView) {
        assert!(self.stack.len() >= 2, "incorrect state");
        let mut top_of_stack = self.stack.pop().unwrap();
        let mut next_of_stack = self.stack.pop().unwrap();

        if self.adjusted_score(&top_of_stack) < self.adjusted_score(&next_of_stack) {
            top_of_stack.converged = top_of_stack.converged || next_of_stack.converged;
            self.stack.push(top_of_stack);
        } else {
            next_of_stack.converged = top_of_stack.converged || next_of_stack.converged;
            self.stack.push(next_of_stack);
        }
    }

}



