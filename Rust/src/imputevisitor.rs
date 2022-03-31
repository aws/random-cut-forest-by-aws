use num::abs;
use rand::{random, Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use crate::{L1distance, nodestore::NodeStore, nodeview::NodeView, visitor::{UniqueMultiVisitor, Visitor, VisitorDescriptor}};


#[repr(C)]
pub struct ImputeVisitor {
    damp: fn(usize, usize) -> f64,
    score_seen: fn(usize, usize) -> f64,
    score_unseen: fn(usize, usize) -> f64,
    normalizer: fn(f64, usize) -> f64,
    ignore_mass: usize,
    centrality: f64,
    tree_mass: usize,
    rng : ChaCha20Rng,
    missing: Vec<usize>,
    stack: Vec<ImputeVisitorStackElement>,
}

#[repr(C)]
struct ImputeVisitorStackElement{
    converged: bool,
    score : f64,
    random : f32,
    index : usize,
    imputed_point : Vec<f32>,
    distance : f32
}

impl ImputeVisitor {
    pub fn new(
        missing: &[usize],
        centrality: f64,
        tree_mass: usize,
        seed : u64,
        ignore_mass: usize,
        damp: fn(usize, usize) -> f64,
        score_seen: fn(usize, usize) -> f64,
        score_unseen: fn(usize, usize) -> f64,
        normalizer: fn(f64, usize) -> f64,
    ) -> Self {
        ImputeVisitor {
            centrality,
            tree_mass,
            rng : ChaCha20Rng::seed_from_u64(seed),
            ignore_mass,
            damp,
            score_seen,
            score_unseen,
            normalizer,
            missing: Vec::from(missing),
            stack : Vec::new()
        }
    }

    /// the following function allows the score to vary between the score used in
    /// anomaly detection and fully random sample based on the parameter centrality
    /// these two cases correspond to centrality = 1 and centrality = 0 respectively

    fn adjusted_score(&self, e : &ImputeVisitorStackElement)->f64{
           self.centrality * (self.normalizer) (e.score,self.tree_mass) +
               (1.0 - self.centrality) * e.random as f64
    }
}




impl Visitor<f64> for ImputeVisitor {
    fn accept_leaf(&mut self, point: &[f32], node_view: &dyn NodeView) {
        let mass = node_view.get_mass();
        let mut score = 0.0;
        let leaf_point = node_view.get_leaf_point();
        let mut new_point = Vec::from(point);
        for i in 0..self.missing.len() {
            new_point[self.missing[i]] = leaf_point[self.missing[i]];
        }
        let mut converged = false;
        if point.eq(&new_point) && mass > self.ignore_mass {
            score =
                (self.damp)(mass, self.tree_mass) * (self.score_seen)(node_view.get_depth(), mass);
            converged = true;
        } else {
            score = (self.score_unseen)(node_view.get_depth(), mass);
        }
        let dist = L1distance(&new_point,&leaf_point) as f32;
        self.stack
            .push(ImputeVisitorStackElement{
                converged,
                score,
                index: node_view.get_leaf_index(),
                random : self.rng.gen::<f32>(),
                imputed_point: new_point,
                distance : dist});
    }

    fn accept(&mut self, _point: &[f32], node_view: &dyn NodeView) {
        assert!(self.stack.len() > 0, " there should have been an accept_leaf call which would have created a non-null stack");
        let mut top_of_stack = self.stack.pop().unwrap();
        if !top_of_stack.converged {
            let prob = if self.ignore_mass == 0 {
                node_view.get_probability_of_cut(&top_of_stack.imputed_point)
            } else {
                node_view.get_shadow_box().probability_of_cut(&top_of_stack.imputed_point)
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

    fn get_result(&self) -> f64 {
        self.stack.last().unwrap().score
    }

    fn has_converged(&self) -> bool {
        self.stack.last().unwrap().converged
    }

    fn descriptor(&self) -> VisitorDescriptor {
        VisitorDescriptor {
            use_point_copy_for_accept: true,
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

impl UniqueMultiVisitor<f64, (usize,f32)> for ImputeVisitor {
    fn get_arguments(&self) -> (usize,f32) {
        assert_eq!(self.stack.len(), 1, "incorrect state");
        let top_of_stack = self.stack.last().unwrap();
        (top_of_stack.index,top_of_stack.distance)
    }

    fn trigger(&self, _point: &[f32], node_view: &dyn NodeView) -> bool {
        self.missing.contains(&node_view.get_cut_dimension())
    }

    fn combine_branches(&mut self, _point: &[f32], _node_view: &dyn NodeView) {
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

    fn unique_answer(&self) -> &[f32] {
        assert!(
            self.stack.len() >= 1,
            "incorrect state, at least one leaf must have been visited"
        );
        &self.stack.last().unwrap().imputed_point
    }
}
