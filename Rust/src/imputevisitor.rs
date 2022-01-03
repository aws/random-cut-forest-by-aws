use std::cmp::min;
use std::io::empty;
use crate::boundingbox::BoundingBox;
use crate::newnodestore::NewNodeStore;
use crate::newnodestore::NodeStoreView;
use crate::nodeview::NodeView;
use crate::pointstore::PointStoreView;
use crate::rcf::Max;
use crate::visitor::{StreamingMultiVisitor, UniqueMultiVisitor, Visitor, VisitorDescriptor};

#[repr(C)]
pub struct ImputeVisitor {
    damp: fn(usize,usize) -> f64,
    score_seen : fn(usize,usize) -> f64,
    score_unseen : fn(usize,usize) -> f64,
    normalizer : fn(f64,usize) -> f64,
    ignore_mass : usize,
    centrality : f64,
    tree_mass : usize,
    missing : Vec<usize>,
    values : Vec<(bool,f64,usize,Vec<f32>)>
}

impl ImputeVisitor {
    pub fn new(missing : &[usize], centrality: f64, tree_mass: usize, ignore_mass : usize, damp: fn(usize,usize) -> f64, score_seen : fn(usize,usize) -> f64, score_unseen : fn(usize,usize) -> f64,
               normalizer : fn(f64,usize) -> f64) -> Self {
        ImputeVisitor{
            centrality,
            tree_mass,
            ignore_mass,
            damp,
            score_seen,
            score_unseen,
            normalizer,
            missing : Vec::from(missing),
            values : Vec::new()
        }
    }
}

impl Visitor<f64> for ImputeVisitor {
    fn accept_leaf(&mut self, point: &[f32], node_view: &dyn NodeView) {
        let mass = node_view.get_mass();
        let mut score= 0.0;
        let leaf_point = node_view.get_leaf_point();
        let mut new_point = Vec::from(point);
        for i in 0..self.missing.len(){
            new_point[self.missing[i]] = leaf_point[self.missing[i]];
        }
        let mut converged = false;
        if point.eq(&new_point) && mass > self.ignore_mass {
            score = (self.damp)(mass, self.tree_mass) * (self.score_seen)(node_view.get_depth(), mass);
            converged = true;
        } else {
            score = (self.score_unseen)(node_view.get_depth(), mass);
        }

        self.values.push((converged,score,node_view.get_leaf_index(),new_point));
    }

    fn accept(&mut self, point: &[f32], node_view: &dyn NodeView) {
        let (converged,score,index,result) = self.values.pop().unwrap();
        if !converged {
            let prob = if self.ignore_mass == 0 { node_view.get_probability_of_cut(&result) } else { node_view.get_shadow_box().probability_of_cut(&result) };
            if prob == 0.0 {
                self.values.push((true,score,index,result));
            } else {
                let new_score = (1.0 - prob) * score + prob * (self.score_unseen)(node_view.get_depth(), node_view.get_mass());
                self.values.push((false,new_score,index,result));
            }
        }
    }

    fn get_result(&self) -> f64 {
        self.values.last().unwrap().1
    }

    fn has_converged(&self) -> bool {
            self.values.last().unwrap().0
    }

    fn descriptor(&self) -> VisitorDescriptor {
        VisitorDescriptor{
            use_point_copy_for_accept: true,
            use_box_for_accept: false,
            use_child_boxes_for_accept: false,
            use_mass_distribution_for_accept: false,
            maintain_shadow_box_for_accept: false,
            use_box_for_trigger: false,
            use_child_boxes_for_trigger: false,
            use_child_mass_distribution_for_trigger: false,
            trigger_manipulation_needs_node_view_accept_fields: false
        }
    }
}

impl UniqueMultiVisitor<f64,usize> for ImputeVisitor {
    fn get_arguments(&self) -> usize {
        assert_eq!(self.values.len(), 1, "incorrect state");
        self.values.last().unwrap().2
    }

    fn trigger(&self, point: &[f32], node_view: &dyn NodeView) -> bool {
        self.missing.contains(&node_view.get_cut_dimension())
    }

    fn combine_branches(&mut self, point: &[f32], node_view: &dyn NodeView) {
        assert!(self.values.len() >= 2, "incorrect state");
        let (first_converged, first_score, first_index, first_result) = self.values.pop().unwrap();
        let (second_converged, second_score, second_index, second_result) = self.values.pop().unwrap();
        if first_score < second_score {
            self.values.push((first_converged || second_converged, first_score, first_index, first_result));
        } else {
            self.values.push((first_converged || second_converged, second_score, second_index, second_result));
        }
    }

    fn unique_answer(&self) -> &[f32] {
        assert!(self.values.len() >= 1, "incorrect state, at least one leaf must have been visited");
        &self.values.last().unwrap().3
    }
}



