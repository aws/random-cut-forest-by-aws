use crate::l1distance;
use crate::rcf::{damp, normalizer, score_seen, score_unseen};

#[repr(C)]
pub struct VisitorInfo {
    pub ignore_mass: usize,
    pub score_seen: fn(usize, usize) -> f64,
    pub score_unseen: fn(usize, usize) -> f64,
    pub damp: fn(usize, usize) -> f64,
    pub normalizer: fn(f64, usize) -> f64,
    pub distance: fn(&[f32],&[f32]) -> f64
}


pub trait Visitor<NodeView, Result> {
    fn accept(&mut self, point : &[f32], visitor_info: &VisitorInfo, node_view: &NodeView);
    fn accept_leaf(&mut self, point : &[f32], visitor_info: &VisitorInfo, node_view: &NodeView);
    fn is_converged(&self) -> bool;
    fn result(&self,visitor_info:&VisitorInfo) -> Result;
    fn use_shadow_box(&self) -> bool;
}

pub trait SimpleMultiVisitor<NodeView, Result> : Visitor<NodeView, Result>{
    fn combine_branches(&mut self, point: &[f32], _node_view: &NodeView, visitor_info:&VisitorInfo);
}

pub trait UniqueMultiVisitor<NodeView,Result>: SimpleMultiVisitor<NodeView, Result> {
    fn trigger(&self, point: &[f32], node_view: &NodeView,visitor_info:&VisitorInfo) -> bool;
    fn unique_answer(&self,visitor_info:&VisitorInfo) -> Vec<f32>;
}

pub trait StreamingMultiVisitor<NodeView,Result>: UniqueMultiVisitor<NodeView,Result> {
    fn initialize_branch_split(&mut self, point: &[f32], node_view: &NodeView,visitor_info:&VisitorInfo);
    fn second_branch(&mut self, point: &[f32], node_view: &NodeView, visitor_info:&VisitorInfo);
}

impl VisitorInfo {
    pub fn default() -> Self {
        VisitorInfo {
            ignore_mass: 0,
            score_seen,
            score_unseen,
            damp,
            normalizer,
            distance : l1distance
        }
    }
    pub fn use_score(ignore_mass:usize,
               score_seen: fn(usize, usize) -> f64,
               score_unseen: fn(usize, usize) -> f64,
               damp: fn(usize, usize) -> f64,
               normalizer: fn(f64, usize) -> f64) -> Self {
        VisitorInfo {
            ignore_mass,
            score_seen,
            score_unseen,
            damp,
            normalizer,
            distance : l1distance
        }
    }
    pub fn use_distance(distance: fn(&[f32], &[f32]) -> f64) -> Self {
        VisitorInfo {
            ignore_mass: 0,
            score_seen,
            score_unseen,
            damp,
            normalizer,
            distance,
        }
    }
}