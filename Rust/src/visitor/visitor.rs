use crate::l1distance;
use crate::rcf::{damp, normalizer, score_seen, score_unseen};
use crate::samplerplustree::nodeview::{AllowTraversal, LeafPointNodeView, MinimalNodeView, NodeView};


pub trait VisitorResult<T> {
    fn get_result(&self) -> T;
    fn has_converged(&self) -> bool;
    fn use_shadow_box(&self) -> bool;
}


pub trait AVisitor<Q,T> : VisitorResult<T> + AllowTraversal<Q> where Q : MinimalNodeView{
    //fn accept_leaf(&mut self, point: &[f32], node_view: &dyn MinimalNodeView);
    //fn accept(&mut self, point: &[f32], node_view: &dyn MinimalNodeView);
}

pub trait SimpleVisitor<T> : VisitorResult<T> {
    fn accept_leaf(&mut self, point: &[f32], node_view: &dyn MinimalNodeView);
    fn accept(&mut self, point: &[f32], node_view: &dyn MinimalNodeView);
}

pub trait LeafPointVisitor<T> : VisitorResult<T>{
    fn accept_leaf(&mut self, point: &[f32], node_view: &dyn LeafPointNodeView);
    fn accept(&mut self, point: &[f32], node_view: &dyn LeafPointNodeView);
}

pub trait Visitor<T> : VisitorResult<T>{
    fn accept_leaf(&mut self, point: &[f32], node_view: &dyn NodeView);
    fn accept(&mut self, point: &[f32], node_view: &dyn NodeView);
}

pub trait MissingCoordinatesMultiLeafPointVisitor<T>: LeafPointVisitor<T> {
    fn combine_branches(&mut self, point: &[f32], node_view: &dyn LeafPointNodeView);
}

pub trait MissingCoordinatesMultiVisitor<T>: Visitor<T> {
    fn combine_branches(&mut self, point: &[f32], node_view: &dyn NodeView);
}

pub trait UniqueMultiVisitor<T>: MissingCoordinatesMultiVisitor<T> {
    fn trigger(&self, point: &[f32], node_view: &dyn NodeView) -> bool;
    fn unique_answer(&self) -> &[f32];
}

pub trait StreamingMultiVisitor<T>: UniqueMultiVisitor<T> {
    fn initialize_branch_split(&mut self, point: &[f32], node_view: &dyn NodeView);
    fn second_branch(&mut self, point: &[f32], node_view: &dyn NodeView);
}

#[repr(C)]
pub struct VisitorInfo {
    pub ignore_mass: usize,
    pub score_seen: fn(usize, usize) -> f64,
    pub score_unseen: fn(usize, usize) -> f64,
    pub damp: fn(usize, usize) -> f64,
    pub normalizer: fn(f64, usize) -> f64,
    pub distance: fn(&[f32],&[f32]) -> f64
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