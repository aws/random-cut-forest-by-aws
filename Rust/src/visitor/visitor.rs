use crate::{
    l1distance,
    rcf::{
        damp, displacement_normalizer, identity, normalizer, score_seen, score_seen_displacement,
        score_unseen, score_unseen_displacement,
    },
    types::Result,
};

#[repr(C)]
pub struct VisitorInfo {
    pub ignore_mass: usize,
    pub score_seen: fn(usize, usize) -> f64,
    pub score_unseen: fn(usize, usize) -> f64,
    pub damp: fn(usize, usize) -> f64,
    pub normalizer: fn(f64, usize) -> f64,
    pub distance: fn(&[f32], &[f32]) -> f64,
}

pub trait Visitor<NodeView, R> {
    fn accept(&mut self, point: &[f32], visitor_info: &VisitorInfo, node_view: &NodeView) -> Result<()>;
    fn accept_leaf(&mut self, point: &[f32], visitor_info: &VisitorInfo, node_view: &NodeView) -> Result<()>;
    fn is_converged(&self) -> Result<bool>;
    fn result(&self, visitor_info: &VisitorInfo) -> Result<R>;
    fn use_shadow_box(&self) -> bool;
}

pub trait SimpleMultiVisitor<NodeView, R>: Visitor<NodeView, R> {
    fn combine_branches(
        &mut self,
        point: &[f32],
        _node_view: &NodeView,
        visitor_info: &VisitorInfo,
    ) -> Result<()>;
}

pub trait UniqueMultiVisitor<NodeView, R>: SimpleMultiVisitor<NodeView, R> {
    fn trigger(&self, point: &[f32], node_view: &NodeView, visitor_info: &VisitorInfo) -> bool;
    fn unique_answer(&self, visitor_info: &VisitorInfo) -> Vec<f32>;
}

pub trait StreamingMultiVisitor<NodeView, R>: UniqueMultiVisitor<NodeView, R> {
    fn initialize_branch_split(
        &mut self,
        point: &[f32],
        node_view: &NodeView,
        visitor_info: &VisitorInfo,
    );
    fn second_branch(&mut self, point: &[f32], node_view: &NodeView, visitor_info: &VisitorInfo);
}

impl VisitorInfo {
    pub fn default() -> Self {
        VisitorInfo {
            ignore_mass: 0,
            score_seen,
            score_unseen,
            damp,
            normalizer,
            distance: l1distance,
        }
    }
    pub fn displacement() -> Self {
        VisitorInfo {
            ignore_mass: 0,
            score_seen: score_seen_displacement,
            score_unseen: score_unseen_displacement,
            damp,
            normalizer: displacement_normalizer,
            distance: l1distance,
        }
    }
    pub fn density() -> Self {
        VisitorInfo {
            ignore_mass: 0,
            score_seen: score_unseen_displacement,
            score_unseen: score_unseen_displacement,
            damp,
            normalizer: identity,
            distance: l1distance,
        }
    }
    pub fn use_score(
        ignore_mass: usize,
        score_seen: fn(usize, usize) -> f64,
        score_unseen: fn(usize, usize) -> f64,
        damp: fn(usize, usize) -> f64,
        normalizer: fn(f64, usize) -> f64,
    ) -> Self {
        VisitorInfo {
            ignore_mass,
            score_seen,
            score_unseen,
            damp,
            normalizer,
            distance: l1distance,
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
