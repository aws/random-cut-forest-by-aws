mod boundingbox;
mod cut;
mod imputevisitor;
mod intervalstoremanager;
pub mod multidimdatawithkey;
mod nodestore;
mod nodeview;
mod pointstore;
mod randomcuttree;
pub mod rcf;
mod sampler;
mod samplerplustree;
pub mod samplesummary;
mod scalarscorevisitor;
mod types;
mod visitor;
mod conditionalfieldsummarizer;
extern crate rand;

use num::abs;
use crate::rcf::{create_rcf, RCF};
extern crate rand_chacha;

pub fn L1distance(a: &[f32], b : &[f32]) -> f64{
    a.iter().zip(b).map(|(x,y)| abs(x-y) as f64).sum()
}



