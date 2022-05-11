pub mod common;
mod pointstore;
pub mod rcf;
mod samplerplustree;
mod types;
pub mod visitor;
mod util;

extern crate rand;
extern crate rand_chacha;

use num::abs;


pub fn l1distance(a: &[f32], b : &[f32]) -> f64{
    a.iter().zip(b).map(|(&x,&y)| abs(x as f64 - y as f64)).sum()
}

pub fn linfinitydistance(a: &[f32], b : &[f32]) -> f64{
    let mut dist = 0.0;
    for i in 0..a.len(){
        let t= abs(a[i] as f64 - b[i] as f64);
        if dist < t {
            dist = t;
        };
    }
    dist
}

pub fn l2distance(a: &[f32], b : &[f32]) -> f64{
    f64::sqrt(a.iter().zip(b).map(|(&x,&y)| (abs(x as f64 -y as f64) * abs ( x as f64 - y as f64))).sum())
}

