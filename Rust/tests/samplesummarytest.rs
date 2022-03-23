
extern crate rand;
extern crate rand_chacha;
extern crate rcflib;

use num::abs;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rcflib::{L1distance, multidimdatawithkey};
use rcflib::multidimdatawithkey::MultiDimDataWithKey;
use rcflib::rcf::{create_rcf, RCF};
use rcflib::samplesummary::summarize;

/// try cargo test --release
/// these tests are designed to be longish


#[test]
fn sample_summary_test() {

    let data_size = 100000;
    let dimensions = 40;
    let mut vec1 = vec![0.0f32;dimensions];
    let mut vec2 = vec![0.0f32;dimensions];
    vec1[0] = 5.0;
    vec2[0] = - 5.0;
    let scalevec = vec![0.1f32;dimensions];
    let scale = vec![vec![0.1f32;dimensions],vec![0.1f32;dimensions]];
    let mean = vec![vec1,vec2];
    let data_with_key = multidimdatawithkey::MultiDimDataWithKey::mixture(
        data_size,
        &mean,
        &scale,
        &vec![0.5f32,0.5f32],
        0
    );

    let mut input = Vec::new();
    for i in 0..data_with_key.data.len() {
         input.push((data_with_key.data[i].clone(),1.0f32));
    }
    let result = summarize(&input,L1distance,5);
    assert!(result.summary_points.len() == 2, " shoulf be two centers");
    /// the top two should correspond to +/- 5.0 in first dimension
    assert!( abs(result.summary_points[0][0] - 5.0) <0.2 || abs(result.summary_points[0][0] + 5.0) <0.2);
    assert!( abs(result.summary_points[1][0] - 5.0) <0.2 || abs(result.summary_points[1][0] + 5.0) <0.2);
    let gap = abs(result.summary_points[0][0] - result.summary_points[1][0]);
    assert!(gap > 9.8 && gap < 10.02);
    /// remainder of coordinates are near 0
    for i in 1..dimensions {
        assert!(abs(result.summary_points[0][i]) < 0.2);
        assert!(abs(result.summary_points[1][i]) < 0.2);
    }

    }