
extern crate rand;
extern crate rand_chacha;
extern crate rcflib;

use num::abs;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rcflib::{L1distance, L2distance, LInfinitydistance, multidimdatawithkey};
use rcflib::multidimdatawithkey::MultiDimDataWithKey;
use rcflib::samplesummary::summarize;

/// try cargo test --release
/// these tests are designed to be longish

use parameterized_test::create;

#[cfg(test)]
parameterized_test::create! { sample_summary_distance_test, (test_dimension,distance), {
    let data_size = 1000000;
    let dimensions = 40;
    let mut mean = Vec::new();
    let mut scale = Vec::new();
    for i in 0..test_dimension {
        let mut vec1 = vec![0.0f32;dimensions];
        let mut vec2 = vec![0.0f32;dimensions];
        vec1[i] = 5.0;
        vec2[i] = - 5.0;
        mean.push(vec1);
        mean.push(vec2);
        scale.push(vec![0.1f32;dimensions]);
        scale.push(vec![0.1f32;dimensions]);
    };

    let data_with_key = multidimdatawithkey::MultiDimDataWithKey::mixture(
        data_size,
        &mean,
        &scale,
        &vec![(0.5 / test_dimension as f32);2*test_dimension],
        0
    );

    let mut input = Vec::new();
    for i in 0..data_with_key.data.len() {
         input.push((data_with_key.data[i].clone(),1.0f32));
    }
    let mut result = summarize(&input,distance,2*test_dimension+3);
    assert!(result.summary_points.len() == 2 * test_dimension, " should be two centers per dimension");
    // the top two should correspond to +/- 5.0 in first dimension
    for i in 0..test_dimension {
        result.summary_points.sort_by(|a,b| a[i].partial_cmp(&b[i]).unwrap());
        assert!(abs(result.summary_points[0][i] + 5.0) < 0.2);
        assert!(abs(result.summary_points[2*test_dimension-1][i] - 5.0) < 0.2);
        for j in 1..(2*test_dimension-1) {
            assert!(abs(result.summary_points[j][i]) < 0.2);
        }
        // remainder of coordinates are near 0
        for j in 0..dimensions {
            if (j != i) {
                assert!(abs(result.summary_points[0][j]) < 0.2);
                assert!(abs(result.summary_points[2 * test_dimension - 1][j]) < 0.2);
            }
        }
    }
    }}

sample_summary_distance_test! {
    a1 : (1,L1distance),
    b1 : (1,L2distance),
    c1 : (1,LInfinitydistance),
    a2 : (2,L1distance),
    b2 : (3,L2distance),
    c2 : (4,LInfinitydistance),
    a3 : (5,L1distance),
    b3 : (5,L2distance),
    c3 : (5,LInfinitydistance),

}