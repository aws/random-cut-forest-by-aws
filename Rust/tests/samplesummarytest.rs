
extern crate rand;
extern crate rand_chacha;
extern crate rcflib;

use num::abs;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;


/// try cargo test --release
/// these tests are designed to be longish

use parameterized_test::create;
use rand::prelude::ThreadRng;
use rand_core::RngCore;
use rcflib::common::multidimdatawithkey::MultiDimDataWithKey;
use rcflib::common::samplesummary::summarize;
use rcflib::{l1distance, l2distance,linfinitydistance};

#[cfg(test)]
parameterized_test::create! { sample_summary_distance_test, (test_dimension,distance), {
    assert!(core(1000000,test_dimension,0,distance));
    }}

fn core(data_size: usize, test_dimension: usize, seed: u64,distance : fn(&[f32],&[f32]) -> f64) -> bool{
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let data_size = 200000;
    let mut mean = Vec::new();
    let mut scale = Vec::new();
    let yard_stick = distance(&vec![0.0;test_dimension],&vec![1.0;test_dimension]) as f32;
    for i in 0..test_dimension {
        let mut vec1 = vec![0.0f32;test_dimension];
        let mut vec2 = vec![0.0f32;test_dimension];
        vec1[i] = 2.0 * yard_stick;
        vec2[i] = - 2.0 * yard_stick;
        mean.push(vec1);
        mean.push(vec2);
        scale.push(vec![0.1f32;test_dimension]);
        scale.push(vec![0.1f32;test_dimension]);
    };

    let data_with_key = MultiDimDataWithKey::mixture(
        data_size,
        &mean,
        &scale,
        &vec![(0.5 / test_dimension as f32);2*test_dimension],
        seed
    );

    let mut input = Vec::new();
    for i in 0..data_with_key.data.len() {
        input.push((data_with_key.data[i].clone(),1.0f32));
    }
    let mut result = summarize(&input,distance,2*test_dimension+3,false);
    let mut answer = result.summary_points.len() == 2 * test_dimension;
    // should be two centers per dimension
    // the top two should correspond to +/- 5.0 in first dimension
    for i in 0..test_dimension {
        result.summary_points.sort_by(|a,b| a[i].partial_cmp(&b[i]).unwrap());
        answer = answer && abs(result.summary_points[0][i] + 2.0 * yard_stick) < 0.2;
        answer = answer && abs(result.summary_points[2*test_dimension-1][i] - 2.0 * yard_stick) < 0.2;
        for j in 1..(2*test_dimension-1) {
            answer = answer && abs(result.summary_points[j][i]) < 0.2;
        }
    }
    answer
}

sample_summary_distance_test! {
    a1 : (1,l1distance),
    b1 : (1,l2distance),
    c1 : (1,linfinitydistance),
    a2 : (2,l1distance),
    b2 : (3,l2distance),
    c2 : (4,linfinitydistance),
    a3 : (5,l1distance),
    b3 : (5,l2distance),
    c3 : (5,linfinitydistance),

}

#[test]
fn benchmark(){
    let mut generator = ThreadRng::default();
    let one_seed: u64 = generator.gen();
    println!(" single seed is {}", one_seed);
    let mut rng = ChaCha20Rng::seed_from_u64(one_seed);

    let mut error = 0;
    for i in 0..10 {
        let seed =rng.next_u64();
        let d = rng.gen_range(3..23);
        error += (core(200000,d,seed,l1distance) == false) as i32;
    }
    assert!(error<5);
}

