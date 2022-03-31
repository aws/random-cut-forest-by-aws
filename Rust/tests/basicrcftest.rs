
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
fn two_distribution_test_static() {

    let data_size = 100000;
    let dimensions = 20;
    let mut vec1 = vec![0.0f32;dimensions];
    let mut vec2 = vec![0.0f32;dimensions];
    vec1[0] = 5.0;
    vec2[0] = - 5.0;
    let scale = vec![vec![0.1f32;dimensions],vec![0.1f32;dimensions]];
    let mean = vec![vec1,vec2];
    let data_with_key = multidimdatawithkey::MultiDimDataWithKey::mixture(
        data_size,
        &mean,
        &scale,
        &vec![0.5f32,0.5f32],
        0
    );

    let shingle_size = 1;
    let number_of_trees = 30;
    let capacity = 256;
    let initial_accept_fraction = 0.1;
    let _point_store_capacity = capacity * number_of_trees + 1;
    let time_decay = 0.1 / capacity as f64;
    let bounding_box_cache_fraction = 1.0;
    let random_seed = 17;
    let parallel_enabled: bool = false;
    let store_attributes: bool = false;
    let internal_shingling: bool = false;
    let internal_rotation = false;

    let mut forest: Box<dyn RCF> = create_rcf(
        dimensions,
        shingle_size,
        capacity,
        number_of_trees,
        random_seed,
        store_attributes,
        parallel_enabled,
        internal_shingling,
        internal_rotation,
        time_decay,
        initial_accept_fraction,
        bounding_box_cache_fraction,
    );

    for i in 0..data_with_key.data.len() {
         forest.update(&data_with_key.data[i],0);
    }

    assert!(forest.score(&vec![0.0f32;dimensions]) > 1.5);

    
    }