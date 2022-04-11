
extern crate rand;
extern crate rand_chacha;
extern crate rcflib;

use num::abs;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rcflib::common::multidimdatawithkey::MultiDimDataWithKey;
use rcflib::rcf::{create_rcf, RCF};

/// try cargo test --release
/// these tests are designed to be longish

#[test]
fn anomalydetection_attribution_and_update() {
    let shingle_size = 8;
    let base_dimension = 5;
    let data_size = 100000;
    let number_of_trees = 30;
    let capacity = 256;
    let initial_accept_fraction = 0.1;
    let dimensions = shingle_size * base_dimension;
    let _point_store_capacity = capacity * number_of_trees + 1;
    let time_decay = 0.1 / capacity as f64;
    let bounding_box_cache_fraction = 1.0;
    let random_seed = 17;
    let parallel_enabled: bool = false;
    let store_attributes: bool = false;
    let internal_shingling: bool = true;
    let internal_rotation = false;
    let noise = 5.0;

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
    let mut rng = ChaCha20Rng::seed_from_u64(42);
    let mut amplitude =  Vec::new();
    for _i in 0..base_dimension {
        amplitude.push( (1.0 + 0.2 * rng.gen::<f32>())*100.0);
    }
    let data_with_key = MultiDimDataWithKey::multi_cosine(
        data_size,
        &vec![60;base_dimension],
        &amplitude,
        noise,
        0,
        base_dimension.into(),
    );

    let mut score: f64 = 0.0;
    let _next_index = 0;

    for i in 0..data_with_key.data.len() {

        let attribution = forest.attribution(&data_with_key.data[i]);
        let new_score = forest.score(&data_with_key.data[i]);
        assert!(abs(new_score - attribution.total()) < 1e-6);

        /*
        if next_index < data_with_key.change_indices.len() && data_with_key.change_indices[next_index] == i {
            println!(" score at change {} position {} ", new_score, i);
            next_index += 1;
        }
        */

        score += attribution.total();
        forest.update(&data_with_key.data[i], 0);
    }

    println!(
        "Average score {} ",
        (score / data_with_key.data.len() as f64)
    );
    assert!(score < data_with_key.data.len() as f64, " average score is above 1");
    println!("Success! {}", forest.get_entries_seen());
    println!("PointStore Size {} ", forest.get_point_store_size());
    println!("Total size {} bytes (approx)", forest.get_size());
}
