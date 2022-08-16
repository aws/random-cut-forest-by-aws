extern crate rand;
extern crate rand_chacha;
extern crate rcflib;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rcflib::{
    common::multidimdatawithkey::MultiDimDataWithKey,
    rcf::{create_rcf, RCF},
};

/// try cargo test --release
/// these tests are designed to be longish

#[test]
fn impute_different_period() {
    let shingle_size = 30;
    let base_dimension = 3;
    let data_size = 100000;
    let number_of_trees = 100;
    let capacity = 256;
    let initial_accept_fraction = 0.1;
    let dimensions = shingle_size * base_dimension;
    let _point_store_capacity = capacity * number_of_trees + 1;
    let time_decay = 0.1 / capacity as f64;
    let bounding_box_cache_fraction = 1.0;
    let random_seed = 17;
    let parallel_enabled: bool = true;
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
    let mut amplitude = Vec::new();
    for _i in 0..base_dimension {
        amplitude.push((1.0 + 0.2 * rng.gen::<f32>()) * 100.0);
    }
    let mut period_rng = ChaCha20Rng::seed_from_u64(7);
    let mut period = Vec::new();
    for _i in 0..base_dimension {
        period.push(((1.0 + 0.2 * period_rng.gen::<f32>()) * 60.0) as usize);
    }
    let data_with_key = MultiDimDataWithKey::multi_cosine(
        data_size,
        &period,
        &amplitude,
        noise,
        0,
        base_dimension.into(),
    );

    let _next_index = 0;
    let mut error = 0.0;
    let mut count = 0;

    for i in 0..data_with_key.data.len() {
        if i > 200 {
            let next_values = forest.extrapolate(1).unwrap().values;
            assert!(next_values.len() == base_dimension);
            error += next_values
                .iter()
                .zip(&data_with_key.data[i])
                .map(|(x, y)| ((x - y) as f64 * (x - y) as f64))
                .sum::<f64>();
            count += base_dimension;
        }
        forest.update(&data_with_key.data[i], 0);
    }

    println!("Success! {}", forest.entries_seen());
    println!("PointStore Size {} ", forest.point_store_size());
    println!("Total size {} bytes (approx)", forest.size());
    println!(
        " RMSE {},  noise {} ",
        f64::sqrt(error / count as f64),
        noise
    );
}
