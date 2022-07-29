extern crate rand;
extern crate rand_chacha;
extern crate rcflib;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rcflib::{
    common::multidimdatawithkey,
    rcf::{create_rcf, RCF},
    trcf::basictrcf::BasicTRCF
};


#[test]
fn test_basic_trcf() {
    let shingle_size = 8;
    let base_dimension = 5;
    let data_size = 1000;
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

    let mut trcf = BasicTRCF::new(
        dimensions,
        shingle_size,
        capacity,
        number_of_trees,
        random_seed, parallel_enabled,
        time_decay,
        0.01,
        initial_accept_fraction,
        bounding_box_cache_fraction,
    );

    let mut rng = ChaCha20Rng::seed_from_u64(42);
    let mut amplitude = Vec::new();
    for _i in 0..base_dimension {
        amplitude.push((1.0 + 0.2 * rng.gen::<f32>()) * 60.0);
    }

    let data_with_key = multidimdatawithkey::MultiDimDataWithKey::multi_cosine(
        data_size,
        &vec![60; base_dimension],
        &amplitude,
        noise,
        42,
        base_dimension.into(),
    );

    let mut next_index = 0;

    for i in 0..data_with_key.data.len() {
        if next_index < data_with_key.change_indices.len() && data_with_key.change_indices[next_index] == i {
            print!("timestamp {} INJECT [ {}", i, data_with_key.changes[next_index][0]);
            for j in 1..base_dimension {
                print!(", {}", data_with_key.changes[next_index][j]);
            }
            println!("]");
            next_index += 1;
        }
        let result = trcf.process(&data_with_key.data[i], 0).unwrap();
        if result.anomaly_grade > 0.0 {
            print!("timestamp {} ", i);
            if (result.relative_index.unwrap() != 0) {
                let gap = -result.relative_index.unwrap();
                if gap == 1 {
                    print!("1 step ago, ");
                } else {
                    print!("{} steps ago, ", gap);
                }
            }
            if result.forecast_reasonable {
                if let Some(expected_list) = result.expected_values_list {
                    let expected = &expected_list[0];
                    let past = &result.past_values.unwrap();
                    print!("DETECT [ {}", (past[0] - expected[0]));
                    for j in 1..base_dimension {
                        print!(", {}", (past[j] - expected[j]));
                    }
                    print!("]");
                }
            }
            println!(" score {}, grade {}", result.score, result.anomaly_grade);
        }
    }
}

#[test]
fn test_basic_trcf_scale() {
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

    let mut trcf = BasicTRCF::new(
        dimensions,
        shingle_size,
        capacity,
        number_of_trees,
        random_seed, parallel_enabled,
        time_decay,
        0.01,
        initial_accept_fraction,
        bounding_box_cache_fraction,
    );

    let mut rng = ChaCha20Rng::from_entropy();
    let mut amplitude = Vec::new();
    for _i in 0..base_dimension {
        amplitude.push((1.0 + 0.2 * rng.gen::<f32>()) * 60.0);
    }
    let data_with_key = multidimdatawithkey::MultiDimDataWithKey::multi_cosine(
        data_size,
        &vec![60; base_dimension],
        &amplitude,
        noise,
        rng.gen::<u64>(),
        base_dimension.into(),
    );

    let mut potential_anomalies:Vec<usize> = Vec::new();
    let mut late= 0;

    for i in 0..data_with_key.data.len() {
        let result = trcf.process(&data_with_key.data[i], 0).unwrap();
        if result.anomaly_grade > 0.0 {
            // some anomalies will be detected late
            // we will keep the vector unsorted, so out of order detection will be penalized
            potential_anomalies.push((i as i32 + result.relative_index.unwrap()) as usize);
        }
    }

    println!("{} anomalies injected in {} points", data_with_key.changes.len(), data_size);
    let mut common =0;
    let mut i:usize =0;
    let mut j:usize =0;
    while i<potential_anomalies.len() && j<data_with_key.change_indices.len() {
        if potential_anomalies[i] == data_with_key.change_indices[j] {
            i += 1;
            j += 1;
            common += 1;
        } else if potential_anomalies[i] < data_with_key.change_indices[j] {
            i += 1;
        } else {
            j += 1;
        }
    }

    println!("{} detected, precision {}, recall {}",potential_anomalies.len(),
             common as f32/potential_anomalies.len() as f32,
             common as f32/data_with_key.change_indices.len() as f32);
}
