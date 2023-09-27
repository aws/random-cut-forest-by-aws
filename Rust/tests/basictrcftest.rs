extern crate rand;
extern crate rand_chacha;
extern crate rcflib;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rcflib::{
    common::multidimdatawithkey,
    trcf::basictrcf::BasicTRCF,
    trcf::types::TransformMethod::NONE
};
use rcflib::trcf::basictrcf::BasicTRCFBuilder;
use rcflib::trcf::types::ForestMode::STANDARD;
use rcflib::trcf::types::{TransformMethod};
use rcflib::trcf::types::TransformMethod::{DIFFERENCE, NORMALIZE, NORMALIZE_DIFFERENCE, SUBTRACT_MA, WEIGHTED};
use crate::rcflib::rcf::RCFOptionsBuilder;
use crate::rcflib::trcf::basictrcf::TRCFOptionsBuilder;
#[cfg(test)]
parameterized_test::create! { test_basic_trcf, (method), {
basic_trcf(method);
}}

#[cfg(test)]
parameterized_test::create! { basic_trcf_scale, (method,base_dimension,verbose), {
    trcf_scale(method,base_dimension,verbose,false);
}}

#[cfg(test)]
parameterized_test::create! { trcf_scale_spikes, (method,base_dimension,verbose), {
    trcf_scale(method,base_dimension,verbose,true);
}}

fn basic_trcf(transform_method : TransformMethod) {
    let shingle_size = 8;
    let base_dimension = 5;
    let data_size = 1000;
    let number_of_trees = 50;
    let capacity = 256;
    let initial_accept_fraction = 0.1;
    let _point_store_capacity = capacity * number_of_trees + 1;
    let time_decay = 0.1 / capacity as f64;
    let bounding_box_cache_fraction = 1.0;
    let random_seed = 17;
    let parallel_enabled: bool = false;
    let noise = 5.0;

    let mut trcf = BasicTRCFBuilder::new(base_dimension,shingle_size)
        .tree_capacity(capacity).number_of_trees(number_of_trees).random_seed(random_seed)
        .transform_method(transform_method)
        .forest_mode(STANDARD).parallel_enabled(parallel_enabled).verbose(true)
        .time_decay(time_decay).transform_decay(time_decay).initial_accept_fraction(initial_accept_fraction)
        .bounding_box_cache_fraction(bounding_box_cache_fraction).build().unwrap();


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
    ).unwrap();

    let mut next_index = 0;

    println!("{}", transform_method);
    for i in 0..data_with_key.data.len() {
        if next_index < data_with_key.change_indices.len() && data_with_key.change_indices[next_index] == i {
            print!("timestamp {} INJECT [ {}", i, data_with_key.changes[next_index][0]);
            for j in 1..base_dimension {
                print!(", {}", data_with_key.changes[next_index][j]);
            }
            println!("]");
            next_index += 1;
        }
        let result = trcf.process(&data_with_key.data[i], i as u64).unwrap();
        if result.anomaly_grade > 0.0 {
            print!("timestamp {} ", i);
            let gap = -result.last_anomaly.as_ref().unwrap().relative_index;
            if gap != 0 {
                if gap == 1 {
                    print!("1 step ago, ");
                } else {
                    print!("{} steps ago, ", gap);
                }
            }

            let expected = &result.last_anomaly.as_ref().unwrap().expected_values_list[0];
            let past = &result.last_anomaly.as_ref().unwrap().past_values;
            print!("DETECT [ {}", (past[0] - expected[0]));
            for j in 1..base_dimension {
                print!(", {}", (past[j] - expected[j]));
            }
            print!("]");

            println!(" score {}, grade {}", result.score, result.anomaly_grade);
        }
    }
}

test_basic_trcf! {
    a1: NONE,
    a2: NORMALIZE,
    a3: SUBTRACT_MA,
    a4: NORMALIZE_DIFFERENCE,
    a5: DIFFERENCE,
    a6: WEIGHTED,
}

basic_trcf_scale! {
    b1: (NONE,3,false),
    b2: (NONE,3,true),
    b3: (NORMALIZE,3,false),
    b4: (NORMALIZE,3,true),
    b5: (SUBTRACT_MA,3,false),
    b6: (SUBTRACT_MA,3,true),
    b7: (NORMALIZE_DIFFERENCE,3,false),
    b8: (NORMALIZE_DIFFERENCE,3,true),
    b9: (DIFFERENCE,3,false),
    b10: (DIFFERENCE,3,true),
    b11: (WEIGHTED,3,false),
    b12: (WEIGHTED,3,true),
}

trcf_scale_spikes! {
    c1: (NONE,1,false),
    c2: (NONE,1,true),
    c3: (NORMALIZE,1,false),
    c4: (NORMALIZE,1,true),
    c5: (SUBTRACT_MA,1,false),
    c6: (SUBTRACT_MA,1,true),
    c7: (NORMALIZE_DIFFERENCE,1,false),
    c8: (NORMALIZE_DIFFERENCE,1,true),
    c9: (DIFFERENCE,1,false),
    c10: (DIFFERENCE,1,true),
    c11: (WEIGHTED,1,false),
    c12: (WEIGHTED,1,true),
}

fn trcf_scale(transform_method:TransformMethod, base_dimension: usize, verbose:bool, add_spikes: bool) {
    let shingle_size = 8;
    let data_size = 100000;
    let number_of_trees = 50;
    let capacity = 256;
    let initial_accept_fraction = 0.1;
    let _point_store_capacity = capacity * number_of_trees + 1;
    let time_decay = 0.1 / capacity as f64;
    let bounding_box_cache_fraction = 1.0;
    let random_seed = 17;
    let parallel_enabled: bool = false;
    let noise = 5.0;

    println!("At scale {}, add spikes? {} verbose = {} ", transform_method, add_spikes, verbose);
    let mut trcf : BasicTRCF = BasicTRCFBuilder::new(base_dimension,shingle_size)
        .tree_capacity(capacity).number_of_trees(number_of_trees).random_seed(random_seed)
        .transform_method(transform_method)
        .forest_mode(STANDARD).parallel_enabled(parallel_enabled).verbose(verbose)
        .time_decay(time_decay).transform_decay(time_decay).initial_accept_fraction(initial_accept_fraction)
        .bounding_box_cache_fraction(bounding_box_cache_fraction).build().unwrap();

    let mut rng = ChaCha20Rng::from_entropy();
    let mut amplitude = Vec::new();
    for _i in 0..base_dimension {
        amplitude.push((1.0 + 0.2 * rng.gen::<f32>()) * 60.0);
    }
    let mut data_with_key = multidimdatawithkey::MultiDimDataWithKey::multi_cosine(
        data_size,
        &vec![60; base_dimension],
        &amplitude,
        noise,
        rng.gen::<u64>(),
        base_dimension.into(),
    ).unwrap();

    let mut potential_anomalies:Vec<usize> = Vec::new();

    let mut next = 100 + (rng.gen::<f32>()*100.0) as usize;
    for i in 0..data_with_key.data.len() {
        if add_spikes && i == next {
            data_with_key.data[i][0] += 100.0 * ( 1.0 + 0.05*rng.gen::<f32>());
            next = 100 + (rng.gen::<f32>()*100.0) as usize;
        }

        let result = trcf.process(&data_with_key.data[i], 0).unwrap();
        if result.anomaly_grade > 0.0 {
            // some anomalies will be detected late
            // we will keep the vector unsorted, so out of order detection will be penalized
            potential_anomalies.push((i as i32 + result.last_anomaly.as_ref().unwrap().relative_index) as usize);
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
