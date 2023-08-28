extern crate rand;
extern crate rand_chacha;
extern crate rcflib;

use num::abs;
/// try cargo test --release
/// these tests are designed to be longish
use rand::{prelude::ThreadRng, Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rand_core::RngCore;
use rcflib::{
    common::{multidimdatawithkey::MultiDimDataWithKey},
    l1distance, l2distance,
};
use rcflib::common::cluster::{Center, multi_cluster_as_object_with_weight_array, multi_cluster_as_ref, multi_cluster_as_weighted_ref, multi_cluster_obj, persist, single_centroid_cluster_slice_with_weight_arrays, single_centroid_cluster_weighted_vec, single_centroid_cluster_weighted_vec_with_distance_over_slices, single_centroid_unweighted_cluster_slice};


fn gen_data(data_size:usize, test_dimension:usize, seed:u64,yard_stick : f32) -> MultiDimDataWithKey {
    let mut mean = Vec::new();
    let mut scale = Vec::new();
    for i in 0..test_dimension {
        let mut vec1 = vec![0.0f32; test_dimension];
        let mut vec2 = vec![0.0f32; test_dimension];
        vec1[i] = 2.0 * yard_stick;
        vec2[i] = -2.0 * yard_stick;
        mean.push(vec1);
        mean.push(vec2);
        scale.push(vec![0.1f32; test_dimension]);
        scale.push(vec![0.1f32; test_dimension]);
    }
    MultiDimDataWithKey::mixture(
        data_size,
        &mean,
        &scale,
        &vec![0.5 / test_dimension as f32; 2 * test_dimension],
        seed,
    ).unwrap()
}

fn test_center(result: &mut Vec<Center>,test_dimension:usize, yard_stick : f32) -> bool {
    let mut answer = true;
    for i in 0..test_dimension {
        result.sort_by(|a, b| a.representative()[i].partial_cmp(&b.representative()[i]).unwrap());
        answer = answer && abs(result[0].representative()[i] + 2.0 * yard_stick) < 0.2;
        answer = answer
            && abs(result[2 * test_dimension - 1].representative()[i] - 2.0 * yard_stick) < 0.2;
        for j in 1..(2 * test_dimension - 1) {
            answer = answer && abs(result[j].representative()[i]) < 0.2;
        }
    }
    answer
}

fn bad_distance<T :?Sized>(_a : &T, _b:&T) -> f64{
    -1.0
}

#[test]
fn test_config() {
    let test_dimension = 3;
    let yard_stick = l1distance(&vec![0.0; test_dimension], &vec![1.0; test_dimension]) as f32;
    let data_with_key = gen_data(1000,test_dimension,0u64,yard_stick);
    let mut input = Vec::new();
    for i in 0..data_with_key.data.len() {
        input.push((data_with_key.data[i].clone(), 1.0f32));
    }
    let result = single_centroid_cluster_weighted_vec_with_distance_over_slices(&input, bad_distance, 2 * test_dimension + 3, false);

    match &result {
        Ok(_x) => assert!(false),
        Err(_y) => assert!(true),
    };

    let result = single_centroid_cluster_weighted_vec_with_distance_over_slices(&input, l2distance, 0, false);

    match &result {
        Ok(_x) => assert!(false),
        Err(_y) => assert!(true),
    };

    let result = single_centroid_cluster_weighted_vec_with_distance_over_slices(&input, l2distance, 200, false);

    match &result {
        Ok(_x) => assert!(false),
        Err(_y) => assert!(true),
    };

    let result = single_centroid_cluster_weighted_vec_with_distance_over_slices(&input, l2distance, 20, false);

    match &result {
        Ok(_x) => assert!(true),
        Err(_y) => assert!(false),
    };
}

fn core(
    data_size: usize,
    test_dimension: usize,
    seed: u64,
    distance: fn(&[f32], &[f32]) -> f64,
) -> bool {
    println!(" starting {}",test_dimension);
    let yard_stick = distance(&vec![0.0; test_dimension], &vec![1.0; test_dimension]) as f32;
    let data_with_key = gen_data(data_size,test_dimension,seed,yard_stick);
    let mut input = Vec::new();
    for i in 0..data_with_key.data.len() {
        input.push((data_with_key.data[i].clone(), 1.0f32));
    }
    let mut result = single_centroid_cluster_weighted_vec_with_distance_over_slices(&input, distance, 2 * test_dimension + 3, false).unwrap();
    let answer = (result.len() == 2 * test_dimension) && test_center(&mut result,test_dimension,yard_stick);
    println!(" done {} {}",test_dimension,answer);
    answer
}

#[test]
fn benchmark_cluster() {
    let mut generator = ThreadRng::default();
    let one_seed: u64 = generator.gen();
    println!(" single seed is {}", one_seed);
    let mut rng = ChaCha20Rng::seed_from_u64(one_seed);

    let mut error = 0;
    for _ in 0..10 {
        let seed = rng.next_u64();
        let d = rng.gen_range(3..23);
        error += (core(200000, d, seed, l1distance) == false) as i32;
    }
    assert!(error < 5);
}

fn core_as_slice_uniform(
    data_size: usize,
    test_dimension: usize,
    seed: u64,
    distance: fn(&[f32], &[f32]) -> f64,
) -> bool {
    println!(" starting {}",test_dimension);
    let yard_stick = distance(&vec![0.0; test_dimension], &vec![1.0; test_dimension]) as f32;
    let data_with_key = gen_data(data_size,test_dimension,seed,yard_stick);

    let mut input:Vec<&[f32]> = Vec::new();
    for i in 0..data_with_key.data.len() {
        input.push(&data_with_key.data[i]);
    }
    let mut result = single_centroid_unweighted_cluster_slice(&input, distance, 2 * test_dimension + 3, false).unwrap();
    let answer = (result.len() == 2 * test_dimension) && test_center(&mut result,test_dimension,yard_stick);
    println!(" done {} {}",test_dimension,answer);
    answer
}

#[test]
fn benchmark_slice_uniform() {
    let mut generator = ThreadRng::default();
    let one_seed: u64 = generator.gen();
    println!(" single seed is {}", one_seed);
    let mut rng = ChaCha20Rng::seed_from_u64(one_seed);

    let mut error = 0;
    for _ in 0..10 {
        let seed = rng.next_u64();
        let d = rng.gen_range(3..23);
        error += (core_as_slice_uniform(200000, d, seed, l1distance) == false) as i32;
    }
    assert!(error < 5);
}


fn core_as_slice_weighted(
    data_size: usize,
    test_dimension: usize,
    seed: u64,
    distance: fn(&[f32], &[f32]) -> f64,
) -> bool {
    println!(" starting {}",test_dimension);
    let yard_stick = distance(&vec![0.0; test_dimension], &vec![1.0; test_dimension]) as f32;
    let data_with_key = gen_data(data_size,test_dimension,seed,yard_stick);
    let mut input:Vec<&[f32]> = Vec::new();
    for i in 0..data_with_key.data.len() {
        input.push(&data_with_key.data[i]);
    }
    let weights = vec![1.0f32;data_with_key.data.len()];
    let mut result = single_centroid_cluster_slice_with_weight_arrays(&input, &weights, distance, 2 * test_dimension + 3, false).unwrap();
    let answer = (result.len() == 2 * test_dimension) && test_center(&mut result,test_dimension,yard_stick);
    println!(" done {} {}",test_dimension,answer);
    answer
}

#[test]
fn benchmark_slice_weighted() {
    let mut generator = ThreadRng::default();
    let one_seed: u64 = generator.gen();
    println!(" single seed is {}", one_seed);
    let mut rng = ChaCha20Rng::seed_from_u64(one_seed);

    let mut error = 0;
    for _ in 0..10 {
        let seed = rng.next_u64();
        let d = rng.gen_range(3..23);
        error += (core_as_slice_weighted(200000, d, seed, l1distance) == false) as i32;
    }
    assert!(error < 5);
}

fn vec_dist(a: &Vec<f32>, b: &Vec<f32>) -> f64 {
    l1distance(&a,&b)
}


fn core_vec(
    data_size: usize,
    test_dimension: usize,
    seed: u64,
    distance: fn(&Vec<f32>, &Vec<f32>) -> f64,
) -> bool {
    println!(" starting {}",test_dimension);
    let yard_stick = distance(&vec![0.0; test_dimension], &vec![1.0; test_dimension]) as f32;
    let data_with_key = gen_data(data_size,test_dimension,seed,yard_stick);
    let mut input = Vec::new();
    for i in 0..data_with_key.data.len() {
        input.push((data_with_key.data[i].clone(),1.0f32));
    }

    let mut result = single_centroid_cluster_weighted_vec(&input, distance, 2 * test_dimension + 3, false).unwrap();
    let answer = (result.len() == 2 * test_dimension) && test_center(&mut result,test_dimension,yard_stick);
    println!(" done {} {}",test_dimension,answer);
    answer
}

#[test]
fn benchmark_vec() {
    let mut generator = ThreadRng::default();
    let one_seed: u64 = generator.gen();
    println!(" single seed is {}", one_seed);
    let mut rng = ChaCha20Rng::seed_from_u64(one_seed);

    let mut error = 0;
    for _ in 0..10 {
        let seed = rng.next_u64();
        let d = rng.gen_range(3..23);
        error += (core_vec(200000, d, seed, vec_dist) == false) as i32;
    }
    assert!(error < 5);
}

fn multi_as_vec(
    data_size: usize,
    test_dimension: usize,
    seed: u64,
    distance: fn(&[f32], &[f32]) -> f64,
) -> bool {
    println!(" starting {}",test_dimension);
    let yard_stick = distance(&vec![0.0; test_dimension], &vec![1.0; test_dimension]) as f32;
    let data_with_key = gen_data(data_size,test_dimension,seed,yard_stick);
    let mut input:Vec<Vec<f32>> = Vec::new();
    for i in 0..data_with_key.data.len() {
        input.push(data_with_key.data[i].clone());
    }
    let mut result = multi_cluster_obj(&input,   vec_dist, 5,0.1,true,2 * test_dimension + 3, false).unwrap();
    let mut answer = result.len() == 2 * test_dimension;
    for i in 0..test_dimension {
        result.sort_by(|a, b| a.representatives()[0].0[i].partial_cmp(&b.representatives()[0].0[i]).unwrap());
        answer = answer && abs(result[0].representatives()[0].0[i] + 2.0 * yard_stick) < 0.5;
        answer = answer
            && abs(result[2 * test_dimension - 1].representatives()[0].0[i] - 2.0 * yard_stick) < 0.5;
        for j in 1..(2 * test_dimension - 1) {
            answer = answer && abs(result[j].representatives()[0].0[i]) < 0.5;
        }
    }
    println!(" done {} {}",test_dimension,answer);
    answer
}

#[test]
fn benchmark_multi_vec() {
    let mut generator = ThreadRng::default();
    let one_seed: u64 = generator.gen();
    println!(" single seed is {}", one_seed);
    let mut rng = ChaCha20Rng::seed_from_u64(one_seed);

    let mut error = 0;
    for _ in 0..10 {
        let seed = rng.next_u64();
        let d = rng.gen_range(3..23);
        error += (multi_as_vec(200000, d, seed, l1distance) == false) as i32;
    }
    assert!(error < 5);
}

fn multi_as_ref(
    data_size: usize,
    test_dimension: usize,
    seed: u64,
    distance: fn(&[f32], &[f32]) -> f64,
) -> bool {
    println!(" starting {}",test_dimension);
    let yard_stick = distance(&vec![0.0; test_dimension], &vec![1.0; test_dimension]) as f32;
    let data_with_key = gen_data(data_size,test_dimension,seed,yard_stick);
    let mut input = Vec::new();
    for i in 0..data_with_key.data.len() {
        input.push(&data_with_key.data[i]);
    }
    let mut result = multi_cluster_as_ref(&input,   vec_dist, 5,0.1,true,2 * test_dimension + 3, false).unwrap();
    let mut answer = result.len() == 2 * test_dimension;
    for i in 0..test_dimension {
        result.sort_by(|a, b| a.representatives()[0].0[i].partial_cmp(&b.representatives()[0].0[i]).unwrap());
        answer = answer && abs(result[0].representatives()[0].0[i] + 2.0 * yard_stick) < 0.5;
        answer = answer
            && abs(result[2 * test_dimension - 1].representatives()[0].0[i] - 2.0 * yard_stick) < 0.5;
        for j in 1..(2 * test_dimension - 1) {
            answer = answer && abs(result[j].representatives()[0].0[i]) < 0.5;
        }
    }
    println!(" done {} {}",test_dimension,answer);
    answer
}

#[test]
fn benchmark_multi_ref() {
    let mut generator = ThreadRng::default();
    let one_seed: u64 = generator.gen();
    println!(" single seed is {}", one_seed);
    let mut rng = ChaCha20Rng::seed_from_u64(one_seed);

    let mut error = 0;
    for _ in 0..10 {
        let seed = rng.next_u64();
        let d = rng.gen_range(3..23);
        error += (multi_as_ref(200000, d, seed, l1distance) == false) as i32;
    }
    assert!(error < 5);
}

fn multi_as_weighted_ref(
    data_size: usize,
    test_dimension: usize,
    seed: u64,
    distance: fn(&[f32], &[f32]) -> f64,
) -> bool {
    println!(" starting {}",test_dimension);
    let yard_stick = distance(&vec![0.0; test_dimension], &vec![1.0; test_dimension]) as f32;
    let data_with_key = gen_data(data_size,test_dimension,seed,yard_stick);
    let mut input = Vec::new();
    for i in 0..data_with_key.data.len() {
        input.push((&data_with_key.data[i],1.0f32));
    }
    let mut result = multi_cluster_as_weighted_ref(&input,   vec_dist, 5,0.1,true,2 * test_dimension + 3, false).unwrap();
    let mut answer = result.len() == 2 * test_dimension;
    for i in 0..test_dimension {
        result.sort_by(|a, b| a.representatives()[0].0[i].partial_cmp(&b.representatives()[0].0[i]).unwrap());
        answer = answer && abs(result[0].representatives()[0].0[i] + 2.0 * yard_stick) < 0.5;
        answer = answer
            && abs(result[2 * test_dimension - 1].representatives()[0].0[i] - 2.0 * yard_stick) < 0.5;
        for j in 1..(2 * test_dimension - 1) {
            answer = answer && abs(result[j].representatives()[0].0[i]) < 0.5;
        }
    }
    println!(" done {} {}",test_dimension,answer);
    answer
}

#[test]
fn benchmark_multi_weighted_ref() {
    let mut generator = ThreadRng::default();
    let one_seed: u64 = generator.gen();
    println!(" single seed is {}", one_seed);
    let mut rng = ChaCha20Rng::seed_from_u64(one_seed);

    let mut error = 0;
    for _ in 0..10 {
        let seed = rng.next_u64();
        let d = rng.gen_range(3..23);
        error += (multi_as_weighted_ref(200000, d, seed, l1distance) == false) as i32;
    }
    assert!(error < 5);
}


fn multi_as_vec_weighted(
    data_size: usize,
    test_dimension: usize,
    seed: u64,
    distance: fn(&[f32], &[f32]) -> f64,
) -> bool {
    println!(" starting {}",test_dimension);
    let yard_stick = distance(&vec![0.0; test_dimension], &vec![1.0; test_dimension]) as f32;
    let data_with_key = gen_data(data_size,test_dimension,seed,yard_stick);
    let mut input:Vec<Vec<f32>> = Vec::new();
    for i in 0..data_with_key.data.len() {
        input.push(data_with_key.data[i].clone());
    }
    let weights = vec![1.0f32;data_with_key.data.len()];
    let ref_result = multi_cluster_as_object_with_weight_array(&input,  &weights, vec_dist, 5,0.1,true,2 * test_dimension + 3, false).unwrap();
    let mut result = persist(&ref_result);
    let mut answer = result.len() == 2 * test_dimension;
    for i in 0..test_dimension {
        result.sort_by(|a, b| a.representatives()[0].0[i].partial_cmp(&b.representatives()[0].0[i]).unwrap());
        answer = answer && abs(result[0].representatives()[0].0[i] + 2.0 * yard_stick) < 0.5;
        answer = answer
            && abs(result[2 * test_dimension - 1].representatives()[0].0[i] - 2.0 * yard_stick) < 0.5;
        for j in 1..(2 * test_dimension - 1) {
            answer = answer && abs(result[j].representatives()[0].0[i]) < 0.5;
        }
    }
    println!(" done {} {}",test_dimension,answer);
    answer
}

#[test]
fn benchmark_multi_vec_weighted() {
    let mut generator = ThreadRng::default();
    let one_seed: u64 = generator.gen();
    println!(" single seed is {}", one_seed);
    let mut rng = ChaCha20Rng::seed_from_u64(one_seed);

    let mut error = 0;
    for _ in 0..10 {
        let seed = rng.next_u64();
        let d = rng.gen_range(3..23);
        error += (multi_as_vec_weighted(200000, d, seed, l1distance) == false) as i32;
    }
    assert!(error < 5);
}
