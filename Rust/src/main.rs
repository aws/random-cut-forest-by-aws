mod pointstore;
mod boundingbox;
mod cut;
mod sampler;
mod randomcuttree;
mod samplerplustree;
mod rcf;
mod multidimdatawithkey;
mod intervalstoremanager;
mod newnodestore;
mod abstractnodeview;
mod visitor;
mod scalarscorevisitor;

extern crate rand;

use num::abs;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use crate::rcf::RCF;
use crate::rcf::RCFSmall;
use crate::rcf::RCFMedium;
use crate::rcf::RCFLarge;

extern crate rand_chacha;

fn main() {
    let shingle_size = 8;
    let base_dimension = 5;
    let data_size = 100000;
    let number_of_trees = 30;
    let capacity = 256;
    let dimension = shingle_size * base_dimension;
    let point_store_capacity = (capacity*number_of_trees + 1);
    let time_decay = 0.1/capacity as f64;
    let initial_accept_fraction = 0.1;
    let bounding_box_fraction = 0.3;
    let random_seed = 17;
    let mut forest : Box<dyn RCF>;
    if (dimension < u8::MAX as usize)
        && (point_store_capacity*shingle_size  <= u16::MAX as usize)
        && (capacity -1 <= u8::MAX as usize) {
        println!(" choosing smallest");
        forest = Box::new(RCFSmall::new(shingle_size * base_dimension, shingle_size, 256, number_of_trees, random_seed, true, true, time_decay, initial_accept_fraction, bounding_box_fraction));
    } else if (dimension < u16::MAX as usize)
            && (capacity - 1  <= u16::MAX as usize) {
        println!(" choosing medium");
        forest = Box::new(RCFMedium::new(shingle_size * base_dimension, shingle_size, capacity, number_of_trees, random_seed, true, true, time_decay,initial_accept_fraction,bounding_box_fraction));
    } else {
        println!(" choosing large");
        forest = Box::new(RCFLarge::new(shingle_size * base_dimension, shingle_size, capacity, number_of_trees, random_seed, true, true, time_decay, initial_accept_fraction, bounding_box_fraction));
    };

    let mut data_with_key = multidimdatawithkey::MultiDimDataWithKey::new(data_size, 60, 100.0, 5.0, 0, base_dimension.into());

    let mut score: f64 = 0.0;
    let mut next_index: usize = 0;
    let mut rng = ChaCha20Rng::seed_from_u64(0);
    let mut factors = vec![0.0; 100];
    for i in 0..data_with_key.data.len() {
        let new_score = forest.score(&data_with_key.data[i]);

       // assert!(abs(new_score - forest.score(&data_with_key.data[i])) < 1e-5);
        /*
        if next_index < data_with_key.change_indices.len() && data_with_key.change_indices[next_index] == i {
            println!(" score at change {} position {} ", new_score, i);
            next_index += 1;
        }
        */
        score += new_score;
        forest.update(data_with_key.data[i].clone(), 0);
    }
    println!("Average score {} ", (score / data_with_key.data.len() as f64));
    println!("Success! {}", forest.get_entries_seen());
    println!("PointStore Size {} ", forest.get_point_store_size());
    println!("Total {}", forest.get_size());
}
