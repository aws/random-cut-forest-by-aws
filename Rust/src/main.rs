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
mod visitor;
mod scalarscorevisitor;
mod imputevisitor;
mod nodeview;

extern crate rand;

use num::abs;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use crate::othermain::alternate_main;
use crate::rcf::{create_rcf, RCF};
use crate::rcf::RCFTiny;
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
    let initial_accept_fraction = 0.1;
    let dimensions = shingle_size * base_dimension;
    let point_store_capacity = (capacity*number_of_trees + 1);
    let time_decay = 0.1/capacity as f64;
    let bounding_box_cache_fraction = 1.0;
    let random_seed = 17;
    let parallel_enabled : bool = false;
    let store_attributes : bool = false;
    let internal_shingling: bool = true;
    let internal_rotation = false;

    let mut forest: Box<dyn RCF> = create_rcf(dimensions, shingle_size, capacity, number_of_trees, random_seed, store_attributes, parallel_enabled, internal_shingling,internal_rotation,time_decay, initial_accept_fraction, bounding_box_cache_fraction);
    let mut data_with_key = multidimdatawithkey::MultiDimDataWithKey::new(data_size, 60, 100.0, 5.0, 0, base_dimension.into());


    let mut score: f64 = 0.0;
    let mut next_index = 0;

        for i in 0..data_with_key.data.len() {
            let new_score = forest.score(&data_with_key.data[i]);
            //println!("{} {} score {}",y,i,new_score);
            /*
        if next_index < data_with_key.change_indices.len() && data_with_key.change_indices[next_index] == i {
            println!(" score at change {} position {} ", new_score, i);
            next_index += 1;
        }
        */
            score += new_score;
            forest.update(&data_with_key.data[i], 0);
        }

    println!("Average score {} ", (score / data_with_key.data.len() as f64));
    println!("Success! {}", forest.get_entries_seen());
    println!("PointStore Size {} ", forest.get_point_store_size());
    println!("Total size {} bytes (approx)", forest.get_size());

}
