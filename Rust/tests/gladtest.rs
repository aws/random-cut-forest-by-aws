extern crate rand;
extern crate rand_chacha;
extern crate rcflib;

use std::f32::consts::PI;
/// try cargo test --release
/// these tests are designed to be longish
use rand::{prelude::ThreadRng, Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rand_core::RngCore;
use rcflib::{
    common::{multidimdatawithkey::MultiDimDataWithKey, samplesummary::summarize},
    l1distance, l2distance, linfinitydistance,
};
use rcflib::common::cluster::{Center, multi_cluster_as_object_with_weight_array, multi_cluster_as_ref, multi_cluster_as_weighted_ref, multi_cluster_obj, MultiCenter, persist, single_centroid_cluster_slice_with_weight_arrays, single_centroid_cluster_weighted_vec, single_centroid_cluster_weighted_vec_with_distance_over_slices, single_centroid_unweighted_cluster_slice};
use rcflib::common::multidimdatawithkey::new_vec;
use rcflib::errors::RCFError;
use rcflib::glad::GlobalLocalAnomalyDetector;


fn rotate_clockwise(point: &[f32], theta: f32) -> Vec<f32> {
    let mut result = vec![0.0f32; 2];
    result[0] = theta.cos() * point[0] + theta.sin() * point[1];
    result[1] = -theta.sin() * point[0] + theta.cos() * point[1];
    return result;
}

fn gen_numeric_data(data_size:usize, seed:u64, shift : (f32,f32), number_of_fans: usize) -> MultiDimDataWithKey {
    let mut vec_mean = vec![shift.0,shift.1];
    let scale = vec![1.0,0.5/number_of_fans as f32];
    let mut data :Vec<Vec<f32>> = Vec::new();
    let mut labels:Vec<usize> = Vec::new();
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    for i in 0..data_size {
        let vec = new_vec(&vec_mean, &scale, &mut rng);
        if rng.gen::<f64>() < 0.005 {
            let j :usize = (rng.next_u32() as usize)%number_of_fans;
            data.push(rotate_clockwise(&vec,(2.0*PI * i as f32)/data_size as f32 + PI*(1.0 + 2.0*j as f32)/number_of_fans as f32));
            labels.push( number_of_fans + 2*j) ;
        } else {
            let j :usize = (rng.next_u32() as usize)%number_of_fans;
            data.push(rotate_clockwise(&vec,(2.0*PI * i as f32)/data_size as f32 + PI*(2.0*j as f32)/number_of_fans as f32));
            labels.push(j) ;
        }
    }
    MultiDimDataWithKey{
        data,
        change_indices: Vec::new(),
        labels,
        changes: Vec::new()
    }
}

fn vec_dist(a: &Vec<f32>, b : &Vec<f32>) -> f64 {
    l2distance(a,b)
}

fn bad_distance<T :?Sized>(a : &T, b:&T) -> f64{
    -0.0001
}

#[test]
fn numeric_glad() {
    let data_size = 1000000; // should be sufficiently large for covering a 360 degree rotation, for |capacity| points
    let number_of_fans = 3;
    let mut generator = ThreadRng::default();
    let one_seed: u64 = generator.gen();
    println!(" single seed is {}", one_seed);
    let mut rng = ChaCha20Rng::seed_from_u64(one_seed);
    let capacity = 2000;
    let time_decay = 1.0 /capacity as f64;
    let data_with_key = gen_numeric_data(data_size,one_seed,(5.0 + 1.0/number_of_fans as f32,0.0),number_of_fans);
    let mut glad = GlobalLocalAnomalyDetector::<Vec<f32>>::new(2000,0,time_decay,5, 0.1,true);

    glad.set_z_factor(6.0 + number_of_fans as f32/4.0);

    let mut false_neg = 0;
    let mut false_pos = 0;
    let mut true_pos = 0;
    let mut print_clusters = false;
    // set the above to see the cluster centers printed with associated relative mass
    // each block is separated by two println!()
    // a simple visualization tool can plot an animation of the clusters
    // for example in gnuplot, try something like
    // set terminal gif transparent animate delay 5
    // do for [i = 0:360] { plot [-10:10][-10:10] "typescript" i i u 1:2:3 w p palette pt 7 t "" }
    let mut first = true;
    for j in 0..data_with_key.data.len() {
        let answer = glad.process(&data_with_key.data[j],1.0,vec_dist,vec_dist, false).unwrap();
        if answer.grade != 0.0 {
            if data_with_key.labels[j] < number_of_fans {
                false_pos += 1;
            } else {
                true_pos += 1;
            }
        } else {
            if data_with_key.labels[j] >= number_of_fans {
                false_neg += 1;
            }
        }

        if (j*360/data_size)%2 != 0 {
            if print_clusters && !first {
                println!();
                println!();
            }
            first = true;
        } else {
            if print_clusters && first {
                let a = glad.clusters();
                for i in 0..a.len() {
                    let item =&a[i];
                    for rep in item.representatives() {
                        println!("{} {} {} {}", rep.0[0], rep.0[1], i , rep.1);
                    }
                }
                first = false;
            }
        }

    }
    println!(" precision {} recall {} out of {} injected anomalies", (true_pos as f32)/(true_pos + false_pos) as f32, true_pos as f32/(true_pos + false_neg) as f32, (true_pos + false_neg));

    // negative weight is error
    assert!(glad.process(&data_with_key.data[0],-1.0,vec_dist,vec_dist,false).is_err());
    // negative distance is error
    assert!(glad.process(&data_with_key.data[0],1.0,vec_dist,bad_distance,false).is_err());
}


pub fn toy_d(a:&Vec<char>, b: &Vec<char>) -> f64 {
    if a.len() > b.len() {
        return toy_d(b, a);
    }
    let mut one = vec![0.0;(b.len()+1)];
    let mut two = vec![0.0;(b.len()+1)];

    for j in 0..b.len()+1 {
        one[j] = j as f64;
    }
    for i in 1..a.len()+1 {

        two[0] = i as f64;
        for ((x, y), z) in two[1..].iter_mut().zip(&one[..b.len()]).zip(b) {
            *x = if a[i-1] == *z {*y} else {*y + 1.0};
        }

        for (x, y) in two.iter_mut().zip(&one) {
            *x = if *x < *y + 1.0 {*x} else {*y+1.0};
        }

        for j in 1..b.len()+1 {
            if two[j] > two[j - 1] + 1.0 {
                two[j] = two[j - 1] + 1.0;
            }
        }

        // change one
        for(x,y) in one.iter_mut().zip(&two){
            *x = *y;
        }
    }
    one[b.len()]
}

pub fn get_ab_array(size: usize, probability_of_a: f64, rng: &mut ChaCha20Rng, change_in_middle : bool, fraction : f64) -> Vec<char> {
    let mut answer = Vec::new();
    let new_size = size + (rng.next_u32() as usize)%(size / 5);
    for i in 0..new_size {
        let toss = if change_in_middle && (i as f64 > (1.0 - fraction) * new_size as f64 || (i as f64) < (new_size as f64 ) * fraction) {
            1.0 - probability_of_a
        } else {
            probability_of_a
        };
        if rng.gen::<f64>() < toss {
            answer.push('\u{2014}');
        } else {
            answer.push('\u{005F}');
        }
    }
    answer
}

const ANSI_RESET :&str = "\u{001B}[0m";
const ANSI_RED : &str = "\u{001B}[31m";
const ANSI_BLUE : &str = "\u{001B}[34m";

pub fn print_array(a:&[char]) {
    for i in 0..a.len() {
        if a[i] == '\u{2014}' {
            print!("{}{}{}", ANSI_RED,a[i],ANSI_RESET);
        } else {
            print!("{}{}{}",ANSI_BLUE,a[i], ANSI_RESET);
        }
    }
}

fn print_clusters(clusters: &Vec<MultiCenter<Vec<char>>>) {
    for i in 0..clusters.len()  {
        println!(" Cluster {},  weight {:.3}, average radius {:.3} ",i,clusters[i].weight(),clusters[i].average_radius());
        for item in &clusters[i].representatives() {
            print!("(wt {:.2}, len {})", item.1,item.0.len());
            print_array(&item.0);
            println!();
        }
        println!();
        println!();
    }
}

#[test]
fn string_glad() {
    let data_size = 200000; // should be sufficiently large for covering a 360 degree rotation, for |capacity| points
    let mut generator = ThreadRng::default();
    let one_seed: u64 = generator.gen();
    println!(" single seed is {}", one_seed);
    let mut rng = ChaCha20Rng::seed_from_u64(one_seed);
    let string_size = 70;
    let capacity= 2000;
    let change_in_middle = true;
    // the following should be away from 0.5 in [0.5,1]
    let gap_prob_of_a = 0.85;
    let time_decay = 1.0 /capacity as f64;
    let anomaly_rate = 0.05;
    let mut injected: bool;
    let mut number_of_injected = 0;

    let mut false_neg = 0;
    let mut false_pos = 0;
    let mut true_pos = 0;

    let print_clusters_strings = true;

    let mut glad = GlobalLocalAnomalyDetector::<Vec<char>>::new(2000,0,time_decay,5, 0.1,false);

    // we will not store the points but perform streaming

    for i in 0..data_size {
        if i>0 && i%10000 == 0 {
            println!(" at {} ",i);
            if print_clusters_strings {
                print_clusters(&glad.clusters());
            }
        }
        let mut point = Vec::new();
        if rng.gen::<f64>() < anomaly_rate {
            injected = true;
            number_of_injected += 1;
            point = get_ab_array(string_size + 10, 0.5, &mut rng, false, 0.0);
        } else {
            let flag = change_in_middle && rng.gen::<f64>() < 0.25;
            let prob = if rng.gen::<f64>() < 0.5 {
                gap_prob_of_a
            } else {
                (1.0 - gap_prob_of_a )
            };
            injected = false;
            point = get_ab_array(string_size, prob, &mut rng, flag, 0.25 * i as f64/ data_size as f64);
        }
        let answer = glad.process(&point,1.0,toy_d,toy_d, false).unwrap();
        if answer.grade != 0.0 {
            if !injected {
                false_pos += 1;
            } else {
                true_pos += 1;
            }
        } else {
            if injected && i > capacity/2 {
                false_neg += 1;
            }
        }
    }
    println!("injected {}", number_of_injected);
    println!(" precision {} recall {} out of {} injected anomalies", (true_pos as f32)/(true_pos + false_pos) as f32, true_pos as f32/(true_pos + false_neg) as f32, (true_pos + false_neg));
}