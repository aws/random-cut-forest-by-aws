extern crate rand;
extern crate rand_chacha;
extern crate rcflib;

use std::collections::{HashMap, HashSet};
use rand::Rng;
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;
use rcflib::common::deviation::Deviation;
use rcflib::common::multidimdatawithkey;
use rcflib::rcf::{ RCFOptionsBuilder};
use rcflib::trcf::basictrcf::TRCFOptionsBuilder;
use rcflib::trcf::multitrcf::{MultiTRCF, MultiTRCFBuilder};
use rcflib::trcf::types::ScoringStrategy::EXPECTED_INVERSE_HEIGHT;
use rcflib::trcf::types::TransformMethod;
use rcflib::trcf::types::TransformMethod::{NONE,NORMALIZE};

#[cfg(test)]
parameterized_test::create! { multi_trcf_basic, (method,parallel_enabled), {
    multi_trcf(method,parallel_enabled);
}}

multi_trcf_basic! {
    d1: (NONE,false),
    d2: (NONE,true),
}

#[test]
pub fn multi_trcf_single_threaded(){
    multi_trcf(NORMALIZE,false);
}

#[test]
pub fn multi_trcf_multi_threaded(){
    multi_trcf(NORMALIZE,true);
}

pub fn multi_trcf(transform_method:TransformMethod,parallel_enabled:bool) {
    let shingle_size = 10;
    let input_dimensions = 1;
    let data_size = 1000;
    let noise = 5.0;
    let number_of_series = 1000;
    // as number_of_series is increased the prec-recall for a fixed number of arms
    let scoring_strategy = EXPECTED_INVERSE_HEIGHT;

    let mut total_injected =0;
    let mut total_found = 0;
    let mut total_overlap = 0;
    let mut late = 0;
    let number_of_models = 3; // more than 10 may not be a great idea
    let mut multi_trcf : MultiTRCF = MultiTRCFBuilder::new(input_dimensions, shingle_size, number_of_models, 2*number_of_series)
        .scoring_strategy(scoring_strategy).parallel_enabled(parallel_enabled).build().unwrap();

    let mut period_map = HashMap::new();

    let mut rng = ChaCha20Rng::seed_from_u64(42);
    let mut data_with_key = Vec::new();
    for y in 0..number_of_series {
        let mut amplitude = Vec::new();
        let mut period = Vec::new();
        for _i in 0..input_dimensions {
            amplitude.push((1.0 + 0.2 * rng.gen::<f32>()) * 60.0);
            // ranges from [30,90)
            period.push(30 + ((rng.gen::<f32>())* 60.0) as usize);
        }

        period_map.insert(y as u64,period.clone());

        data_with_key.push(multidimdatawithkey::MultiDimDataWithKey::multi_cosine(
            data_size,
            &period,
            &amplitude,
            noise,
            rng.gen::<u64>(),
            input_dimensions.into(),
        ).unwrap());
    }
    let mut next_index = vec![0;number_of_series];
    let mut late_discovered = vec![HashSet::new();shingle_size];

    for i in 0..data_size {
        let mut injected  = HashSet::new();
        let mut map : HashMap<u64,(&[f32],u64)> = HashMap::new();
        for j in 0..number_of_series {
            if next_index[j] < data_with_key[j].change_indices.len() && data_with_key[j].change_indices[next_index[j]] == i {
                next_index[j] += 1;
                injected.insert(j);
            }
            map.insert(j as u64, (&data_with_key[j].data[i],i as u64));
        }
        total_injected += injected.len();

        let result = multi_trcf.process(map).unwrap();
        let y : usize = result.iter().map(|x| if x.relative_index == 0 && injected.contains(&(x.id as usize)) {1} else {0}).sum();
        let z : usize = result.iter().map(|x| if x.relative_index != 0 {
            let q = (i as i32 + x.relative_index) as usize;
            if late_discovered[ q % shingle_size].contains(&(x.id as usize)) {1} else {0}
        } else {0}).sum();
        total_overlap += y;
        total_found += result.len();
        late += z;
        late_discovered[ i % shingle_size] = injected;
    }
    println!("number of time series: {} size {} each, {} arms,  parallel enabled: {}",number_of_series,data_size,number_of_models,parallel_enabled);
    println!("shingle size: {}, normalization: {}, anomalies in total {}",shingle_size,transform_method,total_injected);
    println!("spot precision {} recall {}",((total_overlap as f64)/(total_found as f64)),
        ((total_overlap as f64)/(total_injected as f64)));
    println!("with late detection, precision {} recall {}",(((total_overlap + late) as f64)/(total_found as f64)),
             (((total_overlap+late) as f64)/(total_injected as f64)));
    println!("nontrivial bandit switches {}, affirmations {}",multi_trcf.switches(),multi_trcf.affirmations());
    print!("model updates across different arms:");
    for y in multi_trcf.updates() {
        print!("({}, {}) ",y.0,y.1)
    }
    println!();
    // lets check the period_map; just for the first dimension
    let mut result = multi_trcf.states().iter().map(|x| (x.bandit.current_model(),x.id,period_map.get(&x.id).unwrap()[0]))
        .collect::<Vec<(usize,u64,usize)>>();
    result.sort();
    let mut a = result[0].0;
    let mut stat = vec![Deviation::new(0.0).unwrap();number_of_models];
    for y in result {
        if a == y.0 {
            stat[y.0].update(y.2 as f64);
        } else {
            a = y.0;
        }
    }
    for x in 0..number_of_models {
        println!("Model {},  chosen by {}, average period {}, deviation {} ",x,stat[x].count,stat[x].mean(),stat[x].deviation());
    }
}
