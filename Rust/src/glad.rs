
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rand_core::RngCore;
use crate::common::cluster::{multi_cluster_as_object_with_weight_array, multi_cluster_as_weighted_obj, MultiCenter, persist};
use crate::util::check_argument;
use crate::trcf::basicthresholder::BasicThresholder;
use crate::common::intervalstoremanager;
use crate::common::intervalstoremanager::IntervalStoreManager;
use crate::types::Result;

pub const DEFAULT_MAX_CLUSTERS :usize = 10;
pub const SCORE_MAX : f32 = 10.0;
pub const DEFAULT_IGNORE_SMALL_CLUSTER_REPRESENTATIVE : f32 = 0.005;
//ignore clusters that are 10 or more times away from the closest
pub const CLUSTER_COMPARISON_THRESHOLD : f64 = 10.0;

#[repr(C)]
pub struct GenericAnomalyDescriptor<T> {
    pub representative_list: Vec<(T, f32)>,
    pub score: f64,
    pub threshold: f32,
    pub grade: f32,
}

#[repr(C)]
pub struct GlobalLocalAnomalyDetector<T:Clone + Sync> {
    capacity: usize,
    current_size: usize,
    random_seed: u64,
    heap: Vec<(f64,usize)>,
    object_list : Vec<(T,f32)>,
    time_decay: f64,
    most_recent_time_decay_update: u64,
    accumulated_decay: f64,
    interval_manager: IntervalStoreManager<usize>,
    basic_thresholder: BasicThresholder,
    last_cluster: u64,
    do_not_recluster_within : u64,
    entries_seen: u64,
    sequence_number: u64,
    last_mean: f32,
    evicted : Option<(T,f32)>,
    clusters: Vec<MultiCenter<T>>,
    max_allowed: usize,
    shrinkage: f32,
    is_compact: bool,
    number_of_representatives: usize,
    ignore_below: f32,
    initial_accept_fraction: f64,
    //global_distance : fn(&T,&T) -> f64
}

impl<T:Clone + Sync> GlobalLocalAnomalyDetector<T> {
    pub fn new(capacity: usize, random_seed: u64, time_decay: f64, number_of_representatives: usize, shrinkage: f32, is_compact: bool) -> Self{
        let mut basic_thresholder = BasicThresholder::new_adjustible(time_decay as f32,false);
        basic_thresholder.set_absolute_threshold(1.2);
        if !is_compact {
            basic_thresholder.set_z_factor(2.5);
        }
        GlobalLocalAnomalyDetector{
            capacity,
            current_size: 0,
            random_seed,
            heap: vec![],
            object_list: vec![],
            time_decay,
            most_recent_time_decay_update: 0,
            accumulated_decay: 0.0,
            interval_manager: IntervalStoreManager::new(capacity),
            basic_thresholder,
            last_cluster: 0,
            do_not_recluster_within: (capacity / 2) as u64,
            entries_seen: 0,
            sequence_number: 0,
            last_mean: -1.0, // forcing the first clustering
            evicted: Option::None,
            clusters: Vec::new(),
            max_allowed: 10,
            shrinkage,
            is_compact,
            number_of_representatives,
            ignore_below: DEFAULT_IGNORE_SMALL_CLUSTER_REPRESENTATIVE,
            initial_accept_fraction: 0.125,
            //global_distance: ()
        }
    }

    fn initial_accept_probability(&self, fill_fraction: f64) -> f64 {
        return if fill_fraction < self.initial_accept_fraction {
            1.0
        } else if self.initial_accept_fraction >= 1.0 {
            0.0
        } else {
            1.0 - (fill_fraction - self.initial_accept_fraction)
                / (1.0 - self.initial_accept_fraction)
        };
    }

    fn fill_fraction(&self) -> f64 {
        if self.current_size == self.capacity {
            return 1.0;
        };
        (self.current_size as f64 / self.capacity as f64)
    }

    fn compute_weight(&self, random_number: f64, weight: f32) -> f64 {
        f64::ln(-f64::ln(random_number) / weight as f64) -
            ((self.entries_seen - self.most_recent_time_decay_update) as f64
                * self.time_decay - self.accumulated_decay)
    }

    fn swap_down(&mut self, start_index: usize) {
        let mut current: usize = start_index;
        while 2 * current + 1 < self.current_size {
            let mut max_index: usize = 2 * current + 1;
            if 2 * current + 2 < self.current_size
                && self.heap[2 * current + 2].0 > self.heap[max_index].0
            {
                max_index = 2 * current + 2;
            }
            if self.heap[max_index].0 > self.heap[current].0 {
                self.swap_weights(current, max_index);
                current = max_index;
            } else {
                break;
            }
        }
    }

    fn swap_weights(&mut self, a: usize, b: usize) {
        let tmp = self.heap[a];
        self.heap[a] = self.heap[b];
        self.heap[b] = tmp;
    }

    fn evict_max(&mut self) -> (f64, usize) {
        let evicted_point = self.heap[0];
        self.current_size -= 1;
        let current: usize = self.current_size.into();
        self.heap[0] = self.heap[current];
        self.heap[0] = self.heap[current];
        self.swap_down(0);
        evicted_point
    }

    fn sample(&mut self, object: &T, weight: f32) -> bool {
        self.sequence_number += 1;
        self.entries_seen += 1;
        let mut initial = false;
        let mut rng = ChaCha20Rng::seed_from_u64(self.random_seed);
        self.random_seed = rng.next_u64();
        let random_number: f64 = rng.gen();
        let heap_weight = self.compute_weight(random_number, weight);
        if self.current_size < self.capacity {
            let other_random: f64 = rng.gen();
            initial = other_random < self.initial_accept_probability(self.fill_fraction());
        }
        if initial || (heap_weight < self.heap[0].0) {
            if !initial {
                let old_index = self.evict_max().1;
                self.evicted = Some(self.object_list[old_index].clone());
                self.interval_manager.release(old_index);
            }
            let index = self.interval_manager.get();
            if index < self.object_list.len() {
                self.object_list[index] = (object.clone(), weight);
            } else {
                self.object_list.push((object.clone(), weight));
            }
            if (self.heap.len() == self.current_size){
                self.heap.push((heap_weight, index));
            } else {
                self.heap[self.current_size] = (heap_weight, index);
            }
            let mut current = self.current_size;
            self.current_size += 1;

            while current > 0 {
                let tmp = (current - 1) / 2;
                if self.heap[tmp].0 < self.heap[current].0 {
                    self.swap_weights(current, tmp);
                    current = tmp;
                } else {
                    break;
                }
            }
            return true;
        };
        false
    }

    pub fn set_z_factor(&mut self, z_factor : f32){
        self.basic_thresholder.set_z_factor(z_factor);
    }

    pub fn score(&self, current: &T, local_distance: fn(&T, &T) -> f64, consider_occlusion: bool) -> Result<Vec<(T, f32)>> {
        if (self.clusters.len() == 0) {
            return Ok(Vec::new());
        } else {
            let mut candidate_list: Vec<(usize, (f64, &T), f64)> = Vec::new();
            for j in 0..self.clusters.len() {
                let rad = self.clusters[j].average_radius();
                let close = self.clusters[j].distance_to_point_and_ref(current, self.ignore_below, local_distance)?;
                candidate_list.push((j, close, rad));
            }
            candidate_list.sort_by(|a, b| a.1.0.partial_cmp(&b.1.0).unwrap());

            if (candidate_list[0].1.0 == 0.0) {
                return Ok(vec![(candidate_list[0].1.1.clone(), 0.0)]);
            }
            let mut index = 0;
            while (index < candidate_list.len()) {
                let head = candidate_list[index];
                if (consider_occlusion) {
                    for j in index + 1..candidate_list.len() {
                        let occlude = (local_distance)(head.1.1, candidate_list[j].1.1);
                        check_argument(occlude>=0.0, "distances cannot be negative")?;
                        if candidate_list[j].2 > f64::sqrt(occlude * occlude + head.2 * head.2) {
                            candidate_list.remove(j);
                        }
                    }
                }
                index += 1;
            }
            let mut answer = Vec::new();
            let distance_threshold = candidate_list[0].1.0 * CLUSTER_COMPARISON_THRESHOLD;
            for head in &candidate_list {
                if head.1.0 < distance_threshold {
                    let temp_measure = if head.2 > 0.0 && head.1.0 < SCORE_MAX as f64 * head.2 {
                        (head.1.0 / head.2) as f32
                    } else {
                        SCORE_MAX
                    };
                    answer.push((head.1.1.clone(), temp_measure));
                }
            }
            Ok(answer)
        }
    }

    pub fn process(&mut self, object: &T, weight: f32, global_distance: fn(&T, &T) -> f64, local_distance: fn(&T, &T) -> f64, consider_occlusion: bool)
                   -> Result<GenericAnomalyDescriptor<T>> {
        check_argument(weight >= 0.0, "weight cannot be negative")?;
        // recompute clusters first; this enables easier merges and deserialization
        if (self.sequence_number > self.last_cluster + self.do_not_recluster_within) {
            let current_mean = self.basic_thresholder.primary_mean() as f32;
            if (f32::abs(current_mean - self.last_mean) > 0.1 || current_mean > 1.7f32
                || self.sequence_number > self.last_cluster + 20 * self.do_not_recluster_within) {
                self.last_cluster = self.sequence_number;
                self.last_mean = current_mean;
                let temp = multi_cluster_as_weighted_obj(&self.object_list,
                                                         global_distance, 5, 0.1, self.is_compact, self.max_allowed, false)?;
                self.clusters = persist(&temp);
            }
        }
        let mut score_list = self.score(object, local_distance, consider_occlusion)?;
        let threshold = self.basic_thresholder.threshold();
        let mut grade: f32 = 0.0;
        let score: f32 = if score_list.len() == 0 { 0.0 } else {
            score_list.iter().map(|a| a.1).min_by(|a,b| a.partial_cmp(b).unwrap()).unwrap()
        };

        if score_list.len() > 0 {
            if score < SCORE_MAX {
                // an exponential attribution
                let sum: f64 = score_list.iter().map(|a|
                    if a.1 == SCORE_MAX { 0.0f64 } else {
                        f64::exp(-( a.1 * a.1) as f64)
                    }
                ).sum();
                for mut item in &mut score_list {
                    let t = if item.1 == f32::MAX { 0.0 } else {
                        f64::min(1.0, f64::exp(-(item.1 * item.1) as f64) / sum)
                    };
                    item.1 = t as f32;
                }
            } else {
                let y = score_list.len();
                for mut item in &mut score_list {
                    item.1 = 1.0 / (y as f32);
                }
            }
            grade = self.basic_thresholder.anomaly_grade(score, false);
            let other = self.basic_thresholder.z_factor();
            self.basic_thresholder.update_both(score, f32::min(score, other));
        }
        self.sample(object, weight);

        return Ok(GenericAnomalyDescriptor {
            representative_list: score_list,
            score: score as f64,
            threshold,
            grade
        })
    }

    pub fn clusters(&self) -> Vec<MultiCenter<T>> {
        self.clusters.clone()
    }
}

