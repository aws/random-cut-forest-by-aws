use std::cmp::{max, min};
use std::f32::NAN;
use std::ops::{Deref, Index};
use std::slice;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rand_core::RngCore;
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};
use crate::util::check_argument;
use crate::types::Result;

const PHASE2_THRESHOLD: usize = 2;
const SEPARATION_RATIO_FOR_MERGE: f64 = 0.8;
const WEIGHT_THRESHOLD: f64 = 1.25;
const LENGTH_BOUND: usize = 5000;
/**
* In the following, the goal is to cluster objects of type T, given a distance function over a pair
* of references. The clustering need not create any new object of type T, but would find representative
* points that express the cluster. The dictionary of points is expressed as pairs of reference and
* corresponding weights of objects. However for vectors/slives over f32 and single representative scenario
* the clustering does allow computation of an approximation median to create new "central" points.
* Q is the struct that corresponds to a representative of a cluster. Note that
* a cluster can have multiple representatives (see for example https://en.wikipedia.org/wiki/CURE_algorithm)
* thus the entire information is a vector of tuples corresponding to (representative, weight of representative)
*
**/

pub trait IntermediateCluster<Z, Q, T: ?Sized> {
    // aclsuter should provide a measure of the point set convered by the cluster
    fn weight(&self) -> f64;
    // A cluster is an extended object of a type different from the base type of a
    // point indicated by T (and only the reference is used in the distance function
    // every clustering algorithm implicitly/explicitly defines this function
    // the return value is (distance, the integer identifier of the cluster representative)
    // if the cluster is represented by a single value then that identifier would be 0
    fn distance_to_point<'a>(&self, dictionary: &'a [Z], get_point: fn (usize, &'a [Z]) -> &'a T, point: &T, distance: fn(&T, &T) -> f64) -> (f64,usize);
    // Likewise, the distance function needs to be extended (implicitly/explicitly)
    // to a distance function between clusters
    // the return value is the distance and the pair of identifiers in the corresponding clusters which
    // define that closest distance
    fn distance_to_cluster<'a>(&self, dictionary: &'a [Z], get_point: fn (usize, &'a [Z]) -> &'a T, other: &dyn IntermediateCluster<Z, Q, T>, distance: fn(&T, &T) -> f64) -> (f64, usize, usize);
    // a function that assigns a point indexed by usize from a list of samples to
    // the cluster; note the weight used in the function need not be the entire weight of the
    // sampled point (for example in case of soft assignments)
    fn add_point(&mut self, index: usize, weight: f32, dist: f64, representative: usize);
    // given a set of previous assignments, recomputes the optimal set of representatives
    // this is the classic optimization step for k-Means; but the analogue exists for every
    // clustering; note that it is possible that recompute does nothing
    fn recompute<'a>(&mut self, points: &'a [Z], get_point: fn (usize,&'a [Z]) -> &'a T, distance: fn(&T, &T) -> f64) -> f64;
    // resets  the statistics of a clusters; preparing for a sequence of add_point followed
    // by recompute
    fn reset(&mut self);
    // a function that allows a cluster to absorb another cluster in an agglomerative \
    // clustering algorithm
    fn extent_measure(&self) -> f64;
    // a function that indicates cluster quality
    fn average_radius(&self) -> f64;
    // a function that absorbs another cluster
    fn absorb<'a>(&mut self, dictionary: &'a [Z], get_point: fn (usize,&'a [Z]) -> &'a T, another: &dyn IntermediateCluster<Z, Q, T>, distance: fn(&T, &T) -> f64);
    // a function to return a list of representatives corresponding to pairs (Q,weight)
    fn representatives(&self) -> Vec<(Q, f32)>;
    // a function that helps scale (by multiplication) the cluster weight
    fn scale_weight(&mut self, factor: f64);
}


fn pick<T>(points: &[(T, f32)], wt: f32) -> usize {
    let mut position = 0;
    let mut running = wt;
    for i in 0..points.len() {
        position = i;
        if running - points[i].1 <= 0.0 {
            break;
        } else {
            running -= points[i].1;
        }
    }
    position
}



fn median<'a, Z,Q:?Sized>(dimensions:usize,points: &'a [Z], get_point: fn(usize,&'a [Z]) -> &'a Q,list: &mut [(usize, f32)]) -> Vec<f32>
where Q: Index<usize, Output = f32>
{
    let mut answer = vec![0.0f32;dimensions];
    let total: f64 = list.iter().map(|a| a.1 as f64).sum();
    for i in 0..dimensions {
        list.sort_by(|a, b| get_point(a.0,&points)[i].partial_cmp(&get_point(b.0,points)[i]).unwrap());
        let position = pick(list, (total / 2.0) as f32);
        answer[i] = get_point(list[position].0,&points)[i];
    }
    answer
}

#[repr(C)]
pub struct Center {
    representative: Vec<f32>,
    weight: f64,
    points: Vec<(usize, f32)>,
    sum_of_radii: f64,
}

impl Center {
    pub fn new(representative: usize, point:&[f32], weight: f32, _params:usize) -> Self {
        Center {
            representative: Vec::from(point),
            weight: weight as f64,
            points: Vec::new(),
            sum_of_radii: 0.0,
        }
    }

    pub fn new_as_vec(representative: usize, point:&Vec<f32>, weight: f32,_params:usize) -> Self {
        Center {
            representative: point.clone(),
            weight: weight as f64,
            points: Vec::new(),
            sum_of_radii: 0.0,
        }
    }

    pub fn average_radius(&self) -> f64 {
        if self.weight == 0.0 {
            0.0
        } else {
            self.sum_of_radii / self.weight
        }
    }

    pub fn distance(&self, point: &[f32], dist: fn(&[f32],&[f32]) -> f64) -> f64 {
        (dist)(&self.representative,point)
    }

    pub fn representative(&self) -> Vec<f32> {
        self.representative.clone()
    }

    pub fn weight(&self) -> f64 {
        self.weight
    }

    fn re_optimize<'a, Z, Q:?Sized>(&mut self, points:&'a [Z],
                      get_point: fn(usize,&'a [Z]) ->&'a Q,
                      picker: fn(usize,&'a [Z],fn(usize,&'a [Z]) -> &'a Q, a: &mut [(usize,f32)]) -> Vec<f32>)
    {
        if self.weight == 0.0 {
            let dimensions = self.representative.len();
            // the following computes an approximate median
            if self.points.len() < 500 {
                self.representative = picker(dimensions, points, get_point, &mut self.points);
            } else {
                let mut samples = Vec::new();
                let mut rng = ChaCha20Rng::seed_from_u64(0);
                for i in 0..self.points.len() {
                    if rng.gen::<f64>() < (200.0 * self.points[i].1 as f64) / self.weight {
                        samples.push((self.points[i].0, 1.0));
                    }
                }
                self.representative = picker(dimensions, points, get_point, &mut samples);
            };
        }
    }

    fn recompute_rad<'a,Z>(&mut self, points: &'a [Z], get_point: fn(usize,&'a [Z]) -> &'a [f32], dist: fn(&[f32],&[f32]) -> f64)
        -> f64
    {
        let old_value = self.sum_of_radii;
        self.sum_of_radii = 0.0;
        for j in 0..self.points.len() {
            self.sum_of_radii += self.points[j].1 as f64
                * dist(&self.representative, get_point(self.points[j].0,&points)) as f64;
        }
        old_value - self.sum_of_radii
    }

    fn recompute_rad_vec<'a,Z>(&mut self, points: &'a [Z], get_point: fn(usize,&'a [Z]) -> &'a Vec<f32>, dist: fn(&Vec<f32>,&Vec<f32>) -> f64)
                           -> f64
    {
        let old_value = self.sum_of_radii;
        self.sum_of_radii = 0.0;
        for j in 0..self.points.len() {
            self.sum_of_radii += self.points[j].1 as f64
                * dist(&self.representative, get_point(self.points[j].0,&points)) as f64;
        }
        old_value - self.sum_of_radii
    }

    fn add_point(&mut self, index: usize, weight: f32, dist: f64) {
        self.points.push((index, weight));
        self.weight += weight as f64;
        self.sum_of_radii += weight as f64 * dist;
    }

    fn reset(&mut self) {
        self.points.clear();
        self.weight = 0.0;
        self.sum_of_radii = 0.0;
    }

    fn absorb_list(&mut self, other_weight: f64, other_list: &Vec<(Vec<f32>,f32)>, closest: (f64,usize)){
        let t = f64::exp(2.0 * (self.weight - other_weight) / (self.weight + other_weight));
        let factor = t / (1.0 + t);
        let dimensions = self.representative.len();
        for i in 0..dimensions {
            self.representative[i] = (factor * (self.representative[i] as f64)
                + (1.0 - factor) * (other_list[closest.1].0[i] as f64)) as f32;
        }

        self.sum_of_radii += (self.weight * (1.0 - factor) + factor * other_weight) * closest.0;
    }
}


impl<Z> IntermediateCluster<Z,Vec<f32>, [f32]> for Center {
    fn weight(&self) -> f64 {
        self.weight()
    }

    fn scale_weight(&mut self, factor: f64){
        assert!(!factor.is_nan() && factor>0.0," has to be positive");
        self.weight = (self.weight as f64 * factor);
    }

    fn distance_to_point<'a>(&self, _points:&'a [Z],_get_point: fn(usize,&'a [Z]) ->&'a [f32],point: &[f32], distance: fn(&[f32], &[f32]) -> f64) -> (f64,usize) {
        ((distance)(&self.representative, point),0)
    }

    fn distance_to_cluster<'a>(
        &self,
        _points:&'a [Z],
        _get_point: fn(usize,&'a [Z]) ->&'a [f32],
        other: &dyn IntermediateCluster<Z,Vec<f32>, [f32]>,
        distance: fn(&[f32], &[f32]) -> f64,
    ) -> (f64,usize,usize) {
        let tuple = other.distance_to_point(_points,_get_point,&self.representative, distance);
        (tuple.0,0,tuple.1)
    }

    fn add_point(&mut self, index: usize, weight: f32, dist: f64, representative:usize) {
        assert!(representative==0,"can have only one representative");
        assert!(!weight.is_nan() && weight >= 0.0f32, "non-negative weight");
        self.add_point(index,weight,dist);
    }

    fn recompute<'a>(&mut self, points:&'a [Z],get_point: fn(usize,&'a [Z]) ->&'a [f32], distance: fn(&[f32], &[f32]) -> f64) -> f64 {
        self.re_optimize(&points,get_point, median);
        self.recompute_rad(&points,get_point, distance)
    }

    fn reset(&mut self) {
        self.reset();
    }

    fn extent_measure(&self) -> f64 {
        self.average_radius()
    }

    fn average_radius(&self) -> f64 {
        self.average_radius()
    }

    fn absorb<'a>(
        &mut self,
        points:&'a [Z],
        get_point: fn(usize,&'a [Z]) ->&'a [f32],
        another: &dyn IntermediateCluster<Z, Vec<f32>, [f32]>,
        distance: fn(&[f32], &[f32]) -> f64,
    ) {
        let closest = another.distance_to_point(points,get_point, &self.representative,distance);
        self.absorb_list(another.weight(),&another.representatives(),closest);
    }

    fn representatives(&self) -> Vec<(Vec<f32>, f32)> {
        vec![(self.representative.clone(), self.weight as f32); 1]
    }
}

impl<Z> IntermediateCluster<Z,Vec<f32>, Vec<f32>> for Center {
    fn weight(&self) -> f64 {
        self.weight()
    }

    fn scale_weight(&mut self, factor: f64){
        assert!(!factor.is_nan() && factor>0.0," has to be positive");
        self.weight = (self.weight as f64 * factor);
    }

    fn distance_to_point<'a>(&self, _points:&'a [Z],_get_point: fn(usize,&'a [Z]) ->&'a Vec<f32>,point: &Vec<f32>, distance: fn(&Vec<f32>, &Vec<f32>) -> f64) -> (f64,usize) {
        ((distance)(&self.representative, point),0)
    }

    fn distance_to_cluster<'a>(
        &self,
        _points:&'a [Z],
        _get_point: fn(usize,&'a [Z]) ->&'a Vec<f32>,
        other: &dyn IntermediateCluster<Z,Vec<f32>, Vec<f32>>,
        distance: fn(&Vec<f32>, &Vec<f32>) -> f64,
    ) -> (f64,usize,usize) {
        let tuple = other.distance_to_point(_points,_get_point,&self.representative, distance);
        (tuple.0,0,tuple.1)
    }

    fn add_point(&mut self, index: usize, weight: f32, dist: f64, representative:usize) {
        assert!(representative==0,"can have only one representative");
        assert!(!weight.is_nan() && weight >= 0.0f32, "non-negative weight");
        self.add_point(index,weight,dist);
    }

    fn recompute<'a>(&mut self, points:&'a [Z],get_point: fn(usize,&'a [Z]) ->&'a Vec<f32>, distance: fn(&Vec<f32>, &Vec<f32>) -> f64) -> f64 {
        self.re_optimize(&points,get_point, median);
        self.recompute_rad_vec(&points,get_point, distance)
    }

    fn reset(&mut self) {
        self.reset();
    }

    fn extent_measure(&self) -> f64 {
        self.average_radius()
    }

    fn average_radius(&self) -> f64 {
        self.average_radius()
    }

    fn absorb<'a>(
        &mut self,
        points:&'a [Z],
        get_point: fn(usize,&'a [Z]) ->&'a Vec<f32>,
        another: &dyn IntermediateCluster<Z, Vec<f32>, Vec<f32>>,
        distance: fn(&Vec<f32>, &Vec<f32>) -> f64,
    ) {
        let closest = another.distance_to_point(points,get_point, &self.representative,distance);
        self.absorb_list(another.weight(),&another.representatives(),closest);
    }

    fn representatives(&self) -> Vec<(Vec<f32>, f32)> {
        vec![(self.representative.clone(), self.weight as f32); 1]
    }
}



fn process_point<'a,Z,U,Q,T :?Sized>(dictionary: &'a [Z], get_point: fn(usize,&'a [Z])->&'a T, index: usize, centers: &mut [U], weight : f32, distance: fn(&T, &T) -> f64) -> Result<()>
    where
        U: IntermediateCluster<Z,Q, T> + Send,
        T: std::marker::Sync,
{
    let mut dist = vec![(0.0, 1); centers.len()];
    let mut min_distance = (f64::MAX, 1);
    for j in 0..centers.len() {
        dist[j] = centers[j].distance_to_point(dictionary, get_point,get_point(index,dictionary), distance);
        check_argument(dist[j].0>=0.0," distances cannot be negative")?;
        if min_distance.0 > dist[j].0 {
            min_distance = dist[j];
        }
    };
    //check_argument(min_distance.0>=0.0," distances cannot be negative")?;
    if min_distance.0 == 0.0 {
        for j in 0..centers.len() {
            if dist[j].0 == 0.0 {
                centers[j].add_point(index, weight, 0.0, dist[j].1);
            }
        }
    } else {
        let mut sum = 0.0;
        for j in 0..centers.len() {
            if dist[j].0 <= WEIGHT_THRESHOLD * min_distance.0 {
                sum += min_distance.0 / dist[j].0;
            }
        }
        for j in 0..centers.len() {
            if dist[j].0 <= WEIGHT_THRESHOLD * min_distance.0 {
                centers[j].add_point(
                    index,
                    (weight as f64 * min_distance.0 / (sum * dist[j].0)) as f32,
                    dist[j].0, dist[j].1
                );
            }
        }
    }
    Ok(())
}


fn assign_and_recompute<'a, Z, Q, U, T: ?Sized>(
    dictionary: &'a [Z],
    weights: &'a [f32],
    get_point: fn(usize,&'a [Z]) -> &'a T,
    get_weight: fn(usize,&'a [Z],&'a [f32]) -> f32,
    samples: &[(usize,f32)],
    centers: &mut [U],
    distance: fn(&T, &T) -> f64,
    parallel_enabled: bool,
) -> Result<f64>
    where
        U: IntermediateCluster<Z,Q, T> + Send,
        T: std::marker::Sync,
        Z: std::marker::Sync,
{
    for j in 0..centers.len() {
        centers[j].reset();
    }

    if (samples.len() == 0){
        for i in 0..dictionary.len() {
            process_point(dictionary,get_point,i,centers,get_weight(i,dictionary,weights),distance)?;
        }
    } else {
        for i in 0..samples.len() {
            process_point(dictionary,get_point,i,centers,samples[i].1,distance)?;
        }
    }

    let gain: f64 = if parallel_enabled {
        centers
            .par_iter_mut()
            .map(|x| x.recompute(dictionary, get_point,distance))
            .sum()
    } else {
        centers
            .iter_mut()
            .map(|x| x.recompute(dictionary, get_point,distance))
            .sum()
    };
    Ok(gain)
}


fn down_sample<'a,Z>(points: &'a [Z], weights:&'a [f32], get_weight: fn(usize,&'a [Z],&'a [f32]) -> f32, seed:u64, approximate_bound: usize) ->  Vec<(usize, f32)> {
    let mut total_weight: f64 = 0.0;
    for j in 0..points.len() {
        total_weight += get_weight(j, &points, &weights) as f64;
    };

    let mut rng = ChaCha20Rng::seed_from_u64(seed as u64);
    let mut sampled_points = Vec::new();
    let mut remainder = 0.0f64;
    for j in 0..points.len() {
        let point_weight = get_weight(j,&points,&weights);
        if point_weight > (0.005 * total_weight) as f32 {
            sampled_points.push((j, point_weight));
        } else {
            remainder += point_weight as f64;
        }
    }
    for j in 0..points.len() {
        let point_weight = get_weight(j,&points,&weights);
        if point_weight <= (0.005 * total_weight) as f32
            && rng.gen::<f64>() < approximate_bound as f64 / (points.len() as f64)
        {
            let t = point_weight as f64
                * (points.len() as f64 / approximate_bound as f64)
                * (remainder / total_weight);
            sampled_points.push((j, t as f32));
        }
    }
    sampled_points
}


fn pick_from<'a,Z>(points: &'a [Z], weights: &'a [f32], get_weight: fn(usize,&'a [Z],&'a [f32]) -> f32, wt: f32) -> (usize,f32) {
    let mut position = 0;
    let mut weight = get_weight(position,points,weights);
    let mut running = wt;
    for i in 0..points.len() {
        position = i;
        weight = get_weight(position,points,weights);
        if running - weight <= 0.0 {
            break;
        } else {
            running -= weight;
        }
    }
    (position,weight)
}



pub fn general_iterative_clustering<'a, U, V, Q, Z, T: ?Sized>(
    max_allowed: usize,
    dictionary: &'a [Z],
    weights: &'a [f32],
    get_point: fn(usize,&'a [Z]) -> &'a T,
    get_weight: fn(usize,&'a [Z],&'a [f32]) -> f32,
    approximate_bound: usize,
    seed: u64,
    parallel_enabled: bool,
    create: fn(usize, &'a T, f32, V) -> U,
    create_params: V,
    distance: fn(&T, &T) -> f64,
    phase_2_reassign: bool,
    enable_phase_3: bool,
    overlap_parameter: f64,
) -> Result<Vec<U>>
where
    U: IntermediateCluster<Z,Q, T> + Send,
    T: std::marker::Sync,
    Z: std::marker::Sync,
    V: Copy,
{
    check_argument(max_allowed < 51, " for large number of clusters, other methods may be better, consider recursively removing clusters")?;
    check_argument(max_allowed > 0, " number of clusters has to be greater or equal to 1")?;
    let mut rng = ChaCha20Rng::seed_from_u64(seed);

    let mut centers: Vec<U> = Vec::new();

    let mut samples : Vec<(usize,f32)> = if dictionary.len() > approximate_bound {
        down_sample(dictionary,weights,get_weight,rng.next_u64(),approximate_bound)
    } else {
        Vec::new()
    };
    //
    // we now peform an initialization; the sampling corresponds a denoising
    // note that if we are look at 2k random points, we are likely hitting every group of points
    // with weight 1/k whp
    let sampled_sum: f32 = if dictionary.len() > approximate_bound {
        samples.iter().map(|x| x.1).sum()
    } else {
        (0..dictionary.len()).into_iter().map(|x| get_weight(x,dictionary,weights)).sum()
    };

    for _k in 0..10 * max_allowed {
        let wt = (rng.gen::<f64>() * sampled_sum as f64) as f32;
        let mut min_dist = f64::MAX;
        let (index,weight) = if dictionary.len() > approximate_bound {
            let i = pick(&samples, wt);
            (i,samples[i].1)
        } else {
            pick_from(dictionary,weights,get_weight,wt)
        };
        for i in 0..centers.len() {
            let t = centers[i].distance_to_point(dictionary,get_point,get_point(index,&dictionary), distance);
            if t.0 < min_dist {
                min_dist = t.0;
            };
        }
        if min_dist > 0.0 {
            centers.push(create(index,get_point(index,dictionary), weight,create_params));
        }
    }

    assign_and_recompute(&dictionary,weights, get_point, get_weight,&samples,&mut centers, distance, parallel_enabled)?;

    // sort in increasing order of weight
    centers.sort_by(|o1, o2| o1.weight().partial_cmp(&o2.weight()).unwrap());
    while centers.len() > 0 && centers[0].weight() == 0.0 {
        centers.remove(0);
    }

    let mut phase_3_distance = 0.0f64;
    let mut keep_reducing_centers = centers.len() > max(max_allowed, 1);

    while keep_reducing_centers {
        let mut measure = 0.0f64;
        let mut measure_dist = f64::MAX;
        let mut lower = 0;
        let mut first = lower;
        let mut second = lower + 1;
        let mut found_merge = false;
        while lower < centers.len() - 1 && !found_merge {
            let mut min_dist = f64::MAX;
            let mut min_nbr = usize::MAX;
            for j in lower + 1..centers.len() {
                let dist = centers[lower].distance_to_cluster(&dictionary,get_point,&centers[j], distance);
                if min_dist > dist.0 {
                    min_nbr = j;
                    min_dist = dist.0;
                }
                let numerator = centers[lower].extent_measure()
                    + centers[j].extent_measure()
                    + phase_3_distance;
                if numerator >= overlap_parameter * dist.0 {
                    if measure * dist.0 < numerator {
                        first = lower;
                        second = j;
                        if dist.0 == 0.0f64 {
                            found_merge = true;
                        } else {
                            measure = numerator / dist.0;
                        }
                        measure_dist = dist.0;
                    }
                }
            }
            if lower == 0 && !found_merge {
                measure_dist = min_dist;
                second = min_nbr;
            }
            lower += 1;
        }

        let inital = centers.len();
        if inital > max_allowed || found_merge || (enable_phase_3 && measure > overlap_parameter) {
            let (small, large) = centers.split_at_mut(second);
            large.first_mut().unwrap().absorb(&dictionary,get_point, &small[first], distance);
            centers.swap_remove(first);
            if phase_2_reassign && centers.len() <= PHASE2_THRESHOLD * max_allowed + 1{
                assign_and_recompute(&dictionary, weights,get_point,get_weight, &samples, &mut centers, distance, parallel_enabled);
            }

            centers.sort_by(|o1, o2| o1.weight().partial_cmp(&o2.weight()).unwrap());
            while centers.len() > 0 && centers[0].weight() == 0.0 {
                centers.remove(0);
            }

            if inital > max_allowed && centers.len() <= max_allowed {
                // phase 3 kicks in; but this will execute at most once
                // note that measureDist can be 0 as well
                phase_3_distance = measure_dist;
            }
        } else {
            keep_reducing_centers = false;
        }
    }

    centers.sort_by(|o1, o2| o2.weight().partial_cmp(&o1.weight()).unwrap()); // decreasing order
    let center_sum: f64 = centers.iter().map(|x| x.weight() as f64).sum();
    for i in 0..centers.len() {
        centers[i].scale_weight(1.0/center_sum);
    }
    Ok(centers)
}



fn pick_slice_to_slice<'a>(index: usize, entry:&'a [&[f32]]) -> &'a [f32]{
    &entry[index]
}

fn pick_first_slice_to_slice<'a>(index: usize, entry:&'a [(&[f32],f32)]) -> &'a [f32]{
    &entry[index].0
}

fn pick_first_to_slice<'a>(index: usize, entry:&'a [(Vec<f32>,f32)]) -> &'a [f32]{
    &entry[index].0
}

fn pick_to_slice<'a>(index: usize, entry:&'a [Vec<f32>]) -> &'a [f32]{
    &entry[index]
}


fn pick_tuple_weight<T>(index:usize, entry:&[(T,f32)], weights: &[f32]) -> f32{
    entry[index].1
}

fn pick_weight<T>(index:usize, entry:&[T], weights: &[f32]) -> f32{
    weights[index]
}

fn one<'a,Z>(_i:usize,_points : &'a[Z], _weight : &'a [f32]) -> f32{
    1.0
}

pub fn single_centroid_cluster_weighted_vec_with_distance_over_slices(
    dictionary: &[(Vec<f32>, f32)],
    distance: fn(&[f32], &[f32]) -> f64,
    max_allowed: usize,
    parallel_enabled: bool,
) -> Result<Vec<Center>> {
    general_iterative_clustering(
        max_allowed,
        dictionary,
        empty_weights,
        pick_first_to_slice,
        pick_tuple_weight,
        LENGTH_BOUND,
        max_allowed as u64,
        parallel_enabled,
        Center::new,
        0,
        distance,
        false,
        true,
        SEPARATION_RATIO_FOR_MERGE,
    )
}

const empty_weights:&Vec<f32> = &Vec::new();

pub fn single_centroid_unweighted_cluster_vec_as_slice(
    dictionary: &[Vec<f32>],
    distance: fn(&[f32], &[f32]) -> f64,
    max_allowed: usize,
    parallel_enabled: bool,
) -> Result<Vec<Center>> {
    general_iterative_clustering(
        max_allowed,
        dictionary,
        empty_weights,
        pick_to_slice,
        one,
        LENGTH_BOUND,
        max_allowed as u64,
        parallel_enabled,
        Center::new,
        0,
        distance,
        false,
        true,
        SEPARATION_RATIO_FOR_MERGE,
    )
}

pub fn single_centroid_unweighted_cluster_slice(
    dictionary: &[&[f32]],
    distance: fn(&[f32], &[f32]) -> f64,
    max_allowed: usize,
    parallel_enabled: bool,
) -> Result<Vec<Center>> {
    general_iterative_clustering(
        max_allowed,
        dictionary,
        empty_weights,
        pick_slice_to_slice,
        one,
        LENGTH_BOUND,
        max_allowed as u64,
        parallel_enabled,
        Center::new,
        0,
        distance,
        false,
        true,
        SEPARATION_RATIO_FOR_MERGE,
    )
}


pub fn single_centroid_cluster_slice_with_weight_arrays(
    dictionary: &[&[f32]],
    weights : &[f32],
    distance: fn(&[f32], &[f32]) -> f64,
    max_allowed: usize,
    parallel_enabled: bool,
) -> Result<Vec<Center>> {
    general_iterative_clustering(
        max_allowed,
        dictionary,
        &weights,
        pick_slice_to_slice,
        pick_weight,
        LENGTH_BOUND,
        max_allowed as u64,
        parallel_enabled,
        Center::new,
        0,
        distance,
        false,
        true,
        SEPARATION_RATIO_FOR_MERGE,
    )
}


pub fn single_centroid_cluster_weighted_vec(
    dictionary: &[(Vec<f32>, f32)],
    distance: fn(&Vec<f32>, &Vec<f32>) -> f64,
    max_allowed: usize,
    parallel_enabled: bool,
) -> Result<Vec<Center>> {
    general_iterative_clustering(
        max_allowed,
        dictionary,
        empty_weights,
        pick_first_to_ref,
        pick_tuple_weight,
        LENGTH_BOUND,
        max_allowed as u64,
        parallel_enabled,
        Center::new_as_vec,
        0,
        distance,
        false,
        true,
        SEPARATION_RATIO_FOR_MERGE,
    )
}

fn pick_ref<'a,T :?Sized>(index: usize, entry:&[&'a T]) -> &'a T{
    entry[index]
}

fn pick_ref_tuple_first<'a,T :?Sized>(index: usize, entry:&[(&'a T,f32)]) -> &'a T{
    entry[index].0
}

fn pick_first_to_ref<'a,T>(index: usize, entry:&'a [(T,f32)]) -> &'a T{
    &entry[index].0
}

fn pick_to_ref<'a,T>(index: usize, entry:&'a [T]) -> &'a T{
    &entry[index]
}

pub fn single_centroid_cluster_vec(
    dictionary: &[Vec<f32>],
    distance: fn(&Vec<f32>, &Vec<f32>) -> f64,
    max_allowed: usize,
    parallel_enabled: bool,
) -> Result<Vec<Center>> {
    general_iterative_clustering(
        max_allowed,
        dictionary,
        empty_weights,
        pick_to_ref,
        one,
        LENGTH_BOUND,
        max_allowed as u64,
        parallel_enabled,
        Center::new_as_vec,
        0,
        distance,
        false,
        true,
        SEPARATION_RATIO_FOR_MERGE,
    )
}

#[repr(C)]
pub struct MultiCenterRef<'b, T :?Sized> {
    representatives: Vec<(&'b T,f32)>,
    number_of_representatives: usize,
    is_compact: bool,
    shrinkage: f32,
    weight: f64,
    sum_of_radii: f64,
}

impl<'b,T :?Sized> MultiCenterRef<'b,T>{

    pub fn representatives(& self) -> Vec<(&'b T,f32)>{
        self.representatives.clone()
    }

    pub fn new(representative: usize, point: &'b T, weight: f32, params : (usize,f32,bool)) -> Self {
        let (number_of_representatives, shrinkage,is_compact) = params;
        assert!(number_of_representatives>0,"has to be positive");
        assert!(shrinkage>=0.0 && shrinkage<= 1.0," has to between [0,1]");
        MultiCenterRef {
            representatives: vec![(point, weight as f32);1],
            number_of_representatives,
            shrinkage,
            is_compact,
            weight: weight as f64,
            sum_of_radii: 0.0,
        }
    }

    pub fn average_radius(&self) -> f64 {
        if self.weight == 0.0 {
            0.0
        } else {
            self.sum_of_radii / self.weight
        }
    }

    pub fn weight(&self) -> f64 {
        self.weight
    }
}

impl<'b, Z, T:?Sized> IntermediateCluster<Z, &'b T,T> for MultiCenterRef<'b,T> {
    fn weight(&self) -> f64 {
        self.weight()
    }

    fn distance_to_point<'a>(&self, _points:&'a [Z],_get_point: fn(usize,&'a [Z]) ->&'a T,point: &T, distance: fn(&T, &T) -> f64) -> (f64,usize) {
        let original = ((distance)(point, self.representatives[0].0), 0);
        let mut closest = original;
        for i in 1..self.representatives.len() {
            let t = ((distance)(point, self.representatives[i].0), i);
            if closest.0 > t.0 {
                closest = t;
            }
        }
        ((closest.0 * (1.0 - self.shrinkage as f64) + self.shrinkage as f64 * original.0), closest.1)
    }

    fn distance_to_cluster<'a>(
        &self,
        _points:&'a [Z],
        _get_point: fn(usize,&'a [Z]) ->&'a T,
        other: &dyn IntermediateCluster<Z,&'b T, T>,
        distance: fn(&T, &T) -> f64,
    ) -> (f64,usize,usize) {
        let list = other.representatives();
        let original = ((distance)(list[0].0, self.representatives[0].0), 0, 0);
        let mut closest = original;
        for i in 1..self.representatives.len() {
            for j in 1..list.len() {
                let t = ((distance)(list[j].0, self.representatives[i].0), i, j);
                if closest.0 > t.0 {
                    closest = t;
                }
            }
        }
        ((closest.0 * (1.0 - self.shrinkage as f64) + self.shrinkage as f64 * original.0), closest.1, closest.2)
    }

    fn add_point(&mut self, index: usize, weight: f32, dist: f64, representative: usize) {
        self.representatives[representative].1 += weight;
        self.sum_of_radii += weight as f64 * dist;
        self.weight += weight as f64;
    }

    fn recompute<'a>(&mut self, points:&'a [Z],get_point: fn(usize,&'a [Z]) ->&'a T, distance: fn(&T, &T) -> f64) -> f64 {
        self.representatives.sort_by(|a,b| a.1.partial_cmp(&b.1).unwrap());
        0.0
    }

    fn reset(&mut self) {
        self.sum_of_radii = 0.0;
        self.weight = 0.0;
        for i in 0..self.representatives.len() {
            self.representatives[i].1 = 0.0;
        }
    }

    fn extent_measure(&self) -> f64 {
        0.5 * self.average_radius() / self.number_of_representatives as f64
    }

    fn average_radius(&self) -> f64 {
        self.average_radius()
    }

    fn absorb<'a>(
        &mut self,
        points:&'a [Z],
        get_point: fn(usize,&'a [Z]) ->&'a T,
        another: &dyn IntermediateCluster<Z, &'b T, T>,
        distance: fn(&T, &T) -> f64,
    ) {
        self.sum_of_radii += if self.is_compact {
            another.average_radius()*another.weight()
        } else {
            another.extent_measure()*another.weight()
        };
        self.weight += another.weight();
        let mut representatives = Vec::new();
        representatives.append(&mut self.representatives);
        representatives.append(&mut another.representatives());
        self.representatives = Vec::with_capacity(self.number_of_representatives);

        let mut max_index: usize = 0;
        let mut weight = representatives[0].1;
        for i in 1..representatives.len() {
            if representatives[i].1 > weight {
                weight = representatives[i].1;
                max_index = i;
            }
        }
        self.representatives.push(representatives[max_index]);
        representatives.swap_remove(max_index);


        /**
         * create a list of representatives based on the farthest point method, which
         * correspond to a well scattered set. See
         * https://en.wikipedia.org/wiki/CURE_algorithm
         */
        while (representatives.len() > 0 && self.representatives.len() < self.number_of_representatives) {
            let mut farthest_weighted_distance = 0.0;
            let mut farthest_index: usize = usize::MAX;
            for j in 0..representatives.len() {
                if representatives[j].1 as f64 > (weight as f64) / (2.0 * self.number_of_representatives as f64) {
                    let mut new_weighted_distance = (distance)(self.representatives[0].0,
                                                               representatives[j].0) * representatives[j].1 as f64;
                    assert!(new_weighted_distance >= 0.0, " weights or distances cannot be negative");
                    for i in 1..self.representatives.len() {
                        let t = (distance)(self.representatives[i].0,
                                           representatives[j].0) * representatives[j].1 as f64;
                        assert!(t >= 0.0, " weights or distances cannot be negative");
                        if (t < new_weighted_distance) {
                            new_weighted_distance = t;
                        }
                    }
                    if new_weighted_distance > farthest_weighted_distance {
                        farthest_weighted_distance = new_weighted_distance;
                        farthest_index = j;
                    }
                }
            }
            if farthest_weighted_distance == 0.0 {
                break;
            }
            self.representatives.push(representatives[farthest_index]);
            representatives.swap_remove(farthest_index);
        }

        // absorb the remainder into existing representatives
        for j in 0..representatives.len() {
            let dist = (distance)(representatives[0].0, self.representatives[0].0);
            assert!(dist >= 0.0, "distance cannot be negative");
            let mut min_dist = dist;
            let mut min_index: usize = 0;
            for i in 1..self.representatives.len() {
                let new_dist = (distance)(self.representatives[i].0, representatives[j].0);
                assert!(new_dist >= 0.0, "distance cannot be negative");
                if (new_dist < min_dist) {
                    min_dist = new_dist;
                    min_index = i;
                }
            }
            self.representatives[min_index].1 += representatives[j].1;
            self.sum_of_radii += representatives[j].1 as f64 * min_dist;
        }
        self.representatives.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap());
    }

    fn representatives(&self) -> Vec<(&'b T, f32)> {
        self.representatives()
    }

    fn scale_weight(&mut self, factor: f64) {
        for i in 0..self.representatives.len() {
            self.representatives[i].1 = (self.representatives[i].1 as f64 * factor) as f32;
        }
    }

}

pub fn multi_cluster_obj<'a,T:Sync>(
    dictionary: &'a [T],
    distance: fn(&T, &T) -> f64,
    number_of_representatives: usize,
    shrinkage: f32,
    is_compact: bool,
    max_allowed: usize,
    parallel_enabled: bool,
) -> Result<Vec<MultiCenterRef<'a, T>>> {
    general_iterative_clustering(
        max_allowed,
        dictionary,
        empty_weights,
        pick_to_ref,
        one,
        LENGTH_BOUND,
        max_allowed as u64,
        parallel_enabled,
        MultiCenterRef::new,
        (number_of_representatives,shrinkage,is_compact),
        distance,
        false,
        true,
        SEPARATION_RATIO_FOR_MERGE,
    )
}


pub fn multi_cluster_as_ref<'a,T:Sync>(
    dictionary: &'a [&T],
    distance: fn(&T, &T) -> f64,
    number_of_representatives: usize,
    shrinkage: f32,
    is_compact: bool,
    max_allowed: usize,
    parallel_enabled: bool,
) -> Result<Vec<MultiCenterRef<'a, T>>> {
    general_iterative_clustering(
        max_allowed,
        dictionary,
        empty_weights,
        pick_ref,
        one,
        LENGTH_BOUND,
        max_allowed as u64,
        parallel_enabled,
        MultiCenterRef::new,
        (number_of_representatives,shrinkage,is_compact),
        distance,
        false,
        true,
        SEPARATION_RATIO_FOR_MERGE,
    )
}

pub fn multi_cluster_as_weighted_obj<'a,T:Sync>(
    dictionary: &'a [(T,f32)],
    distance: fn(&T, &T) -> f64,
    number_of_representatives: usize,
    shrinkage: f32,
    is_compact: bool,
    max_allowed: usize,
    parallel_enabled: bool,
) -> Result<Vec<MultiCenterRef<'a, T>>> {
    general_iterative_clustering(
        max_allowed,
        dictionary,
        empty_weights,
        pick_first_to_ref,
        pick_tuple_weight,
        LENGTH_BOUND,
        max_allowed as u64,
        parallel_enabled,
        MultiCenterRef::new,
        (number_of_representatives,shrinkage,is_compact),
        distance,
        false,
        true,
        SEPARATION_RATIO_FOR_MERGE,
    )
}

pub fn multi_cluster_as_weighted_ref<'a,T:Sync + ?Sized>(
    dictionary: &'a [(&T,f32)],
    distance: fn(&T, &T) -> f64,
    number_of_representatives: usize,
    shrinkage: f32,
    is_compact: bool,
    max_allowed: usize,
    parallel_enabled: bool,
) -> Result<Vec<MultiCenterRef<'a, T>>> {

    general_iterative_clustering(
        max_allowed,
        dictionary,
        empty_weights,
        pick_ref_tuple_first,
        pick_tuple_weight,
        LENGTH_BOUND,
        max_allowed as u64,
        parallel_enabled,
        MultiCenterRef::new,
        (number_of_representatives,shrinkage,is_compact),
        distance,
        false,
        true,
        SEPARATION_RATIO_FOR_MERGE,
    )
}

pub fn multi_cluster_as_object_with_weight_array<'a,T :Sync>(
    dictionary: &'a [T],
    weights: &'a [f32],
    distance: fn(&T, &T) -> f64,
    number_of_representatives: usize,
    shrinkage: f32,
    is_compact: bool,
    max_allowed: usize,
    parallel_enabled: bool,
) -> Result<Vec<MultiCenterRef<'a, T>>> {
    general_iterative_clustering(
        max_allowed,
        dictionary,
        weights,
        pick_to_ref,
        pick_weight,
        LENGTH_BOUND,
        max_allowed as u64,
        parallel_enabled,
        MultiCenterRef::new,
        (number_of_representatives,shrinkage,is_compact),
        distance,
        false,
        true,
        SEPARATION_RATIO_FOR_MERGE,
    )

}

#[repr(C)]
#[derive(Clone)]
pub struct MultiCenter<T : Clone> {
    representatives: Vec<(T,f32)>,
    shrinkage: f32,
    weight: f64,
    sum_of_radii: f64,
}

impl<T : Clone> MultiCenter<T>{

    pub fn create<'a>( refc : &MultiCenterRef<'a,T> ) -> Self {
        let mut rep_list = Vec::new();
        for j in refc.representatives() {
            rep_list.push((j.0.clone(),j.1));
        }
        MultiCenter {
            representatives: rep_list,
            weight: refc.weight,
            shrinkage : refc.shrinkage,
            sum_of_radii: refc.sum_of_radii,
        }
    }

    pub fn representatives(& self) -> Vec<(T,f32)>{
        self.representatives.clone()
    }

    pub fn representative(& self, number: usize) -> T {
        self.representatives[number].0.clone()
    }

    pub fn average_radius(&self) -> f64 {
        if self.weight == 0.0 {
            0.0
        } else {
            self.sum_of_radii / self.weight
        }
    }

    pub fn weight(&self) -> f64 {
        self.weight
    }

    pub fn distance_to_point(&self, point: &T, ignore: f32, distance: fn(&T, &T) -> f64) -> Result<(f64,usize)> {
        let original = ((distance)(point, &self.representatives[0].0), 0);
        check_argument(original.0>=0.0,"distances cannot be negative")?;
        let mut closest = original;
        for i in 1..self.representatives.len() {
            if self.representatives[i].1 > ignore {
                let t = ((distance)(point, &self.representatives[i].0), i);
                check_argument(t.0 >= 0.0, "distances cannot be negative")?;
                if closest.0 > t.0 {
                    closest = t;
                }
            }
        }
        Ok(((closest.0 * (1.0 - self.shrinkage as f64) + self.shrinkage as f64 * original.0), closest.1))
    }

    pub fn distance_to_point_and_ref<'a>(&'a self, point: &T, ignore: f32, distance: fn(&T, &T) -> f64) -> Result<(f64,&'a T)> {
        let original = ((distance)(point, &self.representatives[0].0), 0);
        check_argument(original.0>=0.0,"distances cannot be negative")?;
        let mut closest = original;
        for i in 1..self.representatives.len() {
            if self.representatives[i].1 > ignore {
                let t = ((distance)(point, &self.representatives[i].0), i);
                check_argument(t.0 >= 0.0, "distances cannot be negative")?;
                if closest.0 > t.0 {
                    closest = t;
                }
            }
        }
        Ok(((closest.0 * (1.0 - self.shrinkage as f64) + self.shrinkage as f64 * original.0), &self.representatives[closest.1].0))
    }
}

pub fn persist<'a,T:Clone>(list:&Vec<MultiCenterRef<'a,T>>) -> Vec<MultiCenter<T>> {
    let mut answer = Vec::new();
    for item in list {
       answer.push(MultiCenter::create(item));
    }
    answer
}