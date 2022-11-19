use std::cmp::max;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};

const PHASE2_THRESHOLD: usize = 2;
const SEPARATION_RATIO_FOR_MERGE: f64 = 0.8;
const WEIGHT_THRESHOLD: f64 = 1.25;
const LENGTH_BOUND: usize = 1000;

pub trait Cluster<Q, T: ?Sized> {
    // aclsuter should provide a measure of the point set convered by the cluster
    fn weight(&self) -> f64;
    // A cluster is an extended object of a type different from the base type of a
    // point indicated by T (and only the reference is used in the distance function
    // every clustering algorithm implicitly/explicitly defines this function
    fn distance_to_point(&self, point: &T, distance: fn(&T, &T) -> f64) -> f64;
    // Likewise, the distance function needs to be extended (implicitly/explicitly)
    // to a distance function between clusters
    fn distance_to_cluster(&self, other: &dyn Cluster<Q, T>, distance: fn(&T, &T) -> f64) -> f64;
    // a function that assigns a point indexed by usize from a list of samples to
    // the cluster; note the weight used in the function need not be the entire weight of the
    // sampled point (for example in case of soft assignments)
    fn add_point(&mut self, index: usize, weight: f32, dist: f64);
    // given a set of previous assignments, recomputes the optimal set of representatives
    // this is the classic optimization step for k-Means; but the analogue exists for every
    // clustering; note that it is possible that recompute does nothing
    fn recompute(&mut self, points: &[(&T, f32)], distance: fn(&T, &T) -> f64) -> f64;
    // resets  the statistics of a clusters; preparing for a sequence of add_point followed
    // by recompute
    fn reset(&mut self);
    // a function that indicates cluster quality
    fn average_radius(&self) -> f64;
    // a function that allows a cluster to absorb another cluster in an agglomerative \
    // clustering algorithm
    fn absorb(&mut self, another: &dyn Cluster<Q, T>, distance: fn(&T, &T) -> f64);
    // a function to return a list of representatives corresponding to pairs (Q,weight)
    fn representatives(&self) -> Vec<(Q, f32)>;
    // a function to distill all representatives into a single representative under a given
    // distance function over the base type of the points
    fn primary_representative(&self, _distance: fn(&T, &T) -> f64) -> Q;
}

#[repr(C)]
pub struct Center {
    representative: Vec<f32>,
    weight: f64,
    points: Vec<(usize, f32)>,
    sum_of_radii: f64,
}

impl Center {
    pub fn new(representative: &[f32], weight: f32) -> Self {
        Center {
            representative: Vec::from(representative),
            weight: weight as f64,
            points: Vec::new(),
            sum_of_radii: 0.0,
        }
    }

    // the following function takes a list of points and
    // uses the assigned indices in self.points (via add_point) to compute the median
    // clearly this is not optimal for means/L_infinity -- but median is more robust
    // and some interpretation in the context of upstream sampling
    //note that this is not a public function
    fn optimize_small(&mut self, points: &[(&[f32], f32)]) {
        let dimensions = self.representative.len();
        let total: f64 = self.points.iter().map(|a| a.1 as f64).sum();
        for i in 0..dimensions {
            self.points
                .sort_by(|a, b| points[a.0].0[i].partial_cmp(&points[b.0].0[i]).unwrap());
            let position = pick(&self.points, (total / 2.0) as f32);
            self.representative[i] = points[self.points[position].0].0[i];
        }
    }

    // same as optimize_small, but for a large number of assigned points -- the list
    // is now a sample. The same function as optimize_small does not suffice because
    // the borrow checker would not allow samples from self.points be moved and yet borrow self
    // note that this function is called via rayon par_iter() as well
    fn optimize(&mut self, points: &[(&[f32], f32)], list: &mut [(usize, f32)]) {
        let dimensions = self.representative.len();
        let total: f64 = list.iter().map(|a| a.1 as f64).sum();
        for i in 0..dimensions {
            list.sort_by(|a, b| points[a.0].0[i].partial_cmp(&points[b.0].0[i]).unwrap());
            let position = pick(list, (total / 2.0) as f32);
            self.representative[i] = points[list[position].0].0[i];
        }
    }
}

impl Cluster<Vec<f32>, [f32]> for Center {
    fn weight(&self) -> f64 {
        self.weight
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

    fn average_radius(&self) -> f64 {
        if self.weight == 0.0 {
            0.0
        } else {
            self.sum_of_radii / self.weight
        }
    }

    fn recompute(&mut self, points: &[(&[f32], f32)], distance: fn(&[f32], &[f32]) -> f64) -> f64 {
        let old_value = self.sum_of_radii;
        self.sum_of_radii = 0.0;
        if self.weight == 0.0 {
            assert!(self.points.len() == 0, "adding points with weight 0.0 ?");
            return 0.0;
        }

        // the following computes an approximate median
        if self.points.len() < 500 {
            self.optimize_small(points);
        } else {
            let mut samples = Vec::new();
            let mut rng = ChaCha20Rng::seed_from_u64(0);
            for i in 0..self.points.len() {
                if rng.gen::<f64>() < (200.0 * self.points[i].1 as f64) / self.weight {
                    samples.push((self.points[i].0, 1.0));
                }
            }
            self.optimize(points, &mut samples);
        };

        for j in 0..self.points.len() {
            self.sum_of_radii += self.points[j].1 as f64
                * distance(&self.representative, points[self.points[j].0].0) as f64;
        }
        old_value - self.sum_of_radii
    }

    fn distance_to_point(&self, point: &[f32], distance: fn(&[f32], &[f32]) -> f64) -> f64 {
        (distance)(&self.representative, point)
    }

    fn distance_to_cluster(
        &self,
        other: &dyn Cluster<Vec<f32>, [f32]>,
        distance: fn(&[f32], &[f32]) -> f64,
    ) -> f64 {
        other.distance_to_point(&self.representative, distance)
    }

    fn absorb(
        &mut self,
        another: &dyn Cluster<Vec<f32>, [f32]>,
        distance: fn(&[f32], &[f32]) -> f64,
    ) {
        let other_weight = another.weight();
        let t = f64::exp(2.0 * (self.weight - other_weight) / (self.weight + other_weight));
        let factor = t / (1.0 + t);
        let list = another.representatives();
        let mut closest = &list[0].0;
        let mut dist = distance(&self.representative, closest);
        for i in 1..list.len() {
            let t = distance(&self.representative, &list[i].0);
            if t < dist {
                closest = &list[i].0;
                dist = t;
            }
        }

        let dimensions = self.representative.len();
        for i in 0..dimensions {
            self.representative[i] = (factor * (self.representative[i] as f64)
                + (1.0 - factor) * (closest[i] as f64)) as f32;
        }

        self.sum_of_radii += (self.weight * (1.0 - factor) + factor * other_weight) * dist;
    }

    fn representatives(&self) -> Vec<(Vec<f32>, f32)> {
        vec![(self.representative.clone(), self.weight as f32); 1]
    }

    fn primary_representative(&self, _distance: fn(&[f32], &[f32]) -> f64) -> Vec<f32> {
        self.representative.clone()
    }
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

fn assign_and_recompute<Q, U, T: ?Sized>(
    points: &[(&T, f32)],
    centers: &mut [U],
    distance: fn(&T, &T) -> f64,
    parallel_enabled: bool,
) -> f64
where
    U: Cluster<Q, T> + Send,
    T: std::marker::Sync,
{
    for j in 0..centers.len() {
        centers[j].reset();
    }
    for i in 0..points.len() {
        let mut dist = vec![0.0; centers.len()];
        let mut min_distance = f64::MAX;
        for j in 0..centers.len() {
            dist[j] = centers[j].distance_to_point(points[i].0, distance);
            if min_distance > dist[j] {
                min_distance = dist[j];
            }
        }
        if min_distance == 0.0 {
            for j in 0..centers.len() {
                if dist[j] == 0.0 {
                    centers[j].add_point(i, points[i].1, 0.0);
                }
            }
        } else {
            let mut sum = 0.0;
            for j in 0..centers.len() {
                if dist[j] <= WEIGHT_THRESHOLD * min_distance {
                    sum += min_distance / dist[j];
                }
            }
            for j in 0..centers.len() {
                if dist[j] <= WEIGHT_THRESHOLD * min_distance {
                    centers[j].add_point(
                        i,
                        (points[i].1 as f64 * min_distance / (sum * dist[j])) as f32,
                        dist[j],
                    );
                }
            }
        }
    }
    let gain: f64 = if parallel_enabled {
        centers
            .par_iter_mut()
            .map(|x| x.recompute(points, distance))
            .sum()
    } else {
        centers
            .iter_mut()
            .map(|x| x.recompute(points, distance))
            .sum()
    };
    gain
}

fn add_center<T: ?Sized, U, Q>(
    centers: &mut Vec<U>,
    points: &[(&T, f32)],
    distance: fn(&T, &T) -> f64,
    index: usize,
    create: fn(&T, f32) -> U,
) where
    U: Cluster<Q, T> + Send,
{
    let mut min_dist = f64::MAX;
    for i in 0..centers.len() {
        let t = centers[i].distance_to_point(points[index].0, distance);
        if t < min_dist {
            min_dist = t;
        };
    }
    if min_dist > 0.0 {
        centers.push(create(points[index].0, points[index].1));
    }
}

pub fn iterative_clustering<U, Q, T: ?Sized>(
    max_allowed: usize,
    sampled_points: &[(&T, f32)],
    create: fn(&T, f32) -> U,
    distance: fn(&T, &T) -> f64,
    parallel_enabled: bool,
) -> Vec<U>
where
    U: Cluster<Q, T> + Send,
    T: std::marker::Sync,
{
    general_iterative_clustering(
        max_allowed,
        sampled_points,
        max_allowed as u64,
        parallel_enabled,
        create,
        distance,
        false,
        true,
        SEPARATION_RATIO_FOR_MERGE,
    )
}

pub fn general_iterative_clustering<U, Q, T: ?Sized>(
    max_allowed: usize,
    sampled_points: &[(&T, f32)],
    seed: u64,
    parallel_enabled: bool,
    create: fn(&T, f32) -> U,
    distance: fn(&T, &T) -> f64,
    phase_1_reassign: bool,
    enable_phase_3: bool,
    overlap_parameter: f64,
) -> Vec<U>
where
    U: Cluster<Q, T> + Send,
    T: std::marker::Sync,
{
    let mut rng = ChaCha20Rng::seed_from_u64(seed);

    let mut centers: Vec<U> = Vec::new();
    //
    // we now peform an initialization; the sampling corresponds a denoising
    // note that if we are look at 2k random points, we are likely hitting every group of points
    // with weight 1/k whp

    let sampled_sum: f32 = sampled_points.iter().map(|x| x.1).sum();
    for _k in 0..10 * max_allowed {
        let wt = (rng.gen::<f64>() * sampled_sum as f64) as f32;
        add_center(
            &mut centers,
            &sampled_points,
            distance,
            pick(&sampled_points, wt),
            create,
        );
    }

    assign_and_recompute(&sampled_points, &mut centers, distance, parallel_enabled);

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
                let dist = centers[lower].distance_to_cluster(&centers[j], distance);
                if min_dist > dist {
                    min_nbr = j;
                    min_dist = dist;
                }
                let numerator = centers[lower].average_radius()
                    + centers[j].average_radius()
                    + phase_3_distance;
                if numerator >= overlap_parameter * dist {
                    if measure * dist < numerator {
                        first = lower;
                        second = j;
                        if dist == 0.0f64 {
                            found_merge = true;
                        } else {
                            measure = numerator / dist;
                        }
                        measure_dist = dist;
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
            large.first_mut().unwrap().absorb(&small[first], distance);
            centers.swap_remove(first);
            if phase_1_reassign || centers.len() <= PHASE2_THRESHOLD * max_allowed {
                assign_and_recompute(&sampled_points, &mut centers, distance, parallel_enabled);
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
    return centers;
}
