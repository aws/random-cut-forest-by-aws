use std::cmp::min;
use num::abs;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

///
/// The goal of the summarization below is as follows: on being provided a collection of sampled weighted points
/// represented by a slice &[(Vec<f32>,f32)] where each of the Vec<f32> has the same length/dimension
/// and the f32 in the pair is the corresponding weight.
/// The algorithm uses the philosophy of RCFs, in repeatedly using randomization. It proceeds as follows:
/// 1. It uses an initial sampling which serves as a basis of efficiency as well as denoising, borrowing from
/// https://en.wikipedia.org/wiki/CURE_algorithm, in that algorithm's robustness to outliers.
/// 2. It uses a sampling mechanism to initialize some clusters based on https://en.wikipedia.org/wiki/Data_stream_clustering
/// where the radom sampling achieves half of the the same effects as hierarchical compression.
///3.  It repeatedly merges the most overlapping clusters, failing that, eliminates the least weighted cluster to achieve
/// the same effect as hieararchical compression.
///
/// The algorithm takes a distance function as an input, and tends to produce spherical (measured in the input
/// distance function) clusters. These types of algorithms are unlikely to be useful for large number of output clusters.
/// The output is the SampleSummary, which provides basic statistics of mean, median and deviation
/// in addition it performs a grouping/clustering, assuming that the maximum number of clusters are not large
/// the routine below bounds the number to be max_number_per_dimension times the dimension of Vec<f32>
/// and a smaller number can also be provided in the summarize() function
///
///


const max_number_per_dimension : usize = 5;

#[repr(C)]
pub struct SampleSummary {
    pub summary_points: Vec<Vec<f32>>,

    ///a measure of comparison among the typical points;
    pub relative_weight: Vec<f32>,


    /// number of samples, often the number of summary, but can handle weighted points
    /// (possibly indicating confidence or othe measure) in the future
    pub total_weight: f32,


    /// the global mean, median
    pub mean: Vec<f32>,
    pub median: Vec<f32>,

    /// This is the global deviation, without any filtering on the TreeSamples
    pub deviation: Vec<f32>,
}

impl SampleSummary{
    pub fn new(total_weight: f32, summary_points : Vec<Vec<f32>>, relative_weight : Vec<f32>,
               median : Vec<f32>, mean : Vec<f32>, deviation : Vec<f32>) -> Self {
        SampleSummary{
            total_weight,
            summary_points : summary_points.clone(),
            relative_weight : relative_weight.clone(),
            median: median.clone(),
            mean: mean.clone(),
            deviation : deviation.clone()
        }
    }
}

#[repr(C)]
struct Center<Q,T> {
    representative: Q,
    weight: f64,
    points: Vec<(usize,f32)>,
    sum_of_radii : f64,
    optimize : fn(&mut Q,&[(T,f32)],&mut [(usize,f32)])
}

impl<Q,T: Copy> Center<Q,T> {
    pub fn new(representative : Q, optimize: fn(&mut Q,&[(T,f32)],&mut [(usize,f32)])) -> Self {
        Center{
            representative,
            weight : 0.0,
            points : Vec::new(),
            sum_of_radii : 0.0,
            optimize
        }
    }

    pub fn add_point (&mut self, index:usize, weight: f32){
        self.points.push( (index,weight));
        self.weight += weight as f64;
    }

    pub fn reset(&mut self){
        self.points.clear();
        self.weight = 0.0;
    }

    pub fn average_radius(&self) -> f64 {
        if self.weight == 0.0 {
            0.0
        } else {
            self.sum_of_radii/self.weight
        }
    }

    pub fn recompute(&mut self,points:&[(T,f32)],distance : fn(&Q,T) -> f64){
        self.sum_of_radii  = 0.0;
        if self.weight == 0.0 {
            assert!(self.points.len() == 0, "adding points with weight 0.0 ?");
            //self.coordinate = vec![0.0;self.coordinate.len()];
            return;
        }

        // the following computes an approximate median
        if self.points.len() < 500 {
            (self.optimize)(&mut self.representative, points, &mut self.points)
        } else {
            let mut samples = Vec::new();
            let mut rng =ChaCha20Rng::seed_from_u64(0);
            for i in 0..self.points.len() {
                if rng.gen::<f64>()  < (200.0 * self.points[i].1 as f64)/self.weight {
                    samples.push((self.points[i].0,1.0));
                }
            };
            (self.optimize)(&mut self.representative,points,&mut samples)
        };

        for j in 0..self.points.len() {
                self.sum_of_radii += self.points[j].1 as f64 * distance(&self.representative,points[self.points[j].0].0) as f64;
        }
    }

}

pub fn optimize_l1(representative: &mut Vec<f32>, points:&[(&[f32],f32)], list: &mut[(usize,f32)]){
    let dimensions = representative.len();
    let total : f64 = list.iter().map(|a| a.1 as f64).sum();
    for i in 0..dimensions {
        list.sort_by(|a, b| points[a.0].0[i].partial_cmp(&points[b.0].0[i]).unwrap());
        let position = pick(list,(total/2.0) as f32);
        representative[i] = points[list[position].0].0[i];
    }
}

const weight_threshold : f64 = 1.25;

fn pick<T>(points: &[(T,f32)], wt : f32) -> usize{
    let mut position = 0;
    let mut running = wt;
    for i in 0..points.len() {
        position = i;
        if running -points[i].1 <= 0.0 {
            break;
        } else {
            running -= points[i].1;
        }
    }
    position
}


fn assign<Q,T : Copy>(points : &[(T,f32)], centers: &mut[Center<Q,T>], distance : fn(&Q,T) -> f64) {

    for j in 0..centers.len() {
        centers[j].reset();
    }
    // the generator will keep varying as the number changes
    let mut rng = ChaCha20Rng::seed_from_u64(centers.len() as u64);
    for i in 0..points.len() {
        let mut dist = vec![0.0; centers.len()];
        let mut min_distance = f64::MAX;
        for j in 0..centers.len() {
            dist[j] = distance(&centers[j].representative, points[i].0);
            if min_distance > dist[j] {
                min_distance = dist[j];
            }
        }
        if min_distance == 0.0 {
            for j in 0..centers.len() {
                if dist[j] == 0.0 {
                    centers[j].add_point(i, points[i].1);
                }
            }
        } else {
            let mut sum = 0.0;
            for j in 0..centers.len() {
                if dist[j] <= weight_threshold * min_distance {
                    sum += min_distance / dist[j];
                }
            }
            for j in 0..centers.len() {
                if dist[j] <= weight_threshold * min_distance {
                    centers[j].add_point(i, (points[j].1 as f64 * min_distance / (sum * dist[j])) as f32);
                }
            }
        }
    }
}

fn add_center(points: &[(&[f32],f32)], centers : &mut Vec<Center<Vec<f32>,&[f32]>>, distance: fn (&Vec<f32>,&[f32])-> f64, index: usize,optimize: fn(&mut Vec<f32>,&[(&[f32],f32)],&mut [(usize,f32)])) {
    let mut min_dist = f64::MAX;
    for i in 0..centers.len() {
        let t = distance(&centers[i].representative, points[index].0);
        if t < min_dist {
            min_dist = t;
        };
    }
    if min_dist > 0.0 {
        centers.push(Center::new(Vec::from(points[index].0),optimize));
    }
}

const separation_ratio_for_merge : f64 = 0.8;


pub fn summarize(points : &[(Vec<f32>,f32)], distance : fn(&Vec<f32>,&[f32]) -> f64, max_number : usize) -> SampleSummary {
    assert!(max_number < 51, " for large number of clusters, other methods may be better, consider recursively removing clusters");
    if max_number == 0 {
        return SampleSummary{
            summary_points: vec![],
            relative_weight: vec![],
            total_weight: 0.0,
            mean: vec![],
            median: vec![],
            deviation: vec![]
        }
    };

    assert!(points.len() > 0, "cannot be empty list");
    let dimensions = points[0].0.len();
    assert!(dimensions > 0, " cannot have 0 dimensions");
    let total_weight : f64 = points.iter().map(|x| x.1 as f64).sum();
    assert!(total_weight>0.0, "weights cannot be all zero");
    assert!(total_weight.is_finite(), " cannot have infinite weights");
    let mut mean = vec![0.0f32; dimensions];
    let mut deviation = vec![0.0f32; dimensions];
    let mut sum_values_sq = vec![0.0f64; dimensions];
    let mut sum_values = vec![0.0f64; dimensions];
    for i in 0..points.len() {
        assert!(points[i].0.len() == dimensions, "incorrect dimensions");
        assert!(points[i].1 >= 0.0, "point weights have to be non-negative");
        for j in 0..dimensions {
            assert!(points[i].0[j].is_finite(), " cannot have infinite values in coordinate");
            sum_values[j] += points[i].1 as f64 * points[i].0[j] as f64;
            sum_values_sq[j] += points[i].1 as f64 * points[i].0[j]  as f64 * points[i].0[j] as f64;
        }
    };
    for j in 0..dimensions {
        mean[j] = (sum_values[j] / total_weight) as f32;
        let t: f64 = sum_values_sq[j] / total_weight - sum_values[j] * sum_values[j] / (total_weight * total_weight);
        deviation[j] = f64::sqrt(if t > 0.0 { t } else { 0.0 }) as f32;
    };
    let mut median = vec![0.0f32; dimensions];
    let num = points.len();
    for j in 0..dimensions {
        let mut v : Vec<f32>= points.iter().map(|x| x.0[j]).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        median[j] = v[num / 2];
    };

    let mut centers: Vec<Center<Vec<f32>,&[f32]>> = Vec::new();
    centers.push(Center::new(median.clone(),optimize_l1));
    let max_allowed = min(dimensions * max_number_per_dimension,
                          max_number);
    let mut rng = ChaCha20Rng::seed_from_u64(max_allowed as u64);

    /// the following sampling is for efficiency reasons
    /// the sampling produces references to save duplication
    /// however that implies that ownership wise, those references should be
    /// created and eventually destroyed in this same block
     let mut sampled_points: Vec<(&[f32],f32)> = Vec::new();
    if points.len() < 10000 {
        for j in 0..points.len() {
            sampled_points.push((&points[j].0,points[j].1));
        }
    } else {
        let mut remainder = 0.0f64;
        for j in 0..points.len() {
            if points[j].1 > (total_weight / 1000.0) as f32 {
                sampled_points.push((&points[j].0, points[j].1));
            } else {
                remainder += points[j].1 as f64;
            }
        }
        for j in 0..points.len() {
            if points[j].1 <= (total_weight / 1000.0) as f32 && rng.gen::<f64>() < 10000.0 / points.len() as f64 {
                let t = points[j].1 as f64 * (points.len() as f64/10000.0) * (remainder / total_weight);
                sampled_points.push((&points[j].0, t as f32));
            }
        }
    }

    ///
    /// we now peform an initialization; the sampling corresponds a denoising
    /// note that if we are look at 2k random points, we are likely hitting every group of points
    /// with weight 1/k whp
    ///
    let sampled_sum : f32 = sampled_points.iter().map(|x| x.1).sum();
    for _k in 0..2 * max_allowed{
        let wt = (rng.gen::<f64>() * sampled_sum as f64) as f32;
        add_center(&sampled_points,&mut centers,distance,pick(&sampled_points,wt), optimize_l1);
    }

    assign(&sampled_points,&mut centers,distance);

    for i in 0..centers.len() {
        centers[i].recompute(&sampled_points,distance);
    }

    // sort in increasing order of weight
    centers.sort_by(|o1, o2| o1.weight.partial_cmp(&o2.weight).unwrap());

    let mut measure = 2.0 * separation_ratio_for_merge;
    while measure > 2.0 * separation_ratio_for_merge || centers.len() > max_allowed || centers[0].weight < 1.0/(10.0 *max_allowed as f64){

        let mut first = usize::MAX;
        let mut second = usize::MAX;
        measure = 0.0;
        for i in 0..centers.len() - 1 {
            for j in i + 1..centers.len() {
                let dist = distance(&centers[i].representative, &centers[j].representative);
                let temp_measure = (centers[i].average_radius() + centers[j].average_radius()) / dist;
                if measure < temp_measure {
                    first = i;
                    second = j;
                    measure = temp_measure;
                }
            }
        };
        if measure >= separation_ratio_for_merge {
            if centers[first].weight < centers[second].weight {
                centers.swap_remove(first);
            } else {
                centers.swap_remove(second);
            }
        } else if centers.len() > max_allowed || centers[0].weight < 1.0/(10.0 *max_allowed as f64){
            // not well separated, remove small weight cluster centers
            // increasing order of weight
            centers.swap_remove(0);
        }
        assign(&sampled_points,&mut centers,distance);
        for i in 0..centers.len() {
            centers[i].recompute(&sampled_points,distance);
        }
    };

    centers.sort_by(|o1, o2| o2.weight.partial_cmp(&o1.weight).unwrap()); // decreasing order
    let mut summary_points: Vec<Vec<f32>> = Vec::new();
    let mut relative_weight: Vec<f32> = Vec::new();
    let center_sum : f64= centers.iter().map(|x| x.weight).sum();
    for i in 0..centers.len() {
        summary_points.push(centers[i].representative.clone());
        relative_weight.push((centers[i].weight /center_sum) as f32);
    };
    SampleSummary {
        summary_points,
        relative_weight,
        total_weight: total_weight as f32,
        mean,
        median,
        deviation
    }
}