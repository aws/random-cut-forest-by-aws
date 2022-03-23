use std::cmp::min;
use num::abs;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

///
/// The goal of the summarization below is as follows: on being provided a collection of sampled weighted points
/// represented by a slice &[(Vec<f32>,f32)] where each of the Vec<f32> has the same length/dimension
/// and the f32 in the pair is the corresponding weight.
/// it performs a soft multi-centroid summarization similar to https://en.wikipedia.org/wiki/CURE_algorithm
/// it uses L1 clustering since that is natural to RCFs, but can be amended if need arises
///
/// the output is the SampleSummary, which provides basic statistics of mean, median and deviation
/// in addition it performs a grouping/clustering, assuming that the maximum number of clusters are not large
/// the routine below bounds the number to be max_number_per_dimension times the dimension of Vec<f32>
/// and a smaller number can also be provided in the summarize() function
///
///


const max_number_per_dimension : usize = 5;

#[repr(C)]
pub struct SampleSummary {
    pub summary_points: Vec<Vec<f32>>,

    /**
     * a measure of comparison among the typical points;
     */
    pub relative_weight: Vec<f32>,

    /**
     * number of samples, often the number of summary
     */
    pub total_weight: f32,
    /**
     * the global mean, median
     */
    pub mean: Vec<f32>,
    pub median: Vec<f32>,

    /**
     * This is the global deviation, without any filtering on the TreeSamples
     */
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
struct ProjectedPoint {
    index: usize,
    weight: f32
}

#[repr(C)]
struct Center {
    coordinate: Vec<f32>,
    weight: f64,
    points: Vec<ProjectedPoint>,
    sum_of_radii : f64
}


impl ProjectedPoint {
    pub fn new (index:usize, weight : f32) -> Self {
        ProjectedPoint {
            index,
            weight,
        }
    }
}

impl Center {
    pub fn new(dimensions:usize) -> Self {
        Center{
            coordinate : vec![0.0;dimensions],
            weight : 0.0,
            points : Vec::new(),
            sum_of_radii : 0.0
        }
    }

    pub fn initial(coordinate: &[f32]) -> Self{
        Center{
            coordinate : Vec::from(coordinate),
            weight : 0.0,
            points : Vec::new(),
            sum_of_radii : 0.0
        }
    }

    pub fn add (&mut self, index:usize, weight: f32){
        self.points.push( ProjectedPoint::new(index,weight));
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

    fn median(points:&[(Vec<f32>,f32)], list: &mut[ProjectedPoint], dimensions:usize) -> Vec<f32>{
        let mut answer = vec![0.0f32;dimensions];
        let total : f64 = list.iter().map(|a| a.weight as f64).sum();
        for i in 0..dimensions {
            list.sort_by(|a, b| points[a.index].0[i].partial_cmp(&points[b.index].0[i]).unwrap());
            let mut running_weight = total / 2.0;
            let mut position = 0;
            while running_weight >= 0.0 && position < list.len() {
                if running_weight as f32 >= list[position].weight {
                    running_weight -= list[position].weight as f64;
                    position += 1;
                } else {
                    break;
                }
            }
            answer[i] = points[list[position].index].0[i];
        }
        answer
    }

    pub fn recompute(&mut self,points:&[(Vec<f32>,f32)],distance : fn(&[f32],&[f32]) -> f64){
        self.sum_of_radii  = 0.0;
        if self.weight == 0.0 {
            assert!(self.points.len() > 0, "adding no points? ");
            self.coordinate = vec![0.0;self.coordinate.len()];
            return;
        }

        let new_center = if (self.points.len() < 500) {
            Self::median(points,&mut self.points,self.coordinate.len())
        } else {
            let mut samples = Vec::new();
            let mut rng =ChaCha20Rng::seed_from_u64(0);
            for i in 0..self.points.len() {
                if rng.gen::<f64>()  < (200.0 * self.points[i].weight as f64)/self.weight {
                    samples.push(ProjectedPoint::new(self.points[i].index,1.0));
                }
            };
            Self::median(points,&mut samples,self.coordinate.len())
        };

        for i in 0..self.coordinate.len() {
            self.coordinate[i] = new_center[i];
        }

        for j in 0..self.points.len() {
                self.sum_of_radii += self.points[j].weight as f64 * distance(&self.coordinate,&points[self.points[j].index].0) as f64;
        }
    }

}

const weight_threshold : f64 = 1.25;


fn assign(points : &[(Vec<f32>,f32)], centers: &mut[Center], distance : fn(&[f32],&[f32]) -> f64) {

    for j in 0..centers.len() {
        centers[j].reset();
    }

    for i in 0..points.len() {
        let mut dist = vec![0.0;centers.len()];
        let mut min_distance = f64::MAX;
        for j in 0..centers.len() {
            dist[j] = distance(&centers[j].coordinate, &points[i].0);
            if min_distance > dist[j] {
                min_distance = dist[j];
            }
        }
        if min_distance == 0.0 {
            for j in 0..centers.len() {
                if dist[j] == 0.0 {
                    centers[j].add(i, points[i].1);
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
                    centers[j].add(i, (points[j].1 as f64 * min_distance / (sum * dist[j])) as f32);
                }
            }
        }
    }
}

const separation_ratio_for_merge : f64 = 0.8;

pub fn summarize(points : &[(Vec<f32>,f32)], distance : fn(&[f32],&[f32]) -> f64, max_number : usize) -> SampleSummary {
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
    let total_weight = points.iter().map(|x| x.1).sum();
    assert!(total_weight>0.0, "weights cannot be all zero");
    let mut mean = vec![0.0f32; dimensions];
    let mut deviation = vec![0.0f32; dimensions];
    let mut sum_values_sq = vec![0.0f64; dimensions];
    let mut sum_values = vec![0.0f64; dimensions];
    for i in 0..points.len() {
        for j in 0..dimensions {
            sum_values[j] += points[i].1 as f64 * points[i].0[j] as f64;
            sum_values_sq[j] += points[i].1 as f64 * points[i].0[j]  as f64 * points[i].0[j] as f64;
        }
    };
    for j in 0..dimensions {
        mean[j] = (sum_values[j] / total_weight as f64) as f32;
        let t: f64 = sum_values_sq[j] / total_weight as f64 - sum_values[j] * sum_values[j] / (total_weight as f64 * total_weight as f64);
        deviation[j] = f64::sqrt(if t > 0.0 { t } else { 0.0 }) as f32;
    };
    let mut median = vec![0.0f32; dimensions];
    let num = points.len();
    for j in 0..dimensions {
        let mut v : Vec<f32>= points.iter().map(|x| x.0[j]).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        median[j] = v[num / 2];
    };

    let mut centers: Vec<Center> = Vec::new();
    centers.push(Center::initial(&median));
    let max_allowed = min(dimensions * max_number_per_dimension,
                          max_number);
    for k in 0..2 * max_allowed {
        let mut max_dist = 0.0;
        let mut max_index = usize::MAX;
        for j in 0..points.len() {
            let mut min_dist = f64::MAX;
            for i in 0..centers.len() {
                let t= distance(&points[j].0, &centers[i].coordinate);
                if t < min_dist {
                    min_dist = t;
                };
            }
            if min_dist > max_dist {
                max_dist = min_dist;
                max_index = j;
            }
        };
        if max_dist == 0.0 {
            break;
        } else {
            centers.push(Center::initial(&points[max_index].0));
        };
    };

    assign(points,&mut centers,distance);

    for i in 0..centers.len() {
        centers[i].recompute(points,distance);
    }

    let mut measure = 2.0 * separation_ratio_for_merge;
    while measure > 2.0 * separation_ratio_for_merge || centers.len() > max_allowed {
        let mut first = usize::MAX;
        let mut second = usize::MAX;
        measure = 0.0;
        for i in 0..centers.len() - 1 {
            for j in i + 1..centers.len() {
                let dist = distance(&centers[i].coordinate, &centers[j].coordinate);
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
        } else if centers.len() > max_allowed {
            // not well separated, remove small weight cluster centers
            // increasing order of weight
            centers.sort_by(|o1, o2| o1.weight.partial_cmp(&o2.weight).unwrap());
            centers.swap_remove(0);
        }
        assign(points,&mut centers,distance);
        for i in 0..centers.len() {
            centers[i].recompute(points,distance);
        }
    };

    centers.sort_by(|o1, o2| o2.weight.partial_cmp(&o1.weight).unwrap()); // decreasing order
    let mut summary_points: Vec<Vec<f32>> = Vec::new();
    let mut relative_weight: Vec<f32> = Vec::new();
    for i in 0..centers.len() {
        summary_points.push(centers[i].coordinate.clone());
        relative_weight.push((centers[i].weight /total_weight as f64) as f32);
    };
    SampleSummary {
        summary_points,
        relative_weight,
        total_weight,
        mean,
        median,
        deviation
    }
}