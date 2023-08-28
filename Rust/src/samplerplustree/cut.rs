use rand::Rng;
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;
use crate::samplerplustree::boundingbox::BoundingBox;
use crate::util::{maxf32, minf32};

/**
*  this is a class that helps manage the cut information; the nodes do not store information in
*  this format
*/

pub struct Cut {
    pub dimension: usize,
    pub value: f32,
}

impl Cut {
    pub fn new(dimension: usize, value: f32) -> Self {
        Cut { dimension, value }
    }

    // factor should be in [0,1) in the follwoing; but we would not rule that
    // out so that use out-of-range values to test the function
    // the only invariant we must satisfy is that:
    // if the range_sum of the input bounding box is 0 (single point) and
    // the point is not equal to the point defining the box then the cut must
    // be nontrivial
    pub fn random_cut_and_separation(
        bounding_box: &BoundingBox,
        factor: f64,
        point: &[f32],
    ) -> (Cut, bool) {
        let min_values = bounding_box.get_min_values();
        let max_values = bounding_box.get_max_values();
        let mut first_gap = point.len();
        let mut last_gap = 0;
        let mut range: f64 = min_values
            .iter()
            .zip(max_values)
            .zip(point)
            .map(|((x, y), z)| {
                if z < x {
                    (x - z) as f64
                } else if y < z {
                    (z - y) as f64
                } else {
                    0.0
                }
            })
            .sum();
        if range == 0.0 {
            return (Cut::new(usize::MAX, 0.0), false);
        }
        range += bounding_box.get_range_sum();
        range *= factor;

        let mut dim: usize = 0;
        let mut new_cut: f32 = f32::MAX;

        while dim < point.len() {
            let minv = minf32(min_values[dim], point[dim]);
            let maxv = maxf32(max_values[dim], point[dim]);

            let gap: f32 = maxv - minv;

            if gap > 0.0 {
                last_gap = dim;
                if first_gap == point.len() {
                    first_gap = dim; // will not change subsequently
                }
                let new_range = range - gap as f64;
                if new_range <= 0.0 {
                    new_cut = minv + range as f32; // precision lost here
                    if new_cut <= minv || new_cut >= maxv {
                        new_cut = minv;
                    }
                    break;
                    // this implies that gap > 0; which means that there will be no issues
                    // because either min == max both not equal to the point
                    // or rangesum of the original box is not 0.0
                }
                range = new_range;
            }
            dim += 1;
        }

        if dim != point.len() {
            let minvalue = min_values[dim];
            let maxvalue = max_values[dim];

            let separation: bool = ((point[dim] <= new_cut) && (new_cut < minvalue))
                || ((maxvalue <= new_cut) && (new_cut < point[dim]));

            if bounding_box.get_range_sum() != 0.0 || separation {
                return (Cut::new(dim.try_into().unwrap(), new_cut), separation);
            };
        };

        let mut rng = ChaCha20Rng::seed_from_u64(17);//ChaCha20Rng::from_entropy();
        let index = if rng.gen::<f32>() < 0.5 { first_gap } else { last_gap };

        let new_cut = minf32(min_values[index], point[index]);
        let separation: bool = ((point[index] == new_cut) && (new_cut < min_values[index]))
            || ((min_values[index] == new_cut) && (new_cut < point[index]));
        // note it is possible that range is positive due to max_value[index] == point[index]
        // not being the same as min_value[index]; but that is not a problematic scenario
        return (Cut::new(index.try_into().unwrap(), new_cut), separation);
    }
}

#[cfg(test)]
mod tests {
    use crate::samplerplustree::boundingbox::BoundingBox;
    use crate::samplerplustree::cut::Cut;

    #[test]
    fn test_floating_point() {
        let vec1 = vec![0.0001f32, 0.00001f32];
        let vec2 = vec![0.0001f32, 0.000011f32];

        // using unwrap in test
        let b_box= BoundingBox::new(&vec1,&vec1).unwrap();
        Cut::random_cut_and_separation(&b_box,2.0,&vec2); // exagegration
        Cut::random_cut_and_separation(&b_box,-2.0,&vec2); // exaggeration
        Cut::random_cut_and_separation(&b_box,1.0,&vec2); // should not happen
        Cut::random_cut_and_separation(&b_box,0.0,&vec2); // can happen
    }
}