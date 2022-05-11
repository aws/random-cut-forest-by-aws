use crate::samplerplustree::boundingbox::BoundingBox;

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

    pub fn random_cut_and_separation(
        bounding_box: &BoundingBox,
        factor: f64,
        point: &[f32],
    ) -> (Cut, bool) {
        let min_values = bounding_box.get_min_values();
        let max_values = bounding_box.get_max_values();
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
            let minv = if point[dim] < min_values[dim] {
                point[dim]
            } else {
                min_values[dim]
            };
            let maxv = if point[dim] > max_values[dim] {
                point[dim]
            } else {
                max_values[dim]
            };

            let gap: f32 = maxv - minv;
            if gap > range as f32 || (gap == range as f32 && dim == point.len()-1) {
                new_cut = minv + range as f32; // precision lost here
                if new_cut <= minv || new_cut >= maxv {
                    new_cut = minv;
                }
                break;
            }
            range = range - gap as f64;
            dim += 1;
        }

        let minvalue = min_values[dim];
        let maxvalue = max_values[dim];

        let separation: bool = ((point[dim] <= new_cut) && (new_cut < minvalue))
            || ((maxvalue <= new_cut) && (new_cut < point[dim]));
        (Cut::new(dim.try_into().unwrap(), new_cut), separation)
    }
}
