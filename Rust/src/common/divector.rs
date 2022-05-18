use crate::samplerplustree::boundingbox::BoundingBox;

#[repr(C)]
#[derive(Clone)]
pub struct DiVector {
    pub high: Vec<f64>,
    pub low: Vec<f64>,
}

impl DiVector {
    pub fn empty(dimension: usize)  -> Self {
        DiVector{
            high: vec![0.0;dimension],
            low: vec![0.0;dimension]
        }
    }

    pub fn new(high : &[f64], low :&[f64]) -> Self{
        assert!(high.len() == low.len(), " incorrect lengths");
        DiVector{
            high : Vec::from(high),
            low : Vec::from(low)
        }
    }

    pub fn assign_as_probability_of_cut(&mut self, bounding_box: &BoundingBox, point: &[f32]) {
        let minsum: f64 = self.low
            .iter_mut()
            .zip(bounding_box.get_min_values())
            .zip(point)
            .map(|((x, &y), &z)| if y - z > 0.0 {
                *x = (y - z) as f64;
                *x
            } else {
                *x = 0.0;
                *x
            })
            .sum();
        let maxsum: f64 = self.high
            .iter_mut()
            .zip(point)
            .zip(bounding_box.get_max_values())
            .map(|((x, &y), &z)| if y - z > 0.0 {
                *x = (y - z) as f64;
                *x
            } else {
                *x = 0.0;
                *x
            })
            .sum();

        let sum = minsum + maxsum;
        if sum != 0.0 {
            self.scale(1.0/(bounding_box.get_range_sum() + sum));
        }
    }

    pub fn assign_as_probability_of_cut_with_missing_coordinates(
        &mut self,
        bounding_box:&BoundingBox,
        point: &[f32],
        missing_coordinates: &[bool]
    ) {
        let minsum: f64 = self.low
            .iter_mut()
            .zip(bounding_box.get_min_values())
            .zip(point)
            .zip(missing_coordinates)
            .map(|(((x, &y), &z), &b)| if !b && y - z > 0.0 {
                *x = (y - z) as f64;
                *x
            } else {
                *x = 0.0;
                *x
            })
            .sum();
        let maxsum: f64 = self.high
            .iter_mut()
            .zip(point)
            .zip(bounding_box.get_max_values())
            .zip(missing_coordinates)
            .map(|(((x, &y), &z), &b)| if !b && y - z > 0.0 {
                *x = (y - z) as f64;
                *x
            } else {
                *x = 0.0;
                *x
            })
            .sum();

        let sum = minsum + maxsum;
        if sum != 0.0 {
            self.scale(1.0/(bounding_box.get_range_sum() + sum));
        }
    }


    pub fn assign(&mut self, other :&DiVector){
        for (x, &y) in self.high.iter_mut().zip(&other.high) {
            *x = y;
        }
        for (x, &y) in self.low.iter_mut().zip(&other.low) {
            *x = y;
        }
    }

    pub fn add_from(&mut self, other: &DiVector, factor : f64) {
        other.add_to_scaled(self,factor);
    }

    pub fn add_to(&self, other: &mut DiVector) {

        for (x, &y) in other.high.iter_mut().zip(&self.high) {
            *x += y;
        }
        for (x, &y) in other.low.iter_mut().zip(&self.low) {
            *x += y;
        }

    }

    pub fn add_to_scaled(&self, other: &mut DiVector, factor : f64) {

        for (x, &y) in other.high.iter_mut().zip(&self.high) {
            *x += y * factor;
        }
        for (x, &y) in other.low.iter_mut().zip(&self.low) {
            *x += y * factor;
        }

    }

    pub fn divide(&mut self, num: usize){
        self.scale(1.0/num as f64)
    }


    pub fn scale(&mut self, factor : f64) {
        for x in self.high.iter_mut() {
            *x *= factor;
        }
        for x in self.low.iter_mut() {
            *x *= factor;
        }
    }

    pub fn total(&self) -> f64 {
        self.high.iter().sum::<f64>() + self.low.iter().sum::<f64>()
    }

    pub fn normalize(&mut self, value : f64) {
        let current = self.total();
        if current <= 0.0 {
            let v = value/(2.0 * self.high.len() as f64);
            for x in self.high.iter_mut() {
                *x = v;
            }
            for x in self.low.iter_mut() {
                *x = v;
            }
        } else {
            self.scale(value/current);
        }
    }

    pub fn dimensions(&self) -> usize {
        self.high.len()
    }

    pub fn high_low_sum(&self, index: usize) -> f64 {
        self.high[index] + self.low[index]
    }

}