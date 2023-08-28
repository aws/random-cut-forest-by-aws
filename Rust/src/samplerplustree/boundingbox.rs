use crate::util::check_argument;
use crate::types::Result;

#[repr(C)]
#[derive(Clone)]
pub struct BoundingBox {
    range_sum: f64,
    min_values: Vec<f32>,
    max_values: Vec<f32>,
}

impl BoundingBox {
    pub fn new(first_values: &[f32], second_values: &[f32]) -> Result<Self> {
        check_argument(first_values.len() == second_values.len(), " mismatched lengths")?;
        let minv: Vec<f32> = first_values
            .iter()
            .zip(second_values)
            .map(|(x, y)| if *x < *y { *x } else { *y })
            .collect();
        let maxv: Vec<f32> = first_values
            .iter()
            .zip(second_values)
            .map(|(x, y)| if *x > *y { *x } else { *y })
            .collect();

        let sum = minv.iter().zip(&maxv).map(|(x, y)| (y - x) as f64).sum();
        Ok(BoundingBox {
            min_values: minv,
            max_values: maxv,
            range_sum: sum,
        })
    }

    pub fn check_contains_and_add_point(&mut self, values: &[f32]) -> bool {
        self.add_two_arrays(values, values)
    }

    pub fn add_box(&mut self, x: &BoundingBox) {
        self.add_two_arrays(x.get_min_values(), x.get_max_values());
    }

    fn add_two_arrays(&mut self, minvalues: &[f32], maxvalues: &[f32]) -> bool {
        let old_sum = self.range_sum;

        for (x, y) in self.min_values.iter_mut().zip(minvalues) {
            *x = if *x < *y { *x } else { *y };
        }
        for (x, y) in self.max_values.iter_mut().zip(maxvalues) {
            *x = if *x < *y { *y } else { *x };
        }

        self.range_sum = self
            .min_values
            .iter()
            .zip(self.get_max_values())
            .map(|(x, y)| (y - x) as f64)
            .sum();

        old_sum == self.range_sum
    }

    pub fn get_range_sum(&self) -> f64 {
        self.range_sum
    }

    pub fn get_min_values(&self) -> &[f32] {
        &self.min_values
    }

    pub fn get_max_values(&self) -> &[f32] {
        &self.max_values
    }

    pub fn probability_of_cut(&self, point: &[f32]) -> f64 {
        let minsum: f32 = self
            .min_values
            .iter()
            .zip(point)
            .map(|(&x, &y)| if x - y > 0.0 { x - y } else { 0.0 })
            .sum();
        let maxsum: f32 = point
            .iter()
            .zip(self.get_max_values())
            .map(|(&x, &y)| if x - y > 0.0 { x - y } else { 0.0 })
            .sum();
        let sum = maxsum + minsum;

        if sum == 0.0 {
            return 0.0;
        } else if self.range_sum == 0.0 {
            return 1.0;
        }
        (sum as f64) / (self.range_sum + sum as f64)
    }

    pub fn probability_of_cut_with_missing_coordinates(
        &self,
        point: &[f32],
        missing_coordinates: &[bool],
    ) -> f64 {
        let minsum: f32 = self
            .min_values
            .iter()
            .zip(point)
            .zip(missing_coordinates)
            .map(|((&x, &y), &z)| if !z && x - y > 0.0 { x - y } else { 0.0 })
            .sum();
        let maxsum: f32 = point
            .iter()
            .zip(self.get_max_values())
            .zip(missing_coordinates)
            .map(|((&x, &y), &z)| if !z && x - y > 0.0 { x - y } else { 0.0 })
            .sum();
        let sum = maxsum + minsum;

        if sum == 0.0 {
            return 0.0;
        } else if self.range_sum == 0.0 {
            return 1.0;
        }
        (sum as f64) / (self.range_sum + sum as f64)
    }
}
