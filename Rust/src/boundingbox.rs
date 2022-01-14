#[repr(C)]
pub struct BoundingBox {
    range_sum: f64,
    min_values: Vec<f32>,
    max_values: Vec<f32>,
}

impl BoundingBox {
    pub fn new(first_values: &[f32], second_values: &[f32]) -> Self {
        assert!(first_values.len() == second_values.len());
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
        BoundingBox {
            min_values: minv,
            max_values: maxv,
            range_sum: sum,
        }
    }

    pub fn check_contains_and_add_point(&mut self, values: &[f32]) -> bool {
        self.two_arrays(values, values)
    }

    fn two_arrays(&mut self, minvalues: &[f32], maxvalues: &[f32]) -> bool {
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
        if old_sum > self.range_sum {
            panic!();
        }
        old_sum == self.range_sum
    }

    pub fn contains(&self, values: &[f32]) -> bool {
        let not_inside = self
            .min_values
            .iter()
            .zip(values)
            .zip(self.get_max_values())
            .any(|((x, y), z)| x > y || y > z);
        return !not_inside;
    }

    pub fn copy_from(&mut self, other_box: &BoundingBox) {
        assert!(self.min_values.len() == other_box.min_values.len());
        self.range_sum = other_box.range_sum;
        self.min_values.copy_from_slice(other_box.get_min_values());
        self.max_values.copy_from_slice(other_box.get_max_values());
    }

    pub fn get_range_sum(&self) -> f64 {
        self.range_sum
    }

    pub fn get_range(&self, dim: usize) -> f64 {
        (self.max_values[dim] - self.min_values[dim]).into()
    }

    pub fn get_min_value(&self, dim: usize) -> f32 {
        self.min_values[dim]
    }

    pub fn get_max_value(&self, dim: usize) -> f32 {
        self.max_values[dim]
    }

    pub fn get_dimensions(&self) -> usize {
        self.min_values.len()
    }

    // replaces with a point, and optional rotation
    pub fn replace_with_point(&mut self, point: &[f32]) {
        assert!(
            point.len() == self.min_values.len(),
            " incorrect box replacement"
        );
        for i in 0..point.len() {
            self.max_values[i] = point[i];
            self.min_values[i] = point[i];
        }
    }

    pub fn copy(&self) -> BoundingBox {
        BoundingBox::new(&self.min_values, &self.max_values)
    }

    pub fn get_rangesum(&self) -> f64 {
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

    pub fn probability_of_cut_di_vector(&self, point: &[f32]) -> Vec<f32> {
        let mut answer = vec![0.0; 2 * point.len()];
        answer[0..point.len()]
            .iter_mut()
            .zip(&self.min_values)
            .zip(point)
            .map(|((x, &y), &z)| {
                *x = if y - z > 0.0 { y - z } else { 0.0 };
            });
        answer[point.len()..2 * point.len()]
            .iter_mut()
            .zip(point)
            .zip(&self.max_values)
            .map(|((x, &y), &z)| {
                *x = if y - z > 0.0 { y - z } else { 0.0 };
            });
        let sum: f32 = answer.iter().sum();

        if sum != 0.0 {
            if self.range_sum == 0.0 {
                answer.resize(2 * point.len(), 1.0);
            } else {
                let newsum = self.range_sum + sum as f64;
                answer.iter_mut().map(|x| {
                    *x = ((*x as f64) / newsum) as f32;
                });
            }
        }
        return answer;
    }
}
