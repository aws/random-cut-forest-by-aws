use crate::cut::Cut;

#[repr(C)]
pub struct BoundingBox {
    range_sum: f64,
    min_values: Vec<f32>,
    max_values: Vec<f32>,
}



impl BoundingBox {
    pub fn new(first_values: &[f32], second_values: &[f32]) -> Self {
        assert!(first_values.len() == second_values.len());
        let mut minv: Vec<f32> = first_values.iter().zip(second_values).map(|(x, y)| if *x < *y { *x } else { *y }).collect();
        let mut maxv: Vec<f32> = first_values.iter().zip(second_values).map(|(x, y)| if *x > *y { *x } else { *y }).collect();


        let sum = minv.iter().zip(&maxv).map(|(x, y)| (y - x) as f64).sum();
        BoundingBox {
            min_values: minv,
            max_values: maxv,
            range_sum: sum
        }
    }

    pub fn check_contains_and_add_point(&mut self, values: &[f32]) -> bool {
        self.two_arrays(values, values)
    }

    fn two_arrays(&mut self, minvalues: &[f32], maxvalues: &[f32]) -> bool {
        let old_sum = self.range_sum;

        for (x, y) in self.min_values.iter_mut().zip(minvalues) {
            *x = if (*x < *y) { *x } else { *y };
        }
        for (x, y) in self.max_values.iter_mut().zip(maxvalues) {
            *x = if (*x < *y) { *y } else { *x };
        }

        self.range_sum = self.min_values.iter().zip(self.get_max_values()).map(|(x, y)| (y - x) as f64).sum();
        if old_sum > self.range_sum {
            panic!();
        }
        old_sum == self.range_sum
    }

    pub fn contains(&self, values: &[f32]) -> bool {
        let not_inside = self.min_values.iter().zip(values).zip(self.get_max_values()).any(|((x, y), z)|
            x > y || y > z);
        return !not_inside;
    }


    pub fn add_box_and_check_absorbs(&mut self, other_box: &BoundingBox) -> bool {
        self.two_arrays(other_box.get_min_values(), other_box.get_max_values())
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


    pub fn get_cut_and_separation(&self, factor: f64, point: &[f32], verbose:bool) -> (Cut, bool, bool) {

        //let mut range : f64 = 0.0;
        let mut range: f64 = self.min_values.iter().zip(self.get_max_values())
            .zip(point).map(|((x, y), z)|
            { if z < x { (x - z) as f64 } else if y < z { (z - y) as f64 } else { 0.0 } }).sum();
        if range == 0.0 {
            return (Cut::new(usize::MAX, 0.0), false, true);
        }
        range += self.range_sum;
        range *= factor;

        let mut dim: usize = 0;
        let mut new_cut : f32  = f32::MAX;

        while dim < point.len() {
            let minv = if point[dim] < self.min_values[dim] { point[dim] } else { self.min_values[dim] };
            let maxv = if point[dim] > self.max_values[dim] { point[dim] } else { self.max_values[dim] };

            let gap: f32 = maxv - minv;
            if gap > range as f32 {
                new_cut = minv + range as f32; // precision lost here
                if new_cut <= minv || new_cut >= maxv {
                    new_cut = minv;
                }
                break;
            }
            range = range - gap as f64;
            dim += 1;
        }

        let minvalue = self.min_values[dim];
        let maxvalue = self.max_values[dim];

        let separation: bool = ((point[dim] <= new_cut) && (new_cut < minvalue)) ||
            ((maxvalue <= new_cut) && (new_cut < point[dim]));
        (Cut::new(dim.try_into().unwrap(), new_cut), separation, false)
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
        let minsum: f32 = self.min_values.iter().zip(point).map(|(&x, &y)| { if x - y > 0.0 { x - y } else { 0.0 } }).sum();
        let maxsum: f32 = point.iter().zip(self.get_max_values()).map(|(&x, &y)| { if x - y > 0.0 { x - y } else { 0.0 } }).sum();
        let sum = maxsum + minsum;

        if sum == 0.0 {
            return 0.0;
        } else if self.range_sum == 0.0 {
            return 1.0;
        }
        (sum as f64) / (self.range_sum + sum as f64)
    }

    pub fn probability_of_cut_di_vector(&self, point: &[f32]) -> Vec<f32> {
        let mut answer = vec![0.0;2*point.len()];
        answer[0..point.len()].iter_mut().zip(&self.min_values).zip(point).map(|((x, &y), &z)| { *x = if y - z > 0.0 { y - z } else { 0.0 };});
        answer[point.len()..2*point.len()].iter_mut().zip(point).zip(&self.max_values).map(|((x, &y), &z)| { *x = if y - z > 0.0 { y - z } else { 0.0 };});
        let sum :f32 = answer.iter().sum();

        if sum != 0.0 {
            if self.range_sum == 0.0 {
                answer.resize(2 * point.len(), 1.0);
            } else {
                let newsum = self.range_sum + sum as f64;
                answer.iter_mut().map(|x| { *x = ((*x as f64)/ newsum) as f32;});
            }
        }
        return answer;
    }
}
