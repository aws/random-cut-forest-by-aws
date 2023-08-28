use crate::util::check_argument;
use crate::types::Result;

/**
 * A RangeVector is used when we want to track a quantity and its upper and
 * lower bounds
 */
#[repr(C)]
#[derive(Clone)]
pub struct RangeVector<T> {
    pub values: Vec<T>,
    pub upper: Vec<T>,
    pub lower: Vec<T>
}

impl RangeVector<f32> {
    pub fn new(dimensions: usize) -> Self {
        RangeVector {
            values: vec![0.0; dimensions],
            upper: vec![0.0; dimensions],
            lower: vec![0.0; dimensions]
        }
    }
}

impl RangeVector<f64> {
    pub fn new(dimensions: usize) -> Self {
        RangeVector {
            values: vec![0.0; dimensions],
            upper: vec![0.0; dimensions],
            lower: vec![0.0; dimensions]
        }
    }
}


impl<T: PartialOrd + Clone + Copy  + std::ops::AddAssign + std::ops::MulAssign> RangeVector<T> {
    pub fn from(values : Vec<T>) -> Self {
        RangeVector{
            values : values.clone(),
            upper : values.clone(),
            lower : values.clone()
        }
    }

    pub fn create(values: &[T], upper: &[T], lower:&[T]) -> Result<Self> {
        check_argument(values.len() == upper.len() && upper.len() == lower.len(), " incorrect lengths")?;
        for i in 0..values.len() {
            check_argument(values[i] <= upper[i], " incorrect upper bound")?;
            check_argument(lower[i] <= values [i], "incorrect lower bounds")?;
        }
        Ok(RangeVector{
            values :Vec::from(values),
            upper : Vec::from(upper),
            lower : Vec::from(lower)
        })
    }

    pub fn shift(&mut self, i:usize, shift: T) {
        self.values[i] += shift;
        self.upper[i] += shift;
        self.lower[i] += shift;
        // managing precision explicitly
        if self.upper[i] < self.values[i] {
            self.upper[i] = self.values[i];
        }
        if self.lower[i] > self.values[i] {
            self.lower[i] = self.values[i];
        }
    }

    pub fn cascaded_add(&mut self, base: &[T]) -> Result<()>{
        check_argument(base.len() >0 , "must be of positive length")?;
        let horizon = self.values.len()/base.len();
        check_argument(horizon * base.len() == self.values.len(), " incorrect function call")?;
        for j in 0..base.len() {
            self.shift(j,base[j]);
        }
        for i in 1..horizon {
            for j in 0..base.len() {
                self.shift(i * base.len() + j, self.values[(i-1)*base.len() + j]);
            }
        }
        Ok(())
    }

    pub fn scale(&mut self, i:usize, scale: T) {
        self.values[i] *= scale;
        self.upper[i] *= scale;
        self.lower[i] *= scale;
        // managing precision explicitly
        if self.upper[i] < self.values[i] {
            self.upper[i] = self.values[i];
        }
        if self.lower[i] > self.values[i] {
            self.lower[i] = self.values[i];
        }
    }
}
