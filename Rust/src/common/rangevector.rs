

/**
 * A RangeVector is used when we want to track a quantity and its upper and
 * lower bounds
 */
#[repr(C)]
#[derive(Clone)]
pub struct RangeVector {
    pub values: Vec<f32>,
    pub upper: Vec<f32>,
    pub lower: Vec<f32>
}

impl RangeVector {
    pub fn new(dimensions: usize) -> Self {
        RangeVector {
            values: vec![0.0; dimensions],
            upper: vec![0.0; dimensions],
            lower: vec![0.0; dimensions]
        }
    }

    pub fn from(values : Vec<f32>) -> Self {
        RangeVector{
            values : values.clone(),
            upper : values.clone(),
            lower : values.clone()
        }
    }

    pub fn create(values: &[f32], upper: &[f32], lower:&[f32]) -> Self {
        assert!(values.len() == upper.len() && upper.len() == lower.len(), " incorrect lengths");
        for i in 0..values.len() {
            assert!(values[i] <= upper[i], " incorrect upper bound at {}", i);
            assert!(lower[i] <= values [i], "incorrect lower bounds at {}",i);
        }
        RangeVector{
            values :Vec::from(values),
            upper : Vec::from(upper),
            lower : Vec::from(lower)
        }
    }

    pub fn shift(&mut self, i:usize, shift: f32) {
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

    pub fn scale(&mut self, i:usize, scale: f32) {
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
