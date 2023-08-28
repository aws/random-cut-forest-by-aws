use crate::util::check_argument;
use crate::types::Result;

/**
 * This class maintains a simple discounted statistics. Setters are avoided
 * except for discount rate which is useful as initialization from raw scores
 */
#[repr(C)]
#[derive(Clone)]
pub struct Deviation {
    pub discount: f64,
    pub weight: f64,
    pub sum_squared:f64,
    pub sum: f64,
    pub count: i32
}

impl Deviation {
    pub fn new(discount: f64) -> Result<Self> {
        check_argument(discount>=0.0 && discount < 1.0, "incorrect discount value")?;
        Ok(Deviation {
            discount,
            weight: 0.0,
            sum:0.0,
            sum_squared:0.0,
            count:0
        })
    }

    pub fn default() -> Self {
        Deviation {
            discount: 0.0,
            weight: 0.0,
            sum:0.0,
            sum_squared:0.0,
            count:0
        }
    }

    pub fn create(discount:f64,weight:f64,sum:f64,sum_squared:f64,count:i32) -> Self{
        Deviation{
            discount,
            weight,
            sum,
            sum_squared,
            count
        }
    }

    pub fn reset(&mut self) {
        self.weight = 0.0;
        self.count = 0;
        self.sum = 0.0;
        self.sum_squared = 0.0;
    }

    pub fn mean(&self) -> f64 {
        if self.is_empty() {
            0.0
        } else {
            self.sum / self.weight
        }
    }

    pub fn update(&mut self, score: f64) {
        let factor = if self.discount == 0.0 {1.0} else {
            let a = 1.0 - self.discount;
            let b= 1.0 - 1.0 / (self.count + 2) as f64;
            if a<b {a} else {b}
        };
        self.sum = self.sum * factor + score;
        self.sum_squared = self.sum_squared * factor + score * score;
        self.weight = self.weight * factor + 1.0;
        self.count += 1;
    }

    pub fn deviation(&self) -> f64{
        if self.is_empty() {
            return 0.0;
        }
        let temp = self.sum / self.weight;
        let answer = self.sum_squared / self.weight - temp * temp;
        if answer > 0.0 {
            f64::sqrt(answer)
        } else {
            0.0
        }
    }

    pub fn is_empty(&self) -> bool{
        self.weight <= 0.0
    }

    pub fn discount(&self) -> f64 {
        self.discount
    }

    pub fn set_discount(&mut self, discount:f64) {
        self.discount = discount;
    }

    pub fn sum(&self) -> f64 {
        self.sum
    }

    pub fn sum_squared(&self) -> f64{
        self.sum_squared
    }

    pub fn weight(&self) -> f64{
        self.weight
    }

    pub fn count(&self) -> i32{
        self.count
    }

    pub fn set_count(&mut self,count:i32) {
        self.count = count;
    }
}
