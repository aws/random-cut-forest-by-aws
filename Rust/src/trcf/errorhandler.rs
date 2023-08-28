
use crate::common::descriptor::{Descriptor, ErrorInformation};
use crate::common::divector::DiVector;
use crate::common::rangevector::RangeVector;
use crate::trcf::types::Calibration;
use crate::util::{check_argument, maxf32, minf32};
use crate::types::{Result};

#[repr(C)]
#[derive(Clone)]
pub struct ErrorHandler {
    add_error : bool,
    input_length: usize,
    sequence_index: usize,
    percentile: f32,
    forecast_horizon: usize,
    error_horizon: usize,
    past_forecasts: Vec<RangeVector<f32>>,
    actuals: Vec<Vec<f32>>,
    error_distribution: RangeVector<f32>,
    error_rmse: DiVector,
    error_mean: Vec<f32>,
    interval_precision: Vec<f32>,
    last_deviations: Vec<f32>
}

impl ErrorHandler {
    pub fn new(add_error: bool, input_length: usize, forecast_horizon: usize, error_horizon: usize, percentile: f32) -> Self {
        ErrorHandler {
            add_error,
            input_length,
            sequence_index: 0,
            percentile,
            forecast_horizon,
            error_horizon,
            past_forecasts: Vec::new(),
            actuals: Vec::new(),
            error_distribution: RangeVector::<f32>::new(input_length*forecast_horizon),
            error_rmse: DiVector::empty(input_length*forecast_horizon),
            error_mean: vec![0.0;input_length*forecast_horizon],
            interval_precision: vec![0.0;input_length*forecast_horizon],
            last_deviations: vec![0.0;input_length]
        }
    }

    pub fn update_actuals(&mut self, input : &[f32], deviations : &[f32]) -> Result<()>{
        let array_length = self.past_forecasts.len();
        let input_length = input.len();
        check_argument(self.input_length == input_length, "incorrect input")?;
        if self.sequence_index > 0 {
            let input_index = (self.sequence_index + array_length - 1) % array_length;
            if self.sequence_index < array_length + 1 {
                check_argument(self.actuals.len() == input_index, "incorrect accounting")?;
                self.actuals.push(Vec::from(input));
            } else {
                for (x,y) in self.actuals[input_index].iter_mut().zip(input) {
                    *x = *y;
                }
            }
        }

        self.sequence_index += 1;
        self.recompute_errors()?;
        for (x,y) in self.last_deviations.iter_mut().zip(deviations){
            *x = *y;
        };
        Ok(())
    }

    pub fn augment_descriptor(&self,descriptor : &mut Descriptor) {
        descriptor.error_information = Some(
            ErrorInformation{
                interval_precision: self.interval_precision.clone(),
                error_distribution: self.error_distribution.clone(),
                error_rmse: self.error_rmse.clone(),
                error_mean: self.error_mean.clone()
            }
        )
    }

    pub fn update_forecasts(&mut self, range_vector: &RangeVector<f32>) -> Result<()> {
        check_argument(range_vector.values.len() == self.input_length *self.forecast_horizon, "incorrect input")?;
        let array_length = self.past_forecasts.len();
        let stored_forecast_index = (self.sequence_index + array_length - 1) % (array_length);
        if stored_forecast_index < array_length + 1 {
            check_argument(self.past_forecasts.len() == stored_forecast_index, "incorrect accounting")?;
            self.past_forecasts.push(range_vector.clone());
        } else {
            for (x,y) in self.past_forecasts[stored_forecast_index].values.iter_mut().zip(&range_vector.values) {
                *x = *y;
            }
            for (x,y) in self.past_forecasts[stored_forecast_index].lower.iter_mut().zip(&range_vector.lower) {
                *x = *y;
            }
            for (x,y) in self.past_forecasts[stored_forecast_index].upper.iter_mut().zip(&range_vector.upper) {
                *x = *y;
            }
        };
        Ok(())
    }


    fn length(sequence_index : usize, error_horizon : usize, index : usize) -> usize {
        if sequence_index > error_horizon + index + 1 {error_horizon}
        else  if sequence_index < index + 1 {0}
        else {sequence_index - index - 1}
    }

    fn recompute_errors(&mut self) -> Result<()>{
        let array_length = self.past_forecasts.len();
        let input_index = (self.sequence_index - 2 + array_length) % array_length;
        let mut median_error = vec![0.0f32;self.error_horizon];

        for x in self.interval_precision.iter_mut(){
            *x =0.0;
        }

        for i in 0..self.forecast_horizon {
            let len = Self::length(self.sequence_index, self.error_horizon, i);
            for j in 0..self.input_length {
                let pos = i * self.input_length + j;
                if len > 0 {
                    let mut positive_sum = 0.0f64;
                    let mut positive_count = 0;
                    let mut negative_sum = 0.0f64;
                    let mut positive_sq_sum = 0.0f64;
                    let mut negative_sq_sum = 0.0f64;
                    for k in 0..len  {
                        let past_index = (input_index - i - k + array_length) % array_length;
                        let index = (input_index - k + array_length) % array_length;
                        let error = (self.actuals[index][j] - self.past_forecasts[past_index].values[pos]) as f64;
                        median_error[k] = error as f32;
                        let within = self.past_forecasts[past_index].upper[pos] >= self.actuals[index][j]
                            && self.actuals[index][j] >= self.past_forecasts[past_index].lower[pos];
                        self.interval_precision[pos] += if within { 1.0 } else { 0.0 };

                        if error >= 0.0 {
                            positive_sum += error;
                            positive_sq_sum += error * error;
                            positive_count += 1;
                        } else {
                            negative_sum += error;
                            negative_sq_sum += error * error;
                        }
                    }
                    self.error_mean[pos] = (positive_sum + negative_sum) as f32 / len as f32;
                    self.error_rmse.high[pos] = if positive_count == 0 { 0.0 } else {
                        f64::sqrt(positive_sq_sum / positive_count as f64)
                    };
                    self.error_rmse.low[pos] = if positive_count == len { 0.0 } else {
                        - f64::sqrt(negative_sq_sum / (len - positive_count) as f64)
                    };

                    if len as f32 * self.percentile >= 1.0 {
                        median_error[0..(len as usize)].sort_by(|o1, o2| o1.partial_cmp(&o2).unwrap());
                        self.error_distribution.values[pos] = Self::interpolated_median(&median_error, len)?;
                        self.error_distribution.upper[pos] = Self::interpolated_upper_rank(&median_error, len, len as f32 * self.percentile);
                        self.error_distribution.lower[pos] = Self::interpolated_lower_rank(&median_error, len as f32 * self.percentile);
                    }
                    self.interval_precision[pos] = self.interval_precision[pos] / len as f32;
                } else {
                    self.error_mean[pos] = 0.0;
                    self.error_rmse.high[pos] = 0.0;
                    self.error_rmse.low[pos] = 0.0;
                    self.error_distribution.values[pos] = 0.0;
                    self.error_distribution.upper[pos] = 0.0;
                    self.error_distribution.lower[pos] = 0.0;
                    self.interval_precision[pos] = 0.0;
                }
            }
        };
        Ok(())
    }

    pub fn calibrate(&self,calibration : Calibration, ranges : &mut RangeVector<f32>) -> Result<()>{
        check_argument(self.input_length * self.forecast_horizon == ranges.values.len(), "mismatched lengths")?;
        for i in 0..self.forecast_horizon {
            let len = Self::length(self.sequence_index, self.error_horizon, i);
            for j in 0..self.input_length {
                let pos = i * self.input_length + j;
                if len > 0 {
                    if calibration != Calibration::NONE {
                        if len as f32 * self.percentile < 1.0 {
                            let deviation = self.last_deviations[j];
                            ranges.upper[pos] = maxf32(ranges.upper[pos], ranges.values[pos] + (1.3 * deviation));
                            ranges.lower[pos] = minf32(ranges.lower[pos], ranges.values[pos] - (1.3 * deviation));
                        } else {
                            match calibration {
                                Calibration::SIMPLE => { Self::adjust(pos, ranges, &self.error_distribution)?; },
                                Calibration::MINIMAL => { Self::adjust_minimal(pos, ranges, &self.error_distribution)?; },
                                _ => {}
                            }
                        }
                    }
                }
            }
        };
        Ok(())
    }

    fn interpolated_median(ascending_array : &[f32], len : usize) -> Result<f32>{
        check_argument(ascending_array.len() >= len, "incorrect length parameter")?;
        let lower = if len % 2 == 0 { ascending_array[len / 2 - 1] }
                else { (ascending_array[len / 2] + ascending_array[len / 2 - 1]) / 2.0 };
        let upper = if len % 2 == 0 { ascending_array[len / 2] }
                else { (ascending_array[len / 2] + ascending_array[len / 2 - 1]) / 2.0 };

        if lower <= 0.0 && 0.0 <= upper {
            return Ok(0.0);
        } else {
            return Ok((upper + lower) / 2.0);
        }
    }

    fn interpolated_lower_rank(ascending_array: &[f32], frac_rank : f32) -> f32{
        let rank = f32::floor(frac_rank) as usize;
        return ascending_array[rank - 1]
                + (frac_rank - rank as f32) * (ascending_array[rank] - ascending_array[rank - 1]);
    }

    fn interpolated_upper_rank(ascending_array : &[f32], len : usize, frac_rank : f32) -> f32 {
        let rank = f32::floor(frac_rank) as usize;
        return ascending_array[len - rank]
                + (frac_rank - rank as f32) * (ascending_array[len - rank - 1] - ascending_array[len - rank]);
    }

    fn adjust(pos: usize, range_vector : &mut RangeVector<f32>, other : &RangeVector<f32>) -> Result<()>{
        check_argument(other.values.len() == range_vector.values.len(), " mismatch in lengths")?;
        check_argument(pos < other.values.len(), " cannot be this large")?;
        range_vector.values[pos] += other.values[pos];
        range_vector.upper[pos] = maxf32(range_vector.values[pos], range_vector.upper[pos] + other.upper[pos]);
        range_vector.lower[pos] = minf32(range_vector.values[pos], range_vector.lower[pos] + other.lower[pos]);
        Ok(())
    }

   fn adjust_minimal(pos: usize, range_vector : &mut RangeVector<f32>, other : &RangeVector<f32>)  -> Result<()> {
       check_argument(other.values.len() == range_vector.values.len(), " mismatch in lengths")?;
       check_argument(pos < other.values.len(), "cannot be this large")?;
       let old_val = range_vector.values[pos];
       range_vector.values[pos] += other.values[pos];
       range_vector.upper[pos] = maxf32(range_vector.values[pos], old_val + other.upper[pos]);
       range_vector.lower[pos] = minf32(range_vector.values[pos], old_val + other.lower[pos]);
       Ok(())
   }
}
