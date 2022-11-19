
use crate::common::anomalydescriptor::AnomalyDescriptor;
use crate::common::rangevector::RangeVector;
use crate::rcf::{create_rcf, RCF};
use crate::types::Result;
use crate::trcf::predictorcorrector::PredictorCorrector;

pub struct BasicTRCF {
    dimensions: usize,
    capacity: usize,
    number_of_trees: usize,
    rcf : Box<dyn RCF>,
    predictor_corrector: PredictorCorrector,
    shingle_size: usize,
    internal_timestamp : usize,
    bounding_box_cache_fraction: f64,
    last_anomaly_descriptor : AnomalyDescriptor,
}


impl BasicTRCF {
    pub fn new(
        dimensions: usize,
        shingle_size: usize,
        capacity: usize,
        number_of_trees: usize,
        random_seed: u64,
        parallel_enabled: bool,
        time_decay: f64,
        anomaly_rate: f32,
        initial_accept_fraction: f64,
        bounding_box_cache_fraction: f64,
    ) -> Self {
        BasicTRCF{
            dimensions,
            capacity,
            number_of_trees,
            rcf: create_rcf(dimensions,shingle_size,capacity,number_of_trees,random_seed,false,parallel_enabled,true,false,time_decay,initial_accept_fraction,bounding_box_cache_fraction),
            predictor_corrector : PredictorCorrector::new(anomaly_rate),
            shingle_size,
            bounding_box_cache_fraction,
            internal_timestamp : 0,
            last_anomaly_descriptor: AnomalyDescriptor::new(&Vec::new(),0),
        }
    }

    pub fn process(&mut self, point: &[f32], timestamp: usize) -> Result<AnomalyDescriptor>{
        let mut result = AnomalyDescriptor::new(point,timestamp);
        result.internal_timestamp = self.internal_timestamp;
        self.internal_timestamp += 1;
        result.rcf_point = Some(self.rcf.shingled_point(point));
        result.forecast_reasonable = self.rcf.entries_seen() > 200;
        self.predictor_corrector.detect_and_modify(&mut result,&self.last_anomaly_descriptor,self.shingle_size, &self.rcf,false,false);
        if result.anomaly_grade > 0.0 {
            if let Some(x) = &result.expected_rcf_point {
                let base_dimension = self.dimensions / self.shingle_size;
                let start: usize = (self.shingle_size as i32 + result.relative_index.unwrap() - 1) as usize * base_dimension;
                result.expected_values_list = Some(vec![Vec::from(&x[start..start+base_dimension])]);
                result.past_values = Some(Vec::from(&result.rcf_point.as_ref().unwrap()[start..start+base_dimension]));
                result.likelihood_of_values=Some(vec![1.0f32]);
            }
            self.last_anomaly_descriptor = result.clone();
        }

        self.rcf.update(point,timestamp as u64)?;
        Ok(result)
    }

    pub fn extrapolate(&self, look_ahead: usize) -> Result<RangeVector> {
        self.rcf.extrapolate(look_ahead)
    }


}



