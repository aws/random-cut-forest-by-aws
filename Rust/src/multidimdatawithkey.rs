extern crate rand;
use rand::SeedableRng;
extern crate rand_chacha;
use rand_chacha::ChaCha20Rng;
use crate::multidimdatawithkey::rand::RngCore;
use crate::multidimdatawithkey::rand::Rng;
use std::f32::consts::PI;


pub struct MultiDimDataWithKey {
    pub data : Vec<Vec<f32>>,
    pub change_indices : Vec<usize>,
    pub changes : Vec<Vec<f32>>
}

impl MultiDimDataWithKey {
    pub fn new(num: usize, period: usize, amplitude: f32, noise: f32, seed: u64,
               base_dimension: usize) -> Self {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let mut noiserng = ChaCha20Rng::seed_from_u64(seed + 1);
        let mut phase: Vec<usize> = Vec::new();
        let mut amp = Vec::new();

        for _i in 0..base_dimension {
            phase.push(rng.next_u64() as usize % period);
            let v: f32 = 0.2 * rng.gen::<f32>() * amplitude + amplitude;
            amp.push(v);
        }

        let mut data: Vec<Vec<f32>> = Vec::new();
        let mut change_indices: Vec<usize> = Vec::new();
        let mut changes: Vec<Vec<f32>> = Vec::new();

        for i in 0..num {
            let mut elem = vec![0.0; base_dimension];
            let flag = noiserng.gen::<f32>() < 0.01;
            let mut new_change = vec![0.0; base_dimension];
            let mut used: bool = false;
            for j in 0..base_dimension {
                elem[j] = amp[j] * (2.0 * PI * (i + phase[j]) as f32 / period as f32).cos()
                    + noise * noiserng.gen::<f32>();
                if flag && noiserng.gen::<f64>() < 0.3 {
                    let factor: f32 = 5.0 * (1.0 + noiserng.gen::<f32>());
                    let mut change: f32 = factor * noise;
                    if noiserng.gen::<f32>() < 0.5 {
                        change = -change;
                    }
                    elem[j] += change;
                    new_change[j] = change;
                    used = true;
                }
            }
            data.push(elem);
            if used {
                change_indices.push(i);
                changes.push(new_change);
            }
        }
        MultiDimDataWithKey {
            data,
            change_indices,
            changes
        }
    }
}