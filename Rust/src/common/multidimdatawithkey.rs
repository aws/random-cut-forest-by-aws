extern crate rand;

extern crate rand_chacha;
use std::f32::consts::PI;

use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand_core::RngCore;

use crate::rand::Rng;

pub struct MultiDimDataWithKey {
    pub data: Vec<Vec<f32>>,
    pub change_indices: Vec<usize>,
    pub labels: Vec<usize>,
    pub changes: Vec<Vec<f32>>,
}

impl MultiDimDataWithKey {
    pub fn multi_cosine(
        num: usize,
        period: &[usize],
        amplitude: &[f32],
        noise: f32,
        seed: u64,
        base_dimension: usize,
    ) -> Self {
        assert!(
            period.len() == base_dimension,
            " need a period for each dimension "
        );
        assert!(
            amplitude.len() == base_dimension,
            " need an amplitude for each dimension"
        );
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let mut noiserng = ChaCha20Rng::seed_from_u64(seed + 1);
        let mut phase: Vec<usize> = Vec::new();

        for i in 0..base_dimension {
            phase.push(rng.next_u64() as usize % period[i]);
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
                elem[j] = amplitude[j]
                    * (2.0 * PI * (i + phase[j]) as f32 / period[j] as f32).cos()
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
            labels: Vec::new(),
            changes,
        }
    }

    pub fn mixture(
        num: usize,
        mean: &[Vec<f32>],
        scale: &[Vec<f32>],
        weight: &[f32],
        seed: u64,
    ) -> Self {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        assert!(num > 0, " number of elements cannot be 0");
        assert!(mean.len() > 0, " cannot be null");
        let base_dimension = mean[0].len();
        assert!(
            mean.len() == scale.len(),
            " need scales and means to be 1-1"
        );
        assert!(
            weight.len() == mean.len(),
            " need weights and means to be 1-1"
        );
        for i in 0..mean.len() {
            assert!(
                mean[i].len() == base_dimension,
                " must have the same dimensions"
            );
            assert!(
                scale[i].len() == base_dimension,
                "sclaes must have the same dimension as the mean"
            );
            assert!(weight[i] >= 0.0, " weights cannot be negative");
        }
        let sum: f32 = weight.iter().sum();

        let mut data = Vec::new();
        let mut labels = Vec::new();
        for _j in 0..num {
            let mut i = 0;
            let mut wt: f32 = sum * rng.gen::<f32>();
            while wt > weight[i] {
                wt -= weight[i];
                i += 1;
            }
            data.push(new_vec(&mean[i], &scale[i], &mut rng));
            labels.push(i);
        }

        MultiDimDataWithKey {
            data,
            labels,
            change_indices: vec![],
            changes: vec![],
        }
    }
}

fn next_element(mean: f32, scale: f32, rng: &mut ChaCha20Rng) -> f32 {
    let mut r: f32 = f64::sqrt(-2.0f64 * f64::ln(rng.gen::<f64>())) as f32;
    // the following is to discard inf being returned from ln()
    while r.is_infinite() {
        r = f64::sqrt(-2.0f64 * f64::ln(rng.gen::<f64>())) as f32;
    }

    let switch: f32 = rng.gen();
    if 0.5 < switch {
        mean + scale * r * f32::cos(2.0 * PI * rng.gen::<f32>())
    } else {
        mean + scale * r * f32::sin(2.0 * PI * rng.gen::<f32>())
    }
}

fn new_vec(mean: &[f32], scale: &[f32], rng: &mut ChaCha20Rng) -> Vec<f32> {
    let dimensions = mean.len();
    let mut answer = Vec::new();
    for i in 0..dimensions {
        answer.push(next_element(mean[i], scale[i], rng));
    }
    answer
}
