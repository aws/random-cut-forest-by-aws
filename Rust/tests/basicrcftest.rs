extern crate rand;
extern crate rand_chacha;
extern crate rcflib;

use num::abs;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rcflib::{
    common::multidimdatawithkey::MultiDimDataWithKey,
    rcf::{create_rcf, RCF},
    visitor::visitor::VisitorInfo,
};

/// try cargo test --release
/// these tests are designed to be longish

#[test]
fn two_distribution_test_static() {
    let data_size = 1000;
    let dimensions = 20;
    let yard_stick = 5.0;
    let mut vec1 = vec![0.0f32; dimensions];
    let mut vec2 = vec![0.0f32; dimensions];
    vec1[0] = yard_stick;
    vec2[0] = -yard_stick;
    let scale = vec![vec![0.1f32; dimensions], vec![0.1f32; dimensions]];
    let mean = vec![vec1.clone(), vec2.clone()].clone();
    let data_with_key =
        MultiDimDataWithKey::mixture(data_size, &mean, &scale, &vec![0.5f32, 0.5f32], 0);

    let shingle_size = 1;
    let number_of_trees = 50;
    let capacity = 256;
    let initial_accept_fraction = 0.1;
    let _point_store_capacity = capacity * number_of_trees + 1;
    let time_decay = 0.1 / capacity as f64;
    let bounding_box_cache_fraction = 1.0;
    let random_seed = 17;
    let parallel_enabled: bool = true;
    let store_attributes: bool = false;
    let internal_shingling: bool = false;
    let internal_rotation = false;

    let mut forest: Box<dyn RCF> = create_rcf(
        dimensions,
        shingle_size,
        capacity,
        number_of_trees,
        random_seed,
        store_attributes,
        parallel_enabled,
        internal_shingling,
        internal_rotation,
        time_decay,
        initial_accept_fraction,
        bounding_box_cache_fraction,
    );

    let mut another_forest: Box<dyn RCF> = create_rcf(
        dimensions,
        shingle_size,
        capacity * 2,
        number_of_trees,
        random_seed,
        store_attributes,
        parallel_enabled,
        internal_shingling,
        internal_rotation,
        time_decay,
        initial_accept_fraction,
        bounding_box_cache_fraction,
    );

    for i in 0..data_with_key.data.len() {
        forest.update(&data_with_key.data[i], 0);
        another_forest.update(&data_with_key.data[i], 0);
    }

    let anomaly = vec![0.0f32; dimensions];

    assert!(forest.score(&anomaly) > 1.5);
    assert!(forest.displacement_score(&anomaly) * f64::log2(capacity as f64) > 1.5);
    let interpolant = forest.interpolation_visitor_traversal(&anomaly, &VisitorInfo::default());
    let attribution = forest.attribution(&anomaly);
    assert!(attribution.high[0] > 0.75);
    assert!(attribution.low[0] > 0.75);
    for i in 1..dimensions {
        assert!(attribution.high[i] < 0.1);
        assert!(attribution.low[i] < 0.1);
        assert!(abs(attribution.low[i] - interpolant.measure.low[i]) < 1e-6);
        assert!(abs(attribution.high[i] - interpolant.measure.high[i]) < 1e-6);
    }
    assert!(abs(attribution.low[0] - interpolant.measure.low[0]) < 1e-6);
    assert!(abs(attribution.high[0] - interpolant.measure.high[0]) < 1e-6);

    // a three signma radius
    assert!(
        abs(interpolant.distance.high[0] - yard_stick as f64 * interpolant.probability_mass.high[0])
            < 0.3
    );
    assert!(
        abs(interpolant.distance.low[0] - yard_stick as f64 * interpolant.probability_mass.low[0])
            < 0.3
    );
    assert!(interpolant.distance.high[1] < 0.1);
    assert!(interpolant.distance.low[1] < 0.1);
    assert!(interpolant.probability_mass.high[1] < 0.1);
    assert!(interpolant.probability_mass.low[1] < 0.1);
    assert!(interpolant.probability_mass.high[0] > 0.4);
    assert!(interpolant.probability_mass.low[0] > 0.4);
    let score = forest.score(&anomaly);

    assert!(abs(score - attribution.total()) < 1e-6);
    // score is calibrated for clear cut anomalies, even if sample size doubles ...
    assert!(abs(score - another_forest.score(&anomaly)) < 0.1 * score);

    // scores of non-anomalies are not calibrated to be he same but
    // are below 1 and should be close

    assert!(abs(forest.score(&vec1) - another_forest.score(&vec1)) < 0.1);
    assert!(abs(forest.score(&vec2) - another_forest.score(&vec2)) < 0.1);
    assert!(forest.score(&vec1) < 0.8);
    assert!(forest.score(&vec2) < 0.8);

    let displacement_score = forest.displacement_score(&anomaly);
    // displacement is calibrated for clear cut anomalies
    // samplesize did not matter for such
    assert!(
        abs(displacement_score - another_forest.displacement_score(&anomaly))
            < 0.1 * displacement_score
    );

    // displacement is NOT the same for dense regions; larger samplesize
    // leads to lower score; in fact the gap is close to the ratio of samplesize
    // due to normalization
    assert!(forest.displacement_score(&vec1) > 1.5 * another_forest.displacement_score(&vec1));
    assert!(forest.displacement_score(&vec2) > 1.5 * another_forest.displacement_score(&vec2));

    // multiplied by log_2 ; the displacement score is in the same numeric
    // range [0..log_2(sample size) as the regular score
    assert!(displacement_score * f64::log2(capacity as f64) > 2.0);

    // in contrast to displacement, density is calibrated at the dense points
    assert!(
        abs(forest.density(&vec1) - another_forest.density(&vec1)) < 0.1 * forest.density(&vec1)
    );
    assert!(
        abs(forest.density(&vec2) - another_forest.density(&vec2)) < 0.1 * forest.density(&vec2)
    );
    // and much more than at anomalous points
    assert!(forest.density(&vec1) > capacity as f64 * forest.density(&anomaly));
    assert!(forest.density(&vec2) > capacity as f64 * forest.density(&anomaly));

    // but now, unlike displacement, the  calibration is awry at potential anomalies
    // and moreover is in the other direction; larger samplesize gives larger densities because
    // of spurious points coming closer. This is a core intuition of observations/observability; the
    // calibration of central tendency (often used in forecast, also densities of dense regions)
    // has different requirements compared to callibration at extremeties (anomalies, sparse regions)
    assert!(another_forest.density(&anomaly) > 1.5 * forest.density(&anomaly));
}
