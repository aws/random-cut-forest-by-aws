# Random Cut Forest

This directory contains a Rust implementation of the Random Cut Forest (RCF)
data structure and algorithms for anomaly detection, denstiy estimation,
imputation, and forecast. The goal of this package is to provide a
high-performance implementation of RCF in Rust as well as the backend for the
Python bindings also contained in this repository.

## Usage

To use `random_cut_forest`, add this to your `Cargo.toml`:

```toml
[dependencies]
random_cut_forest = "0.1.0"
```

The two main types provided by this package are `RandomCutForest` and 
`RandomCutForestBuilder`. The latter creates a `RandomCutForest` using a 
combination of required and optional construction parameters.

Below is an example showing RCF construction, training, and anomaly scoring.

```rust
use random_cut_forest::{RandomCutForest, RandomCutForestBuilder};

// build a random cut forest. the dimension is the only required parameter
let mut rcf: RandomCutForest<f32> = RandomCutForestBuilder::new(2)
    .sample_size(256)    // # of samples per tree
    .num_trees(50)       // # of trees in the model
    .build();            // build forest from configuration

// train the model on a collection of vectors
for point in data.iter() {
    rcf.update(point.clone());
}

// compute anomaly scores using the trained model
let anomaly_scores: Vec<f32> = data.iter()
  .map(|p| rcf.anomaly_score(p))
  .collect();
```

## References

* Guha, Sudipto, Nina Mishra, Gourav Roy, and Okke Schrijvers. *"Robust random 
  cut forest based anomaly detection on streams."* In International conference 
  on machine learning, pp. 2712-2721. PMLR, 2016. ([pdf][rcf-paper])

[rcf-paper]: http://proceedings.mlr.press/v48/guha16.pdf