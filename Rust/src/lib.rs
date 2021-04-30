//! A Rust implementation of robust random cut forests, a model-free algorithm
//! for sketching data streams.
//!
//! Random cut forests are an unsupervised algorithm for learning sketches of
//! dynamic data streams. One common use of random cut forests is the discovery
//! of anomalous data. The goal of this package is to provide a high-performance
//! implementation in Rust as well as to provide a Python bindings backend for
//! ease of experimentation in scientific work.
//!
//! ```ignore
//! use random_cut_forest::{RandomCutForest, RandomCutForestBuilder};
//!
//! // build a random cut forest. the dimension is the only required parameter
//! let mut rcf: RandomCutForest<f32> = RandomCutForestBuilder::new(2)
//!     .sample_size(256)    // # of samples per tree
//!     .num_trees(50)       // # of trees in the model
//!     .build();            // build forest from configuration
//!
//! // train the model on a collection of vectors
//! let data: Vec<Vec<f32>>;
//! for point in data.iter() {
//!     rcf.update(point.clone());
//! }
//!
//! // compute anomaly scores using the trained model
//! let anomaly_scores: Vec<f32> = data.iter()
//!     .map(|p| rcf.anomaly_score(p))
//!     .collect();
//! ```
//!
//! ### References
//!
//! Sudipto Guha, Nina Mishra, Gourav Roy, and Okke Schrijvers. *"Robust random
//! cut forest based anomaly detection on streams."* International Conference
//! on Machine Learning, pp. 2712-2721. PMLR, 2016. ()

mod store;
pub use store::{NodeStore, PointStore};

pub mod tree;
pub use tree::{BoundingBox, Cut, Internal, Leaf, Node, NodeTraverser, Tree};