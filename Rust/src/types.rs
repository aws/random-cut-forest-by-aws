extern crate num_traits;
use num_traits::{Zero, One, Float};
use std::iter::Sum;

/// Floating point type for RCF representing either f32 or f64.
///
/// This type represents either a f32 or f64. In order to make the algorithm
/// generic across floats we need to specify some of the necessary traits for
/// various operations. Since this is used everywhere in the library we
/// encapsulate all of these traits into a single trait alias.
pub trait RCFFloat = Float + One + PartialOrd + Sum + Zero;