use crate::{errors::RCFError, types::Result};

pub(crate) fn add_to(a: &f64, b: &mut f64) {
    *b += *a;
}
pub(crate) fn divide(a: &mut f64, b: usize) {
    *a /= b as f64;
}

pub(crate) fn maxf32(a : f32, b:f32) -> f32 {
    if a<b {b} else {a}
}

pub(crate) fn minf32(a : f32, b:f32) -> f32 {
    if a<b {a} else {b}
}

pub(crate) fn absf32(a : f32) -> f32 {
    if a<0.0 {-a} else {a}
}

pub(crate) fn add_nbr(a: &(f64, usize, f64), b: &mut Vec<(f64, usize, f64)>) {
    b.push(*a)
}

pub(crate) fn nbr_finish(_a: &mut Vec<(f64, usize, f64)>, _b: usize) {}

/// If the test condition is false, return an InvalidArgument error with
/// the given error message. Otherwise return Ok.
pub(crate) fn check_argument(test: bool, msg: &'static str) -> Result<()> {
    if test {
        Ok(())
    } else {
        Err(RCFError::InvalidArgument { msg: msg })
    }
}
