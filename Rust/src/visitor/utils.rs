extern crate num_traits;
use num_traits::{Float, One};

#[inline(always)]
pub fn score_seen<T>(depth: T, mass: u32) -> T
    where T: Float + One
{
    let one: T = One::one();
    one / (
        depth + (T::from(mass).unwrap() + one).ln()/T::from(2.0).unwrap().ln())
}

#[inline(always)]
pub fn score_unseen<T>(depth: T) -> T
    where T: Float + One
{
    let one: T = One::one();
    one/(depth + one)
}

#[inline(always)]
pub fn damp<T>(leaf_mass: u32, tree_mass: u32) -> T
    where T: Float + One
{
    let one: T = One::one();
    one - T::from(leaf_mass).unwrap()/(
        T::from(2.0).unwrap() * T::from(tree_mass).unwrap())
}

#[inline(always)]
pub fn normalize_score<T>(score: T, mass: u32) -> T
    where T: Float + One
{
    let one: T = One::one();
    score * (T::from(mass).unwrap() + one).ln()/T::from(2.0).unwrap().ln()
}