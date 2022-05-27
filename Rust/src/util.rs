pub(crate) fn add_to(a: &f64, b: &mut f64) {
    *b += *a;
}
pub(crate) fn divide(a: &mut f64, b: usize) {
    *a /= b as f64;
}

pub(crate) fn add_nbr(a: &(f64, usize, f64), b: &mut Vec<(f64, usize, f64)>) {
    b.push(*a)
}

pub(crate) fn nbr_finish(_a: &mut Vec<(f64, usize, f64)>, _b: usize) {}
