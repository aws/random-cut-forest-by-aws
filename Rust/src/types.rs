/// A trait that defines a maximum value constant.
pub trait Max {
    const MAX: Self;
}

impl Max for u8 {
    const MAX: u8 = u8::MAX;
}

impl Max for u16 {
    const MAX: u16 = u16::MAX;
}

impl Max for usize {
    const MAX: usize = usize::MAX;
}

pub type PointIndex = usize;

/// The Location trait is used as a shorthand for the various traits needed by store (e.g., point
/// store, node store) locations. These are the values vended by stores to reference a stored
/// value.
pub trait Location:
    Copy + Max + std::cmp::PartialEq + TryFrom<usize> + std::marker::Send + Sync
{
}

impl Location for u8 {}
impl Location for u16 {}
impl Location for usize {}
