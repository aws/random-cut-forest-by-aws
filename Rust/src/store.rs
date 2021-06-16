extern crate slab;
use slab::Slab;

use crate::Node;

/// A type for storing data points by key.
pub type PointStore<T> = Slab<Vec<T>>;

/// A type for storing nodes by key.
pub type NodeStore<T> = Slab<Node<T>>;