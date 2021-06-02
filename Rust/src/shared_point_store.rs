use std::collections::HashMap;

/// Trait for data structures representing shared point stores.
///
/// A shared point store only stores one copy of the point at each sequence index
/// no matter how many trees request to insert that point.
trait PointStore {
    type Point;

    /// Get a referece to a point by key.
    ///
    /// Returns `None` if a point with that key doesn't live in the point store.
    fn get(&self, key: usize) -> Option<&Self::Point>;

    /// Insert a point at a given sequence index into the point store.
    ///
    /// The key of the point in the store is returned.
    fn insert(&mut self, point: Self::Point, sequence_index: usize) -> usize;

    /// Remove a point by key from the point store.
    fn remove(&mut self, key: usize);
}

/// A shared point store that uses a hash map to store points
///
/// Keys in the point store are precisely sequence indices. This implies a
/// pre-condition that a unique point is associated with each sequence index.
/// Insertion will panic if two different points with the same sequence index
/// are inserted.
///
/// The values of the hash map are point, reference count pairs. When the
/// reference count reaches zero then the point is removed from the hash map.
struct HashMapPointStore<T> {
    map: HashMap<usize, (Vec<T>, u16)>
}

impl<T> HashMapPointStore<T> {
    pub fn new() -> Self {
        HashMapPointStore {
            map: HashMap::new(),
        }
    }
}

impl<T> PointStore for HashMapPointStore<T>
    where T: PartialEq
{
    type Point = Vec<T>;

    fn get(&self, key: usize) -> Option<&Self::Point> {
        match self.map.get(&key) {
            Some((point, _)) => Some(point),
            None => None,
        }
    }

    fn insert(&mut self, point: Self::Point, sequence_index: usize) -> usize {
        match self.map.get_mut(&sequence_index) {
            Some((stored_point, ref_count)) => {
                if point != *stored_point {
                    panic!("Attempting to insert a different point at the same sequence index.")
                }
                *ref_count += 1;
            }
            None => {
                self.map.insert(sequence_index, (point, 1));
            },
        }
        sequence_index
    }

    fn remove(&mut self, key: usize) {
        match self.map.get_mut(&key) {
            Some((_, ref_count)) => {
                if *ref_count > 1 {
                    *ref_count -= 1;
                } else {
                    self.map.remove(&key);
                }
            },
            None => (),
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    /// A generic test for any point store on Vec<f32>'s. Used to test multiple
    /// shared point store implementations.
    fn test_point_store<PS: PointStore<Point=Vec<f32>>>(store: &mut PS) {
        let key1 = store.insert(vec![0.0, 0.0], 0);
        let key2 = store.insert(vec![1.0, 1.0], 1);
        let key3 = store.insert(vec![2.0, 2.0], 2);
        let key4 = store.insert(vec![2.0, 2.0], 2);

        // check we get the points we inserted
        assert_eq!(store.get(key1).unwrap(), &vec![0.0, 0.0]);
        assert_eq!(store.get(key2).unwrap(), &vec![1.0, 1.0]);
        assert_eq!(store.get(key3).unwrap(), &vec![2.0, 2.0]);
        assert_eq!(store.get(key4).unwrap(), &vec![2.0, 2.0]);
        assert_eq!(key3, key4);

        // check that unknown keys return None
        assert!(store.get(99999999).is_none());

        // check that removed singleton vectors return None
        store.remove(key1);
        assert!(store.get(key1).is_none());

        store.remove(key2);
        assert!(store.get(key2).is_none());

        // check removal of points with higher reference count
        store.remove(key3);
        assert_eq!(store.get(key3).unwrap(), &vec![2.0, 2.0]);
        store.remove(key3);
        assert!(store.get(key3).is_none());
    }

    /// A generic test for any point store on Vec<f32>'s. Used to test multiple
    /// shared point store implementations.
    ///
    /// This tests the precondition that point insertions must have the same
    /// point for the same sequnce index. Any test that uses this function
    /// must include #[should_panic].
    fn test_point_store_invalid_insert<PS: PointStore<Point=Vec<f32>>>(store: &mut PS) {
        // panic when two different vectors are inserted with the same sequence index
        store.insert(vec![0.0, 0.0], 0);
        store.insert(vec![1.0, 1.0], 0);
    }

    #[test]
    fn test_hash_map_point_store() {
        let mut store = HashMapPointStore::<f32>::new();
        test_point_store(&mut store);
    }

    #[test]
    #[should_panic]
    fn test_hash_map_point_store_invalid_insert() {
        let mut store = HashMapPointStore::<f32>::new();
        test_point_store_invalid_insert(&mut store);
    }
}