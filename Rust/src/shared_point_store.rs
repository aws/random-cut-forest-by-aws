#![allow(unused_variables)]
#![allow(dead_code)]

extern crate num_traits;
use num_traits::Float;

use std::collections::{HashMap, VecDeque};

/// Trait for data structures representing shared point stores.
///
/// A shared point store only stores one copy of the point at each sequence index
/// no matter how many trees request to insert that point.
trait PointStore<T> {
    /// Get a referece to a point by key.
    ///
    /// Returns `None` if a point with that key doesn't live in the point store.
    fn get(&self, key: usize) -> Option<&[T]>;

    /// Insert a point at a given sequence index into the point store.
    ///
    /// The key of the point in the store is returned.
    fn insert(&mut self, point: &[T], sequence_index: usize) -> usize;

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

impl<T> PointStore<T> for HashMapPointStore<T>
    where T: Float
{

    fn get(&self, key: usize) -> Option<&[T]> {
        match self.map.get(&key) {
            Some((point, _)) => Some(point),
            None => None,
        }
    }

    fn insert(&mut self, point: &[T], sequence_index: usize) -> usize {
        match self.map.get_mut(&sequence_index) {
            Some((stored_point, ref_count)) => {
                if point != *stored_point {
                    panic!("Attempting to insert a different point at the same sequence index.")
                }
                *ref_count += 1;
            }
            None => {
                // insert a copy of the 
                self.map.insert(sequence_index, (point.to_vec(), 1));
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

//////////////////////////////////////////////////////////////////////////////
/// 
/// # Shingle-Aware Point Store
/// 
/// TODO
/// - [x] should we really add Display just for panic messages?
/// - [ ] should the buffer always be contiguous?
/// - [ ] (IN PROGRESS) Indexing issues when depending on sequence index to 
///       determine vector position, especially when there are no points in
///       the store.
/// - [ ] consider using range() or range_mut() in insert_shingle()
/// - [ ] Insert needs to be &[T]. Does that mean we'll end up with two 
///       copies of a point?
///
////////////////////////////////////////////////////////////////////////////// 

/// Configuration setting for the type of data inserted into the point store.
/// 
/// The type of the insert is `Vec<T>`. However, depending on what kinds of
/// pre-processing is available, we can interpret this to be a D-dimensional
/// data point or an (S x D)-dimensional shingle.
/// 
/// That is suppose we have a D=2 dimensional data stream,
/// 
/// ```text
///     [a₀, a₁],
///     [b₀, b₁],
///     [c₀, c₁],
///     [d₀, d₁],
///     [e₀, e₁],
///     ...
/// ```
/// 
/// that we want converted to the shingles of shingle size S=3.
/// 
/// ```text
///     s₀ = [a₀, a₁, b₀, b₁, c₀, c₁],
///     s₁ = [b₀, b₁, c₀, c₁, d₀, d₁],
///     s₂ = [c₀, c₁, d₀, d₁, e₀, e₁],
///     ...
/// ```
/// 
/// To insert the next shingle s₃ we have two options:
/// 
/// 1. `ShingleAwareInsertType::Element` -- We only need to provide the new data
///    point `[f₀, f₁]`. The point store will automatically constuct the next
///    shingle, `[d₀, d₁, e₀, e₁, f₀, f₁]`, and return this shingle when `get()`
///    is called.
/// 2. `ShingleAwareInsertType::Shingle` -- The entire shingle `[d₀, d₁, e₀, e₁,
///    f₀, f₁]` needs to be inserted into point store. The redundant information
///    `d₀`, `d₁`, `e₀`, and `e₁` are used to verify the insertion.
/// 
pub enum ShingleAwareInsertType {
    Element,
    Shingle,
}

/// A point store optimized for shingled data.
///
/// Shingling is a data pre-processing step that transforms an input
/// D-dimensional data stream into a (D x S)-dimensional data stream of shingles.
/// Each shingle is a consecutive set of observations.
///
/// For example, below we have a stream of vectors of dimension two. After
/// shingling with a shingle size of three we get a stream of vectors of
/// dimension six.
///
/// ```text
///     [a₀, a₁],
///     [b₀, b₁],
///     [c₀, c₁],
///     [d₀, d₁],
///     [e₀, e₁],
///     [f₀, f₁],
///     ...
///
///     ↓ (after shingling with shingle_size=3) ↓
///
///     s₀ = [a₀, a₁, b₀, b₁, c₀, c₁],
///     s₁ = [b₀, b₁, c₀, c₁, d₀, d₁],
///     s₂ = [c₀, c₁, d₀, d₁, e₀, e₁],
///     s₃ = [d₀, d₁, e₀, e₁, f₀, f₁],
///     ...
/// ```
///
/// The main idea behind the shingle aware point store is take advantage of the
/// fact that consecutive *shingles* have high overlap. In particular, shingles
/// s₀ and s₁ share (shingle_size - 1) x dimension elements.
///
/// ```text
///           shingle s₀
///      ├────────────────────┤
///     [a₀, a₁, b₀, b₁, c₀, c₁, d₀, d₁, e₀, e₁, f₀, f₁]
///              ├────────────────────┤
///                     shingle s₁
///```
/// 
/// # Implementation Details
/// 
/// The shingle aware point store maintains a shingle `buffer` and `ref_counts`.
/// The integer reference counts track how many times a given point has been 
/// inserted into the store and, equivalently in the application to random cut
/// forests, how many trees refer to a given shingle. 
/// 
/// Specifically, for each vector element there is a reference count. The
/// diagram below shows an example with dimension 2 shingle size 3 elements.
/// Note that each dimension 2 vector has associated with it a reference count
/// counting how many shingles in the store are using that point.
/// 
/// ```text
///               shingle_size=3
///
///         dimension=2   stride=(dimension x shingle_size)
///            ├───┤           ├────────────────────┤
///     buf = [a₀, a₁, b₀, b₁, c₀, c₁, d₀, d₁, e₀, e₁, f₀, f₁]
///     ref = [r₀,     r₁,     r₂,     r₃,     r₄    , r₅    ]
///            ↑
///          shift
/// ```
/// 
/// These buffers, implemented as dequeues, are allowed to shrink when points
/// at the front are not longer referenced. In order to keep key consistency,
/// a `shift` is kept to track how many elements are removed.
/// 
struct ShingleAwarePointStore<T> {
    buffer: VecDeque<T>,
    ref_counts: VecDeque<u16>,
    dimension: usize,
    shingle_size: usize,
    shift: usize,
    insert_type: ShingleAwareInsertType,
}



impl<T> ShingleAwarePointStore<T>
    where T: Float + std::fmt::Display
{
    /// Create a new shingle-aware point store.
    ///
    /// A shingle-aware point store needs to know the dimensionality of the
    /// original data and the shingle size.
    fn new(dimension: usize, shingle_size: usize, insert_type: ShingleAwareInsertType) -> Self {
        ShingleAwarePointStore {
            buffer: VecDeque::new(),
            ref_counts: VecDeque::new(),
            dimension: dimension,
            shingle_size: shingle_size,
            shift: 0,
            insert_type: insert_type,
        }
    }

    /// Pops unreferenced elements from the tail of the shingle buffer.
    ///
    /// The first vector element in the diagram below no longer has any
    /// references. (Note that we only need to store one reference count per
    /// vector value.) Shrinking removes the elements from the back of the
    /// deque until the first element with a positive reference is encountered.
    ///
    /// After shrinking we update the buffer's `shift` to maintain consistency
    /// between buffer indexes and sequence indexes.
    ///
    /// ```text
    ///
    ///     buf = [a₀, a₁, b₀, b₁, c₀, c₁, d₀, d₁, e₀, e₁, f₀, f₁]
    ///     ref = [0,      0,      x,      0,      x,      x     ]
    ///            ↑
    ///          shift
    ///
    ///     ↓↓↓ (after shrinking) ↓↓↓
    ///
    ///                     buf = [c₀, c₁, d₀, d₁, e₀, e₁, f₀, f₁]
    ///                     ref = [x,      0,      x,      x     ]
    ///                            ↑
    ///                (shift + #elements deleted)
    /// ```
    fn shrink_buffer(&mut self) {
        while self.buffer.len() > 0 {
            match self.ref_counts.get(0) {
                None => unreachable!(),
                Some(ref_count) => {
                    // remove the reference count and the last point if the
                    // reference count is zero. update the shift
                    if *ref_count > 0 {
                        return;
                    }
                    self.ref_counts.pop_front();
                    for _ in 0..self.dimension {
                        self.buffer.pop_front();
                    }

                    // edge case: don't need to shift if you're removing the
                    // last shingle
                    if self.ref_counts.len() >= self.shingle_size {
                        self.shift += 1;
                    }
                }
            }
        }
    }

    /// Insert a shingle into the point store.
    /// 
    /// This function assumes that the data inserted into the point store is a
    /// shingle.
    /// 
    /// # Panics
    /// 
    /// Insertion panics if the newly inserted shingle is inconsisent with the
    /// data already in the point store. For example, if the shingle at sequence
    /// index `t` is `[a, b, c]` and we try to insert `[b, x, d]` at index `t+1`.
    /// This is due to the element inconsistency `x != c`.
    fn insert_shingle(&mut self, shingle: &[T], sequence_index: usize) {
        assert_eq!(
            shingle.len(), self.dimension * self.shingle_size,
            "Inconsistent dimension of input point with point store. (Expected \
            = {}, point length = {}", self.dimension * self.shingle_size, shingle.len());
        assert!(
            sequence_index >= self.shift,
            "Cannot insert a new shingle in the past.");

        // determine the index where the new point will be added in the ref
        // counts vector. if the index exceeds the current capacity then append
        // the new values.
        let start = sequence_index - self.shift;
        let end = start + self.shingle_size;
        let mut len = self.ref_counts.len();

        // if the sequence index puts the start of the new point beyond the 
        // bounds of the current buffer then extend the buffer to the start.
        // recompute the length
        if start > len {
            for i in len..start {
                self.ref_counts.push_back(0);
                for k in 0..self.dimension {
                    self.buffer.push_back(T::infinity());
                }
            }
            len = self.ref_counts.len();
        }

        if end > len {
            // before doing the actual insertion, verify that the elements
            // that are already in the point store match the new shingle. 
            for i in start..len {                                                                                                                                                                                                                                                                                                  
                let j = i - start;  // element index in shingle
                for k in 0..self.dimension {
                    let shingle_elt = shingle.get(j*self.dimension + k).unwrap();
                    let buffer_elt = self.buffer.get(i*self.dimension + k).unwrap();
                    assert!(shingle_elt == buffer_elt, "Inconsistent shingle data")
                }
            }
    
            // insert the new elements from the input shingle
            for i in len..end {
                self.ref_counts.push_back(0);
                let j = i - start;
                for k in 0..self.dimension {
                    let element_k = shingle.get(j*self.dimension + k).unwrap();
                    self.buffer.push_back(*element_k);
                }
            }
        }

        // either the new point/shingle already lives in the store or we just
        // allocated space for it. increment the reference counts
        for i in start..end {
            *self.ref_counts.get_mut(i).unwrap() += 1;
        }
    }
}


impl<T> PointStore<T> for ShingleAwarePointStore<T>
    where T: Float + std::fmt::Display
{
    /// Get a referece to a point by key.
    ///
    /// Returns `None` if a point with that key doesn't live in the point store.
    fn get(&self, key: usize) -> Option<&[T]> {
        let idx = key - self.shift;
        let ref_count = match self.ref_counts.get(idx) {
            None => 0,
            Some(count) => *count,
        };

        if ref_count == 0 {
            return None;
        }

        let start = (key - self.shift) * self.dimension;
        let end = start + self.dimension * self.shingle_size;
        let slice = &self.buffer.as_slices().0[start..end];
        return Some(slice);
    }

    /// Insert a point at a given sequence index into the point store.
    ///
    /// Because shingling is a pre-processing step, we assume in the shingle-aware
    /// point store that the inserted value is a shingle of size
    /// `dimension x shingle_size`.
    ///
    /// The key of the point in the store is returned.
    fn insert(&mut self, value: &[T], sequence_index: usize) -> usize {
        match self.insert_type {
            ShingleAwareInsertType::Element => unimplemented!(),
            ShingleAwareInsertType::Shingle => self.insert_shingle(value, sequence_index),
        }
        self.buffer.make_contiguous();
        sequence_index
    }

    /// Remove a shingle by key from the point store.
    /// 
    /// The reference count for the shingle at the input key is decremented.
    /// Afterward, the store is shrunk if there are any unused shingles at the
    /// front of the store.
    /// 
    /// # Panics
    /// 
    /// Shingle removal will panic if only a part of the shingle exists in 
    /// the buffer. In this situation, something wrong happened during
    /// buffer management.
    fn remove(&mut self, key: usize) {
        let start = key - self.shift;
        let end = start + self.shingle_size;
        if start > self.ref_counts.len() {
            return;
        }

        if end > self.ref_counts.len() {
            panic!("Inconsistent state in point store: store buffer does not /
            have enough room for the shingle at key={}", key);
        }

        // first check if the shingle lives in the point store in entirety. 
        // we don't want to decrement reference counts to a partial shingle
        for r in self.ref_counts.range(start..end) {
            if *r == 0 {
                return;
            }
        }

        // the full shingle exists. decrement reference counts
        for r in self.ref_counts.range_mut(start..end) {
            if *r > 0 {
                *r -= 1;
            }
        }

        self.shrink_buffer()
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    
    use std::cmp::PartialEq;
    use std::fmt::Debug;
    use std::iter::Iterator;

    fn assert_vec_eq<T: Debug + PartialEq, I: Iterator<Item=T>, J: Iterator<Item=T>>(u: I, v: J) {
        for (ui, vi) in u.zip(v) {
            assert_eq!(ui, vi);
        }
    }

    /// A generic test for any point store on Vec<f32>'s. Used to test multiple
    /// shared point store implementations.
    /// 
    /// Because shingle-aware point store is one of the available shared point
    /// stores we use shingles as the input. These are two-dimensional data
    /// with shingle size three. For non-shingle-aware point stores these test
    /// cases are just as valid.
    fn test_point_store<PS: PointStore<f32>>(store: &mut PS) {
        let key1 = store.insert(&[0.0, 0.1, 1.0, 1.1, 2.0, 2.1], 2);
        let key2 = store.insert(&[1.0, 1.1, 2.0, 2.1, 3.0, 3.1], 3);

        // case: insert same point twice
        let key3 = store.insert(&[2.0, 2.1, 3.0, 3.1, 4.0, 4.1], 4);
        let key4 = store.insert(&[2.0, 2.1, 3.0, 3.1, 4.0, 4.1], 4);

        // case: skip a consecutive sequence index
        let key5 = store.insert(&[4.0, 4.1, 5.0, 5.1, 6.0, 6.1], 6); 

        // check we get the points we inserted
        assert_eq!(store.get(key1).unwrap(), &[0.0, 0.1, 1.0, 1.1, 2.0, 2.1]);
        assert_eq!(store.get(key2).unwrap(), &[1.0, 1.1, 2.0, 2.1, 3.0, 3.1]);
        assert_eq!(store.get(key3).unwrap(), &[2.0, 2.1, 3.0, 3.1, 4.0, 4.1]);
        assert_eq!(store.get(key4).unwrap(), &[2.0, 2.1, 3.0, 3.1, 4.0, 4.1]);
        assert_eq!(store.get(key5).unwrap(), &[4.0, 4.1, 5.0, 5.1, 6.0, 6.1]);
        assert_eq!(key3, key4);

        // check that unknown keys return None
        assert!(store.get(99999999).is_none());

        // point removal
        store.remove(key1);
        assert!(store.get(key1).is_none());
        store.remove(key2);
        assert!(store.get(key2).is_none());
        store.remove(key3);
        assert_eq!(store.get(key3).unwrap(), &[2.0, 2.1, 3.0, 3.1, 4.0, 4.1]);
        store.remove(key3);
        assert!(store.get(key3).is_none());

        // test interleaved insertions and deletions
        let key6 = store.insert(&[5.0, 5.1, 6.0, 6.1, 7.0, 7.1], 7);
        let key7 = store.insert(&[6.0, 6.1, 7.0, 7.1, 8.0, 8.1], 8);
        store.remove(key7);
        assert_eq!(store.get(key6).unwrap(), &[5.0, 5.1, 6.0, 6.1, 7.0, 7.1]);
        assert!(store.get(key7).is_none());

        let key7 = store.insert(&[6.0, 6.1, 7.0, 7.1, 8.0, 8.1], 8);
        assert_eq!(store.get(key7).unwrap(), &[6.0, 6.1, 7.0, 7.1, 8.0, 8.1]);

        let key8 = store.insert(&[7.0, 7.1, 8.0, 8.1, 9.0, 9.1], 9);
    }

    /// A generic test for any point store on Vec<f32>'s. Used to test multiple
    /// shared point store implementations.
    ///
    /// This tests the precondition that point insertions must have the same
    /// point for the same sequnce index. Any test that uses this function
    /// must include #[should_panic].
    fn test_point_store_invalid_insert<PS: PointStore<f32>>(store: &mut PS) {
        // panic when two different vectors are inserted with the same sequence index
        store.insert(&[0.0, 0.1, 1.0, 1.1, 2.0, 2.1], 0);
        store.insert(&[1.0, 1.1, 2.0, 2.1, 3.0, 3.1], 0);
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

    #[test]
    fn test_shingle_aware_point_store() {
        let mut store = HashMapPointStore::<f32>::new();
        test_point_store(&mut store);
    }

    #[test]
    #[should_panic]
    fn test_shingle_aware_point_store_invalid_insert() {
        let mut store = HashMapPointStore::<f32>::new();
        test_point_store_invalid_insert(&mut store);
    }

    #[test]
    fn test_shingle_aware_point_store_internal_state() {
        // first, test the internal state. then run `test_point_store()`
        let shingle_size = 3;
        let dimension = 2;
        let mut store = ShingleAwarePointStore::<f32>::new(2, 3, ShingleAwareInsertType::Shingle);

        assert_eq!(store.buffer.len(), 0);
        assert_eq!(store.ref_counts.len(), 0);

        // insert the first shingle at sequence index
        let point = &[0.0, 0.1, 1.0, 1.1, 2.0, 2.1];
        store.insert(point, 0);

        assert_eq!(store.buffer.len(), 6);
        assert_eq!(store.ref_counts.len(), 3);
        assert_eq!(store.buffer, point);
        for n in store.ref_counts.iter() {
            assert_eq!(n, &1);
        }

        // insert a copy of the first shingle at the same sequence index. the 
        // buffer shouldn't change but the reference counts should increment
        store.insert(point, 0);

        assert_eq!(store.buffer.len(), 6);
        assert_eq!(store.ref_counts.len(), 3);
        assert_eq!(store.buffer, point);
        for n in store.ref_counts.iter() {
            assert_eq!(n, &2);
        }

        // insert the next shingle
        let point = &[1.0, 1.1, 2.0, 2.1, 3.0, 3.1];
        store.insert(point, 1);

        let expected_buffer = &[0.0, 0.1, 1.0, 1.1, 2.0, 2.1, 3.0, 3.1];
        let expected_ref_counts = &[2, 3, 3, 1];
        assert_eq!(store.buffer.len(), 8);
        assert_eq!(store.ref_counts.len(), 4);
        assert_eq!(store.buffer, expected_buffer);
        assert_eq!(store.ref_counts, expected_ref_counts);

        // insert a shingle that is more than one "step" away from the previous
        // shingle in the store (simulating a "missed" shingle store)
        let point = &[10.0, 10.1, 11.0, 11.1, 12.0, 12.1];
        store.insert(point, 6);

        let expected_buffer = &[
            0.0, 0.1, 1.0, 1.1, 2.0, 2.1, 3.0, 3.1,
            f32::infinity(), f32::infinity(), // use -oo where data
            f32::infinity(), f32::infinity(), // does not exist
            10.0, 10.1, 11.0, 11.1, 12.0, 12.1];
        let expected_ref_counts = &[2, 3, 3, 1, 0, 0, 1, 1, 1];
        assert_eq!(store.buffer, expected_buffer);
        assert_eq!(store.ref_counts, expected_ref_counts);

        // we now start removing points, starting with an "internal" shingle
        store.remove(1);

        let expected_ref_counts = &[2, 2, 2, 0, 0, 0, 1, 1, 1];
        assert_eq!(store.buffer, expected_buffer);
        assert_eq!(store.ref_counts, expected_ref_counts);

        // trying to remove it again shouldn't be a problem. that is, don't
        // decrement reference counters if the full shingle isn't there
        store.remove(1);
        assert_eq!(store.buffer, expected_buffer);
        assert_eq!(store.ref_counts, expected_ref_counts);

        // remove the point at the front of the store twice. this should trigger
        // a buffer shrink
        store.remove(0);

        let expected_ref_counts = &[1, 1, 1, 0, 0, 0, 1, 1, 1];
        assert_eq!(store.buffer, expected_buffer);
        assert_eq!(store.ref_counts, expected_ref_counts);
        assert_eq!(store.shift, 0);

        store.remove(0);

        let expected_buffer = &[10.0, 10.1, 11.0, 11.1, 12.0, 12.1];
        let expected_ref_counts = &[1, 1, 1];
        assert_eq!(store.buffer, expected_buffer);
        assert_eq!(store.ref_counts, expected_ref_counts);
        assert_eq!(store.shift, 6);

        // remove the final point from the shingle aware point store. recall
        // that this was inserted at sequence index = 6
        store.remove(6);
        assert_eq!(store.buffer, &[]);
        assert_eq!(store.ref_counts, &[]);
        assert_eq!(store.shift, 6);

        // can we insert shingles back into an empty store? how does sequence
        // indexing handle the situation? we first re-insert the point we 
        // just deleted. We then insert the next point by sequence index
        let point = &[0.0, 0.1, 1.0, 1.1, 2.0, 2.1];
        store.insert(point, 6);
        assert_eq!(store.buffer, point);
        assert_eq!(store.ref_counts, &[1, 1, 1]);

        store.remove(6);
        assert_eq!(store.buffer, &[]);
        assert_eq!(store.ref_counts, &[]);
        assert_eq!(store.shift, 6);

        let point = &[1.0, 1.1, 2.0, 2.1, 3.0, 3.1];
        store.insert(point, 8);

        let expected_buffer = &[
            f32::infinity(), f32::infinity(), // first missing element
            f32::infinity(), f32::infinity(), // second missing element
            1.0, 1.1, 2.0, 2.1, 3.0, 3.1];
        assert_eq!(store.buffer, expected_buffer);
        assert_eq!(store.ref_counts, &[0, 0, 1, 1, 1]);
    }
}