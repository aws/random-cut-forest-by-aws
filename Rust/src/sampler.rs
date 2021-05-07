//! Random sampling of data from a stream.
//!
//! A stream sampler is a mechanism for maintaining a fixed-size random sample
//! from a stream of data. Specifically, we maintain "weighted samples": data
//! samples that have assigned to them a "weight" which is proportional to the
//! probability that the point will be included in the sample. This allows for
//! something like time-decay random sampling where we prefer to accept
//! recently observed samples.
//!
//! ```
//! use random_cut_forest::{SamplerResult, StreamSampler, WeightedSample};
//!
//! // create a sampler that can contain two elements with a time decay
//! // parameter large enough that it should be very rare for a new point
//! // to not be accepted
//! let mut sampler = StreamSampler::new(2, 100000.0);
//!
//! // sample a new value. this is guaranteed to be sampled as long as
//! // the sampler is not full
//! match sampler.sample("hello", 10) {
//!     SamplerResult::Accepted(evicted) => assert!(evicted.is_none()),
//!     SamplerResult::Ignored => panic!(),
//! }
//!
//! // confirm that this value is contained within the samples
//! let samples: Vec<&WeightedSample<&str>> = sampler.iter().collect();
//! assert_eq!(samples.len(), 1);
//! assert_eq!(samples[0].value(), &"hello");
//!
//! // sample a second value. this too should be accepted since the sampler
//! // is not yet full
//! match sampler.sample("world", 20) {
//!     SamplerResult::Accepted(evicted) => assert!(evicted.is_none()),
//!     SamplerResult::Ignored => panic!(),
//! }
//!
//! // the sampler is full, but because of the large decay parameter, it is
//! // almost guaranteed that the third sample will be accepted
//! match sampler.sample("I'm taking over", 123) {
//!     SamplerResult::Accepted(evicted) => assert_eq!(evicted.unwrap().value(), &"hello"),
//!     SamplerResult::Ignored => panic!(),
//! }
//!
//! // again, with a large decay parameter, an element with a small sequence
//! // index will likely not by sampled
//! match sampler.sample("Just passing by", 0) {
//!     SamplerResult::Accepted(_) => panic!(),
//!     SamplerResult::Ignored => println!("This sample was ignored"),
//! }
//! ```

extern crate rand;
use rand::{Rng, SeedableRng};

extern crate rand_chacha;
use rand_chacha::ChaCha8Rng;

use std::cmp::{Ord, PartialOrd, Eq, Ordering};
use std::collections::BinaryHeap;
use std::collections::binary_heap;

/// Weighted samples stored in a stream sampler.
///
/// Weighted samples store a value along with a weight. Depending on the
/// configuration of the sampler, the weight can be used to affect the
/// probability that the point will be accepted into sampler. Within a sampler,
/// weighted samples are ranked by their weight.
///
/// Weighted samples also use a "sequence index", which indicates when this
/// sample was observed relative to other samples in the stream. Stream samplers
/// use this information to determine which points
///
/// # Examples
///
/// ```
/// use random_cut_forest::WeightedSample;
///
/// let x = WeightedSample::new("Hello, ", 42.0);
/// let y = WeightedSample::new("world.", 123.0);
/// let z = WeightedSample::new("Same weight as 'Hello'", 42.0);
///
/// assert!(x < y);
/// assert!(x == z);
/// ```
pub struct WeightedSample<T> {
    value: T,
    weight: f32,
}

impl<T> WeightedSample<T> {
    pub fn new(value: T, weight: f32) -> Self {
        WeightedSample {
            value: value,
            weight: weight,
        }
    }

    /// Get the value stored in the weighted sample.
    pub fn value(&self) -> &T { &self.value }

    /// Get the weight of the sample.
    pub fn weight(&self) -> &f32 { &self.weight }
}

/// Weighted samples are ordered by their weight. Because weighted samples are
/// stored in a heap ([`std::collections::BinaryHeap`]) we need to implement
/// [`Ord`].
///
/// # Note
///
/// Floats do not actually implement `Ord`, so this is a bit of a hack in order
/// to get a heap/priority queue working on weighted samples.
impl<T> Ord for WeightedSample<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.weight < other.weight {
            return Ordering::Less;
        } else if self.weight > other.weight {
            return Ordering::Greater;
        } else {
            return Ordering::Equal;
        }
    }
}

impl<T> PartialOrd for WeightedSample<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.weight.partial_cmp(&other.weight)
    }
}

impl<T> PartialEq for WeightedSample<T> {
    fn eq(&self, other: &Self) -> bool {
        self.weight.eq(&other.weight)
    }
}

impl<T> Eq for WeightedSample<T> { }


/// Returned when a stream samper accepts or is updated with a new value.
///
/// When [`StreamSampler::sample`] is called with a new value, the value is
/// either ignored or accepted into the sampler. If accepted, a weighted sample
/// might be evicted from the sampler.
pub enum SamplerResult<T> {
    Ignored,
    Accepted(Option<WeightedSample<T>>),
}

/// Maintains a fixed-size random sample from a data stream.
///
/// When a new value is submitted to the sampler it decides whether to accept
/// the point into the sample. The decision is based on the current number of
/// samples as well as the weights of these samples. Newer values, indicated by
/// their sequence index, are assigned a larger weight than older values.
///
/// This sampler is based on a weighted reservoir sampling algorithm. As such,
/// it has a time decay parameter `time_decay`. The value of `time_decay`
/// indicates how aggressively the sampler should prefer to store the most
/// recently obesrved samples. When `time_decay == 0.0`, samples are uniformly
/// retained; that is, given `N` observations a sampler of size `S` will keep
/// `S` samples distributed uniformly across the `N` observations without
/// replacement.
///
/// # Examples
///
/// ```
/// use random_cut_forest::{SamplerResult, StreamSampler};
///
/// // create a seeded stream sampler on strings
/// let mut sampler: StreamSampler<&str> = StreamSampler::new(2, 123.4);
/// sampler.seed(42);
/// assert_eq!(sampler.capacity(), 2);
///
/// // add a sample to the empty stream sampler with a sequence index of zero
/// sampler.sample("Hello", 0);
/// assert!(!sampler.is_full());
/// assert_eq!(sampler.size(), 1);
/// assert_eq!(sampler.num_observations(), 1);
///
/// // the sample function returns a SampleResult
/// let result = sampler.sample(", world!", 1);
/// assert!(sampler.is_full());
/// assert_eq!(sampler.size(), 2);
/// assert_eq!(sampler.num_observations(), 2);
///
/// // until the sampler is full it will always return an `Accepted` result
/// if let SamplerResult::Accepted(evicted_sample) = result {
///     assert!(evicted_sample.is_none());
/// } else { panic!("Expected accepted sample") }
///
///
/// // with the random seed above, the next sample should be accepted as well,
/// // evicting some older point from the sample
/// let result = sampler.sample("-- Some Programmer", 20);
/// assert!(sampler.is_full());
/// assert_eq!(sampler.size(), 2);
/// assert_eq!(sampler.num_observations(), 3);
///
/// if let SamplerResult::Accepted(evicted_sample) = result {
///     assert!(evicted_sample.is_some());
///     println!("evicted value = {}", evicted_sample.unwrap().value());
/// } else { panic!("Expected accepted sample") }
/// ```
///
pub struct StreamSampler<T> {
    weighted_samples: BinaryHeap<WeightedSample<T>>,
    sample_size: usize,
    num_observations: usize,
    time_decay: f32,
    rng: ChaCha8Rng,
}


impl<T> StreamSampler<T> {

    /// Create a new stream sampler.
    ///
    /// A `sample_size` must be provided to indicate the number of samples that
    /// the stream sampler can store. Additionally, a decay factor `time_decay`
    /// must be provided to indicate how aggressively the sampler favors keeping
    /// more recently observed sampler.
    ///
    /// # Examples
    ///
    /// ```
    /// use random_cut_forest::StreamSampler;
    ///
    /// // create a new stream sampler on floats with capacity two
    /// let sampler: StreamSampler<&str> = StreamSampler::new(2, 0.1);
    /// ```
    pub fn new(sample_size: usize, time_decay: f32) -> Self {
        if time_decay < 0.0 {
            panic!("Time decay parameter must be non-negative")
        }

        StreamSampler {
            weighted_samples: BinaryHeap::with_capacity(sample_size),
            sample_size: sample_size,
            num_observations: 0,
            time_decay: time_decay,
            rng: ChaCha8Rng::from_entropy(),
        }
    }

    /// Reset the stream samplers random number generator with a specified seed.
    ///
    /// # Examples
    ///
    /// ```
    /// use random_cut_forest::StreamSampler;
    ///
    /// let mut sampler: StreamSampler<&str> = StreamSampler::new(2, 0.1);
    /// sampler.seed(42);
    /// ```
    pub fn seed(&mut self, seed: u64) {
        self.rng = ChaCha8Rng::seed_from_u64(seed);
    }

    /// Sample a new value with a given sequence index.
    ///
    /// A value along with `sequence_index`, indicating the relative order of
    /// this value to other observed values, is provided to be a candidate
    /// addition to this sample. If the value is **not** sampled then a
    /// [`SamplerResult::Ignored`] is returned by this function. Otherwise,
    /// a [`SamplerResult::Accepted`] is returned containing an `evicted_sample`
    /// of type `Option<WeightedSample<T>>`. This is the sample that had to be
    /// evicted in order to make room for the newly accepted sample.
    ///
    /// # Examples
    ///
    /// ```
    /// use random_cut_forest::{SamplerResult, StreamSampler};
    ///
    /// // create a sampler that can hold two samples
    /// let mut sampler: StreamSampler<&str> = StreamSampler::new(2, 0.1);
    /// sampler.seed(0);
    ///
    /// // pass in the value "hello" with sequence index zero
    /// let result = sampler.sample("hello", 0);
    ///
    /// // the result should be accepted and no points are evicted (the sampler
    /// // is currently empty)
    /// if let SamplerResult::Accepted(evicted) = result {
    ///     assert!(evicted.is_none());
    /// } else { panic!("Expected accepted sample") }
    /// ```
    ///
    pub fn sample(&mut self, value: T, sequence_index: usize) -> SamplerResult<T> {
        let weight = self.compute_weight(sequence_index);
        self.num_observations += 1;

        // determine if we should accept the new value into the sample
        let under_sampled = self.num_observations <= self.sample_size;
        let new_observation_has_smaller_weight = match self.weighted_samples.peek() {
            Some(sample) => weight < *sample.weight(),
            None => false,
        };
        let accept_sample = under_sampled || new_observation_has_smaller_weight;

        // if accepted, add to samples and evict a sample if necessary
        if accept_sample {
            let evicted_sample = match self.is_full() {
                true => self.weighted_samples.pop(),
                false => None,
            };
            let candidate_sample = WeightedSample { value: value, weight: weight };
            self.weighted_samples.push(candidate_sample);

            return SamplerResult::Accepted(evicted_sample);
        }

        SamplerResult::Ignored
    }

    /// Transform a sequence index to a weight using this sampler's decay factor.
    ///
    /// The weight of sample is used to determine the priority of the samples;
    /// the sampler maintains those samples with largest observed weight. Given
    /// a sequence index, `n`, the computed weight is `R = u^(1/w)` where
    /// `w = exp(lambda * n)` and `lambda` is the decay parameter.
    ///
    /// In practice we transform these weights into log-space for numerical
    /// stability. The more negative these transformed weights are the more
    /// likely the value will be accepted into the sample.
    ///
    /// # Examples
    ///
    /// ```
    /// use random_cut_forest::StreamSampler;
    /// extern crate rand;
    ///
    /// let mut sampler = StreamSampler::new(128, 0.1);
    /// sampler.sample("hello", 0);
    ///
    /// println!("{}", sampler.compute_weight(1));
    /// println!("{}", sampler.compute_weight(2));
    /// println!("{}", sampler.compute_weight(1000)); // likely to be more negative
    /// ```
    pub fn compute_weight(&mut self, sequence_index: usize) -> f32 {
        let random: f32 = self.rng.gen();
        -(sequence_index as f32) * self.time_decay + (-random.ln()).ln()
    }

    /// Returns an iterator on the elements of the sampler.
    ///
    /// This simply returns the result of [`BinaryHeap.iter()`]. The weighted
    /// samples are visited in arbitrary order.
    pub fn iter(&self) -> binary_heap::Iter<'_, WeightedSample<T>> {
        self.weighted_samples.iter()
    }

    pub fn num_observations(&self) -> usize { self.num_observations }
    pub fn is_full(&self) -> bool { self.sample_size == self.weighted_samples.len() }
    pub fn capacity(&self) -> usize { self.sample_size }
    pub fn size(&self) -> usize { self.weighted_samples.len() }
    pub fn time_decay(&self) -> f32 { self.time_decay }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weighted_sample() {
        let x1 = WeightedSample { value: "Without education we", weight: 0.0 };
        let x2 = WeightedSample { value: "are in a horrible and", weight: 1.0 };
        let x3 = WeightedSample { value: "deadly danger of taking", weight: -2.0 };
        let x4 = WeightedSample { value: "educated people seriously.", weight: 3.0 };
        let x5 = WeightedSample { value: "-- G. K. Chesterton", weight: 0.0 };

        assert!(x3 < x1 && x1 < x2 && x2 < x4);
        assert!(x1 == x5);
    }

    #[test]
    fn test_sampler() {
        // set the time decay parameter large enough that it should be very
        // rare for a new point to not be accepted into the sampler
        let mut sampler = StreamSampler::new(2, 100000.0);
        assert_eq!(sampler.capacity(), 2);
        assert_eq!(sampler.size(), 0);
        assert_eq!(sampler.is_full(), false);

        match sampler.sample("Without education we", 0) {
            SamplerResult::Accepted(evicted) => {
                assert!(evicted.is_none());
                assert_eq!(sampler.size(), 1);
                assert_eq!(sampler.is_full(), false);
            }
            SamplerResult::Ignored => panic!("Expected data accepted")
        }

        match sampler.sample("are in a horrible and", 10) {
            SamplerResult::Accepted(evicted) => {
                assert!(evicted.is_none());
                assert_eq!(sampler.size(), 2);
                assert_eq!(sampler.is_full(), true);
            }
            SamplerResult::Ignored => panic!("Expected data accepted")
        }

        match sampler.sample("deadly danger of taking", 100) {
            SamplerResult::Accepted(evicted) => match evicted {
                    Some(evicted) => {
                        assert_eq!(evicted.value(), &"Without education we");
                        assert_eq!(sampler.size(), 2);
                        assert_eq!(sampler.is_full(), true);
                    }
                    None => panic!("Expected evicted point")
                }
            SamplerResult::Ignored => panic!("Expected data accepted")
        }

        match sampler.sample("educated people seriously.", 1000) {
            SamplerResult::Accepted(evicted) => match evicted {
                Some(evicted) => {
                    assert_eq!(evicted.value(), &"are in a horrible and");
                    assert_eq!(sampler.size(), 2);
                    assert_eq!(sampler.is_full(), true);
                }
                None => panic!("Expected evicted point")
            }
            SamplerResult::Ignored => panic!("Expected data accepted")
        }

        match sampler.sample("-- G. K. Chesterton", 10000) {
            SamplerResult::Accepted(evicted) => match evicted {
                Some(evicted) => {
                    assert_eq!(evicted.value(), &"deadly danger of taking");
                    assert_eq!(sampler.size(), 2);
                    assert_eq!(sampler.is_full(), true);
                }
                None => panic!("Expected evicted point")
                }
            SamplerResult::Ignored => panic!("Expected data accepted")
        }
    }
}