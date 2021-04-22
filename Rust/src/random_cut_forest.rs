extern crate num_traits;
use num_traits::Float;

use crate::SampledTree;

use std::marker::PhantomData;
use std::iter::Sum;

/// A random cut forest model.
///
/// Random cut forests are model-free data structures for sketching data
/// streams. This type is the main interface for training a random cut forest
/// and using the model for various scoring operations such as anomaly detection.
///
/// A random cut forest is a collection of random cut trees; specifically a
/// collection of [`SampledTree`] structs. An update to a random cut forest
/// model corresponds to independently updating each `SampledTree` with the
/// input point. When a scoring algorithm is called, such as anomaly score,
/// each random cut tree reports a score and these scores are aggegated in
/// some way particular to the scoring algorithm.
///
/// It is recommended to use [`RandomCutForestBuilder`] to create a new
/// [`RandomCutForest`] model.
///
/// # Examples
///
/// ```
/// use random_cut_forest::{RandomCutForest, RandomCutForestBuilder};
///
/// // create the default random cut forest on three-dimensional data points
/// let mut forest: RandomCutForest<f32> = RandomCutForestBuilder::new(3).build();
///
/// // update the forest with some data points
/// forest.update(vec![0.0, 0.0, 0.0]);
/// forest.update(vec![1.0, 1.0, 1.0]);
/// forest.update(vec![0.0, 0.5, 0.3]);
/// forest.update(vec![0.6, 0.0, -0.2]);
///
/// // compute anomaly scores
/// // let score = forest.anomaly_score(vec![0.1, 0.2, 0.3]);
/// ```
pub struct RandomCutForest<T> {
    dimension: usize,
    num_observations: usize,
    num_trees: usize,
    sample_size: usize,
    time_decay: f32,
    trees: Vec<SampledTree<T>>,
}

impl<T> RandomCutForest<T>
    where T: Float + Sum
{

    /// Update a random cut forest with a new data point.
    ///
    /// A copy of the data point will be sent to each sampled tree in the forest
    /// for consideration. Each tree independently decides whether to accept the
    /// point into its sample.
    ///
    /// # Panics
    ///
    /// If the dimensionality of the input data point does not match the
    /// dimensionality of the forest.
    ///
    /// # Examples
    ///
    /// ```
    /// use random_cut_forest::{RandomCutForest, RandomCutForestBuilder};
    ///
    /// // create a default RCF on two-dimensional data
    /// let mut forest: RandomCutForest<f32> = RandomCutForestBuilder::new(2).build();
    ///
    /// // update the forest with some data
    /// forest.update(vec![0.0, 0.0]);
    /// forest.update(vec![1.0, 0.0]);
    /// forest.update(vec![0.0, 1.0]);
    /// forest.update(vec![1.0, 1.0]);
    ///
    /// // update panics if the point has an incorrect dimensionality/length
    /// //forest.update(vec![2.0, 3.0, 4.0]);
    /// ```
    pub fn update(&mut self, point: Vec<T>) {
        assert_eq!(point.len(), self.dimension,
            "Dimension mismatch. Expected {}-dimensional input.",
            self.dimension);

        self.num_observations += 1;
        for tree in self.trees.iter_mut() {
            tree.update(point.clone(), self.num_observations)
        }
    }

    /// Return the dimension of the data accepted by this random cut forest.
    pub fn dimension(&self) -> usize { self.dimension }

    /// Return the decay factor of the random samplers used by the forest's trees.
    pub fn time_decay(&self) -> f32 { self.time_decay }

    /// Return the total number of observations made by this forest.
    pub fn num_observations(&self) -> usize { self.num_observations }

    /// Return the number of trees in this forest.
    pub fn num_trees(&self) -> usize { self.num_trees }

    /// Return the number of samples/observations stored in each tree.
    pub fn sample_size(&self) -> usize { self.sample_size }

    /// Return a vector of references to the trees of the forest.
    pub fn trees(&self) -> &Vec<SampledTree<T>> { &self.trees }
}


/// Convenient mechanism for creating [`RandomCutForest`]s.
///
/// Random cut forests are highly configurable and come with large number of
/// parameters, many of which have reasonable default values. This builder
/// makes it easier to construct a random cut forest model
///
/// The builder has the following required parameters for initialization:
///
/// * `dimension`
///
/// The builder uses the following defaults for the remaining parameters:
///
/// * `num_trees = 50`
/// * `sample_size = 256`
/// * `time_decay = 0.0`
///
/// # Examples
///
/// ```
/// use random_cut_forest::{RandomCutForest, RandomCutForestBuilder};
///
/// // create the default random cut forest on three-dimensional data points
/// let dimension = 3;
/// let forest = RandomCutForestBuilder::<f32>::new(dimension).build();
/// assert_eq!(forest.dimension(), 3);
/// assert_eq!(forest.num_trees(), 50);
/// assert_eq!(forest.sample_size(), 256);
/// assert_eq!(forest.time_decay(), 0.0);
///
/// // create a forest with specified parameters. you can also specify the base
/// // type by annotating the target variable
/// let forest: RandomCutForest<f32> = RandomCutForestBuilder::new(dimension)
///     .num_trees(20)
///     .sample_size(128)
///     .time_decay(0.01)
///     .build();
/// assert_eq!(forest.dimension(), 3);
/// assert_eq!(forest.num_trees(), 20);
/// assert_eq!(forest.sample_size(), 128);
/// assert_eq!(forest.time_decay(), 0.01);
/// ```
///
pub struct RandomCutForestBuilder<T> {
    dimension: usize,
    num_trees: usize,
    sample_size: usize,
    time_decay: f32,
    _point_type: PhantomData<T>,
}

impl<T> RandomCutForestBuilder<T>
    where T: Float + Sum
{

    /// Initialize a random cut forest builder.
    ///
    /// The primary required parameter is the dimensionality of the forest.
    /// Reasonable defaults are used for other parameters.
    pub fn new(dimension: usize) -> RandomCutForestBuilder<T> {
        RandomCutForestBuilder {
            dimension: dimension,
            time_decay: 0.0,
            num_trees: 50,
            sample_size: 256,
            _point_type: PhantomData::<T>,
        }
    }

    /// Set the dimension of the random cut forest.
    pub fn dimension(mut self, dimension: usize) -> RandomCutForestBuilder<T> {
        self.dimension = dimension;
        self
    }

    /// Set the number of trees used in the random cut forest.
    pub fn num_trees(mut self, num_trees: usize) -> RandomCutForestBuilder<T> {
        self.num_trees = num_trees;
        self
    }

    /// Set the number of samples retained by each tree in the random cut forest.
    pub fn sample_size(mut self, sample_size: usize) -> RandomCutForestBuilder<T> {
        self.sample_size = sample_size;
        self
    }

    /// Set the random sampling decay factor of the random cut forest.
    pub fn time_decay(mut self, time_decay: f32) -> RandomCutForestBuilder<T> {
        self.time_decay = time_decay;
        self
    }

    /// Build a random cut forest using the parameters set by the builder.
    pub fn build(self) -> RandomCutForest<T> {
        let mut trees: Vec<SampledTree<T>> = Vec::with_capacity(self.num_trees);
        for _ in 0..self.num_trees {
            trees.push(SampledTree::new(self.sample_size, self.time_decay));
        }

        RandomCutForest {
            dimension: self.dimension,
            num_trees: self.num_trees,
            sample_size: self.sample_size,
            time_decay: self.time_decay,
            trees: trees,
            num_observations: 0,
        }
    }
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn update() {
        let mut forest = RandomCutForestBuilder::new(2)
            .num_trees(2)
            .sample_size(4)
            .time_decay(8.0)  // large positive value means updates are almost forced
            .build();

        // update until full
        forest.update(vec![0.0, 0.0]);
        forest.update(vec![1.0, 0.0]);
        forest.update(vec![0.0, 1.0]);
        forest.update(vec![1.0, 1.0]);

        // updates that may require deletion
        forest.update(vec![2.0, 0.0]);
        forest.update(vec![0.0, 2.0]);
        forest.update(vec![-2.0, 0.0]);
        forest.update(vec![0.0, -2.0]);
    }
}