extern crate num_traits;
use num_traits::Float;

use std::cell::{Ref, RefCell, RefMut};
use std::iter::Sum;
use std::rc::Rc;

use crate::{PointStore, SamplerResult, StreamSampler};
use crate::tree::{AddResult, NodeTraverser, Tree};

/// Combination of a tree and a reservoir sampler.
///
/// A random cut tree, represented by the [`Tree`] struct, is a data structure
/// for organizing vectors using bounding boxes and random cuts. In a random
/// cut forest model we wish to limit the number of points contained in the
/// tree. This is done using a reservoir sampler. A `SampledTree` coordinates
/// point additions and deletions between a reservoir sampler, given by a
/// [`StreamSampler`], and the random cut tree.
///
/// # Examples
///
/// ```
/// use random_cut_forest::{Node, SampledTree};
///
/// // create a sampled tree that can hold 32 points witha a decay factor 0.01
/// let mut tree: SampledTree<f32> = SampledTree::new(32, 0.01);
///
/// // if necessary, we can set the random seed used in point sampling as well
/// // as the random cut process
/// tree.seed(42);
///
/// // update the tree with some points. every point will be accepted into the
/// // tree until the sample size is reached.
/// tree.update(vec![0.0, 0.0], 0);
/// tree.update(vec![1.0, -1.0], 1);
/// tree.update(vec![2.0, 3.0], 2);
/// assert_eq!(tree.num_observations(), 3);
///
/// // given a query point, we can traverse the tree by following random cuts
/// let query = vec![0.5, 1.5];
/// let traversal_nodes: Vec<&Node<f32>> = tree.traverse(&query).collect();
/// ```
pub struct SampledTree<T> {
    point_store: Rc<RefCell<PointStore<T>>>,
    tree: Tree<T>,
    sampler: StreamSampler<usize>,
}

impl<T> SampledTree<T>
    where T: Float + Sum
{
    /// Create a new sampled tree.
    ///
    /// Specify the tree's `sample_size`, the number of point maintained by the
    /// tree, and decay factor `time_decay` for the time-decay sampler defined
    /// in [`StreamSampler`].
    ///
    /// # Examples
    ///
    /// ```
    /// use random_cut_forest::SampledTree;
    /// let tree: SampledTree<f32> = SampledTree::new(128, 0.01);
    /// ```
    pub fn new(sample_size: usize, time_decay: f32) -> Self {
        let point_store: Rc<RefCell<PointStore<T>>> = Rc::new(RefCell::new(PointStore::new()));
        SampledTree::new_with_point_store(sample_size, time_decay, point_store)
    }

    /// Create a new sampled tree with a given point store.
    ///
    /// In addition to the parameters in `new()`, specifies a point store to
    /// be used by the sampled tree.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::cell::RefCell;
    /// use std::rc::Rc;
    /// use random_cut_forest::{SampledTree, PointStore};
    ///
    /// // create a point store
    /// let point_store = Rc::new(RefCell::new(PointStore::new()));
    ///
    /// // initialize the sampled tree with a new point store
    /// let tree: SampledTree<f32> = SampledTree::new_with_point_store(
    ///     128, 0.01, point_store);
    /// ```
    pub fn new_with_point_store(
        sample_size: usize,
        time_decay: f32,
        point_store: Rc<RefCell<PointStore<T>>>,
    ) -> Self {
        SampledTree {
            point_store: point_store.clone(),
            tree: Tree::new_with_point_store(point_store.clone()),
            sampler: StreamSampler::new(sample_size, time_decay),
        }
    }

    /// Sets the seed of the `SampledTree`'s tree and stream sampler.
    ///
    /// Randomness is used in [`Tree`] primarily for generating random cuts.
    /// Randomness is used in [`StreamSampler`] to determine which points are
    /// accepted into the sample.
    pub fn seed(&mut self, seed: u64) {
        self.tree.seed(seed);
        self.sampler.seed(seed);
    }

    /// Update the sampled tree with a new point.
    ///
    /// The stream sampler decides if the new point will be accepted into the
    /// tree as a function of the decay factor `time_decay` and the input
    /// `sequence_index` for this point.
    ///
    /// # Examples
    ///
    /// ```
    /// use random_cut_forest::SampledTree;
    /// let mut tree: SampledTree<f32> = SampledTree::new(128, 0.01);
    ///
    /// tree.update(vec![0.0, 0.0], 0);
    /// tree.update(vec![1.0, -1.0], 1);
    /// assert_eq!(tree.num_observations(), 2);
    /// ```
    pub fn update(&mut self, point: Vec<T>, sequence_index: usize) {
        // we need a point key that we can submit to the sampler. the strategy,
        // then, is to first add the point to the tree and then sample using
        // the output key. if the key is accepted by the sampler then we
        // evict if necessary. otherwise, we delete the point we just added
        //
        // TODO: is there a way to do this without cloning? We need a reference
        // to the point if we need to delete afterward. Slabs allow you to
        // obtain the next available key...
        let point_key = match self.tree.add_point(point.clone()) {
            AddResult::AddedPoint(key) => key,
            AddResult::MassIncreased(key) => key,
        };

        match self.sampler.sample(point_key, sequence_index) {
            SamplerResult::Accepted(evicted) => match evicted {
                Some(evicted) => {
                    // TODO: can we satisfy the borrow checker so that we can
                    // perform the delete without needing to clone the point?
                    let evicted_point = {
                        let point_store = self.point_store.borrow();
                        point_store.get(*evicted.value()).unwrap().clone()
                    };
                    self.tree.delete_point(&evicted_point);
                }
                None => ()
            },
            SamplerResult::Ignored => { self.tree.delete_point(&point); }
        }
    }

    /// Traverse the tree with a given query point as input.
    ///
    /// Returns an iterator on the nodes of the tree. The iterator begins at the
    /// tree's root node and follows the branch to the leaf node containing the
    /// point nearest to the query point in L1 norm.
    ///
    /// # Examples
    ///
    /// ```
    /// use random_cut_forest::{Node, SampledTree};
    /// let mut tree: SampledTree<f32> = SampledTree::new(32, 0.01);
    /// tree.update(vec![0.0, 0.0], 0);
    /// tree.update(vec![1.0, 1.0], 1);
    ///
    /// // provide a query point that will always be to the left of any cut
    /// // on the two points inserted above, no matter the cut location
    /// let query = vec![-2.0, -2.0];
    /// let nodes: Vec<&Node<f32>> = tree.traverse(&query).collect();
    /// assert_eq!(nodes.len(), 2);
    ///
    /// // verify the last point of the tree is equal to the left-most point
    /// if let Node::Leaf(leaf) = nodes[1] {
    ///     let point_key = leaf.point();
    ///     let point_store = tree.borrow_point_store();
    ///     let point = point_store.get(point_key).unwrap();
    ///     assert_eq!(point, &vec![0.0, 0.0]);
    /// } else {
    ///     panic!("Last node in traversal should be a leaf!")
    /// }
    /// ```
    ///
    pub fn traverse<'a>(&'a self, point: &'a Vec<T>) -> NodeTraverser<'a, T> {
        self.tree.traverse(point)
    }

    /// Returns the sample size of the sampled tree.
    ///
    /// # Examples
    ///
    /// ```
    /// use random_cut_forest::SampledTree;
    ///
    /// let tree: SampledTree<f32> = SampledTree::new(256, 0.001);
    /// assert_eq!(tree.sample_size(), 256);
    /// ```
    pub fn sample_size(&self) -> usize { self.sampler.capacity() }

    /// Returns the time decay factor of the random sampler.
    ///
    /// # Examples
    ///
    /// ```
    /// use random_cut_forest::SampledTree;
    ///
    /// let tree: SampledTree<f32> = SampledTree::new(256, 0.001);
    /// assert_eq!(tree.time_decay(), 0.001);
    /// ```
    pub fn time_decay(&self) -> f32 { self.sampler.time_decay() }

    /// Returns the total number of observations made by the tree.
    ///
    /// For every point sent to [`SampledTree::update`], the total number of
    /// observations is incremented independent of the sample size or whether
    /// or not the point is accepted into the sample.
    ///
    /// # Examples
    ///
    /// ```
    /// use random_cut_forest::SampledTree;
    ///
    /// let mut tree: SampledTree<f32> = SampledTree::new(2, 1.0);
    /// assert_eq!(tree.num_observations(), 0);
    ///
    /// tree.update(vec![0.0, 0.0], 0);
    /// assert_eq!(tree.num_observations(), 1);
    ///
    /// tree.update(vec![1.0, 1.0], 10);
    /// assert_eq!(tree.num_observations(), 2);
    ///
    /// tree.update(vec![2.0, 2.0], 20);
    /// assert_eq!(tree.num_observations(), 3);
    /// ```
    pub fn num_observations(&self) -> usize { self.sampler.num_observations() }

    /// Returns a reference to the tree in the sampled tree.
    pub fn tree(&self) -> &Tree<T> { &self.tree }

    /// Borrow the sampled tree's point store.
    pub fn borrow_point_store(&self) -> Ref<PointStore<T>> { self.point_store.borrow() }

    /// Mutably borrow the sample's tree's point store.
    pub fn mut_borrow_point_store(&self) -> RefMut<PointStore<T>> { self.point_store.borrow_mut() }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filled_sample() {
        let mut tree: SampledTree<f32> = SampledTree::new(2, 8.0);

        // update until full
        tree.update(vec![0.0, 0.0], 0);
        tree.update(vec![1.0, 0.0], 1);

        // additional points that cause evictions
        tree.update(vec![0.0, 1.0], 100);
    }
}