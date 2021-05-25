extern crate num_traits;
use num_traits::{Float, One, Zero};

use std::iter::Sum;
use std::ops::AddAssign;

use crate::tree::{BoundingBox, Cut, Internal, Leaf, Node, Tree};
use super::{Visitor};
use super::utils::{score_seen, score_unseen, damp};


/// A di-vector consisting of high/positive and low/negative components.
///
/// A di-vector is used when we want to track a quantity in both the positive
/// and negative directions for each dimension in a manifold. For example, when
/// using an [`AttributionVisitor`] to compute the attribution of the anomaly
/// score of a point. We want to know if the anomaly score attributed to the
/// ith coordinate of the point is due to that coordinate being unusually high
/// or unusually low.
///
/// # Examples
///
/// ```
/// use random_cut_forest::visitor::DiVec;
///
/// // create a new di-vector using some high- and low-components
/// let mut divec = DiVec::<f32>::new(vec![1.0, 3.0], vec![0.0, -1.0]);
/// assert_eq!(divec.hi[0], 1.0);
/// assert_eq!(divec.hi[1], 3.0);
///
/// // we can scale the divector componenets
/// divec.scale_mut(2.0);
/// assert_eq!(divec.hi[0], 2.0);
/// assert_eq!(divec.hi[1], 6.0);
/// assert_eq!(divec.lo[0], 0.0);
/// assert_eq!(divec.lo[1], -2.0);
///
/// // normalize so that the L1-norm is equal to a target value
/// divec.renormalize_mut(1.0);
/// assert_eq!(divec.hi[0], 1.0/3.0);
/// assert_eq!(divec.hi[1], 1.0);
/// assert_eq!(divec.lo[0], 0.0);
/// assert_eq!(divec.lo[1], -1.0/3.0);
///
/// // some operators are overloaded
/// let other = DiVec::<f32>::new(vec![2.0/3.0, 0.0], vec![0.0, 1.0/3.0]);
/// divec += other;
/// assert_eq!(divec.hi[0], 1.0);
/// assert_eq!(divec.hi[1], 1.0);
/// assert_eq!(divec.lo[0], 0.0);
/// assert_eq!(divec.lo[1], 0.0);
/// ```
///
#[derive(Clone)]
pub struct DiVec<T> {
    pub hi: Vec<T>,
    pub lo: Vec<T>,
}

impl<T> DiVec<T>
    where T: Float + Sum
{
    /// Create a new di-vector with a given high-component and low-component.
    ///
    /// # Examples
    ///
    /// ```
    /// use random_cut_forest::visitor::DiVec;
    ///
    /// let divec = DiVec::<f32>::new(vec![1.0, 2.0], vec![3.0, 4.0]);
    /// assert_eq!(divec.hi[0], 1.0);
    /// assert_eq!(divec.hi[1], 2.0);
    /// assert_eq!(divec.lo[0], 3.0);
    /// assert_eq!(divec.lo[1], 4.0);
    /// ```
    pub fn new(hi: Vec<T>, lo: Vec<T>) -> Self {
        if hi.len() != lo.len() {
            panic!("Hi and lo components must have the same dimension.")
        }
        DiVec { hi: hi, lo: lo }
    }

    /// Create an empty di-vector with empty high- and low-components of a
    /// given dimension.
    ///
    /// # Examples
    ///
    /// ```
    /// use random_cut_forest::visitor::DiVec;
    ///
    /// let divec = DiVec::<f32>::with_capacity(2);
    /// assert_eq!(divec.hi[0], 0.0);
    /// assert_eq!(divec.hi[1], 0.0);
    /// assert_eq!(divec.lo[0], 0.0);
    /// assert_eq!(divec.lo[1], 0.0);
    /// ```
    pub fn with_capacity(dimensions: usize) -> Self {
        DiVec {
            hi: vec![Zero::zero(); dimensions],
            lo: vec![Zero::zero(); dimensions],
        }
    }

    /// Get the dimension of the di-vector.
    ///
    /// # Example
    ///
    /// ```
    /// use random_cut_forest::visitor::DiVec;
    ///
    /// let divec = DiVec::<f32>::new(vec![1.0, 3.0], vec![0.0, -1.0]);
    /// assert_eq!(divec.dimensions(), 2);
    /// ```
    pub fn dimensions(&self) -> usize {
        self.hi.len()
    }

    /// Scale the di-vector in place.
    ///
    /// # Examples
    ///
    /// ```
    /// use random_cut_forest::visitor::DiVec;
    ///
    /// let mut divec = DiVec::<f32>::new(vec![1.0, 3.0], vec![0.0, -1.0]);
    /// divec.scale_mut(2.0);
    /// assert_eq!(divec.hi[0], 2.0);
    /// assert_eq!(divec.hi[1], 6.0);
    /// assert_eq!(divec.lo[0], 0.0);
    /// assert_eq!(divec.lo[1], -2.0);
    /// ```
    pub fn scale_mut(&mut self, scale: T) {
        for i in 0..self.dimensions() {
            self.hi[i] = self.hi[i] * scale;
            self.lo[i] = self.lo[i] * scale;
        }
    }

    /// Create a scaled version of the divector.
    ///
    /// # Examples
    ///
    /// ```
    /// use random_cut_forest::visitor::DiVec;
    ///
    /// let divec = DiVec::<f32>::new(vec![1.0, 3.0], vec![0.0, -1.0]);
    /// let other = divec.scale(2.0);
    /// assert_eq!(other.hi[0], 2.0);
    /// assert_eq!(other.hi[1], 6.0);
    /// assert_eq!(other.lo[0], 0.0);
    /// assert_eq!(other.lo[1], -2.0);
    ///
    /// assert_eq!(divec.hi[0], 1.0);
    /// assert_eq!(divec.hi[1], 3.0);
    /// assert_eq!(divec.lo[0], 0.0);
    /// assert_eq!(divec.lo[1], -1.0);
    /// ```
    pub fn scale(&self, scale: T) -> DiVec<T> {
        let mut result = self.clone();
        result.scale_mut(scale);
        result
    }

    /// Renormalize the divec to have a target L1 norm.
    ///
    /// If the L1-norm of this divec is positive, then the values in the high
    /// and low components are uniformly rescaled such that the L1-norm is equal
    /// to the new target value.
    ///
    /// # Examples
    ///
    /// ```
    /// use random_cut_forest::visitor::DiVec;
    ///
    /// let mut divec = DiVec::<f32>::new(vec![1.0, 3.0], vec![0.0, -1.0]);
    /// divec.renormalize_mut(1.0);
    ///
    /// assert_eq!(divec.hi[0], 1.0 / 3.0);
    /// assert_eq!(divec.hi[1], 1.0);
    /// assert_eq!(divec.lo[0], 0.0);
    /// assert_eq!(divec.lo[1], -1.0 / 3.0);
    /// ```
    pub fn renormalize_mut(&mut self, target_l1_norm: T) {
        let l1_norm = self.total_hi_lo_sum();
        if l1_norm > Zero::zero() {
            self.scale_mut(target_l1_norm / l1_norm);
        }
    }


    /// Get the high-low sum in a given dimension.
    ///
    /// # Examples
    ///
    /// ```
    /// use random_cut_forest::visitor::DiVec;
    ///
    /// let divec = DiVec::<f32>::new(vec![1.0, 3.0], vec![0.0, -1.0]);
    /// assert_eq!(divec.hi_lo_sum(0), 1.0);
    /// assert_eq!(divec.hi_lo_sum(1), 2.0);
    /// ```
    #[inline(always)]
    pub fn hi_lo_sum(&self, dimension: usize) -> T {
        self.hi[dimension] + self.lo[dimension]
    }

    /// Get the sum of all high-lo
    /// w sums across all dimensions.
    ///
    /// # Examples
    ///
    /// ```
    /// use random_cut_forest::visitor::DiVec;
    ///
    /// let divec = DiVec::<f32>::new(vec![1.0, 3.0], vec![0.0, -1.0]);
    /// assert_eq!(divec.total_hi_lo_sum(), (1.0 + 0.0) + (3.0 - 1.0));
    /// ```
    pub fn total_hi_lo_sum(&self) -> T {
        (0..self.dimensions()).map(|i| self.hi_lo_sum(i)).sum()
    }
}

impl<T> AddAssign for DiVec<T>
    where T: Float + Sum
{
    fn add_assign(&mut self, rhs: Self) {
        if self.dimensions() != rhs.dimensions() {
            panic!("Dimensions must agree to accumulate.")
        }

        for i in 0..self.dimensions() {
            self.hi[i] = self.hi[i] + rhs.hi[i];
            self.lo[i] = self.lo[i] + rhs.lo[i];
        }
    }
}


/// A multi-visitor on nodes for computing anomaly attribution.
///
/// The anomaly attribution encodes two pieces of information about a query
/// point:
///
/// 1. Which dimensions are the primary contributors to the anomalousness of
///    the point?
/// 2. In those dimensions, is the point anomalous because the corresponding
///    value is too high or too low.
///
/// These data are represented in an output [`DiVec`].
///
/// # Description of Algorithm
///
/// At each node in the path we compute the "max-" and "min-gaps" from the
/// bounding box to the query point `P`. These are the differentials, `d1` and
/// `d2` in the diagram below; similar to that of
/// [`AnomalyScoreVisitor`](crate::visitor::AnomalyScoreVisitor). However, in
/// this case we also track whether the differentials put `P` to the left or to
/// the right of the box within each dimension.
///
/// ```text
///    ┌────────────────┐   ┬
///    │P               │   │
///    │                │   │  d2      P = point to score
///    │                │   │          B = bounding box at current node
///    │      ┌─────────┤   ┼
///    │      │    B    │   │  l2
///    └──────┴─────────┘   ┴
///
///    ├──────┼─────────┤
///       d1       l1
/// ```
///
/// For example, in the diagram above the min-gap in the first dimension is
/// equal to `d1`, since `P` is to the left of the node's bounding box in the
/// first dimension. Similarly, the max-gap in the second dimension is `d2`
/// since `P` is to the right of `B` in the second dimension. (The max-gap in the
/// first dimension and the min-gap in the second dimension are both zero.)
///
/// The high- and low-components of the attribution score are updated with a
/// probabilistic rescaling of the score computed at the previous node and
/// the score obtained from this node. The probability weights come from the
/// probability of separation similar to that computed in `AnomalyScoreVisitor`.
/// This end result is rescaled such that the high-low sum of the attribution
/// vector is equal to the usual anomaly score.
pub struct AttributionVisitor<'a, T> {
    // Reference to the tree on which the attribution scores will be computed.
    tree: &'a Tree<T>,

    // Input point to score on the above tree.
    point_to_score: &'a Vec<T>,

    // The attribution score computed during the multi-visitor process.
    attribution: DiVec<T>,

    // A vector tracking the max- and min-gaps between the query point and the
    // current node bounding box
    differences: Vec<T>,

    // Sum of the max- and min-gaps across all dimensions
    sum_of_differences: T,

    // Sum of the lengths of the merged bounding box between the current node
    // and the query point. used for score rescaling.
    sum_of_new_range: T,

    // The anomaly score computed during the visitor process
    anomaly_score: T,

    // For improved performance, we set a flag if the point to score lies in
    // a bounding box. Once this happens, the anomaly score does not update.
    point_inside_box: bool,

    // Similar to point_inside_box but for each coordinate, allowing
    // short-cutting of certain computations
    coordinate_inside_box: Vec<bool>,

    // If true, adjust scoring for previously observed or duplicate points
    hit_duplicates: bool,
}


impl<'a, T> AttributionVisitor<'a, T>
    where T: Float + Sum + Zero
{
    pub fn new(
        tree: &'a Tree<T>,
        point_to_score: &'a Vec<T>,
    ) -> AttributionVisitor<'a, T> {
        AttributionVisitor {
            tree: tree,
            point_to_score: point_to_score,
            attribution: DiVec::with_capacity(point_to_score.len()),
            sum_of_differences: Zero::zero(),
            differences: vec![Zero::zero(); 2*point_to_score.len()],
            sum_of_new_range: Zero::zero(),
            anomaly_score: Zero::zero(),
            point_inside_box: false,
            coordinate_inside_box: vec![false; point_to_score.len()],
            hit_duplicates: false,
        }
    }

    /// Update the attribution using a bounding box and merged box at the current node.
    ///
    /// When updating the attribution score at a node we compare the node's
    /// bounding box to the merged bounding box that would be created by adding
    /// the point to be scored. This function updates the fields `sum_of_differences`
    /// and `differences` to reflect the total difference in side lenghts and
    /// the difference in side lengths in each dimension, respectively.
    fn update_ranges(&mut self, small_box: &BoundingBox<T>, large_box: &BoundingBox<T>) {
        self.sum_of_differences = Zero::zero();
        self.sum_of_new_range = Zero::zero();
        self.differences = vec![Zero::zero(); 2*self.point_to_score.len()];

        for i in 0..self.point_to_score.len() {
            let large_range = large_box.max_values()[i] - large_box.min_values()[i];
            self.sum_of_new_range = self.sum_of_new_range + large_range;

            if self.coordinate_inside_box[i] {
                continue;
            }

            let max_gap = T::max(large_box.max_values()[i] - small_box.max_values()[i], Zero::zero());
            let min_gap = T::max(small_box.min_values()[i] - large_box.min_values()[i], Zero::zero());
            if max_gap + min_gap > Zero::zero() {
                self.sum_of_differences = self.sum_of_differences + (min_gap + max_gap);
                self.differences[2*i] = max_gap;
                self.differences[2*i + 1] = min_gap;
            } else {
                self.coordinate_inside_box[i] = true;
            }
        }
    }
}


impl<'a, T> Visitor<T> for AttributionVisitor<'a, T>
    where T: Float + One + Sum
{
    type Output = DiVec<T>;

    /// Return the normalized attribution di-vector.
    ///
    /// Scales each component by base-two log of the tree mass so that the
    /// resulting attribution vector is independent of the tree's sample size.
    /// The rescaling is equivalent to `normalize_score` used in
    /// [`AnomalyScoreVisitor`](crate::visitor::AnomalyScoreVisitor).
    fn get_result(&self) -> DiVec<T> {
        let one = One::one();
        let scale = (T::from(self.tree.mass()).unwrap() + one).ln() / (one + one).ln();
        let result = self.attribution.scale(scale);
        result
    }

    /// Initialize the attribution di-vector.
    ///
    /// The attribution scores are initialized to the initial max- and min-gaps
    /// between the query point and the nearest leaf node in the tree as
    /// determined by [`Tree::traverse`].
    ///
    /// The initial value depends on whether the point to score is equal to the
    /// point at the leaf node.
    fn accept_leaf(&mut self, leaf: &Leaf, depth: T) {
        // compute the bounding boxes corresponding to the leaf point and the
        // merged box with the point to score
        let store = self.tree.borrow_point_store();
        let point = store.get(leaf.point()).unwrap();
        let bounding_box = &BoundingBox::new_from_point(point);
        let merged_box = &BoundingBox::merged_box_with_point(&bounding_box, self.point_to_score);

        self.update_ranges(bounding_box, merged_box);
        if point == self.point_to_score {
            self.hit_duplicates = true;
        }

        if self.hit_duplicates {
            self.anomaly_score = score_seen(depth, leaf.mass()) * damp(leaf.mass(), self.tree.mass());

            let scale: T = T::from(2 * self.point_to_score.len()).unwrap();
            self.attribution.hi.fill(self.anomaly_score / scale);
            self.attribution.lo.fill(self.anomaly_score / scale);
            self.coordinate_inside_box.fill(false);
        } else {
            self.anomaly_score = score_unseen(depth);

            for i in 0..self.point_to_score.len() {
                self.attribution.hi[i] = self.anomaly_score * self.differences[2*i] / self.sum_of_new_range;
                self.attribution.lo[i] = self.anomaly_score * self.differences[2*i + 1] / self.sum_of_new_range;
            }
        }
    }

    /// Update the attribution scores at an internal node along the path to the root.
    ///
    /// Determine the max- and min-gaps between the query point and the bounding
    /// box at the current internal node. Update the high- and low-attributions
    /// based on these information and the probability of separation in each
    /// dimension.
    ///
    /// As in [`accept_leaf`](AttributionVisitor::accept_leaf), the updates are
    /// based on whether the query point is equal to the nearest leaf node
    /// found during the tree traversal.
    fn accept(&mut self, node: &Internal<T>, depth: T) {
        if self.point_inside_box {
            return;
        }

        // if we hit a duplicate then we use the sibling bounding box to
        // represent a counterfactual "what if the point and the candidate
        // nearest neighbor had not been inserted into the tree?"
        let small_box: BoundingBox<T> = if self.hit_duplicates {
            let sibling = if Cut::is_left_of(self.point_to_score, node.cut()) {
                self.tree.get_node(node.right())
            } else {
                self.tree.get_node(node.left())
            };
            match sibling {
                Node::Leaf(leaf) => {
                    let store = self.tree.borrow_point_store();
                    let point = store.get(leaf.point()).unwrap();
                    BoundingBox::new_from_point(point)
                },
                Node::Internal(internal) => BoundingBox::new(
                    internal.bounding_box().min_values(),
                    internal.bounding_box().max_values()),
            }
        } else {
            BoundingBox::new(
                node.bounding_box().min_values(),
                node.bounding_box().max_values())
        };

        let large_box = &BoundingBox::merged_box_with_point(&small_box, self.point_to_score);
        self.update_ranges(&small_box, large_box);

        // use the separation probability to update the attribution scores
        let separation_probability: T = self.sum_of_differences / self.sum_of_new_range;
        if separation_probability <= Zero::zero() {
            self.point_inside_box = true;
        } else {
            let new_score = score_unseen(depth);
            for i in 0..self.point_to_score.len() {
                let sep_prob_spike_direction = self.differences[2*i] / self.sum_of_new_range;
                let sep_prob_dip_direction = self.differences[2*i + 1] / self.sum_of_new_range;

                let p = -separation_probability + One::one();
                self.attribution.hi[i] = sep_prob_spike_direction*new_score + p*self.attribution.hi[i];
                self.attribution.lo[i] = sep_prob_dip_direction*new_score + p*self.attribution.lo[i];
            }
        }

        // final rescaling ensures agreement with the anomaly score visitor
        if self.hit_duplicates && (self.point_inside_box || depth == Zero::zero()) {
            self.attribution.renormalize_mut(self.anomaly_score);
        }
    }
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn divec_add_assign() {
        let mut v = DiVec::<f32>::new(vec![1.0, 2.0], vec![3.0, 4.0]);
        let w = DiVec::<f32>::new(vec![0.1, 0.2], vec![0.3, 0.4]);
        v += w;

        assert_eq!(v.hi[0], 1.1);
        assert_eq!(v.hi[1], 2.2);
        assert_eq!(v.lo[0], 3.3);
        assert_eq!(v.lo[1], 4.4);
    }
}