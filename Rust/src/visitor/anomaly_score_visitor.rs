extern crate num_traits;
use num_traits::{Float, One, Zero};

use std::iter::Sum;

use crate::visitor::Visitor;
use crate::tree::{BoundingBox, Internal, Leaf, Tree};
use super::utils::{score_seen, score_unseen, damp, normalize_score};


/// A visitor on nodes used to compute an anomaly score.
///
/// Given a tree and a point to score, the tree returns a branch of nodes using
/// [`Tree::traverse()`]. This branch is a path from the leaf node in the tree
/// closest to the point to score (in the L1 norm) to the root node of the tree.
/// The anomaly score visitor initializes an anomaly score at the leaf node and
/// then updates the score using the *separation probability* at each internal
/// node; the probability that a random cut separates the point to score from
/// that node's bounding box.
///
/// The further away the point to score `P` is from the current bounding box,
///  `B`, the more likely a random cut will separate `P` from `B`. In the
/// two-dimensional example, below, the separation probability is equal to
/// a function of the ratios between the box side lengths, `l1` and `l2`, to the
/// distances `d1` and `d2`.
///
/// ```text
///
///    ┌────────────────┐   ┬
///    │               P│   │
///    │                │   │  d2      P = point to score
///    │                │   │          B = bounding box at current node
///    ├─────────┐      │   ┼
///    │    B    │      │   │  l2
///    └─────────┴──────┘   ┴
///
///    ├─────────┼──────┤
///         l1      d1
/// ```
///
/// Once the traversal reaches the root node of the tree, the anomaly score can
/// be obtained by calling[`get_result()`](Self::get_result).
pub struct AnomalyScoreVisitor<'a, T> {
    // A tree on which an anomaly score will be computed
    tree: &'a Tree<T>,

    // Input point to score using the above tree.
    point_to_score: &'a Vec<T>,

    // The anomaly score computed during the visitor process
    anomaly_score: T,

    // For improved performance, we set a flag if the point to score lies in
    // a bounding box. Once this happens, the anomaly score does not update.
    point_inside_box: bool,

    // Similar to point_inside_box but for each coordinate, allowing
    // short-cutting of certain computations
    coordinate_inside_box: Vec<bool>
}

impl<'a, T> AnomalyScoreVisitor<'a, T> where
    T: Float + One + Sum + Zero
{
    /// Initialize an anomaly score visitor with a tree and a point to score.
    ///
    /// The anomaly score of this visitor is initialized to zero.
    pub fn new(
        tree: &'a Tree<T>,
        point_to_score: &'a Vec<T>,
    ) -> AnomalyScoreVisitor<'a, T> {
        AnomalyScoreVisitor {
            tree: tree,
            point_to_score: point_to_score,
            anomaly_score: Zero::zero(),
            point_inside_box: false,
            coordinate_inside_box: vec![false; point_to_score.len()]
        }
    }

    /// Returns the probability that the point to score and the input bounding
    /// box are separated by a random cut.
    fn separation_probability(&mut self, bounding_box: &BoundingBox<T>) -> T {
        let mut new_range_sum: T = Zero::zero();
        let mut range_diff_sum: T = Zero::zero();
        let min_values = bounding_box.min_values();
        let max_values = bounding_box.max_values();

        for i in 0..bounding_box.dimensions() {
            let mut min_value = min_values[i];
            let mut max_value = max_values[i];
            let old_range = max_value - min_value;

            if !self.coordinate_inside_box[i] {
                if max_value < self.point_to_score[i] {
                    max_value = self.point_to_score[i]
                } else if min_value > self.point_to_score[i] {
                    min_value = self.point_to_score[i];
                } else {
                    // in this case the current coordinate lies within the width
                    // of the bounding box in this dimension. we update with
                    // old range and indicate that this coordinate is now inside
                    new_range_sum = new_range_sum + old_range;
                    self.coordinate_inside_box[i] = true;
                    continue;
                }

                let new_range = max_value - min_value;
                new_range_sum = new_range_sum + new_range;
                range_diff_sum = range_diff_sum + new_range - old_range;
            } else {
                new_range_sum = new_range_sum + old_range;
            }
        }

        if new_range_sum <= Zero::zero() {
            panic!("Sum of new range of the shadow box is smaller than zero.");
        }

        range_diff_sum / new_range_sum
    }
}

impl<'a, T> Visitor<T> for AnomalyScoreVisitor<'a, T> where
    T: Float + One + Sum + Zero
{
    type Output = T;

    /// Initialize the anomaly score from a leaf node.
    ///
    /// A leaf node is the first node visited in the anomaly scoring process.
    /// The initial anomaly score depends whether the point at the leaf node
    /// is equal to the point to score.
    fn accept_leaf(&mut self, leaf: &Leaf, depth: T) {
        let point_store = self.tree.borrow_point_store();
        let point = point_store.get(leaf.point()).unwrap();
        if *self.point_to_score == *point {
            self.point_inside_box = true;
            self.anomaly_score = damp::<T>(leaf.mass(), self.tree.mass()) *
                score_seen(depth, leaf.mass());
        } else {
            self.anomaly_score = score_unseen(depth);
        }
    }

    /// Update the anomaly score from an internal node.
    ///
    /// The anomaly score is updated based on the probability that the poitn to
    /// score would be separated by a random cut from the bounding box at this
    /// node. If the point lies inside the bounding box at this node then the
    /// score is simply propagated up the tree.
    fn accept(&mut self, node: &Internal<T>, depth: T) {
        if self.point_inside_box { return; }

        let separation_probability = self.separation_probability(node.bounding_box());
        if separation_probability <= Zero::zero() {
            self.point_inside_box = true;
            return;
        }

        let one: T = One::one();
        self.anomaly_score = separation_probability * score_unseen(depth) +
            (one - separation_probability) * self.anomaly_score;
    }

    /// Normalize and return the anomaly score computed by the visitor.
    ///
    /// The anomaly score is normalized using the mass of the tree before
    /// returning. This is so that the resulting anomaly score is independent
    /// of the number of samples in the tree.
    fn get_result(&self) -> T {
        normalize_score(self.anomaly_score, self.tree.mass())
    }
}