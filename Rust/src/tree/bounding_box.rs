extern crate num_traits;
use num_traits::{Float, Zero};

use std::fmt;

use crate::RCFFloat;

/// Bounding box on collections on points.
///
/// Given a set of *d*- dimensional points, a bounding box is the smallest *d*-
/// dimensional rectangular prism containing all of these points. A bounding box
/// is represented by two vectors
///
/// # Examples
///
/// ```
/// use random_cut_forest::BoundingBox;
///
/// // create a new bounding box from a single point
/// let point: Vec<f32> = vec![1.0, 2.0];
/// let bbox = BoundingBox::new_from_point(&point);
/// assert_eq!(bbox.min_values(), &point);
/// assert_eq!(bbox.max_values(), &point);
///
/// // create a second bounding box by merging the first one with another point
/// let new_point = vec![3.0, -2.0];
/// let merged_bbox = BoundingBox::merged_box_with_point(&bbox, &new_point);
/// println!("{}", &merged_bbox);   // BoundingBox ((1.0, -2.0), (3.0, 2.0))
///
/// // confirm that the two points and the first bounding box are contained
/// // in this larger merged box
/// assert!(merged_bbox.contains_point(&point));
/// assert!(merged_bbox.contains_point(&new_point));
/// assert!(merged_bbox.contains_box(&bbox));
/// ```
pub struct BoundingBox<T> {
    min_values: Vec<T>,
    max_values: Vec<T>,
    dimensions: usize,
    range_sum: T,
}

impl<T> BoundingBox<T> where T: RCFFloat {

    /// Create a new bounding box from a min values vector and a max values
    /// vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use random_cut_forest::BoundingBox;
    ///
    /// let min = vec![-1.0, 0.0];
    /// let max = vec![1.0, 3.0];
    /// let bbox = BoundingBox::new(&min, &max);
    /// assert_eq!(bbox.dimensions(), 2);
    /// assert_eq!(bbox.range_sum(), 5.0);
    /// ```
    pub fn new(min_values: &Vec<T>, max_values: &Vec<T>) -> Self {
        assert_eq!(min_values.len(), max_values.len());

        BoundingBox {
            min_values: min_values.clone(),
            max_values: max_values.clone(),
            dimensions: min_values.len(),
            range_sum: BoundingBox::compute_range_sum(min_values, max_values),
        }
    }

    /// Create a new bounding box from a single point.
    ///
    /// The resulting bounding box has no interior: its min values are equal to
    /// its max values. Therefore, its range sum is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use random_cut_forest::BoundingBox;
    ///
    /// let point: Vec<f32> = vec![1.0, 2.0];
    /// let bbox = BoundingBox::new_from_point(&point);
    /// assert_eq!(bbox.min_values(), &point);
    /// assert_eq!(bbox.max_values(), &point);
    /// assert_eq!(bbox.dimensions(), 2);
    /// assert_eq!(bbox.range_sum(), 0.0);
    /// ```
    pub fn new_from_point(point: &Vec<T>) -> Self {
        BoundingBox {
            min_values: point.clone(),
            max_values: point.clone(),
            dimensions: point.len(),
            range_sum: Zero::zero(),
        }
    }

    /// Returns a new bounding box given by the merging of a bounding box with
    /// a point.
    ///
    /// If the point lies inside the bounding box then this returns a clone of
    /// the same bounding box.
    ///
    /// # Examples
    ///
    /// ```
    /// use random_cut_forest::BoundingBox;
    ///
    /// let min = vec![0.0, 0.0];
    /// let max = vec![1.0, 1.0];
    /// let bbox = BoundingBox::new(&min, &max);
    ///
    /// let point = vec![0.5, 3.0];
    /// let merged = BoundingBox::merged_box_with_point(&bbox, &point);
    /// assert_eq!(merged.max_values(), &vec![1.0, 3.0]);
    /// assert_eq!(merged.range_sum(), 4.0);
    ///
    /// ```
    pub fn merged_box_with_point(
        bounding_box: &BoundingBox<T>,
        point: &Vec<T>) -> Self
    {
        let min_values: Vec<T> = bounding_box.min_values().iter()
            .zip(point)
            .map(|(&x, &y)| Float::min(x, y))
            .collect();

        let max_values: Vec<T> = bounding_box.max_values().iter()
            .zip(point)
            .map(|(&x, &y)| Float::max(x, y))
            .collect();

        let dimensions = min_values.len();
        let range_sum = BoundingBox::compute_range_sum(
            &min_values, &max_values);

        BoundingBox {
            min_values: min_values,
            max_values: max_values,
            dimensions: dimensions,
            range_sum: range_sum,
        }
    }

    /// Returns a new bounding box given by the merging of two bounding boxes.
    ///
    /// The merging of two bounding boxes is given by two points. The first is
    /// the minimum value in each dimension. The second is the maximum value
    /// in each dimension. The points contained in both bounding boxes are also
    /// contained in this large bounding box.
    ///
    /// # Examples
    ///
    /// ```
    /// use random_cut_forest::BoundingBox;
    ///
    /// let min = vec![0.0, 0.0];
    /// let max = vec![2.0, 2.0];
    /// let bbox1 = BoundingBox::new(&min, &max);
    ///
    /// let min = vec![1.0, 1.0];
    /// let max = vec![3.0, 4.0];
    /// let bbox2 = BoundingBox::new(&min, &max);
    ///
    /// let merged = BoundingBox::merged_box_with_box(&bbox1, &bbox2);
    /// assert_eq!(merged.min_values(), &vec![0.0, 0.0]);
    /// assert_eq!(merged.max_values(), &vec![3.0, 4.0]);
    /// assert_eq!(merged.range_sum(), 7.0);
    /// ```
    pub fn merged_box_with_box(
        bounding_box1: &BoundingBox<T>,
        bounding_box2: &BoundingBox<T>) -> Self
    {
        let min_values: Vec<T> = bounding_box1.min_values().iter()
            .zip(bounding_box2.min_values())
            .map(|(&x, &y)| Float::min(x, y))
            .collect();

        let max_values: Vec<T> = bounding_box1.max_values().iter()
            .zip(bounding_box2.max_values())
            .map(|(&x, &y)| Float::max(x, y))
            .collect();

        let dimensions = min_values.len();
        let range_sum = BoundingBox::compute_range_sum(
            &min_values, &max_values);

        BoundingBox {
            min_values: min_values,
            max_values: max_values,
            dimensions: dimensions,
            range_sum: range_sum,
        }
    }

    /// Get the dimensionality of the bounding box.
    pub fn dimensions(&self) -> usize { self.dimensions }

    /// Get the vector of min values of the bounding box.
    pub fn min_values(&self) -> &Vec<T> { &self.min_values }

    /// Get the vector of max values of the bounding box.
    pub fn max_values(&self) -> &Vec<T> { &self.max_values }

    /// Get the sum across all dimensions of lengths of the bounding box.
    pub fn range_sum(&self) -> T { self.range_sum }

    /// Returns true if the given point is contained inside the bounding box
    ///
    /// # Examples
    ///
    /// ```
    /// use random_cut_forest::BoundingBox;
    ///
    /// let min = vec![0.0, 0.0];
    /// let max = vec![2.0, 2.0];
    /// let bbox = BoundingBox::new(&min, &max);
    /// assert!(bbox.contains_point(&vec![1.0, 1.0]));
    /// ```
    pub fn contains_point(&self, point: &Vec<T>) -> bool {
        for i in 0..self.dimensions {
            if point[i] < self.min_values[i] || self.max_values[i] < point[i] {
                return false;
            }
        }
        true
    }

    /// Returns true if the given bounding box is contained inside this
    /// bounding box.
    ///
    /// # Examples
    ///
    /// ```
    /// use random_cut_forest::BoundingBox;
    ///
    /// let min = vec![0.0, 0.0];
    /// let max = vec![8.0, 8.0];
    /// let bbox = BoundingBox::new(&min, &max);
    ///
    /// let min = vec![0.0, 1.0];
    /// let max = vec![2.0, 3.0];
    /// let small_bbox = BoundingBox::new(&min, &max);
    /// assert!(bbox.contains_box(&small_bbox));
    ///
    /// let min = vec![4.0, 6.0];
    /// let max = vec![9.0, 7.0];
    /// let med_bbox = BoundingBox::new(&min, &max);
    /// assert!(!bbox.contains_box(&med_bbox));
    /// ```
    pub fn contains_box(&self, bounding_box: &BoundingBox<T>) -> bool {
        for i in 0..self.dimensions {
            let min = bounding_box.min_values[i];
            let max = bounding_box.max_values[i];
            if min < self.min_values[i] || self.max_values[i] < max {
                return false;
            }
        }
        true
    }

    /// Compute the range sum from a pair of min/max value vectors.
    ///
    /// The range sum is the sum of the differences between the min values and
    /// max values of the bounding box across each component. For example, if
    /// the min values are `[a, b]` and the max values are `[c, d]` then the
    /// range sum is equal to `(c -a) + (d - b)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use random_cut_forest::BoundingBox;
    ///
    /// let min = vec![1.0, 2.0];
    /// let max = vec![4.0, 3.0];
    /// let range_sum = BoundingBox::compute_range_sum(&min, &max);
    /// assert_eq!(range_sum, 4.0);
    /// ```
    pub fn compute_range_sum(min_values: &Vec<T>, max_values: &Vec<T>) -> T {
        let dimensions = min_values.len();
        assert_eq!(dimensions, max_values.len());

        (0..dimensions).map(|i| max_values[i] - min_values[i]).sum()
    }
}

impl<T> fmt::Display for BoundingBox<T>
    where T: RCFFloat + fmt::Debug
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BoundingBox ({:?}, {:?})", self.min_values, self.max_values)
    }
}