extern crate num_traits;
use num_traits::{Float, Zero};

extern crate rand;
use rand::distributions::Uniform;

use std::iter::Sum;

use crate::BoundingBox;


/// Hyperplane cut inside a bounding box.
///
/// This data structure represents the "cut" part of random cut forests. A cut
/// is a hyperplane that paritions data points into two halves. When the cut
/// lies inside a bounding box it is guaranteed to result in two non-empty
/// partitions.
///
/// A cut consits of a `dimension` and a `value`. The dimension is the dimension
/// along which the normal vector of the cut points. Dimensions use zero-based
/// indexing; that is, in a three dimensional data point the three dimensions
/// are `0`, `1`, and `2`. The value of the cut is the location along the
/// Cartesian axis where the cut is located.
///
/// For example, if the dimension of the cut is `2` and the value of the cut
/// is `1.0` then the cut partitions points into two sets: the data points that
/// have dimension 2 component less than 1.0 and those with dimension 2
/// componenet greater than zero.
///
/// # Examples
///
/// ```
/// use random_cut_forest::Cut;
///
/// // create a new cut from a given dimension and value
/// let cut = Cut::new(1, 0.0);
///
/// // check if a some points are to the left or to the right of the cut
/// let p: Vec<f32> = vec![1.0, -1.0];
/// let q: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
/// assert!(Cut::is_left_of(&p, &cut));
/// assert!(!Cut::is_left_of(&q, &cut));
///
/// // generate a random cut inside a bounding box
/// extern crate rand;
/// use random_cut_forest::BoundingBox;
///
/// let bbox = BoundingBox::new(&vec![0.0, 0.0, 0.0], &vec![2.0, 3.0, 4.0]);
/// let mut rng = rand::thread_rng();
/// let random_cut = Cut::new_random_cut(&bbox, &mut rng).unwrap();
///
/// // confirm that the random cut lies inside the bounding box
/// assert!(0 <= random_cut.dimension() && random_cut.dimension() <= 2);
/// assert!(bbox.min_values()[random_cut.dimension()] <= random_cut.value());
/// assert!(random_cut.value() <= bbox.max_values()[random_cut.dimension()]);
/// ```
#[derive(Debug)]
pub struct Cut<T> {
    dimension: usize,
    value: T,
}

impl<T> Cut<T> 
    where T: Float + Sum
{

    /// Create a new cut from a given dimension and value.
    pub fn new(dimension: usize, value: T) -> Self {
        Cut {
            dimension: dimension,
            value: value
        }
    }

    /// Returns a random cut inside a bounding box.
    ///
    /// The cut is guaranteed to have a value within the bounds of the bounding
    /// box, thus partitioning the points inside the bounding box into two
    /// non-empty subsets.
    ///
    /// This function requires a random number generator: any struct that
    /// implements the [`rand::Rng`] trait from the [`rand`] crate.
    ///
    /// # Examples
    ///
    /// ```
    /// use rand::thread_rng;
    /// let mut rng = thread_rng();
    ///
    /// use random_cut_forest::{BoundingBox, Cut};
    ///
    /// let min = vec![0.0, 0.0, 0.0];
    /// let max = vec![1.0, 2.0, 3.0];
    /// let bbox = BoundingBox::new(&min, &max);
    ///
    /// let cut = Cut::new_random_cut(&bbox, &mut rng).unwrap();
    ///
    /// assert!(min[cut.dimension()] <= cut.value());
    /// assert!(cut.value() <= max[cut.dimension()]);
    /// ```
    pub fn new_random_cut<Rng: rand::Rng>(
        bounding_box: &BoundingBox<T>,
        rng: &mut Rng,
    ) -> Result<Self, &'static str> {
        let distribution = Uniform::new(0.0, 1.0);
        let random: f64 = rng.sample(distribution);

        let min = bounding_box.min_values();
        let max = bounding_box.max_values();
        let mut break_point: T = T::from(random).unwrap() * bounding_box.range_sum();

        for i in 0..bounding_box.dimensions() {
            let range = max[i] - min[i];
            if break_point <= range {
                let mut cut_value = min[i] + break_point;
                if cut_value == max[i] && range > Zero::zero() {
                    cut_value = cut_value - Float::epsilon();
                }
                return Ok(Cut::new(i, cut_value));
            }
            break_point = break_point - range;
        }

        return Err("The random cut break point did not lie in the bounding box range.");
    }

    /// Returns true if `point` is to the left of `cut`.
    ///
    /// This simply checks if the component of the point in the cut's dimension
    /// is less than or equal to the cut's value.alloc
    ///
    /// # Examples
    ///
    /// ```
    /// use random_cut_forest::Cut;
    ///
    /// let cut = Cut::new(1, 0.0);
    /// let point = vec![1.0, -2.0, 3.0, -4.0];
    /// assert!(Cut::is_left_of(&point, &cut));
    /// ```
    pub fn is_left_of(point: &Vec<T>, cut: &Cut<T>) -> bool {
        point[cut.dimension] <= cut.value
    }

    /// Get the dimension of the cut.
    pub fn dimension(&self) -> usize { self.dimension }

    /// Get the value of the cut.
    pub fn value(&self) -> T { self.value }
}