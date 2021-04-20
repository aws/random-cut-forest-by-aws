use super::BoundingBox;
use super::Cut;
use crate::RCFFloat;

type NodeKey = usize;
type PointKey = usize;

/// A leaf node in a random cut tree.
///
/// Points contained in a random cut tree are represented by a leaf node. In
/// particular, a leaf node contains a `Option` node key to a parent key.
///
/// # Examples
///
/// ```no_run
/// use rcf::{PointStore, Leaf};
///
/// let mut point_store = PointStore::new();
///
/// // add a point to a point store. this returns a point key for later access
/// let point = vec![1.0, 2.0, 3.0];
/// let point_key = point_store.insert(point);
///
/// // create a new leaf node on this point key
/// let leaf = Leaf::new(point_key);
/// assert_eq!(leaf.point(), point_key);
/// assert!(leaf.parent().is_none());
/// assert_eq!(leaf.mass(), 1);
/// ```
pub struct Leaf {
    parent: Option<NodeKey>,
    mass: u32,
    point: PointKey,
}

impl Leaf {

    /// Create a new leaf node.
    ///
    /// A leaf node must point to a valid point index to a point living inside
    /// a point store. The mass is initialized to `1` and the parent is
    /// initialized to `None`.
    pub fn new(point_key: PointKey) -> Self {
        Leaf {
            parent: None,
            mass: 1,
            point: point_key,
        }
    }

    /// Returns the key of the parent [`Internal`] node.
    pub fn parent(&self) -> Option<usize> { self.parent }

    /// Returns the key of point in a point store represented by this leaf node.
    pub fn point(&self) -> usize { self.point }

    /// Returns the mass of this leaf node.
    pub fn mass(&self) -> u32 { self.mass }

    /// Decrements the mass at this leaf node by one.
    pub fn decrement_mass(&mut self) { self.mass -= 1; }
}

/// An internal node in a random cut tree.
///
/// Internal nodes contain node keys to its left and right children, which must
/// exist. They also own a bounding box on the points contained below this node
/// as well as a cut defining what data belongs to the left and right nodes.
///
/// # Examples
///
/// ```no_run
/// use rcf::{Cut, BoundingBox, Internal, Leaf, Node, NodeStore};
///
/// let mut node_store: NodeStore<f32> = NodeStore::new();
///
/// // create some nodes and add then to a node store to get their keys
/// let left_node = Node::Leaf(Leaf::new(42)); // a Leaf or Internal node
/// let left_key = node_store.insert(left_node);
///
/// let right_node = Node::Leaf(Leaf::new(123)); // a Leaf or Internal node
/// let right_key = node_store.insert(right_node);
///
/// // create a bounding box and a cut on the bounding box
/// let min = vec![0.0, 1.0];
/// let max = vec![2.0, 3.0];
/// let bbox = BoundingBox::new(&min, &max);
/// let cut = Cut::new(0, 0.7);
///
/// // create a new internal node from these data
/// let node = Internal::new(left_key, right_key, bbox, cut);
/// ```
pub struct Internal<T> {
    parent: Option<NodeKey>,
    left: NodeKey,
    right: NodeKey,
    mass: u32,
    bounding_box: BoundingBox<T>,
    cut: Cut<T>,
}

impl<T> Internal<T> {

    /// Create a new internal node.
    ///
    /// A valid internal node has a left node and a right node. The data at an
    /// internal node consists of a bounding box and a cut on that bounding box.
    /// The mass is initialized to `1` and the parent is initialized to `None`.
    pub fn new(
        left: NodeKey,
        right: NodeKey,
        bounding_box: BoundingBox<T>,
        cut: Cut<T>) -> Self
    {
        Internal {
            parent: None,
            left: left,
            right: right,
            mass: 1,
            bounding_box: bounding_box,
            cut: cut,
        }
    }

    /// Returns the key of the parent [`Internal`] node.
    pub fn parent(&self) -> Option<usize> { self.parent }

    /// Returns the node key of the left child.
    pub fn left(&self) -> NodeKey { self.left }

    /// Sets the left child by node key.
    pub fn set_left(&mut self, left: NodeKey) { self.left = left }

    /// Returns the node key of the right child.
    pub fn right(&self) -> NodeKey { self.right }

    /// Sets the right child by node key.
    pub fn set_right(&mut self, right: NodeKey) { self.right = right }

    /// Returns a reference to this node's bounding box.
    pub fn bounding_box(&self) -> &BoundingBox<T> { &self.bounding_box }

    /// Sets this node's bounding box to a new bounding box.
    pub fn set_bounding_box(&mut self, bounding_box: BoundingBox<T>) {
        self.bounding_box = bounding_box
    }

    /// Returns a reference to this node's random cut.
    pub fn cut(&self) -> &Cut<T> { &self.cut }

    /// Sets this node's random cut to a new random cut.
    pub fn set_cut(&mut self, cut: Cut<T>) { self.cut = cut }

    /// Returns the mass of this internal node.
    pub fn mass(&self) -> u32 { self.mass }

    /// Increments the mass at this internal node by one.
    pub fn increment_mass(&mut self) { self.mass += 1 }

    /// Decrements the mass at this internal node by one.
    pub fn decrement_mass(&mut self) { self.mass -= 1 }
}

/// An enum type representing either an [`Internal`] node or a [`Leaf`] node.
///
/// Node stored in a random cut tree are all of type `Node`. The enum consists
/// of two states: `Leaf`, which contains a [`Leaf`] type, and `Internal`,
/// which contains and [`Internal`] type.
///
/// The methods defined for this enum type are mainly for convenience in working
/// agnostically with either leaves or internal nodes.
pub enum Node<T> {
    Leaf(Leaf),
    Internal(Internal<T>),
}

impl<T: RCFFloat> Node<T> {

    /// Create a new leaf node.
    ///
    /// See [`Leaf::new`] for more information.
    pub fn new_leaf(point: PointKey) -> Self {
        Node::Leaf(Leaf::new(point))
    }

    /// Create a new internal node.
    ///
    /// See [`Internal::new`] for more information.
    pub fn new_internal(
        left: NodeKey,
        right: NodeKey,
        bounding_box: BoundingBox<T>,
        cut: Cut<T>) -> Self
    {
        Node::Internal(Internal::new(left, right, bounding_box, cut))
    }

    /// Returns the key of the parent [`Internal`] node.
    pub fn parent(&self) -> Option<NodeKey> {
        match self {
            Node::Leaf(n) => n.parent,
            Node::Internal(n) => n.parent,
        }
    }

    /// Set the parent node by node key.
    pub fn set_parent(&mut self, parent: Option<NodeKey>) {
        match self {
            Node::Leaf(n) => n.parent = parent,
            Node::Internal(n) => n.parent = parent,
        }
    }

    /// Returns the mass of this node.
    pub fn mass(&self) -> u32 {
        match self {
            Node::Leaf(n) => n.mass,
            Node::Internal(n) => n.mass,
        }
    }

    /// Set the mass of this node.
    pub fn set_mass(&mut self, mass: u32) {
        match self {
            Node::Leaf(n) => n.mass = mass,
            Node::Internal(n) => n.mass = mass,
        }
    }

    /// Increment the mass of this node by one.
    pub fn increment_mass(&mut self) {
        match self {
            Node::Leaf(n) => n.mass += 1,
            Node::Internal(n) => n.mass += 1,
        }
    }

    /// Decrement the mass of this node by one.
    pub fn decrement_mass(&mut self) {
        match self {
            Node::Leaf(n) => n.mass -= 1,
            Node::Internal(n) => n.mass -= 1,
        }
    }

    /// Get a reference to the leaf represented by this node.
    ///
    /// # Panics
    ///
    /// If the node is not a `Leaf`.
    pub fn to_leaf(&self) -> Result<&Leaf, &str> {
        match self {
            Node::Leaf(n) => Ok(n),
            Node::Internal(_) => Err("This node is not a leaf."),
        }
    }
}