/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * A copy of the License is located at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * or in the "license" file accompanying this file. This file is distributed
 * on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 * express or implied. See the License for the specific language governing
 * permissions and limitations under the License.
 */

package com.amazon.randomcutforest.tree;

import static com.amazon.randomcutforest.CommonUtils.checkArgument;
import static com.amazon.randomcutforest.CommonUtils.checkNotNull;
import static com.amazon.randomcutforest.CommonUtils.checkState;
import static com.amazon.randomcutforest.tree.Node.isLeftOf;

import java.util.Arrays;
import java.util.Random;

import com.amazon.randomcutforest.MultiVisitor;
import com.amazon.randomcutforest.Sequential;
import com.amazon.randomcutforest.Visitor;
import com.amazon.randomcutforest.sampler.WeightedPoint;

/**
 * A Random Cut Tree is a tree data structure whose leaves represent points
 * inserted into the tree and whose interior nodes represent regions of space
 * defined by Bounding Boxes and Cuts. New nodes and leaves are added to the
 * tree by making random cuts. See {@link #addPoint} for details.
 *
 * The main use of this class is to be updated with points sampled from a
 * stream, and to define traversal methods. Users can then implement a
 * {@link Visitor} which can be submitted to a traversal method in order to
 * compute a statistic from the tree.
 */
public class RandomCutTree implements ITree<Sequential<double[]>> {

    /**
     * By default, trees will not store sequence indexes.
     */
    public static final boolean DEFAULT_STORE_SEQUENCE_INDEXES_ENABLED = false;

    /**
     * By default, nodes will not store center of mass.
     */
    public static final boolean DEFAULT_CENTER_OF_MASS_ENABLED = false;

    /**
     * The flag to determine if sequence indexes are stored
     */

    private final boolean storeSequenceIndexesEnabled;
    /**
     * The flag to determine if center of mass is enabled
     */

    private final boolean centerOfMassEnabled;
    /**
     * The random number generator used create random cuts.
     */
    private final Random random;
    /**
     * The root node of the tree.
     */
    protected Node root;

    protected RandomCutTree(Builder<?> builder) {
        storeSequenceIndexesEnabled = builder.storeSequenceIndexesEnabled;
        centerOfMassEnabled = builder.centerOfMassEnabled;
        if (builder.random != null) {
            random = builder.random;
        } else {
            random = new Random();
        }
    }

    /**
     * @return a new RandomCutTree builder.
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Create a new RandomCutTree with optional arguments set to default values.
     *
     * @param randomSeed The random seed used to create the random number generator
     *                   for this tree.
     * @return a new RandomCutTree with optional arguments set to default values.
     */
    public static RandomCutTree defaultTree(long randomSeed) {
        return builder().randomSeed(randomSeed).build();
    }

    /**
     * @return a new RandomCutTree with optional arguments set to default values.
     */
    public static RandomCutTree defaultTree() {
        return builder().build();
    }

    /**
     * @return true if nodes in this tree retain the center of mass, false
     *         otherwise.
     */
    public boolean centerOfMassEnabled() {
        return centerOfMassEnabled;
    }

    /**
     * @return true if points in this tree are saved with sequence indexes, false
     *         otherwise.
     */
    public boolean storeSequenceIndexesEnabled() {
        return storeSequenceIndexesEnabled;
    }

    /**
     * Return a new {@link Cut}, which is chosen uniformly over the space of
     * possible cuts for the given bounding box.
     *
     * @param random A random number generator
     * @param box    A bounding box that we want to find a random cut for.
     * @return A new Cut corresponding to a random cut in the bounding box.
     */
    static Cut randomCut(Random random, BoundingBox box) {
        double rangeSum = box.getRangeSum();
        checkArgument(rangeSum > 0, "box.getRangeSum() must be greater than 0");

        double breakPoint = random.nextDouble() * rangeSum;

        for (int i = 0; i < box.getDimensions(); i++) {
            double range = box.getRange(i);
            if (breakPoint <= range) {
                double cutValue = box.getMinValue(i) + breakPoint;

                // Random cuts have to take a value in the half-open interval [minValue,
                // maxValue) to ensure that a
                // Node has a valid left child and right child.
                if ((cutValue == box.getMaxValue(i)) && (box.getMinValue(i) < box.getMaxValue(i))) {
                    cutValue = Math.nextAfter(box.getMaxValue(i), box.getMinValue(i));
                }

                return new Cut(i, cutValue);
            }
            breakPoint -= range;
        }

        throw new IllegalStateException("The break point did not lie inside the expected range");
    }

    /**
     * Replace a Node in the tree structure. This method replaces oldNode with
     * newNode as a child of oldNode.getParent().
     *
     * @param oldNode The node we are replacing.
     * @param newNode The new node we are inserting into the tree.
     */
    static void replaceNode(Node oldNode, Node newNode) {

        Node parent = oldNode.getParent();
        if (parent != null) {
            if (parent.getLeftChild() == oldNode) {
                parent.setLeftChild(newNode);
            } else {
                parent.setRightChild(newNode);
            }
        }
        newNode.setParent(parent);
    }

    /**
     * Return the sibling of a non-root node. Note that every non-leaf node in a
     * Random Cut Tree has two children.
     *
     * @param node The node whose sibling we are requesting.
     * @return the sibling of node in the tree.
     */
    static Node getSibling(Node node) {
        checkNotNull(node.getParent(), "node parent must not be null");

        Node parent = node.getParent();
        if (parent.getLeftChild() == node) {
            return parent.getRightChild();
        } else if (parent.getRightChild() == node) {
            return parent.getLeftChild();
        } else {
            throw new IllegalArgumentException("node parent does not link back to node");
        }
    }

    /**
     * Delete the given point from this tree.
     *
     * @param point A point in the tree that we wish to delete.
     */
    @Override
    public void deletePoint(Sequential<double[]> point) {
        checkState(root != null, "root must not be null");
        deletePoint(root, point.getValue(), point.getSequenceIndex());
    }

    /**
     * Delete the given point from this tree.
     *
     * @param weightedPoint A point in the tree that we wish to delete.
     */
    public void deletePoint(WeightedPoint weightedPoint) {
        checkState(root != null, "root must not be null");
        deletePoint(root, weightedPoint.getPoint(), weightedPoint.getSequenceIndex());
    }

    /**
     * This function deletes the point from the tree recursively. We traverse the
     * tree based on the cut stored in each interior node until we reach a leaf
     * node. We then delete the leaf node if the mass of the node is 1, otherwise we
     * reduce the mass by 1.
     *
     * @param node          node that we are visiting in the tree.
     * @param point         the point that is being deleted from the tree.
     * @param sequenceIndex the insertion index of the point being deleted from the
     *                      tree.
     */
    void deletePoint(Node node, double[] point, long sequenceIndex) {
        if (node.isLeaf()) {
            if (!node.leafPointEquals(point)) {
                throw new IllegalStateException(Arrays.toString(point) + " " + Arrays.toString(node.getLeafPoint())
                        + " " + Arrays.equals(node.getLeafPoint(), point) + " Inconsistency in trees in delete step.");
            }
            if (storeSequenceIndexesEnabled) {
                if (!node.getSequenceIndexes().contains(sequenceIndex)) {
                    throw new IllegalStateException("Error in sequence index. Inconsistency in trees in delete step.");
                }
            }

            /*
             * the above assumes that sequence indexes are unique ... which is true for the
             * specific sampler used
             */

            if (node.getMass() > 1) {
                node.decrementMass();
                if (storeSequenceIndexesEnabled) {
                    node.deleteSequenceIndex(sequenceIndex);
                }
                return;
            }

            Node parent = node.getParent();
            if (parent == null) {
                root = null;
                return;
            }

            Node grandParent = parent.getParent();
            if (grandParent == null) {
                root = getSibling(node);
                root.setParent(null);
            } else {
                Node sibling = getSibling(node);
                replaceNode(parent, sibling);
            }
            return;
        }

        // node is not a leaf

        if (isLeftOf(point, node)) {
            deletePoint(node.getLeftChild(), point, sequenceIndex);
        } else {
            deletePoint(node.getRightChild(), point, sequenceIndex);
        }

        BoundingBox leftBox = node.getLeftChild().getBoundingBox();
        BoundingBox rightBox = node.getRightChild().getBoundingBox();
        node.setBoundingBox(leftBox.getMergedBox(rightBox));
        node.decrementMass();
        if (centerOfMassEnabled) {
            node.subtractFromPointSum(point);
        }
    }

    /**
     * Add a new point to the tree.
     *
     * @param point The point to add to the tree.
     */
    @Override
    public void addPoint(Sequential<double[]> point) {
        if (root == null) {
            root = newLeafNode(point.getValue(), point.getSequenceIndex());
        } else {
            addPoint(root, point.getValue(), point.getSequenceIndex());
        }
    }

    /**
     * Add a new point to the tree.
     *
     * @param weightedPoint The point to add to the tree.
     */
    public void addPoint(WeightedPoint weightedPoint) {
        if (root == null) {
            root = newLeafNode(weightedPoint.getPoint(), weightedPoint.getSequenceIndex());
        } else {
            addPoint(root, weightedPoint.getPoint(), weightedPoint.getSequenceIndex());
        }
    }

    /**
     * This function adds a point to the tree recursively. The algorithm for adding
     * a point is as follows:
     * <ol>
     * <li>At the current node we create a new bounding box by merging the point
     * with the existing box.</li>
     * <li>We pick a dimension and a value of cut.</li>
     * <li>If the cut falls outside the existing box we create a new node and
     * replace the current node with the new node. We then add the current node as
     * the child node to the new node. We create another leaf node containing the
     * point that we want to insert and add it as the other child of the node that
     * we have created earlier.</li>
     * <li>If the cut falls inside the existing box. We follow the cut of the
     * existing box and move to right or left child of the current node based on the
     * existing cut.</li>
     * </ol>
     *
     * @param node          the current node in the tree we are on
     * @param point         the point that we want to add to the tree
     * @param sequenceIndex the insertion index of the point being added to the tree
     */

    private void addPoint(Node node, double[] point, long sequenceIndex) {
        if (node.isLeaf() && node.leafPointEquals(point)) {
            // the inserted point is equal to an existing leaf point
            node.incrementMass();
            if (storeSequenceIndexesEnabled) {
                node.addSequenceIndex(sequenceIndex);
            }
            return;
        }

        // either the node is not a leaf, or else it's a leaf node containing a
        // different point

        BoundingBox existingBox = node.getBoundingBox();
        BoundingBox mergedBox = existingBox.getMergedBox(point);

        if (!existingBox.contains(point)) {

            // Propose a random cut

            Cut cut = randomCut(random, mergedBox);
            int splitDimension = cut.getDimension();
            double splitValue = cut.getValue();
            double minValue = existingBox.getMinValue(splitDimension);
            double maxValue = existingBox.getMaxValue(splitDimension);

            // if the proposed cut separates the new point from the existing bounding box:
            // * create a new leaf node for the point
            // * make it a sibling of the existing bounding box
            // * make the new leaf node and the existing node children of a new node with
            // the merged bounding box

            if (minValue > splitValue || maxValue <= splitValue) {
                Node leaf = newLeafNode(point, sequenceIndex);
                Node mergedNode = minValue > splitValue ? newNode(leaf, node, cut, mergedBox)
                        : newNode(node, leaf, cut, mergedBox);
                if (node.getParent() == null) {
                    root = mergedNode;
                } else {
                    replaceNode(node, mergedNode);
                }
                leaf.setParent(mergedNode);
                node.setParent(mergedNode);

                return;
            }
        }

        // Either the new point is contained in this node's bounding box, or else the
        // proposed cut did not separate
        // it from the existing bounding box. Try again at the next level.

        if (isLeftOf(point, node)) {
            addPoint(node.getLeftChild(), point, sequenceIndex);
        } else {
            addPoint(node.getRightChild(), point, sequenceIndex);
        }

        node.setBoundingBox(mergedBox);
        node.incrementMass();
        if (centerOfMassEnabled) {
            node.addToPointSum(point);
        }
    }

    /*
     * In the method below, the newCopy() operator is invoked after traversing the
     * left side by design. It is not currently being used; but in theory this
     * allows inorder traversal which can simulate more functions in comparison to
     * post-order traversal. One can (but would not be advised to) use the
     * properties of the left traversal to inform the right traversal. Such a
     * strategy may be useful in correcting known biases (for example, the algorithm
     * is biased lower etc.) but in most cases the newCopy() should just use the
     * size information of the data structure and copy the information that has been
     * passed to it from its parent.
     */

    /**
     * Starting from the root, traverse the canonical path to a leaf node and visit
     * the nodes along the path. The canonical path is determined by the input
     * point: at each interior node, we select the child node by comparing the
     * node's {@link Cut} to the corresponding coordinate value in the input point.
     * The method recursively traverses to the leaf node first and then invokes the
     * visitor on each node in reverse order. That is, if the path to the leaf node
     * determined by the input point is root, node1, node2, ..., node(N-1), nodeN,
     * leaf; then we will first invoke visitor::acceptLeaf on the leaf node, and
     * then we will invoke visitor::accept on the remaining nodes in the following
     * order: nodeN, node(N-1), ..., node2, node1, and root.
     *
     * @param point   A point which determines the traversal path from the root to a
     *                leaf node.
     * @param visitor A visitor that will be invoked for each node on the path.
     * @param <R>     The return type of the Visitor.
     * @return the value of {@link Visitor#getResult()}} after the traversal.
     */
    public <R> R traverseTree(double[] point, Visitor<R> visitor) {
        checkState(root != null, "this tree doesn't contain any nodes");
        traversePathToLeafAndVisitNodes(point, visitor, root, 0);
        return visitor.getResult();
    }

    private <R> void traversePathToLeafAndVisitNodes(double[] point, Visitor<R> visitor, Node currentNode,
            int depthOfNode) {
        if (currentNode.isLeaf()) {
            visitor.acceptLeaf(currentNode, depthOfNode);
        } else {
            Node childNode = isLeftOf(point, currentNode) ? currentNode.getLeftChild() : currentNode.getRightChild();
            traversePathToLeafAndVisitNodes(point, visitor, childNode, depthOfNode + 1);
            visitor.accept(currentNode, depthOfNode);
        }
    }

    /**
     * This is a traversal method which follows the standard traveral path (defined
     * in {@link #traverseTree(double[], Visitor)}) but at Node in checks to see
     * whether the visitor should split. If a split is triggered, then independent
     * copies of the visitor are sent down each branch of the tree and then merged
     * before propogating the result.
     *
     * @param point   A point which determines the traversal path from the root to a
     *                leaf node.
     * @param visitor A visitor that will be invoked for each node on the path.
     * @param <R>     The return type of the Visitor.
     * @return the value of {@link Visitor#getResult()}} after the traversal.
     */
    public <R> R traverseTreeMulti(double[] point, MultiVisitor<R> visitor) {
        checkNotNull(point, "point must not be null");
        checkNotNull(visitor, "visitor must not be null");
        checkState(root != null, "this tree doesn't contain any nodes");
        traverseTreeMulti(point, visitor, root, 0);
        return visitor.getResult();
    }

    private <R> void traverseTreeMulti(double[] point, MultiVisitor<R> visitor, Node currentNode, int depthOfNode) {
        if (currentNode.isLeaf()) {
            visitor.acceptLeaf(currentNode, depthOfNode);
        } else if (visitor.trigger(currentNode)) {
            traverseTreeMulti(point, visitor, currentNode.getLeftChild(), depthOfNode + 1);
            MultiVisitor<R> newVisitor = visitor.newCopy();
            traverseTreeMulti(point, newVisitor, currentNode.getRightChild(), depthOfNode + 1);
            visitor.combine(newVisitor);
            visitor.accept(currentNode, depthOfNode);
        } else {
            Node childNode = isLeftOf(point, currentNode) ? currentNode.getLeftChild() : currentNode.getRightChild();
            traverseTreeMulti(point, visitor, childNode, depthOfNode + 1);
            visitor.accept(currentNode, depthOfNode);
        }
    }

    private Node newLeafNode(double[] point, long sequenceIndex) {
        Node node = new Node(point);
        node.setMass(1);
        if (storeSequenceIndexesEnabled) {
            node.addSequenceIndex(sequenceIndex);
        }
        return node;
    }

    private Node newNode(Node leftChild, Node rightChild, Cut cut, BoundingBox box) {
        Node node = new Node(leftChild, rightChild, cut, box, centerOfMassEnabled);

        if (leftChild != null) {
            node.addMass(leftChild.getMass());
            if (centerOfMassEnabled) {
                node.addToPointSum(leftChild.getPointSum());
            }
        }

        if (rightChild != null) {
            node.addMass(rightChild.getMass());
            if (centerOfMassEnabled) {
                node.addToPointSum(rightChild.getPointSum());
            }
        }

        return node;
    }

    /**
     * @return the root node in the tree
     */
    public Node getRoot() {
        return root;
    }

    /**
     * @return the total mass in the tree.
     */
    @Override
    public int getMass() {
        return root == null ? 0 : root.getMass();
    }

    public static class Builder<T extends Builder<T>> {
        private boolean storeSequenceIndexesEnabled = DEFAULT_STORE_SEQUENCE_INDEXES_ENABLED;
        private boolean centerOfMassEnabled = DEFAULT_CENTER_OF_MASS_ENABLED;
        private Random random = null;

        public T storeSequenceIndexesEnabled(boolean storeSequenceIndexesEnabled) {
            this.storeSequenceIndexesEnabled = storeSequenceIndexesEnabled;
            return (T) this;
        }

        public T centerOfMassEnabled(boolean centerOfMassEnabled) {
            this.centerOfMassEnabled = centerOfMassEnabled;
            return (T) this;
        }

        public T random(Random random) {
            this.random = random;
            return (T) this;
        }

        public T randomSeed(long randomSeed) {
            random = new Random(randomSeed);
            return (T) this;
        }

        public RandomCutTree build() {
            return new RandomCutTree(this);
        }
    }
}
