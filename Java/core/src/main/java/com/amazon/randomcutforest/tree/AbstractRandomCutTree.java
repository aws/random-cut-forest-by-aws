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

import java.util.Arrays;
import java.util.Optional;
import java.util.Random;

import com.amazon.randomcutforest.IMultiVisitorFactory;
import com.amazon.randomcutforest.IVisitorFactory;
import com.amazon.randomcutforest.MultiVisitor;
import com.amazon.randomcutforest.RandomCutForest;
import com.amazon.randomcutforest.Visitor;
import com.amazon.randomcutforest.config.Config;

/**
 * A Compact Random Cut Tree is a tree data structure whose leaves represent
 * points inserted into the tree and whose interior nodes represent regions of
 * space defined by Bounding Boxes and Cuts. New nodes and leaves are added to
 * the tree by making random cuts.
 *
 * The offsets are encoded as follows: an offset greater or equal maxSize
 * corresponds to a leaf node of offset (offset - maxSize) otherwise the offset
 * corresponds to an internal node
 *
 * The main use of this class is to be updated with points sampled from a
 * stream, and to define traversal methods. Users can then implement a
 * {@link Visitor} which can be submitted to a traversal method in order to
 * compute a statistic from the tree.
 */
public abstract class AbstractRandomCutTree<Point, NodeReference, PointReference>
        implements ITree<PointReference, Point> {

    /**
     * The index value used to represent the absence of a node. For example, when
     * the tree is created the root node index will be NULL. After a point is added
     * and a root node is created, the root node's parent will be NULL, and so on.
     */

    private Random testRandom;
    private long randomSeed;
    protected NodeReference root;
    public final boolean centerOfMassEnabled;
    public final boolean storeSequenceIndexesEnabled;
    protected double boundingBoxCacheFraction = 1.0;
    Random cacheRandom = new Random(0);
    protected int outputAfter;

    protected AbstractRandomCutTree(AbstractRandomCutTree.Builder<?> builder) {
        if (builder.random != null) {
            this.testRandom = builder.random;
        } else {
            this.randomSeed = builder.randomSeed;
        }
        this.centerOfMassEnabled = builder.centerOfMassEnabled;
        this.storeSequenceIndexesEnabled = builder.storeSequenceIndexesEnabled;
        this.boundingBoxCacheFraction = builder.boundingBoxCacheFraction;

        // This should be set to an appropriate value in a subclass
        outputAfter = Integer.MAX_VALUE;
    }

    @Override
    public <T> void setConfig(String name, T value, Class<T> clazz) {
        if (Config.BOUNDING_BOX_CACHE_FRACTION.equals(name)) {
            checkArgument(Double.class.isAssignableFrom(clazz),
                    String.format("Setting '%s' must be a double value", name));
            setBoundingBoxCacheFraction((Double) value);
        } else {
            throw new IllegalArgumentException("Unsupported configuration setting: " + name);
        }
    }

    @Override
    public <T> T getConfig(String name, Class<T> clazz) {
        checkNotNull(clazz, "clazz must not be null");
        if (Config.BOUNDING_BOX_CACHE_FRACTION.equals(name)) {
            checkArgument(clazz.isAssignableFrom(Double.class),
                    String.format("Setting '%s' must be a double value", name));
            return clazz.cast(boundingBoxCacheFraction);
        } else {
            throw new IllegalArgumentException("Unsupported configuration setting: " + name);
        }
    }

    // dynamically change the fraction of the new nodes which caches their bounding
    // boxes
    // 0 would mean less space usage, but slower throughput
    // 1 would imply larger space but better throughput
    public void setBoundingBoxCacheFraction(double fraction) {
        checkArgument(0 <= fraction && fraction <= 1, "incorrect parameter");
        boundingBoxCacheFraction = fraction;
    }

    /**
     * Return a new {@link Cut}, which is chosen uniformly over the space of
     * possible cuts for the given bounding box.
     *
     * @param random A random number generator
     * @param box    A bounding box that we want to find a random cut for.
     * @return A new Cut corresponding to a random cut in the bounding box.
     */
    protected Cut randomCut(Random random, AbstractBoundingBox<?> box) {
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

    // decides the path taken in the abstract tree at update time
    protected abstract boolean leftOf(Point point, int cutDimension, double cutValue);

    // checks equality based on precision
    protected abstract boolean equals(Point oldPoint, Point point);

    // checks equality based on reference
    protected abstract boolean referenceEquals(PointReference oldPointRef, PointReference pointRef);

    // prints a point based on precision
    protected abstract String toString(Point point);

    // creates a bounding box based on two points, it is a very common occurrence
    protected abstract AbstractBoundingBox<Point> getInternalTwoPointBox(PointReference first, PointReference second);

    /**
     * Takes a node reference and produces a bounding box; cannot return null If
     * bounding boxes are cached then the bounding boxes of the descendants are
     * populated as well. Otherwise a bounding box is continually modified in place
     * to build the bounding box.
     *
     * @param nodeReference reference of node
     * @return the bounding box corresponding to the node, irrespective of any
     *         caching
     */
    protected AbstractBoundingBox<Point> getBoundingBox(NodeReference nodeReference) {
        if (isLeaf(nodeReference)) {
            return getLeafBoxFromLeafNode(nodeReference);
        }
        return constructBoxInPlace(nodeReference);
    }

    // function for getting the bounding boxes for leaf nodes
    protected abstract AbstractBoundingBox<Point> getLeafBoxFromLeafNode(NodeReference nodeReference);

    // function for getting the bounding boxes for leaf nodes that can be mutated in
    // place
    protected abstract AbstractBoundingBox<Point> getMutableLeafBoxFromLeafNode(NodeReference nodeReference);

    protected abstract AbstractBoundingBox<Point> constructBoxInPlace(NodeReference nodeReference);

    // gets the actual values of a point from its reference which can be
    // Integer/direct reference
    abstract Point getPointFromPointReference(PointReference pointIndex);

    // gets the reference to a point from a leaf node
    abstract PointReference getPointReference(NodeReference node);

    // gets the leaf point associated with a leaf node
    Point getPointFromLeafNode(NodeReference node) {
        return getPointFromPointReference(getPointReference(node));
    }

    // is a leaf or otherwise
    abstract protected boolean isLeaf(NodeReference node);

    // decreases mass of a node and returns it
    abstract protected int decrementMass(NodeReference node);

    // increases mass of a node and returns it
    abstract protected int incrementMass(NodeReference node);

    // returns the sibling of a node
    abstract protected NodeReference getSibling(NodeReference nodeReference);

    abstract protected NodeReference getParent(NodeReference nodeReference);

    abstract protected void setParent(NodeReference nodeReference, NodeReference parent);

    abstract protected void delete(NodeReference nodeReference);

    // only valid for internal nodes
    abstract protected int getCutDimension(NodeReference nodeReference);

    // only valid for internal nodes
    abstract protected double getCutValue(NodeReference nodeReference);

    abstract protected NodeReference getLeftChild(NodeReference nodeReference);

    abstract protected NodeReference getRightChild(NodeReference nodeReference);

    // replaces child (with designated parent) by otherNode
    abstract void replaceChild(NodeReference parent, NodeReference child, NodeReference otherNode);

    // implements the delete step in RCF
    abstract protected void replaceNodeBySibling(NodeReference grandParent, NodeReference Parent, NodeReference node);

    // creates a new leaf node
    abstract protected NodeReference addLeaf(PointReference pointIndex);

    // creates an internal node
    abstract protected NodeReference addNode(NodeReference leftChild, NodeReference rightChild, int cutDimension,
            double cutValue, int mass);

    // increases the mass of all ancestor nodes when an added point is internal to
    // some bounding box
    // note the bounding boxes of these ancestor nodes will remain unchanged
    abstract protected void increaseMassOfAncestors(NodeReference mergedNode);

    abstract protected void decreaseMassOfAncestors(NodeReference mergedNode);

    abstract protected int getMass(NodeReference nodeReference);

    abstract protected void addSequenceIndex(NodeReference node, long uniqueSequenceNumber);

    abstract protected void deleteSequenceIndex(NodeReference node, long uniqueSequenceNumber);

    // the following does not need the information of the point in the current time
    // however that information may be of use for different type of Points
    abstract void recomputePointSum(NodeReference node);

    void updateAncestorPointSum(NodeReference node) {
        if (centerOfMassEnabled) {
            NodeReference tempNode = node;
            while (tempNode != null) {
                recomputePointSum(tempNode);
                tempNode = getParent(tempNode);
            }
        }
    }

    abstract AbstractBoundingBox<Point> recomputeBox(NodeReference node);

    // manages the bounding boxes and center of mass
    void updateAncestorNodesAfterDelete(NodeReference nodeReference, PointReference pointReference) {
        Point point = getPointFromPointReference(pointReference);
        NodeReference tempNode = nodeReference;
        boolean boxNeedsUpdate = boundingBoxCacheFraction > 0;
        while (boxNeedsUpdate && tempNode != null) {
            AbstractBoundingBox<Point> box = recomputeBox(tempNode);
            boxNeedsUpdate = (box == null) || !(box.contains(point));
            tempNode = getParent(tempNode);
        }
        updateAncestorPointSum(nodeReference);
    }

    /**
     * finds the reference to the leaf node which corresponds to the path followed
     * in the tree by an input point
     * 
     * @param point point
     * @return reference of the leaf node
     */
    NodeReference findLeaf(Point point) {
        if (root == null) {
            return null;
        }
        NodeReference nodeReference = root;
        while (!isLeaf(nodeReference)) {
            nodeReference = (leftOf(point, getCutDimension(nodeReference), getCutValue(nodeReference)))
                    ? getLeftChild(nodeReference)
                    : getRightChild(nodeReference);
        }
        return nodeReference;
    }

    NodeReference findLeafAndVerify(PointReference pointReference) {
        Point point = getPointFromPointReference(pointReference);
        NodeReference nodeReference = findLeaf(point);
        // the following should suffice for compact in most cases
        // unless the deletion is for an equivalent point
        if (referenceEquals(pointReference, getPointReference(nodeReference))) {
            return nodeReference;
        }
        Point oldPoint = getPointFromLeafNode(nodeReference);
        if (!equals(oldPoint, point)) {
            throw new IllegalStateException(toString(point) + " " + toString(getPointFromLeafNode(nodeReference)) + " "
                    + nodeReference + " node " + false + " Inconsistency in trees.");
        }
        return nodeReference;
    }

    /**
     * the following function returns the number of copies of a point in the tree
     * and switches the reference to the provided reference. This may be useful for
     * collating duplicate points across trees.
     * 
     * @param leafReference reference of the leaf node
     * @param newRef        reference of the point stored at the leaf node
     * @return the number of copies of the point at leaf node
     */
    abstract void setLeafPointReference(NodeReference leafReference, PointReference newRef);

    /**
     * the following switches the reference of any copy of the point used to the new
     * reference if the point does not exist then an exception is raised
     * 
     * @param newRef the new reference of the point
     */
    protected void switchLeafReference(PointReference newRef) {
        checkNotNull(newRef, " cannot be null ");
        NodeReference nodeReference = findLeafAndVerify(newRef);
        setLeafPointReference(nodeReference, newRef);
    }

    /**
     * returns the reference to a point used by the tree, or null if the point is
     * not being used
     * 
     * @param newReference a new reference to the point
     * @return the exisitng reference to that point, or null if the actual point is
     *         not in use
     */
    protected PointReference getEquivalentReference(PointReference newReference) {
        Point point = getPointFromPointReference(newReference);
        NodeReference nodeReference = findLeaf(point);
        if (nodeReference != null) {
            PointReference reference = getPointReference(nodeReference);
            if (!equals(point, getPointFromPointReference(reference))) {
                return null;
            }
            return reference;
        }
        return null;
    }

    /**
     * the following returns the number of copies of a point given a reference. If
     * the point is not in the tree then an exception is raised; but if the
     * reference in the tree is not the same then the answer is zero
     * 
     * @param reference a reference to a point
     * @return the number of copies the exact reference is present
     */

    protected int getCopiesOfReference(PointReference reference) {
        checkNotNull(reference, " reference cannot be null ");
        NodeReference nodeReference = findLeafAndVerify(reference);
        return (reference == getPointReference(nodeReference)) ? getMass(nodeReference) : 0;
    }

    /**
     * deletes a point from the tree
     * 
     * @param pointReference the reference of the point
     * @param sequenceNumber the sequence number (in case we are storing that in the
     *                       leaves)
     * @return the reference used by the leaf node (after verifying equality)
     */

    @Override
    public PointReference deletePoint(PointReference pointReference, long sequenceNumber) {
        checkState(root != null, "root must not be null");

        NodeReference nodeReference = findLeafAndVerify(pointReference);
        if (storeSequenceIndexesEnabled) {
            deleteSequenceIndex(nodeReference, sequenceNumber);
        }
        PointReference returnVal = getPointReference(nodeReference);

        decreaseMassOfAncestors(nodeReference);
        // the node is not included as its ancestor
        // the mass of the parent needs to be 0 for it to be free to be reused

        // decrease mass for the delete
        if (decrementMass(nodeReference) > 0) {
            updateAncestorPointSum(getParent(nodeReference));
            return returnVal;
        }

        NodeReference parent = getParent(nodeReference);

        if (parent == null) {
            root = null;
            delete(nodeReference);
            return returnVal;
        }
        // parent is guaranteed to be an internal node

        NodeReference grandParent = getParent(parent);
        if (grandParent == null) {
            root = getSibling(nodeReference);
            setParent(root, null);
        } else {
            replaceNodeBySibling(grandParent, parent, nodeReference);
            updateAncestorNodesAfterDelete(grandParent, pointReference);
        }
        delete(nodeReference);
        delete(parent);
        return returnVal;
    }

    abstract void setCachedBox(NodeReference node, AbstractBoundingBox<Point> savedBox);

    abstract void addToBox(NodeReference node, Point point);

    void updateAncestorNodesAfterAdd(AbstractBoundingBox<Point> savedBox, NodeReference mergedNode, Point point,
            NodeReference parentIndex) {
        increaseMassOfAncestors(mergedNode);
        if (boundingBoxCacheFraction > 0) {
            setCachedBox(mergedNode, savedBox);
            NodeReference tempNode = getParent(mergedNode);
            while (tempNode != null && !tempNode.equals(parentIndex)) {
                addToBox(tempNode, point);
                tempNode = getParent(tempNode);
            }
        }
        if (centerOfMassEnabled) {
            updateAncestorPointSum(mergedNode);
        }
    }

    /**
     * adds a point to the tree
     *
     * @param pointReference the reference of the point to be added
     * @param sequenceNumber the index of the point in PointStore and the
     *                       corresponding timestamp
     *
     * @return the reference of the inserted point, which can be the input or a
     *         reference to a previously seen copy
     */

    public PointReference addPoint(PointReference pointReference, long sequenceNumber) {

        NodeReference leafNodeForAdd;
        if (root == null) {
            leafNodeForAdd = root = addLeaf(pointReference);
        } else {
            Point point = getPointFromPointReference(pointReference);
            NodeReference savedSiblingNode;
            Cut savedCut;
            AbstractBoundingBox<Point> savedBox;
            AbstractBoundingBox<Point> currentUnmergedBox;

            NodeReference followReference = findLeaf(point);

            PointReference leafPointReference = getPointReference(followReference);
            Point oldPoint = (leafPointReference == null) ? null : getPointFromPointReference(leafPointReference);
            if (leafPointReference == null || equals(oldPoint, point)) {
                // the inserted point is equal to an existing leaf point
                if (leafPointReference == null) {
                    setLeafPointReference(followReference, pointReference);
                }
                incrementMass(followReference);
                increaseMassOfAncestors(followReference);
                if (storeSequenceIndexesEnabled) {
                    addSequenceIndex(followReference, sequenceNumber);
                }
                updateAncestorPointSum(getParent(followReference));
                return getPointReference(followReference);
                // at a leaf and found a previous copy
            }

            Random random = (testRandom != null) ? testRandom : new Random(randomSeed);
            randomSeed = (testRandom != null) ? randomSeed : random.nextLong();

            // construct a potential cut
            savedBox = getInternalTwoPointBox(pointReference, leafPointReference);
            savedCut = randomCut(random, savedBox);
            currentUnmergedBox = getMutableLeafBoxFromLeafNode(followReference);
            savedSiblingNode = followReference;

            // now iterative proceed up the tree and try to construct a cut
            assert currentUnmergedBox != null : "incorrect state";

            followReference = getParent(followReference);

            boolean resolved = false;
            while (!resolved && followReference != null) {
                AbstractBoundingBox<Point> existingBox;
                if (boundingBoxCacheFraction > 0) {
                    // if the boxes are being cached, use the box if present, otherwise
                    // generate and cache the box
                    existingBox = getBoundingBox(followReference);
                } else {
                    NodeReference sibling = (leftOf(point, getCutDimension(followReference),
                            getCutValue(followReference))) ? getRightChild(followReference)
                                    : getLeftChild(followReference);
                    // the boxes are not present, merge the bounding box of the sibling of the last
                    // seen child to the stored box in the state and save it
                    existingBox = currentUnmergedBox.addBox(getBoundingBox(sibling));
                    // note that existingBox will remain mutable
                }

                if (existingBox.contains(point)) {
                    resolved = true; // no further cuts are feasible as we move up the tree
                } else {
                    // a cut is feasible at this level
                    // generate a new cut and see if it separates the new point
                    AbstractBoundingBox<Point> mergedBox = existingBox.copy().addPoint(point);
                    Cut cut = randomCut(random, mergedBox);
                    // avoid generation of mergedBox?
                    int splitDimension = cut.getDimension();
                    double splitValue = cut.getValue();
                    double minValue = existingBox.getMinValue(splitDimension);
                    double maxValue = existingBox.getMaxValue(splitDimension);

                    if (minValue > splitValue || maxValue <= splitValue) {
                        // the cut separates the new point; update the state to store information
                        // about the most recent cut
                        savedSiblingNode = followReference;
                        savedCut = cut;
                        savedBox = mergedBox;
                    }
                    currentUnmergedBox = existingBox;
                    followReference = getParent(followReference);
                }
            }

            NodeReference newParent = followReference;

            // resolve the add, by modifying (if present) the bounding boxes corresponding
            // to current node (nodeReference) all the way to the root
            int cutDimension = savedCut.getDimension();
            double cutValue = savedCut.getValue();
            int oldMass = getMass(savedSiblingNode);
            leafNodeForAdd = addLeaf(pointReference);
            NodeReference mergedNode = leftOf(point, cutDimension, cutValue)
                    ? addNode(leafNodeForAdd, savedSiblingNode, cutDimension, cutValue, (oldMass + 1))
                    : addNode(savedSiblingNode, leafNodeForAdd, cutDimension, cutValue, (oldMass + 1));

            NodeReference oldParent = getParent(savedSiblingNode);
            if (oldParent == null) {
                root = mergedNode;
            } else {
                replaceChild(oldParent, savedSiblingNode, mergedNode);
            }

            setParent(leafNodeForAdd, mergedNode);
            setParent(savedSiblingNode, mergedNode);
            // manage bounding boxes, including caching, as well as centerOfMass
            updateAncestorNodesAfterAdd(savedBox, mergedNode, point, newParent);
        }
        if (storeSequenceIndexesEnabled) {
            addSequenceIndex(leafNodeForAdd, sequenceNumber);
        }

        return pointReference;

    }

    /**
     * The following function is used by Visitors
     *
     * @param nodeReference node identifier of a leaf node
     * @return actual values in double precision
     */
    abstract protected double[] getPoint(NodeReference nodeReference);

    /**
     * Used by visitors to test left/right
     * 
     * @param point         Actual point in double precision
     * @param nodeReference identifier of the node
     * @return left/right decision
     */
    protected boolean leftOf(double[] point, NodeReference nodeReference) {
        return point[getCutDimension(nodeReference)] <= getCutValue(nodeReference);
    }

    // provides a view of the root node
    abstract protected INode<NodeReference> getNode(NodeReference node);

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
     * @param point          A point which determines the traversal path from the
     *                       root to a leaf node.
     * @param visitorFactory A visitor that will be invoked for each node on the
     *                       path.
     * @param <R>            The return type of the Visitor.
     * @return the value of {@link Visitor#getResult()}} after the traversal.
     */
    @Override
    public <R> R traverse(double[] point, IVisitorFactory<R> visitorFactory) {
        checkState(root != null, "this tree doesn't contain any nodes");
        Visitor<R> visitor = visitorFactory.newVisitor(this, point);
        traversePathToLeafAndVisitNodes(projectToTree(point), visitor, root, 0);
        return visitorFactory.liftResult(this, visitor.getResult());
    }

    private <R> void traversePathToLeafAndVisitNodes(double[] point, Visitor<R> visitor, NodeReference node,
            int depthOfNode) {
        if (isLeaf(node)) {
            visitor.acceptLeaf(getNode(node), depthOfNode);
        } else {
            NodeReference nextNode = leftOf(point, node) ? getLeftChild(node) : getRightChild(node);
            traversePathToLeafAndVisitNodes(point, visitor, nextNode, depthOfNode + 1);
            visitor.accept(getNode(node), depthOfNode);
        }
    }

    /**
     * This is a traversal method which follows the standard traversal path (defined
     * in {@link #traverse(double[], IVisitorFactory)}) but at Node in checks to see
     * whether the visitor should split. If a split is triggered, then independent
     * copies of the visitor are sent down each branch of the tree and then merged
     * before propagating the result.
     *
     * @param point          A point which determines the traversal path from the
     *                       root to a leaf node.
     * @param visitorFactory A visitor that will be invoked for each node on the
     *                       path.
     * @param <R>            The return type of the Visitor.
     * @return the value of {@link Visitor#getResult()}} after the traversal.
     */

    @Override
    public <R> R traverseMulti(double[] point, IMultiVisitorFactory<R> visitorFactory) {
        checkNotNull(point, "point must not be null");
        checkNotNull(visitorFactory, "visitor must not be null");
        checkState(root != null, "this tree doesn't contain any nodes");
        MultiVisitor<R> visitor = visitorFactory.newVisitor(this, point);
        traverseTreeMulti(projectToTree(point), visitor, root, 0);
        return visitorFactory.liftResult(this, visitor.getResult());
    }

    private <R> void traverseTreeMulti(double[] point, MultiVisitor<R> visitor, NodeReference node, int depthOfNode) {
        if (isLeaf(node)) {
            visitor.acceptLeaf(getNode(node), depthOfNode);
        } else {
            if (visitor.trigger(getNode(node))) {
                traverseTreeMulti(point, visitor, getLeftChild(node), depthOfNode + 1);
                MultiVisitor<R> newVisitor = visitor.newCopy();
                traverseTreeMulti(point, newVisitor, getRightChild(node), depthOfNode + 1);
                visitor.combine(newVisitor);
            } else {
                NodeReference nextNode = leftOf(point, node) ? getLeftChild(node) : getRightChild(node);
                traverseTreeMulti(point, visitor, nextNode, depthOfNode + 1);
            }
            visitor.accept(getNode(node), depthOfNode);
        }
    }

    /**
     *
     * @return the mass of the tree
     */
    @Override
    public int getMass() {
        return root == null ? 0 : getMass(root);
    }

    /**
     *
     * @return the root of the tree
     */

    public NodeReference getRoot() {
        return root;
    }

    /**
     * returns the number of samples that needs to be seen before returning
     * meaningful inference
     */
    public int getOutputAfter() {
        return outputAfter;
    }

    @Override
    public boolean isOutputReady() {
        return getMass() >= outputAfter;
    }

    public double[] projectToTree(double[] point) {
        return Arrays.copyOf(point, point.length);
    }

    public double[] liftFromTree(double[] result) {
        return Arrays.copyOf(result, result.length);
    }

    public int[] projectMissingIndices(int[] list) {
        return Arrays.copyOf(list, list.length);
    }

    public long getRandomSeed() {
        return randomSeed;
    }

    public static class Builder<T> {
        protected boolean storeSequenceIndexesEnabled = RandomCutForest.DEFAULT_STORE_SEQUENCE_INDEXES_ENABLED;
        protected boolean centerOfMassEnabled = RandomCutForest.DEFAULT_CENTER_OF_MASS_ENABLED;
        protected double boundingBoxCacheFraction = RandomCutForest.DEFAULT_BOUNDING_BOX_CACHE_FRACTION;
        protected long randomSeed = new Random().nextLong();
        protected Random random = null;
        protected Optional<Integer> outputAfter = Optional.empty();
        protected int inputDimension;
        protected int dimension;

        public T storeSequenceIndexesEnabled(boolean storeSequenceIndexesEnabled) {
            this.storeSequenceIndexesEnabled = storeSequenceIndexesEnabled;
            return (T) this;
        }

        public T centerOfMassEnabled(boolean centerOfMassEnabled) {
            this.centerOfMassEnabled = centerOfMassEnabled;
            return (T) this;
        }

        public T boundingBoxCacheFraction(double boundingBoxCacheFraction) {
            this.boundingBoxCacheFraction = boundingBoxCacheFraction;
            return (T) this;
        }

        public T randomSeed(long randomSeed) {
            this.randomSeed = randomSeed;
            return (T) this;
        }

        public T random(Random random) {
            this.random = random;
            return (T) this;
        }

        public T outputAfter(int outputAfter) {
            this.outputAfter = Optional.of(outputAfter);
            return (T) this;
        }

        public T inputDimension(int inputDimension) {
            this.inputDimension = inputDimension;
            return (T) this;
        }

        public T dimension(int dimension) {
            this.dimension = dimension;
            return (T) this;
        }
    }
}
