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

import java.util.Random;
import java.util.function.Function;

import com.amazon.randomcutforest.MultiVisitor;
import com.amazon.randomcutforest.Visitor;

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
public abstract class AbstractRandomCutTree<Point, NodeReference, PointReference> implements ITree<PointReference> {

    /**
     * The index value used to represent the absence of a node. For example, when
     * the tree is created the root node index will be NULL. After a point is added
     * and a root node is created, the root node's parent will be NULL, and so on.
     */

    private final Random random;
    protected NodeReference rootIndex;
    protected INode<NodeReference> nodeView;
    public final boolean enableCache;
    public final boolean enableCenterOfMass;
    public final boolean enableSequenceIndices;

    public AbstractRandomCutTree(long seed, boolean enableCache, boolean enableCenterOfMass,
            boolean enableSequenceIndices) {
        random = new Random(seed);
        this.enableCache = enableCache;
        this.enableCenterOfMass = enableCenterOfMass;
        this.enableSequenceIndices = enableSequenceIndices;
    }

    /**
     * Return a new {@link Cut}, which is chosen uniformly over the space of
     * possible cuts for the given bounding box.
     *
     * @param random A random number generator
     * @param box    A bounding box that we want to find a random cut for.
     * @return A new Cut corresponding to a random cut in the bounding box.
     */
    static Cut randomCut(Random random, AbstractBoundingBox<?> box) {
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

    // decides the path taken in the abstract tree
    abstract protected boolean leftOf(Point point, int cutDimension, double cutValue);

    // checks equality based on precision
    abstract boolean checkEqual(Point oldPoint, Point point);

    // prints a point based on precision
    abstract String toString(Point point);

    // creates a bounding box based on two points, it is a very common occurrence
    abstract AbstractBoundingBox<Point> getInternalTwoPointBox(Point first, Point second);

    /**
     * Takes a node reference and produces a bounding box; cannot return null If
     * bounding boxes are cached then the bounding boxes of the descendants are
     * populated as well. Otherwise a bounding box is continually modified in place
     * to build the bounding box.
     *
     * @param nodeReference reference of node
     * @return
     */
    AbstractBoundingBox<Point> getBoundingBox(NodeReference nodeReference) {
        if (isLeaf(nodeReference)) {
            return getLeafBoxFromLeafNode(nodeReference);
        }
        if (enableCache) {
            return getBoundingBoxReflate(nodeReference);
        }
        return constructBoxInPlace(nodeReference);
    }

    // function to regenerate bounding boxes and cache them, based on precision
    abstract AbstractBoundingBox<Point> getBoundingBoxReflate(NodeReference nodeReference);

    // function for getting the bounding boxes for leaf nodes
    abstract AbstractBoundingBox<Point> getLeafBoxFromLeafNode(NodeReference nodeReference);

    // constructing a bounding box (almost) in place
    protected AbstractBoundingBox<Point> constructBoxInPlace(NodeReference nodeReference) {
        if (isLeaf(nodeReference)) {
            return getLeafBoxFromLeafNode(nodeReference);
        } else {
            AbstractBoundingBox<Point> currentBox = constructBoxInPlace(getLeftChild(nodeReference));
            return constructBoxInPlace(currentBox, getRightChild(nodeReference));

        }
    }

    AbstractBoundingBox<Point> constructBoxInPlace(AbstractBoundingBox<Point> currentBox, NodeReference nodeReference) {
        if (isLeaf(nodeReference)) {
            return currentBox.addPoint(getPointFromLeafNode(nodeReference));
        } else {
            AbstractBoundingBox<Point> tempBox = constructBoxInPlace(currentBox, getLeftChild(nodeReference));
            // the box may be changed for single points
            return constructBoxInPlace(tempBox, getRightChild(nodeReference));
        }
    }

    // gets the actual values of a point from its reference which can be
    // Integer/direct reference
    abstract Point getPointFromPointReference(PointReference pointIndex);

    // gets the leaf point associated with a leaf node
    abstract Point getPointFromLeafNode(NodeReference node);

    // gets the reference to a point from a leaf node
    abstract PointReference getPointReference(NodeReference node);

    // is a leaf or otherwise
    abstract boolean isLeaf(NodeReference node);

    // decreases mass of a node and returns it
    abstract int decrementMass(NodeReference node);

    // increases mass of a node and returns it
    abstract int incrementMass(NodeReference node);

    // returns the sibling of a node
    abstract NodeReference getSibling(NodeReference nodeReference);

    abstract NodeReference getParent(NodeReference nodeReference);

    abstract void setParent(NodeReference nodeReference, NodeReference parent);

    abstract void delete(NodeReference nodeReference);

    // only valid for internal nodes
    abstract int getCutDimension(NodeReference nodeReference);

    // only valid for internal nodes
    abstract double getCutValue(NodeReference nodeReference);

    abstract NodeReference getLeftChild(NodeReference nodeReference);

    abstract NodeReference getRightChild(NodeReference nodeReference);

    // replaces child (with designated parent) by otherNode
    abstract void replaceNode(NodeReference parent, NodeReference child, NodeReference otherNode);

    // implements the delete step in RCF
    abstract void replaceNodeBySibling(NodeReference grandParent, NodeReference Parent, NodeReference node);

    // creates a new leaf node
    abstract NodeReference addLeaf(NodeReference parent, PointReference pointIndex, int mass);

    // creates an internal node
    abstract NodeReference addNode(NodeReference parent, NodeReference leftChild, NodeReference rightChild,
            int cutDimension, double cutValue, int mass);

    // increases the mass of all ancestor nodes when an added point is internal to
    // some bounding box
    // note the bounding boxes of these ancestor nodes will remain unchanged
    abstract void increaseMassOfAncestorsRecursively(NodeReference mergedNode);

    abstract int getMass(NodeReference nodeReference);

    abstract void addSequences(NodeReference node, long uniqueSequenceNumber);

    abstract void deleteSequences(NodeReference node, long uniqueSequenceNumber);

    /**
     * method to delete a point from the tree
     *
     */

    @Override
    public void deletePoint(PointReference pointReference, long sequenceNumber) {
        checkState(rootIndex != null, "root must not be null");
        deletePoint(rootIndex, getPointFromPointReference(pointReference), sequenceNumber, 0);
    }

    // manages the bounding boxes and center of mass
    abstract boolean updateDeletePointBoxes(NodeReference nodeReference, Point point, boolean isResolved);

    /**
     * This function deletes the point from the tree recursively. We traverse the
     * tree based on the cut stored in each interior node until we reach a leaf
     * node. We then delete the leaf node if the mass of the node is 1, otherwise we
     * reduce the mass by 1. The bounding boxes continue
     *
     * @param nodeReference node that we are visiting in the tree.
     * @param point         the point that is being deleted from the tree.
     * @param level         the level (i.e., the length of the path to the root) of
     *                      the node being evaluated.
     * @return if the point is within the bounding box of the remaining points (in
     *         which case the bounding boxes of the ancestors would not need to be
     *         updated)
     */

    private boolean deletePoint(NodeReference nodeReference, Point point, long sequenceNumber, int level) {

        if (isLeaf(nodeReference)) {
            Point oldPoint = getPointFromLeafNode(nodeReference);
            if (!checkEqual(oldPoint, point)) {
                throw new IllegalStateException(toString(point) + " " + toString(getPointFromLeafNode(nodeReference))
                        + " " + nodeReference + " node " + false + " Inconsistency in trees in delete step here.");
            }
            if (enableSequenceIndices) {
                deleteSequences(nodeReference, sequenceNumber);
            }
            // decrease mass for the delete
            if (decrementMass(nodeReference) > 0) {
                return true;
            }

            NodeReference parent = getParent(nodeReference);

            if (parent == null) {
                rootIndex = null;
                delete(nodeReference);
                return true;
            }
            // parent is guaranteed to be an internal node

            NodeReference grandParent = getParent(parent);
            if (grandParent == null) {
                rootIndex = getSibling(nodeReference);
                setParent(rootIndex, null);
            } else {
                replaceNodeBySibling(grandParent, parent, nodeReference);
            }
            delete(nodeReference);
            delete(parent);
            return false;
        }

        // node is not a leaf, and is an internal node
        boolean resolvedDelete = leftOf(point, getCutDimension(nodeReference), getCutValue(nodeReference))
                ? deletePoint(getLeftChild(nodeReference), point, sequenceNumber, level + 1)
                : deletePoint(getRightChild(nodeReference), point, sequenceNumber, level + 1);

        resolvedDelete = updateDeletePointBoxes(nodeReference, point, resolvedDelete);
        decrementMass(nodeReference);
        return resolvedDelete;
    }

    abstract void updateAddPointBoxes(AbstractBoundingBox<Point> savedBox, NodeReference mergedNode, Point point,
            NodeReference parentIndex);

    /**
     * The following function adjusts the tree (if the issue has not been resolved
     * yet) based on the information in AddPointState which corresponds to the
     * particulars of the cut and the node that is being separated by the cut as a
     * sibling of the point
     *
     * @param point         the actual values
     * @param parentIndex   index of the ancestor node (can be NULL for root) above
     *                      which no changes need to be made
     * @param addPointState the information about the cut.
     */

    void resolve(Point point, NodeReference parentIndex,
            AddPointState<Point, NodeReference, PointReference> addPointState) {
        if (!addPointState.getResolved()) {
            addPointState.setResolved();
            NodeReference siblingNode = addPointState.getSiblingOffset();
            int cutDimension = addPointState.getCutDimension();
            double cutValue = addPointState.getCutValue();
            int oldMass = getMass(siblingNode);
            NodeReference leafNode = addLeaf(null, addPointState.getPointIndex(), 1);
            NodeReference mergedNode = leftOf(point, cutDimension, cutValue)
                    ? addNode(null, leafNode, siblingNode, cutDimension, cutValue, (oldMass + 1))
                    : addNode(null, siblingNode, leafNode, cutDimension, cutValue, (oldMass + 1));

            NodeReference parent = getParent(siblingNode);
            if (parent == null) {
                rootIndex = mergedNode;
            } else {
                replaceNode(parent, siblingNode, mergedNode);
            }

            setParent(leafNode, mergedNode);
            setParent(siblingNode, mergedNode);
            // manage bounding boxes, including caching, as well as centerOfMass
            updateAddPointBoxes(addPointState.getSavedBox(), mergedNode, point, parentIndex);
            // manage mass of points
            increaseMassOfAncestorsRecursively(mergedNode);
            if (enableSequenceIndices) {
                addSequences(leafNode, addPointState.getSequenceNumber());
            }
        }

    }

    /**
     * This function adds a point to the tree recursively starting from the leaf
     * node.
     *
     * @param nodeReference the current node in the tree we are on
     * @param point         the point that we want to add to the tree
     * @param pointIndex    is the location of the new copy of the point in point
     *                      store
     *
     * @return the integer index of the inserted point. If a previous copy is found
     *         then the index of the previous copy is returned. That helps in
     *         maintaining the number of times a particular vector has been seen by
     *         some tree. If no duplicate is found then pointIndex is returned.
     */

    private AddPointState<Point, NodeReference, PointReference> addPoint(NodeReference nodeReference, Point point,
            PointReference pointIndex, long sequenceNumber) {

        if (isLeaf(nodeReference)) {
            Point oldPoint = getPointFromLeafNode(nodeReference);
            if (checkEqual(oldPoint, point)) {
                // the inserted point is equal to an existing leaf point
                incrementMass(nodeReference);
                AddPointState<Point, NodeReference, PointReference> newState = new AddPointState<>(
                        getPointReference(nodeReference));
                // the following will ensure that no further processing happens
                // note that boxes (if present) do not need to be updated
                // the index of the duplicate point is saved
                newState.setResolved();
                increaseMassOfAncestorsRecursively(nodeReference);
                return newState;
            } else {
                AbstractBoundingBox<Point> mergedBox = getInternalTwoPointBox(point, oldPoint);
                Cut cut = randomCut(random, mergedBox);
                // the cut is between a leaf node and the new point; it must exist
                AddPointState<Point, NodeReference, PointReference> newState = new AddPointState<>(pointIndex);
                newState.initialize(nodeReference, cut.getDimension(), cut.getValue(), sequenceNumber, mergedBox);
                return newState;
            }
        }

        NodeReference nextNode;
        NodeReference sibling;

        if ((leftOf(point, getCutDimension(nodeReference), getCutValue(nodeReference)))) {
            nextNode = getLeftChild(nodeReference);
            sibling = getRightChild(nodeReference);
        } else {
            nextNode = getRightChild(nodeReference);
            sibling = getLeftChild(nodeReference);
        }

        // we recurse in a preorder traversal over the tree
        AddPointState<Point, NodeReference, PointReference> addPointState = addPoint(nextNode, point, pointIndex,
                sequenceNumber);

        if (!addPointState.getResolved()) {
            AbstractBoundingBox<Point> existingBox;
            if (enableCache) {
                // if the boxes are being cached, use the box if present, otherwise
                // generate and cache the box
                existingBox = getBoundingBox(nodeReference);
                // uncomment to test
                // checkState(existingBox.equals(constructBoxInPlace(nodeReference)), " error");
            } else {
                // the boxes are not present, merge the bounding box of the sibling of the last
                // seen child (nextNode) to the stored box in the state and save it
                existingBox = addPointState.getCurrentBox().getMergedBox(getBoundingBox(sibling));
                addPointState.setCurrentBox(existingBox);
            }

            if (existingBox.contains(point)) {
                // resolve the add, by modifying (if present) the bounding boxes corresponding
                // to current node (nodeReference) all the way to the root
                resolve(point, nodeReference, addPointState);
                return addPointState;
            }
            // generate a new cut and see if it separates the new point
            AbstractBoundingBox<Point> mergedBox = existingBox.getMergedBox(point);
            Cut cut = randomCut(random, mergedBox);
            int splitDimension = cut.getDimension();
            double splitValue = cut.getValue();
            double minValue = existingBox.getMinValue(splitDimension);
            double maxValue = existingBox.getMaxValue(splitDimension);

            if (minValue > splitValue || maxValue <= splitValue) {
                // the cut separates the new point; update the state to store information
                // about the most recent cut
                addPointState.initialize(nodeReference, splitDimension, splitValue, sequenceNumber, mergedBox);
            }
        }
        return addPointState;
    }

    /**
     * adds a point to the tree
     * 
     * @param sequenceNumber the index of the point in PointStore and the
     *                       corresponding timestamp
     *
     * @return the index of the inserted point, which can be the input or the index
     *         of a previously seen copy
     */

    public PointReference addPoint(PointReference pointReference, long sequenceNumber) {
        int saveMass = getMass();
        Point pointValue = getPointFromPointReference(pointReference);
        if (rootIndex == null) {
            rootIndex = addLeaf(null, pointReference, 1);
            checkState(saveMass + 1 == getMass(), "incorrect add");
            return pointReference;
        } else {
            AddPointState<Point, NodeReference, PointReference> addPointState = addPoint(rootIndex, pointValue,
                    pointReference, sequenceNumber);
            resolve(pointValue, null, addPointState);
            checkState(saveMass + 1 == getMass(), "incorrect add");
            return addPointState.getPointIndex();
        }
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
    public <R> R traverse(double[] point, Function<ITree<?>, Visitor<R>> visitorFactory) {
        checkState(rootIndex != null, "this tree doesn't contain any nodes");
        Visitor<R> visitor = visitorFactory.apply(this);
        traversePathToLeafAndVisitNodes(point, visitor, rootIndex, nodeView, 0);
        return visitor.getResult();
    }

    private <R> void traversePathToLeafAndVisitNodes(double[] point, Visitor<R> visitor, NodeReference currentNode,
            INode<NodeReference> nodeView, int depthOfNode) {
        if (isLeaf(currentNode)) {
            visitor.acceptLeaf(nodeView.getNodeView(currentNode), depthOfNode);
        } else {
            NodeReference childNode = leftOf(point, currentNode) ? getLeftChild(currentNode)
                    : getRightChild(currentNode);
            traversePathToLeafAndVisitNodes(point, visitor, childNode, nodeView.getNodeView(childNode),
                    depthOfNode + 1);
            visitor.accept(nodeView.getNodeView(currentNode), depthOfNode);
        }
    }

    /**
     * This is a traversal method which follows the standard traversal path (defined
     * in {@link #traverse(double[], Function)}) but at Node in checks to see
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
    public <R> R traverseMulti(double[] point, Function<ITree<?>, MultiVisitor<R>> visitorFactory) {
        checkNotNull(point, "point must not be null");
        checkNotNull(visitorFactory, "visitor must not be null");
        checkState(rootIndex != null, "this tree doesn't contain any nodes");
        MultiVisitor<R> visitor = visitorFactory.apply(this);
        traverseTreeMulti(point, visitor, rootIndex, nodeView, 0);
        return visitor.getResult();
    }

    private <R> void traverseTreeMulti(double[] point, MultiVisitor<R> visitor, NodeReference currentNode,
            INode<NodeReference> nodeView, int depthOfNode) {
        if (isLeaf(currentNode)) {
            visitor.acceptLeaf(nodeView.getNodeView(currentNode), depthOfNode);
        } else {
            if (visitor.trigger(nodeView.getNodeView(currentNode))) {
                NodeReference leftChild = getLeftChild(currentNode);
                traverseTreeMulti(point, visitor, leftChild, nodeView.getNodeView(leftChild), depthOfNode + 1);
                MultiVisitor<R> newVisitor = visitor.newCopy();
                NodeReference rightChild = getRightChild(currentNode);
                traverseTreeMulti(point, newVisitor, rightChild, nodeView.getNodeView(rightChild), depthOfNode + 1);
                visitor.combine(newVisitor);
            } else {
                NodeReference childNode = leftOf(point, currentNode) ? getLeftChild(currentNode)
                        : getRightChild(currentNode);
                traverseTreeMulti(point, visitor, childNode, nodeView.getNodeView(childNode), depthOfNode + 1);
            }
            visitor.accept(nodeView.getNodeView(currentNode), depthOfNode);
        }
    }

    @Override
    public int getMass() {
        return (rootIndex == null) ? 0 : getMass(rootIndex);
    }

}
