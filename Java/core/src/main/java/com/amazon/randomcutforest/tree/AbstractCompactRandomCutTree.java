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
import com.amazon.randomcutforest.executor.Sequential;
import com.amazon.randomcutforest.state.store.LeafStoreState;
import com.amazon.randomcutforest.state.store.NodeStoreState;
import com.amazon.randomcutforest.store.IPointStoreView;
import com.amazon.randomcutforest.store.LeafStore;
import com.amazon.randomcutforest.store.NodeStore;

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
public abstract class AbstractCompactRandomCutTree<Point> implements ITree<Integer> {

    /**
     * The index value used to represent the absence of a node. For example, when
     * the tree is created the root node index will be NULL. After a point is added
     * and a root node is created, the root node's parent will be NULL, and so on.
     */
    public static final int NULL = -1;

    private final Random random;
    protected int maxSize;
    protected NodeStore internalNodes;
    protected LeafStore leafNodes;
    protected IPointStoreView<Point> pointStore;
    protected int rootIndex;
    private final CompactNodeView nodeView;
    protected IBoundingBox<Point>[] cachedBoxes;
    protected boolean enableCache;

    public AbstractCompactRandomCutTree(int maxSize, long seed, boolean enableCache) {
        checkArgument(maxSize > 0, "maxSize must be greater than 0");
        this.maxSize = maxSize;
        internalNodes = new NodeStore((short) (this.maxSize - 1));
        leafNodes = new LeafStore((short) this.maxSize);
        random = new Random(seed);
        rootIndex = NULL;
        nodeView = new CompactNodeView(this, NULL);
        this.enableCache = enableCache;
    }

    public AbstractCompactRandomCutTree(int maxSize, long seed, LeafStore leafStore, NodeStore nodeStore, int rootIndex,
            boolean enableCache) {
        checkArgument(maxSize > 0, "maxSize must be greater than 0");
        checkNotNull(leafStore, "leafStore must not be null");
        checkNotNull(nodeStore, "nodeStore must not be null");

        this.maxSize = maxSize;
        this.rootIndex = rootIndex;
        random = new Random(seed);
        internalNodes = nodeStore;
        leafNodes = leafStore;
        nodeView = new CompactNodeView(this, rootIndex);
        this.enableCache = enableCache;
    }

    /**
     * Return a new {@link Cut}, which is chosen uniformly over the space of
     * possible cuts for the given bounding box.
     *
     * @param random A random number generator
     * @param box    A bounding box that we want to find a random cut for.
     * @return A new Cut corresponding to a random cut in the bounding box.
     */
    static Cut randomCut(Random random, IBoundingBox<?> box) {
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
     * @param oldNodeOffset The node we are replacing.
     * @param newNodeOffset The new node we are inserting into the tree.
     */

    void replaceNode(int oldNodeOffset, int newNodeOffset) {

        int parent = getParent(oldNodeOffset);
        if (parent != NULL) {
            // checkState(!leafNodes.isLeaf(parent), "incorrect");
            internalNodes.replaceNode(parent, oldNodeOffset, newNodeOffset);
        }

        setParent(newNodeOffset, parent);
    }

    /**
     * Return the sibling of a non-root node. Note that every non-leaf node in a
     * Random Cut Tree has two children.
     *
     * @param nodeReference The node whose sibling we are requesting.
     * @return the sibling of node in the tree.
     */

    int getSibling(int nodeReference) {
        checkArgument(nodeReference >= 0, " cannot be negative");
        int parent = getParent(nodeReference);

        if (parent != NULL) {
            // checkState(!leafNodes.isLeaf(parent), "incorrect");
            if (internalNodes.getLeftIndex(parent) == nodeReference) {
                return internalNodes.getRightIndex(parent);
            } else if (internalNodes.getRightIndex(parent) == nodeReference) {
                return internalNodes.getLeftIndex(parent);
            } else {
                throw new IllegalArgumentException("node parent does not link back to node");
            }
        } else
            return NULL; // root is leaf
    }

    void setParent(int siblingOffset, int mergedNode) {
        if (leafNodes.isLeaf(siblingOffset)) {
            leafNodes.setParent(siblingOffset, mergedNode);
        } else {
            internalNodes.setParent(siblingOffset, mergedNode);
        }
    }

    int getParent(int siblingOffset) {
        if (leafNodes.isLeaf(siblingOffset)) {
            return leafNodes.getParent(siblingOffset);
        } else {
            return internalNodes.getParent(siblingOffset);
        }
    }

    protected boolean leftOf(double[] point, int nodeReference) {
        return point[internalNodes.getCutDimension(nodeReference)] <= internalNodes.getCutValue(nodeReference);
    }

    abstract protected boolean leftOf(Point point, int cutDimension, double cutValue);

    abstract boolean checkEqual(Point oldPoint, Point point);

    abstract String toString(Point point);

    IBoundingBox<Point> getInternalBoundingBox(int nodeReference) {
        IBoundingBox<Point> answer = getBoundingBoxLeaveNull(nodeReference);
        if (answer == null) {
            answer = (enableCache) ? reflateNode(nodeReference) : constructBoxInPlace(nodeReference);
        }
        return answer;
    }

    abstract IBoundingBox<Point> getLeafBoxFromPoint(int pointIndex);

    IBoundingBox<Point> getBoundingBoxLeaveNull(int nodeReference) {
        if (leafNodes.isLeaf(nodeReference)) {
            return getLeafBoxFromPoint(leafNodes.getPointIndex(nodeReference));
        } else if (enableCache) {
            return cachedBoxes[nodeReference];
        } else {
            return null;
        }
    }

    abstract IBoundingBox<Point> reflateNode(int nodeReference);

    IBoundingBox<Point> constructBoxInPlace(int nodeReference) {
        if (leafNodes.isLeaf(nodeReference)) {
            return getLeafBoxFromPoint(leafNodes.getPointIndex(nodeReference));
        } else {
            IBoundingBox<Point> currentBox = constructBoxInPlace(internalNodes.getLeftIndex(nodeReference));
            return constructBoxInPlace(currentBox, internalNodes.getRightIndex(nodeReference));

        }
    }

    IBoundingBox<Point> constructBoxInPlace(IBoundingBox<Point> currentBox, int nodeReference) {
        if (leafNodes.isLeaf(nodeReference)) {
            return currentBox.addPoint(pointStore.get(leafNodes.getPointIndex(nodeReference)));
        } else {
            IBoundingBox<Point> tempBox = constructBoxInPlace(currentBox, internalNodes.getLeftIndex(nodeReference));
            // the box may be changed for single points
            return constructBoxInPlace(tempBox, internalNodes.getRightIndex(nodeReference));
        }
    }

    abstract IBoundingBox<Point> getInternalMergedBox(Point point, Point oldPoint);

    /**
     * returns the bounding box
     */
    BoundingBox getBoundingBox(int nodeReference) {
        return getInternalBoundingBox(nodeReference).copyBoxToDouble();
    }

    /**
     * method to delete a point from the tree
     * 
     * @param sequential the descriptor of the point
     */

    public void deletePoint(Sequential<Integer> sequential) {
        checkState(rootIndex != NULL, "root must not be null");
        deletePoint(rootIndex, pointStore.get(sequential.getValue()), 0);
    }

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

    private boolean deletePoint(int nodeReference, Point point, int level) {

        if (leafNodes.isLeaf(nodeReference)) {
            Point oldPoint = pointStore.get(leafNodes.getPointIndex(nodeReference));
            if (!checkEqual(oldPoint, point)) {
                throw new IllegalStateException(
                        toString(point) + " " + pointStore.toString(leafNodes.getPointIndex(nodeReference)) + " "
                                + nodeReference + " node " + leafNodes.getPointIndex(nodeReference) + " " + false
                                + " Inconsistency in trees in delete step here.");
            }

            if (leafNodes.getMass(nodeReference) > 1) {
                leafNodes.decrementMass(nodeReference);
                return true;
            }

            int parent = leafNodes.getParent(nodeReference);

            if (parent == NULL) {
                rootIndex = NULL;
                leafNodes.delete(nodeReference);
                return true;
            }

            int grandParent = internalNodes.getParent(parent);
            if (grandParent == NULL) {
                rootIndex = getSibling(nodeReference);
                setParent(rootIndex, NULL);
            } else {
                replaceNode(parent, getSibling(nodeReference));
            }
            leafNodes.delete(nodeReference);
            internalNodes.delete(parent);
            return false;
        }

        // node is not a leaf, and is an internal node
        boolean resolvedDelete = leftOf(point, internalNodes.getCutDimension(nodeReference),
                internalNodes.getCutValue(nodeReference))
                        ? deletePoint(internalNodes.getLeftIndex(nodeReference), point, level + 1)
                        : deletePoint(internalNodes.getRightIndex(nodeReference), point, level + 1);

        if (!resolvedDelete && (cachedBoxes != null) && cachedBoxes[nodeReference] != null) {
            IBoundingBox<Point> leftBox = getBoundingBoxLeaveNull(internalNodes.getLeftIndex(nodeReference));
            IBoundingBox<Point> rightBox = getBoundingBoxLeaveNull(internalNodes.getRightIndex(nodeReference));
            if ((rightBox != null) && (leftBox != null)) {
                cachedBoxes[nodeReference] = leftBox.getMergedBox(rightBox);
                if (cachedBoxes[nodeReference].contains(point)) {
                    resolvedDelete = true;
                }
            } else {
                cachedBoxes[nodeReference] = null;
            }
        }
        internalNodes.decrementMass(nodeReference);
        return resolvedDelete;
    }

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

    void resolve(Point point, int parentIndex, AddPointState<Point> addPointState) {
        if (!addPointState.getResolved()) {
            addPointState.setResolved();
            int siblingOffset = addPointState.getSiblingOffset();
            int cutDimension = addPointState.getCutDimension();
            double cutValue = addPointState.getCutValue();
            int oldMass = getMass(siblingOffset);
            int leafOffset = leafNodes.addLeaf(NULL, addPointState.getPointIndex(), 1);
            int mergedNode = leftOf(point, cutDimension, cutValue)
                    ? internalNodes.addNode(NULL, leafOffset, siblingOffset, cutDimension, cutValue, (oldMass + 1))
                    : internalNodes.addNode(NULL, siblingOffset, leafOffset, cutDimension, cutValue, (oldMass + 1));

            int parent = getParent(siblingOffset);

            if (parent == NULL) {
                rootIndex = mergedNode;
            } else {
                replaceNode(siblingOffset, mergedNode);
            }

            leafNodes.setParent(leafOffset, mergedNode);
            setParent(siblingOffset, mergedNode);

            if (cachedBoxes != null) {
                cachedBoxes[mergedNode] = addPointState.getSavedBox();
                int tempNode = mergedNode;
                while (internalNodes.getParent(tempNode) != parentIndex) {
                    tempNode = internalNodes.getParent(tempNode);
                    cachedBoxes[tempNode].addPoint(point);
                }
            }
            int tempNode = mergedNode;
            while (internalNodes.getParent(tempNode) != NULL) {
                tempNode = internalNodes.getParent(tempNode);
                internalNodes.incrementMass(tempNode);
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

    private AddPointState<Point> addPoint(int nodeReference, Point point, int pointIndex) {

        if (leafNodes.isLeaf(nodeReference)) {
            int oldPointIndex = leafNodes.getPointIndex(nodeReference);
            Point oldPoint = pointStore.get(oldPointIndex);
            if (checkEqual(oldPoint, point)) {
                // the inserted point is equal to an existing leaf point
                leafNodes.incrementMass(nodeReference);
                AddPointState<Point> newState = new AddPointState<>(oldPointIndex);
                // the following will ensure that no further processing happens
                // note that boxes (if present) do not need to be updated
                // the index of the duplicate point is saved
                newState.setResolved();
                int node = leafNodes.getParent(nodeReference);
                while (node != NULL) {
                    internalNodes.incrementMass(node);
                    node = getParent(node);
                }
                return newState;
            } else {
                IBoundingBox<Point> mergedBox = getInternalMergedBox(point, oldPoint);
                Cut cut = randomCut(random, mergedBox);
                // the cut is between a leaf node and the new point; it must exist
                AddPointState<Point> newState = new AddPointState<>(pointIndex);
                newState.initialize(nodeReference, cut.getDimension(), cut.getValue(), mergedBox);
                return newState;
            }
        }

        int nextNode;
        int sibling;
        if (leftOf(point, internalNodes.getCutDimension(nodeReference), internalNodes.getCutValue(nodeReference))) {
            nextNode = internalNodes.getLeftIndex(nodeReference);
            sibling = internalNodes.getRightIndex(nodeReference);
        } else {
            sibling = internalNodes.getLeftIndex(nodeReference);
            nextNode = internalNodes.getRightIndex(nodeReference);
        }
        // we recurse in a preorder traversal ove the tree
        AddPointState<Point> addPointState = addPoint(nextNode, point, pointIndex);

        if (!addPointState.getResolved()) {
            IBoundingBox<Point> existingBox;
            if (enableCache) {
                // if the boxes are being cached, use the box if present, otherwise
                // generate and cache the box
                existingBox = getInternalBoundingBox(nodeReference);
            } else {
                // the boxes are not present, merge the bounding box of the sibling of the last
                // seen child (nextNode) to the stored box in the state and save it
                // we can change the current box in addPointState
                existingBox = addPointState.getCurrentBox().addBox(getInternalBoundingBox(sibling));
                addPointState.setCurrentBox(existingBox);
            }

            if (existingBox.contains(point)) {
                // resolve the add, by modifying (if present) the bounding boxes corresponding
                // to current node (nodeReference) all the way to the root
                resolve(point, nodeReference, addPointState);
                return addPointState;
            }
            // generate a new cut and see if it separates the new point
            IBoundingBox<Point> mergedBox = existingBox.getMergedBox(point);
            Cut cut = randomCut(random, mergedBox);
            int splitDimension = cut.getDimension();
            double splitValue = cut.getValue();
            double minValue = existingBox.getMinValue(splitDimension);
            double maxValue = existingBox.getMaxValue(splitDimension);

            if (minValue > splitValue || maxValue <= splitValue) {
                // the cut separates the new point; update the state to store information
                // about the most recent cut
                addPointState.initialize(nodeReference, splitDimension, splitValue, mergedBox);
            }
        }
        return addPointState;
    }

    /**
     * adds a point to the tree
     * 
     * @param seq the index of the point in PointStore and the corresponding
     *            timestamp
     *
     * @return the index of the inserted point, which can be the input or the index
     *         of a previously seen copy
     */

    public Integer addPoint(Sequential<Integer> seq) {
        int pointIndex = seq.getValue();
        Point pointValue = pointStore.get(pointIndex);
        if (rootIndex == NULL) {
            rootIndex = leafNodes.addLeaf(NULL, pointIndex, 1);
            return pointIndex;
        } else {
            AddPointState<Point> addPointState = addPoint(rootIndex, pointValue, pointIndex);
            resolve(pointValue, NULL, addPointState);
            return addPointState.getPointIndex();
        }
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
        checkState(rootIndex != NULL, "this tree doesn't contain any nodes");
        Visitor<R> visitor = visitorFactory.apply(this);
        traversePathToLeafAndVisitNodes(point, visitor, rootIndex, nodeView, 0);
        return visitor.getResult();
    }

    private <R> void traversePathToLeafAndVisitNodes(double[] point, Visitor<R> visitor, int currentNode,
            CompactNodeView nodeView, int depthOfNode) {
        if (leafNodes.isLeaf(currentNode)) {
            nodeView.setCurrentNodeIndex(currentNode);
            visitor.acceptLeaf(nodeView, depthOfNode);
        } else {
            int childNode = leftOf(point, currentNode) ? internalNodes.getLeftIndex(currentNode)
                    : internalNodes.getRightIndex(currentNode);
            traversePathToLeafAndVisitNodes(point, visitor, childNode, nodeView, depthOfNode + 1);
            nodeView.setCurrentNodeIndex(currentNode);
            visitor.accept(nodeView, depthOfNode);
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
        checkState(rootIndex != NULL, "this tree doesn't contain any nodes");
        MultiVisitor<R> visitor = visitorFactory.apply(this);
        traverseTreeMulti(point, visitor, rootIndex, nodeView, 0);
        return visitor.getResult();
    }

    private <R> void traverseTreeMulti(double[] point, MultiVisitor<R> visitor, int currentNode,
            CompactNodeView nodeView, int depthOfNode) {
        if (leafNodes.isLeaf(currentNode)) {
            nodeView.setCurrentNodeIndex(currentNode);
            visitor.acceptLeaf(nodeView, depthOfNode);
        } else {
            nodeView.setCurrentNodeIndex(currentNode);
            if (visitor.trigger(nodeView)) {
                traverseTreeMulti(point, visitor, internalNodes.getLeftIndex(currentNode), nodeView, depthOfNode + 1);
                MultiVisitor<R> newVisitor = visitor.newCopy();
                nodeView.setCurrentNodeIndex(internalNodes.getRightIndex(currentNode));
                traverseTreeMulti(point, newVisitor, internalNodes.getRightIndex(currentNode), nodeView,
                        depthOfNode + 1);
                visitor.combine(newVisitor);
            } else {
                int childNode = leftOf(point, currentNode) ? internalNodes.getLeftIndex(currentNode)
                        : internalNodes.getRightIndex(currentNode);
                traverseTreeMulti(point, visitor, childNode, nodeView, depthOfNode + 1);
            }
            nodeView.setCurrentNodeIndex(currentNode);
            visitor.accept(nodeView, depthOfNode);
        }
    }

    @Override
    public int getMass() {
        return (rootIndex == NULL) ? 0 : getMass(rootIndex);
    }

    protected int getMass(int nodeReference) {
        return leafNodes.isLeaf(nodeReference) ? leafNodes.getMass(nodeReference)
                : internalNodes.getMass(nodeReference);
    }

    abstract protected double[] getLeafPoint(int nodeReference);

    public void reflateTree() {
        if ((rootIndex == NULL) || (leafNodes.isLeaf(rootIndex))) {
            return;
        }
        reflateNode(rootIndex);
    }

    public NodeStoreState getNodes() {
        return new NodeStoreState(internalNodes);
    }

    public LeafStoreState getLeaves() {
        return new LeafStoreState(leafNodes);
    }

    public int getRootIndex() {
        return rootIndex;
    }

}
