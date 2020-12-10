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
import com.amazon.randomcutforest.state.store.NodeStoreMapper;
import com.amazon.randomcutforest.state.store.NodeStoreState;
import com.amazon.randomcutforest.store.IPointStore;
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
public abstract class AbstractCompactRandomCutTree<P> implements ITree<Integer> {

    /**
     * The index value used to represent the absence of a node. For example, when
     * the tree is created the root node index will be NULL. After a point is added
     * and a root node is created, the root node's parent will be NULL, and so on.
     */
    public static final short NULL = -1;

    private Random random;
    protected int maxSize;
    protected NodeStore internalNodes;
    protected LeafStore leafNodes;
    protected IPointStore<P> pointStore;
    protected int rootIndex;
    private CompactNodeView nodeView;
    protected IBoundingBox<P>[] cachedBoxes;

    public AbstractCompactRandomCutTree(int maxSize, long seed) {
        checkArgument(maxSize > 0, "maxSize must be greater than 0");
        this.maxSize = maxSize;
        internalNodes = new NodeStore((short) (this.maxSize - 1));
        leafNodes = new LeafStore((short) this.maxSize);
        random = new Random(seed);
        rootIndex = NULL;
        nodeView = new CompactNodeView(this, NULL);
    }

    public AbstractCompactRandomCutTree(int maxSize, long seed, LeafStore leafStore, NodeStore nodeStore,
            int rootIndex) {
        checkArgument(maxSize > 0, "maxSize must be greater than 0");
        checkNotNull(leafStore, "leafStore must not be null");
        checkNotNull(nodeStore, "nodeStore must not be null");

        this.maxSize = maxSize;
        this.rootIndex = rootIndex;
        random = new Random(seed);
        NodeStoreMapper newMapper = new NodeStoreMapper();
        internalNodes = nodeStore;
        leafNodes = leafStore;
        nodeView = new CompactNodeView(this, rootIndex);
    }

    /**
     * Return a new {@link Cut}, which is chosen uniformly over the space of
     * possible cuts for the given bounding box.
     *
     * @param random A random number generator
     * @param box    A bounding box that we want to find a random cut for.
     * @return A new Cut corresponding to a random cut in the bounding box.
     */
    static Cut randomCut(Random random, IBoundingBox box) {
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
     * @param nodeOffset The node whose sibling we are requesting.
     * @return the sibling of node in the tree.
     */

    int getSibling(int nodeOffset) {
        checkArgument(nodeOffset >= 0, " cannotbe negative");
        int parent = getParent(nodeOffset);

        if (parent != NULL) {
            // checkState(!leafNodes.isLeaf(parent), "incorrect");
            if (internalNodes.getLeftIndex(parent) == nodeOffset) {
                return internalNodes.getRightIndex(parent);
            } else if (internalNodes.getRightIndex(parent) == nodeOffset) {
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

    protected boolean leftOf(double[] point, int nodeOffset) {
        return point[internalNodes.getCutDimension(nodeOffset)] <= internalNodes.getCutValue(nodeOffset);
    }

    abstract protected boolean leftOf(P point, int cutDimension, double cutValue);

    abstract boolean checkEqual(P oldPoint, P point);

    abstract String toString(P point);

    abstract IBoundingBox<P> getLeafBox(int pointOffset);

    abstract IBoundingBox<P> reflateNode(int nodeOffset);

    IBoundingBox<P> getInternalBoundingBox(int nodeOffset) {
        if (leafNodes.isLeaf(nodeOffset)) {
            return getLeafBox(leafNodes.getPointIndex(nodeOffset));
        } else {
            if (cachedBoxes == null) {
                return reflateNode(nodeOffset);
            } else {
                if (cachedBoxes[nodeOffset] == null) {
                    cachedBoxes[nodeOffset] = reflateNode(nodeOffset);
                }
                return cachedBoxes[nodeOffset];
            }
        }
    }

    abstract IBoundingBox<P> getInternalMergedBox(P point, P oldPoint);

    IBoundingBox<P> getBoundingBoxLeaveNull(int nodeOffset) {
        if (leafNodes.isLeaf(nodeOffset)) {
            return getLeafBox(leafNodes.getPointIndex(nodeOffset));
        } else if (cachedBoxes == null) {
            return null;
        } else {
            return cachedBoxes[nodeOffset];
        }
    }

    /**
     * returns the boundingbox
     */
    BoundingBox getBoundingBox(int nodeOffset) {
        return getInternalBoundingBox(nodeOffset).convertBoxToDouble();
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
     * @param nodeOffset node that we are visiting in the tree.
     * @param point      the point that is being deleted from the tree.
     * @param level      the level (i.e., the length of the path to the root) of the
     *                   node being evaluated.
     * @return if the point is within the bounding box of the remaining points (in
     *         which case the bounding boxes of the ancestors would not need to be
     *         updated)
     */

    private boolean deletePoint(int nodeOffset, P point, int level) {

        if (leafNodes.isLeaf(nodeOffset)) {
            P oldPoint = pointStore.get(leafNodes.getPointIndex(nodeOffset));
            if (!checkEqual(oldPoint, point)) {
                throw new IllegalStateException(
                        toString(point) + " " + pointStore.toString(leafNodes.getPointIndex(nodeOffset)) + " "
                                + nodeOffset + " node " + leafNodes.getPointIndex(nodeOffset) + " " + false
                                + " Inconsistency in trees in delete step here.");
            }

            if (leafNodes.getMass(nodeOffset) > 1) {
                leafNodes.decrementMass(nodeOffset);
                return true;
            }

            int parent = leafNodes.getParent(nodeOffset);

            if (parent == NULL) {
                rootIndex = NULL;
                leafNodes.delete(nodeOffset);
                return true;
            }

            int grandParent = internalNodes.getParent(parent);
            if (grandParent == NULL) {
                rootIndex = getSibling(nodeOffset);
                setParent(rootIndex, NULL);
            } else {
                replaceNode(parent, getSibling(nodeOffset));
            }
            leafNodes.delete(nodeOffset);
            internalNodes.delete(parent);
            return false;
        }

        // node is not a leaf, and is an internal node
        boolean resolvedDelete = leftOf(point, internalNodes.getCutDimension(nodeOffset),
                internalNodes.getCutValue(nodeOffset))
                        ? deletePoint(internalNodes.getLeftIndex(nodeOffset), point, level + 1)
                        : deletePoint(internalNodes.getRightIndex(nodeOffset), point, level + 1);

        if (!resolvedDelete && (cachedBoxes != null) && cachedBoxes[nodeOffset] != null) {
            IBoundingBox<P> leftBox = getBoundingBoxLeaveNull(internalNodes.getLeftIndex(nodeOffset));
            IBoundingBox<P> rightBox = getBoundingBoxLeaveNull(internalNodes.getRightIndex(nodeOffset));
            if ((rightBox != null) && (leftBox != null)) {
                cachedBoxes[nodeOffset] = leftBox.getMergedBox(rightBox);
                if (cachedBoxes[nodeOffset].contains(point)) {
                    resolvedDelete = true;
                }
            } else {
                cachedBoxes[nodeOffset] = null;
            }
        }
        internalNodes.decrementMass(nodeOffset);
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
     * @param addPointState the in formation about the cut.
     */

    void resolve(P point, int parentIndex, AddPointState addPointState) {
        if (!addPointState.getResolved()) {
            addPointState.setResolved();
            int siblingOffset = addPointState.getSiblingOffset();
            int cutDimension = addPointState.getCutDimension();
            double cutValue = addPointState.getCutValue();
            int oldmass = getMass(siblingOffset);
            int leafOffset = leafNodes.add(NULL, addPointState.getPointIndex(), 1);
            int mergedNode = leftOf(point, cutDimension, cutValue)
                    ? internalNodes.addNode(NULL, leafOffset, siblingOffset, cutDimension, cutValue, (oldmass + 1))
                    : internalNodes.addNode(NULL, siblingOffset, leafOffset, cutDimension, cutValue, (oldmass + 1));

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

        }
    }

    /**
     * This function adds a point to the tree recursively starting from the leaf
     * node.
     *
     * @param nodeOffset the current node in the tree we are on
     * @param point      the point that we want to add to the tree
     * @param pointIndex is the location of the new copy of the point in pointstore
     *
     * @return the integer index of the inserted point. If a previous copy is found
     *         then the index of the previous copy is returned. That helps in
     *         maintaining the number of times a particular vector has been seen by
     *         some tree. If no duplicate is found then pointIndex is returned.
     */

    private AddPointState<P> addPoint(int nodeOffset, P point, int pointIndex) {

        if (leafNodes.isLeaf(nodeOffset)) {
            int oldPointIndex = leafNodes.getPointIndex(nodeOffset);
            P oldPoint = pointStore.get(oldPointIndex);
            if (checkEqual(oldPoint, point)) {
                // the inserted point is equal to an existing leaf point
                leafNodes.incrementMass(nodeOffset);
                AddPointState<P> newState = new AddPointState<>(oldPointIndex);
                // the following will ensure that no further processing happens
                // note that boxes (if present) do not need to be updated
                // the index of the duplicate point is saved
                newState.setResolved();
                return newState;
            } else {
                IBoundingBox<P> mergedBox = getInternalMergedBox(point, oldPoint);
                Cut cut = randomCut(random, mergedBox);
                // the cut is between a leaf node and the new point; it must exist
                AddPointState<P> newState = new AddPointState<>(pointIndex);
                newState.initialize(nodeOffset, cut.getDimension(), cut.getValue(), mergedBox);
                return newState;
            }
        }

        // we first increase the mass
        internalNodes.incrementMass(nodeOffset);

        int nextNode;
        int sibling;
        if (leftOf(point, internalNodes.getCutDimension(nodeOffset), internalNodes.getCutValue(nodeOffset))) {
            nextNode = internalNodes.getLeftIndex(nodeOffset);
            sibling = internalNodes.getRightIndex(nodeOffset);
        } else {
            sibling = internalNodes.getLeftIndex(nodeOffset);
            nextNode = internalNodes.getRightIndex(nodeOffset);
        }
        // we recurse in a preorder traversal ove the tree
        AddPointState addPointState = addPoint(nextNode, point, pointIndex);

        if (!addPointState.getResolved()) {
            IBoundingBox<P> existingBox;
            if (cachedBoxes != null) {
                // if the boxes are being cached, use the box if present, otherwise
                // generate and cache the box
                existingBox = getInternalBoundingBox(nodeOffset);
            } else {
                // the boxes are not present, merge the bounding box of the sibling of the last
                // seen child (nextNode) to the stored box in the state and save it
                existingBox = addPointState.getCurrentBox().getMergedBox(getInternalBoundingBox(sibling));
                addPointState.setCurrentBox(existingBox);
            }

            if (existingBox.contains(point)) {
                // resolve the add, by modifying (if present) the bounding boxes corresponding
                // to current node (nodeOffset) all the way to the root
                resolve(point, nodeOffset, addPointState);
                return addPointState;
            }
            // generate a new cut and see if it separates the new point
            IBoundingBox<P> mergedBox = existingBox.getMergedBox(point);
            Cut cut = randomCut(random, mergedBox);
            int splitDimension = cut.getDimension();
            double splitValue = cut.getValue();
            double minValue = existingBox.getMinValue(splitDimension);
            double maxValue = existingBox.getMaxValue(splitDimension);

            if (minValue > splitValue || maxValue <= splitValue) {
                // the cut separates the new point; update the state to store information
                // about the most recent cut
                addPointState.initialize(nodeOffset, splitDimension, splitValue, mergedBox);
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
        P pointValue = pointStore.get(pointIndex);
        if (rootIndex == NULL) {
            rootIndex = leafNodes.add(NULL, pointIndex, 1);
            return pointIndex;
        } else {
            AddPointState<P> addPointState = addPoint(rootIndex, pointValue, pointIndex);
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
     * This is a traversal method which follows the standard traveral path (defined
     * in {@link #traverse(double[], Function)}) but at Node in checks to see
     * whether the visitor should split. If a split is triggered, then independent
     * copies of the visitor are sent down each branch of the tree and then merged
     * before propogating the result.
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
                nodeView.setCurrentNodeIndex(currentNode);
                visitor.accept(nodeView, depthOfNode);
            } else {
                int childNode = leftOf(point, currentNode) ? internalNodes.getLeftIndex(currentNode)
                        : internalNodes.getRightIndex(currentNode);
                traverseTreeMulti(point, visitor, childNode, nodeView, depthOfNode + 1);
                nodeView.setCurrentNodeIndex(currentNode);
                visitor.accept(nodeView, depthOfNode);
            }
        }
    }

    @Override
    public int getMass() {
        return (rootIndex == NULL) ? 0 : getMass(rootIndex);
    }

    protected int getMass(int nodeOffset) {
        return leafNodes.isLeaf(nodeOffset) ? leafNodes.getMass(nodeOffset) : internalNodes.getMass(nodeOffset);
    }

    abstract protected double[] getLeafPoint(int nodeOffset);

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
