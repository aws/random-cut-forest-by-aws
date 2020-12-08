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
    protected short rootIndex;
    private CompactNodeView nodeView;
    private AddPointState<P> addPointState = new AddPointState<>();
    protected IBox<P>[] cachedBoxes;

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
            short rootIndex) {
        checkArgument(maxSize > 0, "maxSize must be greater than 0");
        checkNotNull(leafStore, "leafStore must not be null");
        checkNotNull(nodeStore, "nodeStore must not be null");

        this.maxSize = maxSize;
        this.rootIndex = rootIndex;
        random = new Random(seed);
        internalNodes = nodeStore;
        leafNodes = leafStore;
        nodeView = new CompactNodeView(this, rootIndex);
    }

    protected boolean isLeaf(short index) {
        return index - maxSize >= 0;
    }

    /**
     * Return a new {@link Cut}, which is chosen uniformly over the space of
     * possible cuts for the given bounding box.
     *
     * @param random A random number generator
     * @param box    A bounding box that we want to find a random cut for.
     * @return A new Cut corresponding to a random cut in the bounding box.
     */
    static Cut randomCut(Random random, IBox box) {
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
     * returns the boundingbox
     */
    BoundingBox getBoundingBox(short nodeOffset) {
        return getInternalBoundingBox(nodeOffset).convertBoxToDouble();
    }

    /**
     * Replace a Node in the tree structure. This method replaces oldNode with
     * newNode as a child of oldNode.getParent().
     *
     * @param oldNodeOffset The node we are replacing.
     * @param newNodeOffset The new node we are inserting into the tree.
     */

    void replaceNode(short oldNodeOffset, short newNodeOffset) {

        short parent;
        if (isLeaf(oldNodeOffset)) {
            parent = leafNodes.parentIndex[oldNodeOffset - maxSize];
        } else {
            parent = internalNodes.parentIndex[oldNodeOffset];
        }

        if (parent != NULL) {
            checkState(parent < maxSize, "incorrect");
            if (internalNodes.leftIndex[parent] == oldNodeOffset) {
                internalNodes.leftIndex[parent] = newNodeOffset;
            } else {
                internalNodes.rightIndex[parent] = newNodeOffset;
            }
        }

        if (isLeaf(newNodeOffset)) {
            leafNodes.parentIndex[newNodeOffset - maxSize] = parent;
        } else {
            internalNodes.parentIndex[newNodeOffset] = parent;
        }
    }

    /**
     * Return the sibling of a non-root node. Note that every non-leaf node in a
     * Random Cut Tree has two children.
     *
     * @param nodeOffset The node whose sibling we are requesting.
     * @return the sibling of node in the tree.
     */

    short getSibling(short nodeOffset) {
        checkArgument(nodeOffset >= 0, " cannotbe negative");
        int parent;
        if (isLeaf(nodeOffset)) {
            parent = leafNodes.parentIndex[nodeOffset - maxSize];
        } else {
            parent = internalNodes.parentIndex[nodeOffset];
        }

        if (parent != NULL) {
            checkArgument(parent < maxSize, "incorrect");
            if (internalNodes.leftIndex[parent] == nodeOffset) {
                return internalNodes.rightIndex[parent];
            } else if (internalNodes.rightIndex[parent] == nodeOffset) {
                return internalNodes.leftIndex[parent];
            } else {
                throw new IllegalArgumentException("node parent does not link back to node");
            }
        } else
            return NULL; // root is leaf
    }

    abstract protected boolean leftOf(double[] point, short nodeOffset);

    abstract protected boolean leftOf(P point, int cutDimension, double cutValue);

    // abstract boolean updateBoxAfterDelete(short nodeOffset, P point, boolean
    // resolvedDelete);

    abstract boolean checkEqual(P oldPoint, P point);

    abstract String printString(P point);

    abstract IBox<P> getLeafBox(int offset);

    abstract IBox<P> reflateNode(short nodeOffset);

    IBox<P> getInternalBoundingBox(short nodeOffset) {
        if (nodeOffset - maxSize >= 0) {
            return getLeafBox(nodeOffset - maxSize);
        } else {
            if (cachedBoxes[nodeOffset] == null) {
                cachedBoxes[nodeOffset] = reflateNode(nodeOffset);
            }
            return cachedBoxes[nodeOffset];
        }
    }

    abstract IBox<P> getInternalMergedBox(P point, P oldPoint);

    IBox<P> getBoundingBoxLeaveNull(short nodeOffset) {
        return (nodeOffset - maxSize >= 0) ? getLeafBox(nodeOffset - maxSize) : cachedBoxes[nodeOffset];
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

    private boolean deletePoint(short nodeOffset, P point, int level) {

        if (isLeaf(nodeOffset)) {
            short leafOffset = (short) (nodeOffset - maxSize);
            P oldPoint = pointStore.get(leafNodes.pointIndex[leafOffset]);
            if (!checkEqual(oldPoint, point)) {
                throw new IllegalStateException(
                        printString(point) + " " + pointStore.getString(leafNodes.pointIndex[leafOffset]) + " "
                                + leafOffset + " node " + leafNodes.pointIndex[leafOffset] + " " + false
                                + " Inconsistency in trees in delete step here.");
            }

            if (leafNodes.mass[leafOffset] > 1) {
                --leafNodes.mass[leafOffset];
                return true;
            }

            short parent = leafNodes.parentIndex[leafOffset];
            int saved = leafNodes.pointIndex[leafOffset];
            if (parent == NULL) {
                rootIndex = NULL;
                leafNodes.delete(leafOffset);
                return true;
            }

            int grandParent = internalNodes.parentIndex[parent];
            if (grandParent == NULL) {
                rootIndex = getSibling(nodeOffset);
                if (isLeaf(rootIndex)) {
                    leafNodes.parentIndex[rootIndex - maxSize] = NULL;
                } else {
                    internalNodes.parentIndex[rootIndex] = NULL;
                }
            } else {
                short sibling = getSibling(nodeOffset);
                replaceNode(parent, sibling);
            }
            leafNodes.delete(leafOffset);
            internalNodes.delete(parent);
            return false;
        }

        // node is not a leaf, and is an internal node
        boolean resolvedDelete = leftOf(point, internalNodes.cutDimension[nodeOffset],
                internalNodes.cutValue[nodeOffset]) ? deletePoint(internalNodes.leftIndex[nodeOffset], point, level + 1)
                        : deletePoint(internalNodes.rightIndex[nodeOffset], point, level + 1);

        if (!resolvedDelete && cachedBoxes[nodeOffset] != null) {
            IBox<P> leftBox = getBoundingBoxLeaveNull(internalNodes.leftIndex[nodeOffset]);
            IBox<P> rightBox = getBoundingBoxLeaveNull(internalNodes.rightIndex[nodeOffset]);
            if ((rightBox != null) && (leftBox != null)) {
                cachedBoxes[nodeOffset] = leftBox.getMergedBox(rightBox);
                if (cachedBoxes[nodeOffset].contains(point)) {
                    resolvedDelete = true;
                }
            } else {
                cachedBoxes[nodeOffset] = null;
            }
        }
        --internalNodes.mass[nodeOffset];
        return resolvedDelete;
    }

    /**
     * The following function adjusts the tree (if the issue has not been resolved
     * yet) based on the information in AddPointState which corresponds to the
     * particulars of the cut and the node that is being separated by the cut as a
     * sibling of the point
     *
     * @param pointIndex    the index of the point being inserted
     * @param point         the actual values
     * @param parentIndex   index of the ancestor node (can be NULL for root) above
     *                      which no changes need to be made
     * @param addPointState the in formation about the cut.
     */

    void resolve(int pointIndex, P point, short parentIndex, AddPointState addPointState) {
        if (!addPointState.getResolved()) {
            addPointState.setResolved();
            short siblingOffset = addPointState.getSiblingOffset();
            int cutDimension = addPointState.getCutDimension();
            double cutValue = addPointState.getCutValue();
            int oldmass = (isLeaf(siblingOffset)) ? leafNodes.mass[siblingOffset - maxSize]
                    : internalNodes.mass[siblingOffset];
            short leafOffset = (short) (leafNodes.add(NULL, pointIndex, 1) + maxSize);
            short mergedNode = leftOf(point, cutDimension, cutValue)
                    ? internalNodes.addNode(NULL, leafOffset, siblingOffset, cutDimension, cutValue, oldmass + 1)
                    : internalNodes.addNode(NULL, siblingOffset, leafOffset, cutDimension, cutValue, oldmass + 1);

            int parent;
            if (isLeaf(siblingOffset)) {
                parent = leafNodes.parentIndex[siblingOffset - maxSize];
            } else {
                parent = internalNodes.parentIndex[siblingOffset];
            }
            if (parent == NULL) {
                rootIndex = mergedNode;
            } else {
                replaceNode(siblingOffset, mergedNode);
            }
            leafNodes.parentIndex[leafOffset - maxSize] = mergedNode;
            if (isLeaf(siblingOffset)) {
                leafNodes.parentIndex[siblingOffset - maxSize] = mergedNode;
            } else {
                internalNodes.parentIndex[siblingOffset] = mergedNode;
            }

            cachedBoxes[mergedNode] = addPointState.getSavedBox();
            short tempNode = mergedNode;
            while (internalNodes.parentIndex[tempNode] != parentIndex) {
                tempNode = internalNodes.parentIndex[tempNode];
                cachedBoxes[tempNode].addPoint(point);
            }

        }
    }

    /**
     * The following method corresponds to visiting a set of nodes along the path
     * from a leaf to the root with increasing BoundingBoxes. For each new
     * BoundingBox seen. If there is a cut that separates the point and the node
     * then the method remembers the node, bounding box and the particulars of the
     * new cut. If there is no possibility of a cut (when the point belongs to the
     * box) the state is resolved by adjusting the tree.
     *
     * @param pointIndex    index of the point to be added
     * @param point         actual value of the point
     * @param nodeOffset    offset of the current node
     * @param addPointState store to remember the most recent cut
     */
    void checkContainsAndUpdateState(int pointIndex, P point, short nodeOffset, AddPointState addPointState) {
        if (!addPointState.getResolved()) {
            IBox<P> existingBox = getInternalBoundingBox(nodeOffset);
            if (existingBox.contains(point)) {
                resolve(pointIndex, point, nodeOffset, addPointState);
                return;
            }
            IBox<P> mergedBox = existingBox.getMergedBox(point);
            Cut cut = randomCut(random, mergedBox);
            short splitDimension = (short) cut.getDimension();
            double splitValue = cut.getValue();
            double minValue = existingBox.getMinValue(splitDimension);
            double maxValue = existingBox.getMaxValue(splitDimension);

            if (minValue > splitValue || maxValue <= splitValue) {
                addPointState.initialize(nodeOffset, splitDimension, splitValue, mergedBox);
            }
        }
    }

    /**
     * This function adds a point to the tree recursively starting from the leaf
     * node. The algorithm for adding a point is as follows:
     * <ol>
     * <li>At the current node we create a new bounding box by merging the point
     * with the existing box.</li>
     * <li>We pick a dimension and a value of cut.</li>
     * <li>If the cut falls outside the existing box we record the most recent
     * location.</li>
     * <li>We proceed to the parent.</li>
     * <li>If the point is within the current bounding box then make the change
     * corresponding to the most recent location and "resolve" the situation. If the
     * point is outside the bounding box of the tree then this step is taken at the
     * root node in addpoint(Sequential seg)</li>
     * </ol>
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

    private int addPoint(short nodeOffset, P point, int pointIndex, AddPointState addPointState) {

        if (isLeaf(nodeOffset)) {
            P oldPoint = pointStore.get(leafNodes.pointIndex[nodeOffset - maxSize]);
            if (checkEqual(oldPoint, point)) {
                // the inserted point is equal to an existing leaf point
                ++leafNodes.mass[nodeOffset - maxSize];
                addPointState.setResolved();
                return leafNodes.pointIndex[nodeOffset - maxSize];
            } else {
                IBox<P> mergedBox = getInternalMergedBox(point, oldPoint);
                Cut cut = randomCut(random, mergedBox);
                addPointState.initialize(nodeOffset, (short) cut.getDimension(), cut.getValue(), mergedBox);
                return pointIndex;
            }
        }

        int ret;
        if (leftOf(point, internalNodes.cutDimension[nodeOffset], internalNodes.cutValue[nodeOffset])) {
            ret = addPoint(internalNodes.leftIndex[nodeOffset], point, pointIndex, addPointState);
        } else {
            ret = addPoint(internalNodes.rightIndex[nodeOffset], point, pointIndex, addPointState);
        }

        checkContainsAndUpdateState(pointIndex, point, nodeOffset, addPointState);
        ++internalNodes.mass[nodeOffset];
        return ret;
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
            rootIndex = (short) (leafNodes.add(NULL, pointIndex, 1) + maxSize);
            return pointIndex;
        } else {
            int ret = addPoint(rootIndex, pointValue, pointIndex, addPointState);
            resolve(pointIndex, pointValue, NULL, addPointState);
            return ret;
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

    private <R> void traversePathToLeafAndVisitNodes(double[] point, Visitor<R> visitor, short currentNode,
            CompactNodeView nodeView, int depthOfNode) {
        if (isLeaf(currentNode)) {
            nodeView.setCurrentNodeIndex(currentNode);
            visitor.acceptLeaf(nodeView, depthOfNode);
        } else {
            short childNode = leftOf(point, currentNode) ? internalNodes.leftIndex[currentNode]
                    : internalNodes.rightIndex[currentNode];
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

    private <R> void traverseTreeMulti(double[] point, MultiVisitor<R> visitor, short currentNode,
            CompactNodeView nodeView, int depthOfNode) {
        if (isLeaf(currentNode)) {
            nodeView.setCurrentNodeIndex(currentNode);
            visitor.acceptLeaf(nodeView, depthOfNode);
        } else {
            nodeView.setCurrentNodeIndex(currentNode);
            if (visitor.trigger(nodeView)) {
                traverseTreeMulti(point, visitor, internalNodes.leftIndex[currentNode], nodeView, depthOfNode + 1);
                MultiVisitor<R> newVisitor = visitor.newCopy();
                nodeView.setCurrentNodeIndex(internalNodes.rightIndex[currentNode]);
                traverseTreeMulti(point, newVisitor, internalNodes.rightIndex[currentNode], nodeView, depthOfNode + 1);
                visitor.combine(newVisitor);
                nodeView.setCurrentNodeIndex(currentNode);
                visitor.accept(nodeView, depthOfNode);
            } else {
                short childNode = leftOf(point, currentNode) ? internalNodes.leftIndex[currentNode]
                        : internalNodes.rightIndex[currentNode];
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

    protected int getMass(short nodeOffset) {
        return (nodeOffset - maxSize >= 0) ? leafNodes.mass[nodeOffset - maxSize] : internalNodes.mass[nodeOffset];
    }

    abstract protected double[] getLeafPoint(short nodeOffset);

    public void reflateTree() {
        if ((rootIndex == NULL) || (rootIndex >= maxSize))
            return;
        reflateNode(rootIndex);
    }

    public NodeStoreState getNodes() {
        return new NodeStoreState(internalNodes);
    }

    public LeafStoreState getLeaves() {
        return new LeafStoreState(leafNodes);
    }

    public short getRootIndex() {
        return rootIndex;
    }

}
