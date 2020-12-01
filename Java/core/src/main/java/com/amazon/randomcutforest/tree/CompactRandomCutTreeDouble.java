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
import java.util.Random;
import java.util.function.Function;

import com.amazon.randomcutforest.CommonUtils;
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
public class CompactRandomCutTreeDouble implements ITree<Integer> {

    /**
     * The index value used to represent the absence of a node. For example, when
     * the tree is created the root node index will be NULL. After a point is added
     * and a root node is created, the root node's parent will be NULL, and so on.
     */
    public static final short NULL = -1;

    private final Random random;
    private final int maxSize;
    protected final NodeStore internalNodes;
    protected final LeafStore leafNodes;
    protected final IPointStore<double[]> pointStore;
    protected short rootIndex;
    private final CompactNodeViewDouble nodeView;
    private boolean resolvedDelete;
    private boolean addResolved;
    CandidateForSiblingInAddPoint candidateForSiblingInAddPoint = new CandidateForSiblingInAddPoint();

    public CompactRandomCutTreeDouble(int maxSize, long seed, IPointStore<double[]> pointStore) {
        checkArgument(maxSize > 0, "maxSize must be greater than 0");
        checkNotNull(pointStore, "pointStore must not be null");
        this.maxSize = maxSize;
        this.pointStore = pointStore;
        internalNodes = new NodeStore((short) (this.maxSize - 1));
        leafNodes = new LeafStore((short) this.maxSize);
        random = new Random(seed);
        rootIndex = NULL;
        nodeView = new CompactNodeViewDouble(this, NULL);
    }

    public CompactRandomCutTreeDouble(int maxSize, long seed, IPointStore<double[]> pointStore, LeafStore leafStore,
            NodeStore nodeStore, short rootIndex) {
        checkArgument(maxSize > 0, "maxSize must be greater than 0");
        checkNotNull(pointStore, "pointStore must not be null");
        checkNotNull(leafStore, "leafStore must not be null");
        checkNotNull(nodeStore, "nodeStore must not be null");

        this.maxSize = maxSize;
        this.pointStore = pointStore;
        this.rootIndex = rootIndex;
        random = new Random(seed);
        internalNodes = nodeStore;
        leafNodes = leafStore;
        nodeView = new CompactNodeViewDouble(this, rootIndex);
        // reflateTree();
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

    /** returns the boundingbox */
    BoundingBox getBoundingBox(short nodeOffset) {
        if (nodeOffset - maxSize >= 0) {
            return new BoundingBox(pointStore.get(leafNodes.pointIndex[nodeOffset - maxSize]));
        } else {
            if (internalNodes.boundingBox[nodeOffset] == null) {
                reflateNode(nodeOffset);
            }
            return internalNodes.boundingBox[nodeOffset];
        }
    }

    BoundingBox getBoundingBoxLeaveNull(short nodeOffset) {
        if (nodeOffset - maxSize >= 0) {
            return new BoundingBox(pointStore.get(leafNodes.pointIndex[nodeOffset - maxSize]));
        } else {
            return internalNodes.boundingBox[nodeOffset];
        }
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

    protected boolean leftOf(float[] point, short nodeOffset) {
        return leftOf(CommonUtils.toDoubleArray(point), nodeOffset);
    }

    protected boolean leftOf(double[] point, short nodeOffset) {
        return leftOf(point, internalNodes.cutDimension[nodeOffset], internalNodes.cutValue[nodeOffset]);
    }

    private boolean leftOf(double[] point, int dimension, double val) {
        return point[dimension] <= val;
    }

    /**
     * Checks equality of points -- this has to work in tandem across deletePoint,
     * addPoint and randomCut
     * 
     * @param point      first point in question
     * @param otherPoint second point
     * @return identical or otherwise
     */

    private boolean checkEquals(double[] point, double[] otherPoint) {
        for (int j = 0; j < point.length; j++) {
            if (point[j] != otherPoint[j]) {
                return false;
            }
        }
        return true;
    }

    /**
     * The function merges the two child boxes, provided none of the three
     * (including itself) was non-null before the delete.
     * 
     * @param nodeOffset the current node
     * @param point      the coordinates of the point being deleted
     */

    void updateBoxAfterDelete(short nodeOffset, double[] point) {
        if (resolvedDelete || internalNodes.boundingBox[nodeOffset] == null)
            return;
        BoundingBox leftBox = getBoundingBoxLeaveNull(internalNodes.leftIndex[nodeOffset]);
        BoundingBox rightBox = getBoundingBoxLeaveNull(internalNodes.rightIndex[nodeOffset]);
        if ((rightBox != null) && (leftBox != null)) {
            internalNodes.boundingBox[nodeOffset] = leftBox.getMergedBox(rightBox);
            if (internalNodes.boundingBox[nodeOffset].contains(point)) {
                resolvedDelete = true;
            }
        }

    }

    /**
     * Delete the given point from this tree.
     *
     * @param sequential An offset of the point in the tree that we wish to delete.
     */

    @Override
    public void deletePoint(Sequential<Integer> sequential) {
        checkState(rootIndex != NULL, "root must not be null");
        resolvedDelete = false;
        deletePoint(rootIndex, pointStore.get(sequential.getValue()), 0);
    }

    /**
     * This function deletes the point from the tree recursively. We traverse the
     * tree based on the cut stored in each interior node until we reach a leaf
     * node. We then delete the leaf node if the mass of the node is 1, otherwise we
     * reduce the mass by 1.
     *
     * @param nodeOffset node that we are visiting in the tree.
     * @param point      the point that is being deleted from the tree.
     * @param level      the level (i.e., the length of the path to the root) of the
     *                   node being evaluated.
     * @return the index of the deleted node.
     */

    private int deletePoint(short nodeOffset, double[] point, int level) {

        if (isLeaf(nodeOffset)) {
            short leafOffset = (short) (nodeOffset - maxSize);
            double[] oldPoint = pointStore.get(leafNodes.pointIndex[leafOffset]);
            if (!checkEquals(oldPoint, point)) {
                throw new IllegalStateException(
                        Arrays.toString(point) + " " + Arrays.toString(pointStore.get(leafNodes.pointIndex[leafOffset]))
                                + " " + leafOffset + " node " + leafNodes.pointIndex[leafOffset] + " " + false
                                + " Inconsistency in trees in delete step here.");
            }

            if (leafNodes.mass[leafOffset] > 1) {
                --leafNodes.mass[leafOffset];
                return leafNodes.pointIndex[leafOffset];
            }

            short parent = leafNodes.parentIndex[leafOffset];
            int saved = leafNodes.pointIndex[leafOffset];
            if (parent == NULL) {
                rootIndex = NULL;
                leafNodes.delete(leafOffset);
                return saved;
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
            return saved;
        }

        // node is not a leaf, and is an internal node
        int ret;
        if (leftOf(point, nodeOffset)) {
            ret = deletePoint(internalNodes.leftIndex[nodeOffset], point, level + 1);
        } else {
            ret = deletePoint(internalNodes.rightIndex[nodeOffset], point, level + 1);
        }

        updateBoxAfterDelete(nodeOffset, point);
        --internalNodes.mass[nodeOffset];
        return ret;
    }

    /**
     * The following class corresponds to the state stored during the addPoint
     * operation; the sibling (after the addition) of the point being added, the cut
     * dimension, the cut value and the bounding box. Saving the same bounding box
     * as is used to generate the random cut appears to be necessary from the
     * perspective of precision/loss there of.
     *
     * The assumption is that the process would begin at the leaf and bubble up.
     *
     * The checkContainsAndUpdateCut method provides a mechanism to update the state
     * as we visit parents recursively, in the current context, nodeOffset.
     *
     * The effectChange methiod effects the change, where it takes the additional
     * information of the parent where the addition of the point no longer updates
     * any bounding box.
     */

    class CandidateForSiblingInAddPoint {
        short siblingOffset;
        int cutDimension;
        double cutValue;
        BoundingBox savedBox;

        void initialize(short sibling, short dim, double val, BoundingBox box) {
            siblingOffset = sibling;
            cutDimension = dim;
            cutValue = val;
            savedBox = box;
        }

        void resolve(int pointIndex, double[] point, int parentIndex) {
            addResolved = true;
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
            internalNodes.boundingBox[mergedNode] = savedBox;
            short tempNode = mergedNode;
            while (internalNodes.parentIndex[tempNode] != parentIndex) {
                tempNode = internalNodes.parentIndex[tempNode];
                internalNodes.boundingBox[tempNode].addPoint(point);
            }

        }

        boolean checkContainsAndUpdateCut(double[] point, short nodeOffset) {
            BoundingBox existingBox = getBoundingBox(nodeOffset);
            if (existingBox.contains(point)) {
                return true;
            }
            BoundingBox mergedBox = existingBox.getMergedBox(point);
            Cut cut = randomCut(random, mergedBox);
            int splitDimension = cut.getDimension();
            double splitValue = cut.getValue();
            double minValue = existingBox.getMinValue(splitDimension);
            double maxValue = existingBox.getMaxValue(splitDimension);

            if (minValue > splitValue || maxValue <= splitValue) {
                cutValue = splitValue;
                cutDimension = splitDimension;
                siblingOffset = nodeOffset;
                savedBox = mergedBox;
            }
            return false;
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

    private int addPoint(short nodeOffset, double[] point, int pointIndex) {

        if (isLeaf(nodeOffset)) {
            double[] oldPoint = pointStore.get(leafNodes.pointIndex[nodeOffset - maxSize]);
            if (checkEquals(oldPoint, point)) {
                // the inserted point is equal to an existing leaf point
                ++leafNodes.mass[nodeOffset - maxSize];
                addResolved = true;
                return leafNodes.pointIndex[nodeOffset - maxSize];
            } else {
                BoundingBox mergedBox = BoundingBox.getMergedBox(point, oldPoint);
                Cut cut = randomCut(random, mergedBox);
                candidateForSiblingInAddPoint.initialize(nodeOffset, (short) cut.getDimension(), cut.getValue(),
                        mergedBox);
                return pointIndex;
            }
        }

        int ret;
        if (leftOf(point, nodeOffset)) {
            ret = addPoint(internalNodes.leftIndex[nodeOffset], point, pointIndex);
        } else {
            ret = addPoint(internalNodes.rightIndex[nodeOffset], point, pointIndex);
        }

        if (!addResolved) {
            addResolved = candidateForSiblingInAddPoint.checkContainsAndUpdateCut(point, nodeOffset);
            if (addResolved) {
                candidateForSiblingInAddPoint.resolve(pointIndex, point, nodeOffset);
            }
        }
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
        double[] pointValue = pointStore.get(pointIndex);
        if (rootIndex == NULL) {
            rootIndex = (short) (leafNodes.add(NULL, pointIndex, 1) + maxSize);
            return pointIndex;
        } else {
            addResolved = false;
            int ret = addPoint(rootIndex, pointValue, pointIndex);
            if (!addResolved) {
                candidateForSiblingInAddPoint.resolve(pointIndex, pointValue, NULL);
            }
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
            CompactNodeViewDouble nodeView, int depthOfNode) {
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
            CompactNodeViewDouble nodeView, int depthOfNode) {
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
        if (rootIndex == NULL)
            return 0;
        else
            return getMass(rootIndex);
    }

    protected int getMass(short nodeOffset) {
        if (nodeOffset - maxSize >= 0) {
            return leafNodes.mass[nodeOffset - maxSize];
        } else {
            return internalNodes.mass[nodeOffset];
        }
    }

    protected double[] getLeafPoint(short nodeOffset) {
        return pointStore.get(leafNodes.pointIndex[nodeOffset - maxSize]);
    }

    /*
     * private Node newNode(Node leftChild, Node rightChild, Cut cut, BoundingBox
     * box) { Node node = new Node(leftChild, rightChild, cut, box, false);
     * 
     * return node; }
     */
    public void reflateTree() {
        if ((rootIndex == NULL) || (rootIndex >= maxSize))
            return;
        reflateNode(rootIndex);
    }

    /**
     * creates the bounding box of a node/leaf
     * 
     * @param offset node in question
     * @return the bounding box
     */
    private BoundingBox reflateNode(short offset) {
        if (offset - maxSize >= 0) {
            return new BoundingBox(getLeafPoint(offset));
        }
        internalNodes.boundingBox[offset] = reflateNode(internalNodes.leftIndex[offset])
                .getMergedBox(reflateNode(internalNodes.rightIndex[offset]));
        return internalNodes.boundingBox[offset];
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
