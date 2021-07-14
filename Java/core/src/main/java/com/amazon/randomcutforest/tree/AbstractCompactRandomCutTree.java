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

import java.util.Arrays;
import java.util.HashMap;
import java.util.concurrent.ArrayBlockingQueue;

import com.amazon.randomcutforest.RandomCutForest;
import com.amazon.randomcutforest.Visitor;
import com.amazon.randomcutforest.config.Precision;
import com.amazon.randomcutforest.store.INodeStore;
import com.amazon.randomcutforest.store.IPointStoreView;
import com.amazon.randomcutforest.store.NodeStore;
import com.amazon.randomcutforest.store.SmallNodeStore;

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
public abstract class AbstractCompactRandomCutTree<Point> extends AbstractRandomCutTree<Point, Integer, Integer>
        implements ITree<Integer, Point> {

    /**
     * The index value used to represent the absence of a node. For example, when
     * the tree is created the root node index will be NULL. After a point is added
     * and a root node is created, the root node's parent will be NULL, and so on.
     */
    public static final int NULL = -1;

    /**
     * number of maximum leaves in the tree
     */
    protected int maxSize;

    protected INodeStore nodeStore;
    protected IPointStoreView<Point> pointStore;
    protected IBoxCache<Point> boxCache;
    protected Point[] pointSum;
    protected HashMap<Long, Integer>[] sequenceIndexes;
    protected boolean float32Precision;

    public AbstractCompactRandomCutTree(
            com.amazon.randomcutforest.tree.AbstractCompactRandomCutTree.Builder<?> builder) {
        super(builder);
        checkArgument(builder.maxSize > 0, "maxSize must be greater than 0");
        this.float32Precision = builder.float32Precision;
        this.maxSize = builder.maxSize;
        if (builder.outputAfter.isPresent()) {
            outputAfter = builder.outputAfter.get();
        } else {
            outputAfter = maxSize / 4;
        }

        if (builder.root == NULL) {
            if (builder.float32Precision && maxSize < SmallNodeStore.MAX_SMALLNODESTORE_CAPACITY
                    && builder.dimension < Short.MAX_VALUE) {
                this.nodeStore = new SmallNodeStore(maxSize - 1);
            } else {
                this.nodeStore = new NodeStore(maxSize - 1);
            }
            this.root = null;
        } else {
            checkNotNull(builder.nodeStore, "nodeStore must not be null");
            this.nodeStore = builder.nodeStore;
            this.root = builder.root;
        }

        if (storeSequenceIndexesEnabled) {
            sequenceIndexes = new HashMap[maxSize];
        }
    }

    public INodeStore getNodeStore() {
        return nodeStore;
    }

    @Override
    protected INode<Integer> getNode(Integer node) {
        return new CompactNodeView(this, node);
    }

    @Override
    protected void addSequenceIndex(Integer nodeRef, long sequenceIndex) {
        int leafRef = nodeStore.computeLeafIndex(nodeRef);
        if (sequenceIndexes[leafRef] == null) {
            sequenceIndexes[leafRef] = new HashMap<>();
        }
        int num = 0;
        if (sequenceIndexes[leafRef].containsKey(sequenceIndex)) {
            num = sequenceIndexes[leafRef].get(sequenceIndex);
        }
        sequenceIndexes[leafRef].put(sequenceIndex, num + 1);
    }

    public double getBoundingBoxCacheFraction() {
        return boundingBoxCacheFraction;
    }

    /**
     * The following function reorders the nodes stored in the tree in a breadth
     * first order; Note that a regular binary tree where each internal node has 2
     * chidren, as is the case for AbstractRandomCutTree or any tree produced in a
     * Random Forest ensemble (not restricted to Random Cut Forests), has maxsize -
     * 1 internal nodes for maxSize number of leaves. The leaves are numbered 0 +
     * (maxsize), 1 + (maxSize), ..., etc. in that BFS ordering. The root is node 0.
     *
     * Note that if the binary tree is a complete binary tree, then the numbering
     * would correspond to the well known heuristic where children of node index i
     * are numbered 2*i and 2*i + 1. The trees in AbstractCompactRandomCutTree will
     * not be complete binary trees. But a similar numbering enables us to compress
     * the entire structure of the tree into two bit arrays corresponding to
     * presence of left and right children. The idea can be viewed as similar to
     * Zak's numbering for regular binary trees Lexicographic generation of binary
     * trees, S. Zaks, TCS volume 10, pages 63-82, 1980, that uses depth first
     * numbering. However an extensive literature exists on this topic.
     *
     * The overall relies on the extra advantage that we can use two bit sequences;
     * the left and right child pointers which appears to be simple. While it is
     * feasible to always maintain this order, that would complicate the standard
     * binary search tree pattern and this tranformation is used when the tree is
     * serialized. Note that while there is savings in representing the tree
     * structure into two bit arrays, the bulk of the serialization corresponds to
     * the payload at the nodes (cuts, dimensions for internal nodes and index to
     * pointstore, number of copies for the leaves). The translation to the bits is
     * handled by the NodeStoreMapper. The algorithm here corresponds to just
     * producing the cannoical order.
     *
     * The algorithm first renumbers the nodes in BFS ordering. Then the cached
     * bounding boxes (for the internal nodes) and the sequence Index maps (for the
     * leaves) are swapped based on the renumbering map. The swapping of the
     * bounding boxes are abstract and left to the concrete implemenations that know
     * the precision setting. The algorithm validates its work as it proceeds.
     *
     * Note that if the root is a singleton leaf then it will get mapped to 0 +
     * maxSize in the current node store implementation.
     */
    public void reorderNodesInBreadthFirstOrder() {
        NodeStore result = new NodeStore(maxSize - 1);
        if (root != null) {
            if (!isLeaf(root)) {
                int nodeCounter = 0;
                int currentNode = 0;
                int leafcounter = maxSize - 1;
                int[] map = new int[2 * maxSize - 1];
                Arrays.fill(map, NULL);
                ArrayBlockingQueue<Integer> nodeQueue = new ArrayBlockingQueue<>(maxSize - 1);
                nodeQueue.add(root);
                while (!nodeQueue.isEmpty()) {
                    int head = nodeQueue.poll();
                    int newLeft;
                    int leftChild = getLeftChild(head);
                    if (isLeaf(leftChild)) {
                        // the parent is the current node and the indices, mass are being copied over
                        map[leftChild] = result.addLeaf(currentNode, getPointReference(leftChild), getMass(leftChild));
                        assert map[leftChild] == leafcounter : "incorrect state";
                        newLeft = leafcounter++;
                    } else { // leftchild is an internal node
                        newLeft = ++nodeCounter;
                        nodeQueue.add(leftChild);
                    }
                    int newRight;
                    int rightChild = getRightChild(head);
                    if (isLeaf(rightChild)) {
                        // the parent is the current node and the indices, mass are being copied over
                        map[rightChild] = result.addLeaf(currentNode, getPointReference(rightChild),
                                getMass(rightChild));
                        assert map[rightChild] == leafcounter : "incorrect state";
                        newRight = leafcounter++;
                    } else {
                        newRight = ++nodeCounter;
                        nodeQueue.add(rightChild);
                    }
                    // the parent has to be added by now
                    int parent = (head == root) ? -1 : map[getParent(head)];
                    map[head] = result.addNode(parent, newLeft, newRight, getCutDimension(head), getCutValue(head),
                            getMass(head));
                    assert map[head] == currentNode : "incorrect state";
                    currentNode++;
                }
                assert currentNode == nodeStore.size() : "incorrect state";

                boxCache.swapCaches(map);

                if (storeSequenceIndexesEnabled) {
                    HashMap<Long, Integer>[] newSequence = new HashMap[maxSize];
                    for (int i = 0; i < maxSize; i++) { // iterate over leaves
                        if (map[i + maxSize - 1] != NULL) { // leaf is in use
                            assert isLeaf(map[i + maxSize - 1]) : "error in map";
                            newSequence[map[i + maxSize - 1] - maxSize + 1] = sequenceIndexes[i];
                        }
                    }
                    sequenceIndexes = newSequence;
                }
                root = 0;
            } else {
                root = result.addLeaf(NULL, getPointReference(root), getMass(root));
            }
        }
        nodeStore = result;
        if (centerOfMassEnabled) {
            if (!isLeaf(root)) {
                recomputePointSum(root);
            }
        }
    }

    /**
     * deletes a sequence index from a leaf map; if multiple sequence indices are
     * present (for external shingling, timestamp etc. reasons) then the count is
     * decremented.
     *
     * @param nodeRef       the leaf node reference
     * @param sequenceIndex the sequence index to be deleted
     */
    @Override
    protected void deleteSequenceIndex(Integer nodeRef, long sequenceIndex) {
        int leafRef = nodeStore.computeLeafIndex(nodeRef);
        if (sequenceIndexes[leafRef] == null || !sequenceIndexes[leafRef].containsKey(sequenceIndex)) {
            throw new IllegalStateException("Error in sequence index. Inconsistency in trees in delete step.");
        }
        int num = sequenceIndexes[leafRef].get(sequenceIndex);
        if (num == 1) {
            sequenceIndexes[leafRef].remove(sequenceIndex);
        } else {
            sequenceIndexes[leafRef].put(sequenceIndex, num - 1);
        }
    }

    /**
     * The following algorithm constructs a bounding box for a node. If the node is
     * a leaf then it regenerates a mutable bounding box (which can be
     * merged/modified). If the node is an internal node and the box was already
     * cached then it uses a copy (so that the original is not modified by the in
     * place merge operations). Otherwise it generates a box for the left child
     * (this function) and merges the points from the right child (a different
     * function). Finally, if caching is set then the box is saved (at random).
     *
     * @param nodeReference the reference of the node
     * @return the box corresponding to all the descendants of the node
     */
    @Override
    protected AbstractBoundingBox<Point> constructBoxInPlace(Integer nodeReference) {
        if (isLeaf(nodeReference)) {
            return getMutableLeafBoxFromLeafNode(nodeReference);
        } else if (isBoundingBoxCacheEnabled()) {
            return constructBoxInPlace(constructBoxInPlace(getLeftChild(nodeReference)), getRightChild(nodeReference));
        } else {
            AbstractBoundingBox<Point> oldBox = boxCache.getBox(nodeReference);
            if (oldBox != null) { // note cachemanager can have a null box, after deserialization
                return oldBox.copy();
            }
            AbstractBoundingBox<Point> currentBox = constructBoxInPlace(
                    constructBoxInPlace(getLeftChild(nodeReference)), getRightChild(nodeReference));
            boxCache.setBox(nodeReference, currentBox.copy());
            return currentBox;
        }
    }

    /**
     * The following function merges the descendents of a node into a provided box.
     * If a box had been cached then the box is reused and merged into the provided
     * box. Otherwise we recurse on the children. Note that the computation is
     * predicated on the provided box and cannot be used to update the cached boxes.
     *
     * @param currentBox    a mutable bounding box
     * @param nodeReference the node
     * @return the bounding box corresponding to the union of the bounding box at
     *         the node and the current box
     */
    AbstractBoundingBox<Point> constructBoxInPlace(AbstractBoundingBox<Point> currentBox, Integer nodeReference) {
        if (isLeaf(nodeReference)) {
            return currentBox.addPoint(getPointFromLeafNode(nodeReference));
        } else if (isBoundingBoxCacheEnabled()) {
            AbstractBoundingBox<Point> oldBox = boxCache.getBox(nodeReference);
            if (oldBox != null) {
                return currentBox.addBox(oldBox);
            }
            AbstractBoundingBox<Point> newBox = constructBoxInPlace(nodeReference);
            boxCache.setBox(nodeReference, newBox);
            return currentBox.addBox(newBox);
        } else {
            return constructBoxInPlace(constructBoxInPlace(currentBox, getLeftChild(nodeReference)),
                    getRightChild(nodeReference));
        }
    }

    /**
     * during deletes, it may be necessary to recompute a box. Note that cached
     * boxes cannot be used for this node -- because some descendant nodes may have
     * been deleted! If the corresponding box is not cached then this operation
     * returns null.
     *
     * @param node reference of the node
     * @return the new bounding box, which is cached.
     */
    @Override
    AbstractBoundingBox<Point> recomputeBox(Integer node) {
        if (boxCache.getBox(node) != null) {
            // cannot invoke constructBoxInPlace(node) because that would re-use the old
            // box
            AbstractBoundingBox<Point> newBox = constructBoxInPlace(constructBoxInPlace(getLeftChild(node)),
                    getRightChild(node));
            boxCache.setBox(node, newBox);
            return newBox;
        }
        return null;
    }

    /**
     * saving bounding boxes for a newly created internal node. The decision is
     * controlled by the allowed fraction of boxes to be saved. Setting the fraction
     * to 1.0 saves all boxes. If the box is not saved then it must be made null to
     * overwrite any prior information
     *
     * @param mergedNode internal node
     * @param savedBox   the newly created box for this node.
     */
    void setCachedBox(Integer mergedNode, AbstractBoundingBox<Point> savedBox) {
        boxCache.setBox(mergedNode, savedBox);
    }

    /**
     * adds a point to an existing box (if present) during insertion
     *
     * @param node  the node
     * @param point the point to be added
     */

    void addToBox(Integer node, Point point) {
        boxCache.addToBox(node, point);
    }

    // returns the point based on position in the store
    @Override
    Point getPointFromPointReference(Integer pointIndex) {
        checkArgument(pointIndex != null, "incorrect request");
        return (pointIndex == NULL) ? null : pointStore.get(pointIndex);
    }

    // returns the position in the point store
    @Override
    Integer getPointReference(Integer node) {
        int val = getPointReference(intValue(node));
        return (val == NULL) ? null : val;
    }

    int getPointReference(int index) {
        checkArgument(isLeaf(index), "index is not a valid leaf node index");
        return nodeStore.getPointIndex(index);
    }

    @Override
    void setLeafPointReference(Integer leafNode, Integer pointIndex) {
        nodeStore.setPointIndex(leafNode, pointIndex);
    }

    @Override
    protected boolean isLeaf(Integer node) {
        return nodeStore.isLeaf(intValue(node));
    }

    @Override
    protected int decrementMass(Integer node) {
        return nodeStore.decrementMass(intValue(node));
    }

    @Override
    protected int incrementMass(Integer node) {
        return nodeStore.incrementMass(intValue(node));
    }

    @Override
    protected Integer getSibling(Integer node) {
        return getSibling(intValue(node));
    }

    protected int getSibling(int node) {
        int parent = getParent(node);
        checkArgument(parent != NULL, "node does not have a parent or a sibling");
        return nodeStore.getSibling(parent, node);
    }

    @Override
    protected Integer getParent(Integer node) {
        int parent = getParent(intValue(node));
        return parent != NULL ? parent : null;
    }

    protected int getParent(int node) {
        return nodeStore.getParentIndex(node);
    }

    @Override
    protected void setParent(Integer child, Integer parent) {
        nodeStore.setParentIndex(intValue(child), intValue(parent));
    }

    /**
     * deletes a node from the stores and resets the connectivity information so
     * that state is preserved
     * 
     * @param node the node to be deleted
     */
    @Override
    protected void delete(Integer node) {
        nodeStore.delete(intValue(node));
    }

    @Override
    protected int getCutDimension(Integer node) {
        return nodeStore.getCutDimension(node);
    }

    @Override
    protected double getCutValue(Integer node) {
        return nodeStore.getCutValue(node);
    }

    @Override
    protected Integer getLeftChild(Integer node) {
        return nodeStore.getLeftIndex(node);
    }

    @Override
    protected Integer getRightChild(Integer node) {
        return nodeStore.getRightIndex(node);
    }

    @Override
    void replaceChild(Integer parent, Integer child, Integer otherNode) {
        replaceChild(intValue(parent), intValue(child), intValue(otherNode));
    }

    protected void replaceChild(int parent, int child, int otherNode) {
        nodeStore.replaceChild(parent, child, otherNode);
        setParent(otherNode, parent);
    }

    @Override
    protected void replaceNodeBySibling(Integer grandParent, Integer parent, Integer node) {
        replaceNodeBySibling(intValue(grandParent), intValue(parent), intValue(node));

    }

    protected void replaceNodeBySibling(int grandParent, int parent, int node) {
        int sibling = nodeStore.getSibling(parent, node);
        nodeStore.replaceChild(grandParent, parent, sibling);
        setParent(sibling, grandParent);
    }

    @Override
    protected Integer addLeaf(Integer pointIndex) {
        return nodeStore.addLeaf(NULL, pointIndex, 1);
    }

    @Override
    protected Integer addNode(Integer leftChild, Integer rightChild, int cutDimension, double cutValue, int mass) {
        return addNode(intValue(leftChild), intValue(rightChild), cutDimension, cutValue, mass);
    }

    protected int addNode(int leftChild, int rightChild, int cutDimension, double cutValue, int mass) {
        return nodeStore.addNode(NULL, leftChild, rightChild, cutDimension, cutValue, mass);
    }

    @Override
    protected void increaseMassOfAncestors(Integer mergedNode) {
        nodeStore.increaseMassOfSelfAndAncestors(getParent(mergedNode.intValue()));
    }

    @Override
    protected void decreaseMassOfAncestors(Integer mergedNode) {
        nodeStore.decreaseMassOfSelfAndAncestors(getParent(mergedNode.intValue()));
    }

    @Override
    protected int getMass(Integer node) {
        return nodeStore.getMass(intValue(node));
    }

    protected static int intValue(Integer i) {
        return i == null ? NULL : i;
    }

    public int getMaxSize() {
        return maxSize;
    }

    public Point getPointSum() {
        return (root == NULL) ? null : getPointSum(root);
    }

    public int getRootIndex() {
        return intValue(root);
    }

    @Override
    protected boolean referenceEquals(Integer oldPointRef, Integer pointRef) {
        return oldPointRef.equals(pointRef);
    }

    public Point getPointSum(int ref) {
        if (isLeaf(ref)) {
            return pointStore.getScaledPoint(getPointReference(ref), getMass(ref));
        }
        if (pointSum[ref] == null) {
            recomputePointSum(ref);
        }
        return pointSum[ref];
    }

    public boolean isBoundingBoxCacheEnabled() {
        return boundingBoxCacheFraction > 0.0;
    }

    // TODO: fix ownership of default variables
    public static class Builder<T extends Builder<T>> extends AbstractRandomCutTree.Builder<T> {

        private INodeStore nodeStore = null;
        private int root = NULL;
        private int dimensions = 0;
        private int maxSize = RandomCutForest.DEFAULT_SAMPLE_SIZE;
        private boolean float32Precision = (RandomCutForest.DEFAULT_PRECISION == Precision.FLOAT_32);

        public T nodeStore(INodeStore nodeStore) {
            this.nodeStore = nodeStore;
            return (T) this;
        }

        public T root(int root) {
            this.root = root;
            return (T) this;
        }

        public T maxSize(int maxSize) {
            this.maxSize = maxSize;
            return (T) this;
        }

        public T float32Precision(boolean precision) {
            this.float32Precision = precision;
            return (T) this;
        }

        public T setDimensions(int dimensions) {
            this.dimension = dimensions;
            return (T) this;
        }

    }

}
