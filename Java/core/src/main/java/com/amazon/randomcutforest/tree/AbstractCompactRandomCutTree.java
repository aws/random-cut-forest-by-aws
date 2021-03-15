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

import java.util.HashMap;

import com.amazon.randomcutforest.Visitor;
import com.amazon.randomcutforest.store.ILeafStore;
import com.amazon.randomcutforest.store.INodeStore;
import com.amazon.randomcutforest.store.IPointStoreView;
import com.amazon.randomcutforest.store.LeafStore;
import com.amazon.randomcutforest.store.NodeStore;
import com.amazon.randomcutforest.store.SmallLeafStore;
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
        implements ITree<Integer> {

    /**
     * The index value used to represent the absence of a node. For example, when
     * the tree is created the root node index will be NULL. After a point is added
     * and a root node is created, the root node's parent will be NULL, and so on.
     */
    public static final int NULL = -1;

    protected int maxSize;
    protected final ILeafStore leafStore;
    protected final INodeStore nodeStore;
    protected IPointStoreView<Point> pointStore;
    protected AbstractBoundingBox<Point>[] cachedBoxes;
    protected Point[] pointSum;
    protected boolean enableCache;
    protected HashMap<Long, Integer>[] sequenceIndexes;

    public AbstractCompactRandomCutTree(int maxSize, long seed, boolean enableCache, boolean enableCenterOfMass,
            boolean enableSequenceIndices) {
        super(seed, enableCache, enableCenterOfMass, enableSequenceIndices);
        checkArgument(maxSize > 0, "maxSize must be greater than 0");
        this.maxSize = maxSize;
        if (maxSize < SmallNodeStore.MAX_TREE_SIZE) { // max_short/2 because the tree is almost twice the size of leaves
            leafStore = new SmallLeafStore((short) maxSize);
            nodeStore = new SmallNodeStore((short) (maxSize - 1));
        } else {
            leafStore = new LeafStore(maxSize);
            nodeStore = new NodeStore(maxSize - 1);
        }
        root = null;
        this.enableCache = enableCache;
        if (enableSequenceIndices) {
            sequenceIndexes = new HashMap[maxSize];
        }
        // adjusting the below parameter in [0,1] may change the space time tradeoff
        // but should not affect the computation in any manner
        // setBoundingBoxCacheFraction(0.3);
    }

    public AbstractCompactRandomCutTree(int maxSize, long seed, ILeafStore leafStore, INodeStore nodeStore, int root,
            boolean enableCache) {
        super(seed, enableCache, false, false);
        checkArgument(maxSize > 0, "maxSize must be greater than 0");
        checkNotNull(leafStore, "leafStore must not be null");
        checkNotNull(nodeStore, "nodeStore must not be null");

        this.maxSize = maxSize;
        this.leafStore = leafStore;
        this.nodeStore = nodeStore;
        this.root = root == NULL ? null : root;
        this.enableCache = enableCache;
    }

    public ILeafStore getLeafStore() {
        return leafStore;
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
        int leafRef = getLeafIndexForTreeIndex(nodeRef);
        if (sequenceIndexes[leafRef] == null) {
            sequenceIndexes[leafRef] = new HashMap<Long, Integer>();
        }
        int num = 0;
        if (sequenceIndexes[leafRef].containsKey(sequenceIndex)) {
            num = sequenceIndexes[leafRef].get(sequenceIndex);
        }
        sequenceIndexes[leafRef].put(sequenceIndex, num + 1);
    }

    @Override
    protected void deleteSequenceIndex(Integer nodeRef, long sequenceIndex) {
        int leafRef = getLeafIndexForTreeIndex(nodeRef);
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

    @Override
    protected AbstractBoundingBox<Point> constructBoxInPlace(Integer nodeReference) {
        if (isLeaf(nodeReference)) {
            return getMutableLeafBoxFromLeafNode(nodeReference);
        } else if (cachedBoxes != null && cachedBoxes[nodeReference] != null) {
            return cachedBoxes[nodeReference].copy();
        } else {
            AbstractBoundingBox<Point> currentBox = constructBoxInPlace(
                    constructBoxInPlace(getLeftChild(nodeReference)), getRightChild(nodeReference));
            // the following is useful in cases where the forest is deserialized and
            // even though caches may be set, the individual boxes are null
            if (cachedBoxes != null && cacheRandom.nextDouble() <= boundingBoxCacheFraction) {
                cachedBoxes[nodeReference] = currentBox.copy();
            }
            return currentBox;
        }
    }

    AbstractBoundingBox<Point> constructBoxInPlace(AbstractBoundingBox<Point> currentBox, Integer nodeReference) {
        if (isLeaf(nodeReference)) {
            return currentBox.addPoint(getPointFromLeafNode(nodeReference));
        } else if (cachedBoxes != null && cachedBoxes[nodeReference] != null) {
            // if a box is present for a node in use, that box must be correct
            // this invariant is maintained throughout.
            return currentBox.addBox(cachedBoxes[nodeReference]);
        } else if (cachedBoxes != null && boundingBoxCacheFraction > 0) {
            // there is a possibility of saving the box for the current node
            AbstractBoundingBox<Point> newBox = constructBoxInPlace(nodeReference);
            if (cacheRandom.nextDouble() <= boundingBoxCacheFraction) {
                cachedBoxes[nodeReference] = newBox;
            }
            return currentBox.addBox(newBox);
        } else {
            return constructBoxInPlace(constructBoxInPlace(currentBox, getLeftChild(nodeReference)),
                    getRightChild(nodeReference));
        }
    }

    @Override
    AbstractBoundingBox<Point> recomputeBox(Integer node) {
        if (cachedBoxes[node] != null) {
            // cannot invoke constructBoxInPlace(node) because that would re-use the old
            // box
            cachedBoxes[node] = constructBoxInPlace(constructBoxInPlace(getLeftChild(node)), getRightChild(node));
            return cachedBoxes[node];
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
        if (cacheRandom.nextDouble() <= boundingBoxCacheFraction) {
            cachedBoxes[mergedNode] = savedBox;
        } else {
            cachedBoxes[mergedNode] = null;
        }
    }

    void addToBox(Integer tempNode, Point point) {
        if (cachedBoxes[tempNode] != null) {
            cachedBoxes[tempNode].addPoint(point); // internal boxes can be updated in place
        }
    }

    // returns the point based on position in the store
    @Override
    Point getPointFromPointReference(Integer pointIndex) {
        checkArgument(pointIndex != null, "incorrect request");
        return pointStore.get(pointIndex);
    }

    // returns the position in the point store
    @Override
    Integer getPointReference(Integer node) {
        return getPointReference(intValue(node));
    }

    int getPointReference(int index) {
        checkArgument(isLeaf(index), "index is not a valid leaf node index");
        return leafStore.getPointIndex(getLeafIndexForTreeIndex(index));
    }

    @Override
    protected boolean isLeaf(Integer node) {
        return isLeaf(intValue(node));
    }

    protected boolean isLeaf(int index) {
        return index >= maxSize - 1;
    }

    @Override
    protected int decrementMass(Integer node) {
        return decrementMass(intValue(node));
    }

    protected int decrementMass(int node) {
        return isLeaf(node) ? leafStore.decrementMass(getLeafIndexForTreeIndex(node)) : nodeStore.decrementMass(node);
    }

    @Override
    protected int incrementMass(Integer node) {
        return incrementMass(intValue(node));
    }

    protected int incrementMass(int node) {
        return isLeaf(node) ? leafStore.incrementMass(getLeafIndexForTreeIndex(node)) : nodeStore.incrementMass(node);
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
        if (isLeaf(node)) {
            return leafStore.getParent(getLeafIndexForTreeIndex(node));
        } else {
            return nodeStore.getParent(node);
        }
    }

    @Override
    protected void setParent(Integer child, Integer parent) {
        setParent(intValue(child), intValue(parent));
    }

    protected void setParent(int child, int parent) {
        if (isLeaf(child)) {
            leafStore.setParent(getLeafIndexForTreeIndex(child), parent);
        } else {
            nodeStore.setParent(child, parent);
        }
    }

    @Override
    protected void delete(Integer node) {
        delete(intValue(node));
    }

    protected void delete(int node) {
        if (isLeaf(node)) {
            leafStore.delete(getLeafIndexForTreeIndex(node));
        } else {
            nodeStore.delete(node);
        }
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
        return getTreeIndexForLeafIndex(leafStore.addLeaf(NULL, pointIndex, 1));
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
        nodeStore.increaseMassOfAncestorsAndItself(getParent(mergedNode.intValue()));
    }

    @Override
    protected int getMass(Integer node) {
        return getMass(intValue(node));
    }

    protected int getMass(int node) {
        return isLeaf(node) ? leafStore.getMass(getLeafIndexForTreeIndex(node)) : nodeStore.getMass(node);
    }

    protected int getTreeIndexForLeafIndex(int leafIndex) {
        return leafIndex + maxSize - 1;
    }

    protected int getLeafIndexForTreeIndex(int treeIndex) {
        checkArgument(isLeaf(treeIndex), "treeIndex is not a valid leaf node index");
        return treeIndex - maxSize + 1;
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

    public Point getPointSum(int ref) {
        if (isLeaf(ref)) {
            return pointStore.getScaledPoint(getPointReference(ref), getMass(ref));
        }
        if (pointSum[ref] == null) {
            recomputePointSum(ref);
        }
        return pointSum[ref];
    }
}
