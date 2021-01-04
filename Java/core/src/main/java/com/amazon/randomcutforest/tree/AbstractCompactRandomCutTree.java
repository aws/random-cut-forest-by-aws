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

import java.util.HashSet;

import com.amazon.randomcutforest.Visitor;
import com.amazon.randomcutforest.store.IPointStoreView;

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
    protected final CompactNodeManager nodeManager;
    protected IPointStoreView<Point> pointStore;
    protected AbstractBoundingBox<Point>[] cachedBoxes;
    protected Point[] pointSum;
    protected boolean enableCache;
    protected HashSet[] sequenceIndexes;

    public AbstractCompactRandomCutTree(int maxSize, long seed, boolean enableCache, boolean enableCenterOfMass,
            boolean enableSequenceIndices) {
        super(seed, enableCache, enableCenterOfMass, enableSequenceIndices);
        checkArgument(maxSize > 0, "maxSize must be greater than 0");
        this.maxSize = maxSize;
        nodeManager = new CompactNodeManager(maxSize);
        rootIndex = null;
        this.enableCache = enableCache;
        if (enableSequenceIndices) {
            sequenceIndexes = new HashSet[maxSize];
        }
        // adjusting the below parameter in [0,1] may change the space time tradeoff
        // but should not affect the computation in any manner
        // setBoundingBoxCacheFraction(0.3);
    }

    public AbstractCompactRandomCutTree(int maxSize, long seed, CompactNodeManager nodeManager, int rootIndex,
            boolean enableCache) {
        super(seed, enableCache, false, false);
        checkArgument(maxSize > 0, "maxSize must be greater than 0");
        checkNotNull(nodeManager, "nodeManager must not be null");

        this.maxSize = maxSize;
        this.rootIndex = rootIndex;
        this.nodeManager = nodeManager;
        this.enableCache = enableCache;
    }

    @Override
    protected INode<Integer> getNode(Integer node) {
        return new CompactNodeView(this, node);
    }

    @Override
    protected void addSequenceIndex(Integer nodeRef, long sequenceIndex) {
        if (sequenceIndexes[nodeRef] == null) {
            sequenceIndexes[nodeRef] = new HashSet<Long>();
        }
        sequenceIndexes[nodeRef].add(sequenceIndex);
    }

    @Override
    protected void deleteSequenceIndex(Integer nodeRef, long sequenceIndex) {
        if (sequenceIndexes[nodeRef] == null || !sequenceIndexes[nodeRef].contains(sequenceIndex)) {
            throw new IllegalStateException("Error in sequence index. Inconsistency in trees in delete step.");
        }
        sequenceIndexes[nodeRef].remove(sequenceIndex);
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
            if (cachedBoxes != null && boundingBoxCacheFraction >= 1.0) {
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
        } else {
            AbstractBoundingBox<Point> newBox = constructBoxInPlace(
                    constructBoxInPlace(currentBox, getLeftChild(nodeReference)), getRightChild(nodeReference));
            // the following is useful in cases where the forest is deserialized and
            // even though caches may be set, the individual boxes are null
            if (cachedBoxes != null && boundingBoxCacheFraction >= 1.0) {
                cachedBoxes[nodeReference] = newBox;
            }
            return newBox;
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
        if (cacheRandom.nextDouble() < boundingBoxCacheFraction) {
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

    public CompactNodeManager getNodeManager() {
        return nodeManager;
    }

    public int getRootIndex() {
        return rootIndex;
    }

    @Override
    abstract protected boolean leftOf(Point point, int cutDimension, double cutValue);

    @Override
    abstract protected boolean checkEqual(Point oldPoint, Point point);

    @Override
    abstract protected String toString(Point point);

    // returns the point based on position in the store
    @Override
    Point getPointFromPointReference(Integer pointIndex) {
        checkArgument(pointIndex != null, "incorrect request");
        return pointStore.get(pointIndex);
    }

    // returns the position in the point store
    @Override
    Integer getPointReference(Integer node) {
        checkArgument(node != null, "incorrect request");
        return nodeManager.getPointIndex(node);
    }

    @Override
    protected boolean isLeaf(Integer node) {
        return nodeManager.isLeaf(node);
    }

    @Override
    protected int decrementMass(Integer node) {
        return nodeManager.decrementMass(node);
    }

    @Override
    protected int incrementMass(Integer node) {
        return nodeManager.incrementMass(node);
    }

    @Override
    protected Integer getSibling(Integer node) {
        return nodeManager.getSibling(node);
    }

    @Override
    protected Integer getParent(Integer node) {
        return nodeManager.getParent(node);
    }

    @Override
    protected void setParent(Integer node, Integer parent) {
        nodeManager.setParent(node, parent);
    }

    @Override
    protected void delete(Integer node) {
        nodeManager.delete(node);
    }

    @Override
    protected int getCutDimension(Integer node) {
        return nodeManager.getCutDimension(node);
    }

    @Override
    protected double getCutValue(Integer node) {
        return nodeManager.getCutValue(node);
    }

    @Override
    protected Integer getLeftChild(Integer node) {
        return nodeManager.getLeftChild(node);
    }

    @Override
    protected Integer getRightChild(Integer node) {
        return nodeManager.getRightChild(node);
    }

    @Override
    protected void replaceNode(Integer parent, Integer child, Integer otherNode) {
        nodeManager.replaceNode(parent, child, otherNode);
    }

    @Override
    protected void replaceNodeBySibling(Integer grandParent, Integer parent, Integer node) {
        nodeManager.replaceParentBySiblingOfNode(grandParent, parent, node);
    }

    @Override
    protected Integer addLeaf(Integer pointIndex) {
        return nodeManager.addLeaf(null, pointIndex, 1);
    }

    @Override
    protected Integer addNode(Integer leftChild, Integer rightChild, int cutDimension, double cutValue, int mass) {
        return nodeManager.addNode(null, leftChild, rightChild, cutDimension, cutValue, mass);
    }

    @Override
    protected void increaseMassOfAncestors(Integer mergedNode) {
        nodeManager.increaseMassOfAncestors(mergedNode);
    }

    @Override
    protected int getMass(Integer node) {
        return nodeManager.getMass(node);
    }
}
