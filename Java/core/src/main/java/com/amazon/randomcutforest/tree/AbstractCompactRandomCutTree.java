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
        if (sequenceIndexes[nodeRef] != null) {
            sequenceIndexes[nodeRef].remove(sequenceIndex);
        }
    }

    AbstractBoundingBox<Point> getBoundingBoxReflate(Integer nodeReference) {
        if (isLeaf(nodeReference)) {
            return getLeafBoxFromLeafNode(nodeReference);
        }
        if (cachedBoxes[nodeReference] == null) {
            cachedBoxes[nodeReference] = getBoundingBoxReflate(nodeManager.getLeftChild(nodeReference))
                    .getMergedBox(getBoundingBoxReflate(nodeManager.getRightChild(nodeReference)));
        }
        return cachedBoxes[nodeReference];
    }

    void setCachedBox(Integer mergedNode, AbstractBoundingBox<Point> savedBox) {
        cachedBoxes[mergedNode] = savedBox;
    }

    void addToBox(Integer tempNode, Point point) {
        cachedBoxes[tempNode].addPoint(point); // internal boxes should be updateable in place
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
    protected Integer addLeaf(Integer parent, Integer pointIndex, int mass) {
        return nodeManager.addLeaf(parent, pointIndex, mass);
    }

    @Override
    protected Integer addNode(Integer parent, Integer leftChild, Integer rightChild, int cutDimension, double cutValue,
            int mass) {
        return nodeManager.addNode(parent, leftChild, rightChild, cutDimension, cutValue, mass);
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
