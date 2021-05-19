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

import java.util.Arrays;
import java.util.Set;

import com.amazon.randomcutforest.store.INodeStore;

public class CompactNodeView<Point> implements INode<Integer> {
    final AbstractCompactRandomCutTree<Point> tree;
    int currentNodeOffset;
    IBoxCache<Point> boxCache;
    INodeStore nodeStore;

    public CompactNodeView(AbstractCompactRandomCutTree<Point> tree, int initialNodeIndex) {
        this.tree = tree;
        this.currentNodeOffset = initialNodeIndex;
        boxCache = tree.boxCache;
        nodeStore = tree.nodeStore;
    }

    public int getMass() {
        return nodeStore.getMass(currentNodeOffset);
    }

    public IBoundingBoxView getBoundingBox() {
        return getBox(currentNodeOffset);
    }

    IBoundingBoxView getBox(int node) {
        if (!nodeStore.isLeaf(node)) {
            IBoundingBoxView box = boxCache.getBox(node);
            if (box != null)
                return box;
        }
        return tree.getBoundingBox(node);
    }

    public IBoundingBoxView getSiblingBoundingBox(double[] point) {
        if (tree.leftOf(point, currentNodeOffset)) {
            return getBox(nodeStore.getRightIndex(currentNodeOffset));
        } else {
            return getBox(nodeStore.getLeftIndex(currentNodeOffset));
        }
    }

    public boolean leafPointEquals(double[] point) {
        return Arrays.equals(getLeafPoint(), point);
    }

    public int getCutDimension() {
        return nodeStore.getCutDimension(currentNodeOffset);
    }

    @Override
    public double getCutValue() {
        return nodeStore.getCutValue(currentNodeOffset);
    }

    public double[] getLeafPoint() {
        return tree.getPoint(currentNodeOffset);
    }

    @Override
    public double[] getLiftedLeafPoint() {
        return tree.liftFromTree(tree.getPoint(currentNodeOffset));
    }

    public Set<Long> getSequenceIndexes() {
        checkArgument(nodeStore.isLeaf(currentNodeOffset), " not a leaf node");
        return tree.sequenceIndexes[nodeStore.computeLeafIndex(currentNodeOffset)].keySet();
    }

    @Override
    public boolean isLeaf() {
        return nodeStore.isLeaf(currentNodeOffset);
    }

    @Override
    public INode<Integer> getLeftChild() {
        return new CompactNodeView<>(tree, nodeStore.getLeftIndex(currentNodeOffset));
    }

    @Override
    public INode<Integer> getRightChild() {
        return new CompactNodeView<>(tree, nodeStore.getRightIndex(currentNodeOffset));
    }

    @Override
    public INode<Integer> getParent() {
        return new CompactNodeView<>(tree, nodeStore.getParentIndex(currentNodeOffset));
    }

}
