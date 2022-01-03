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

import static com.amazon.randomcutforest.CommonUtils.toDoubleArray;

import java.util.Collections;
import java.util.Set;

import com.amazon.randomcutforest.store.IPointStoreView;

public class NodeView implements INodeView {
    AbstractNodeStore nodeStore;
    int currentNodeOffset;
    double[] leafPoint;
    IPointStoreView<float[]> pointStoreView;
    BoundingBoxFloat currentBox;

    public NodeView(AbstractNodeStore nodeStore, IPointStoreView<float[]> pointStoreView, int root) {
        this.currentNodeOffset = root;
        this.pointStoreView = pointStoreView;
        this.nodeStore = nodeStore;
    }

    public int getMass() {
        return nodeStore.getMass(currentNodeOffset);
    }

    public IBoundingBoxView getBoundingBox() {
        if (currentBox == null) {
            return nodeStore.getBox(currentNodeOffset);
        }
        return currentBox;
    }

    public IBoundingBoxView getSiblingBoundingBox(double[] point) {
        return (toLeft(point)) ? nodeStore.getRightBox(currentNodeOffset) : nodeStore.getLeftBox(currentNodeOffset);
    }

    public int getCutDimension() {
        return nodeStore.getCutDimension(currentNodeOffset);
    }

    @Override
    public double getCutValue() {
        return nodeStore.getCutValue(currentNodeOffset);
    }

    public double[] getLeafPoint() {
        return leafPoint;
    }

    public Set<Long> getSequenceIndexes() {
        return Collections.emptySet();
    }

    public boolean isLeaf() {
        return nodeStore.isLeaf(currentNodeOffset);
    }

    protected void setCurrentNode(int newNode, int index, boolean setBox) {
        currentNodeOffset = newNode;
        float[] point = pointStoreView.get(index);
        leafPoint = toDoubleArray(point);
    }

    protected void setCurrentNodeOnly(int newNode) {
        currentNodeOffset = newNode;
    }

    public void updateToParent(int parent, int currentSibling, boolean updateBox) {
        currentNodeOffset = parent;
    }

    // this function exists for matching the behavior of RCF2.0 and will be replaced
    // this function explicitly uses the encoding of the new nodestore
    protected boolean toLeft(double[] point) {
        return point[nodeStore.getCutDimension(currentNodeOffset)] <= nodeStore.getCutValue(currentNodeOffset);
    }
}
