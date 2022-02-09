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

import java.util.HashMap;

import com.amazon.randomcutforest.store.IPointStoreView;

public class NodeView implements INodeView {
    AbstractNodeStore nodeStore;
    int currentNodeOffset;
    float[] leafPoint;
    IPointStoreView<float[]> pointStoreView;
    BoundingBox currentBox;

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

    public IBoundingBoxView getSiblingBoundingBox(float[] point) {
        return (toLeft(point)) ? nodeStore.getRightBox(currentNodeOffset) : nodeStore.getLeftBox(currentNodeOffset);
    }

    public int getCutDimension() {
        return nodeStore.getCutDimension(currentNodeOffset);
    }

    @Override
    public double getCutValue() {
        return nodeStore.getCutValue(currentNodeOffset);
    }

    public float[] getLeafPoint() {
        return leafPoint;
    }

    public HashMap<Long, Integer> getSequenceIndexes() {
        checkArgument(nodeStore.isLeaf(currentNodeOffset), "can only be invoked for a leaf");
        if (nodeStore.storeSequenceIndexesEnabled) {
            return nodeStore.sequenceMap.get(nodeStore.getPointIndex(currentNodeOffset));
        } else {
            return new HashMap<>();
        }
    }

    @Override
    public double probailityOfSeparation(float[] point) {
        return nodeStore.probabilityOfCut(currentNodeOffset, point, pointStoreView, currentBox);
    }

    @Override
    public int getLeafPointIndex() {
        checkArgument(isLeaf(), "incorrect call");
        return nodeStore.getPointIndex(currentNodeOffset);
    }

    public boolean isLeaf() {
        return nodeStore.isLeaf(currentNodeOffset);
    }

    protected void setCurrentNode(int newNode, int index, boolean setBox) {
        currentNodeOffset = newNode;
        leafPoint = pointStoreView.get(index);
        if (setBox && nodeStore.boundingboxCacheFraction < AbstractNodeStore.SWITCH_FRACTION) {
            currentBox = new BoundingBox(leafPoint, leafPoint);
        }
    }

    protected void setCurrentNodeOnly(int newNode) {
        currentNodeOffset = newNode;
    }

    public void updateToParent(int parent, int currentSibling, boolean updateBox) {
        currentNodeOffset = parent;
        if (updateBox && nodeStore.boundingboxCacheFraction < AbstractNodeStore.SWITCH_FRACTION) {
            nodeStore.growNodeBox(currentBox, pointStoreView, parent, currentSibling);
        }
    }

    // this function exists for matching the behavior of RCF2.0 and will be replaced
    // this function explicitly uses the encoding of the new nodestore
    protected boolean toLeft(float[] point) {
        return point[nodeStore.getCutDimension(currentNodeOffset)] <= nodeStore.getCutValue(currentNodeOffset);
    }
}
