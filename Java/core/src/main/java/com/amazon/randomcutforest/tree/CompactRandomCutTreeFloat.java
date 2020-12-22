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

import static com.amazon.randomcutforest.CommonUtils.checkNotNull;
import static com.amazon.randomcutforest.CommonUtils.toDoubleArray;

import java.util.Arrays;

import com.amazon.randomcutforest.store.IPointStore;
import com.amazon.randomcutforest.store.LeafStore;
import com.amazon.randomcutforest.store.NodeStore;

public class CompactRandomCutTreeFloat extends AbstractCompactRandomCutTree<float[]> {

    public CompactRandomCutTreeFloat(int maxSize, long seed, IPointStore<float[]> pointStore, boolean cacheEnabled,
            boolean centerOfMassEnabled, boolean enableSequenceIndices) {
        super(maxSize, seed, cacheEnabled, centerOfMassEnabled, enableSequenceIndices);
        checkNotNull(pointStore, "pointStore must not be null");
        super.pointStore = pointStore;
        if (cacheEnabled) {
            cachedBoxes = new BoundingBoxFloat[maxSize - 1];
        }
        if (centerOfMassEnabled) {
            pointSum = new float[maxSize - 1][];
        }
    }

    public CompactRandomCutTreeFloat(int maxSize, long seed, IPointStore<float[]> pointStore, LeafStore leafStore,
            NodeStore nodeStore, int rootIndex, boolean cacheEnabled) {
        super(maxSize, seed, leafStore, nodeStore, rootIndex, cacheEnabled);
        checkNotNull(pointStore, "pointStore must not be null");
        super.pointStore = pointStore;
        cachedBoxes = new BoundingBoxFloat[maxSize - 1];
    }

    @Override
    protected String toString(float[] point) {
        return Arrays.toString(point);
    }

    @Override
    AbstractBoundingBox<float[]> getLeafBoxFromLeafNode(Integer pointIndex) {
        return new BoundingBoxFloat(pointStore.get(nodeManager.getPointIndex(pointIndex)));
    }

    @Override
    AbstractBoundingBox<float[]> getInternalTwoPointBox(float[] first, float[] second) {
        return new BoundingBoxFloat(first, second);
    }

    @Override
    protected boolean checkEqual(float[] oldPoint, float[] point) {
        return Arrays.equals(oldPoint, point);
    }

    @Override
    protected boolean leftOf(float[] point, int dimension, double val) {
        return point[dimension] <= val;
    }

    @Override
    protected float[] getPointFromLeafNode(Integer nodeOffset) {
        return pointStore.get(nodeManager.getPointIndex(nodeOffset));
    }

    @Override
    protected double[] getPoint(Integer nodeOffset) {
        return toDoubleArray(getPointFromLeafNode(nodeOffset));
    }

    AbstractBoundingBox<float[]> getBoundingBoxReflate(Integer nodeReference) {
        if (isLeaf(nodeReference)) {
            return new BoundingBoxFloat(getPointFromLeafNode(nodeReference));
        }
        if (cachedBoxes[nodeReference] == null) {
            cachedBoxes[nodeReference] = getBoundingBoxReflate(nodeManager.getLeftChild(nodeReference))
                    .getMergedBox(getBoundingBoxReflate(nodeManager.getRightChild(nodeReference)));
        }
        return cachedBoxes[nodeReference];
    }

    @Override
    void updateDeletePointSum(int nodeRef, float[] point) {
        if (pointSum[nodeRef] == null) {
            pointSum[nodeRef] = new float[point.length];
        }
        for (int i = 0; i < point.length; i++) {
            pointSum[nodeRef][i] += point[i];
        }
    }

    float[] getPointSum(int ref) {
        return nodeManager.isLeaf(ref) ? getPointFromLeafNode(ref) : pointSum[ref];
    }

    @Override
    void updateAddPointSum(Integer mergedNode, float[] point) {
        if (pointSum[mergedNode] == null) {
            pointSum[mergedNode] = new float[point.length];
        }
        float[] leftSum = getPointSum(nodeManager.getLeftChild(mergedNode));
        float[] rightSum = getPointSum(nodeManager.getRightChild(mergedNode));
        for (int i = 0; i < point.length; i++) {
            pointSum[mergedNode][i] = leftSum[i] + rightSum[i];
        }
        int tempNode = mergedNode;
        while (nodeManager.getParent(tempNode) != NULL) {
            tempNode = nodeManager.getParent(tempNode);
            for (int i = 0; i < point.length; i++) {
                pointSum[tempNode][i] += point[i];
            }
        }
    }
}
