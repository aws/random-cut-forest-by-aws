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

import com.amazon.randomcutforest.store.ILeafStore;
import com.amazon.randomcutforest.store.INodeStore;
import com.amazon.randomcutforest.store.IPointStore;

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

    public CompactRandomCutTreeFloat(int maxSize, long seed, IPointStore<float[]> pointStore, ILeafStore leafStore,
            INodeStore nodeStore, int root, boolean cacheEnabled) {
        super(maxSize, seed, leafStore, nodeStore, root, cacheEnabled);
        // checkNotNull(pointStore, "pointStore must not be null");
        super.pointStore = pointStore;
        if (cacheEnabled) {
            cachedBoxes = new BoundingBoxFloat[maxSize - 1];
        }
    }

    @Override
    public void usePointStore(IPointStore<?> pointStore) {
        super.pointStore = (IPointStore<float[]>) pointStore;
    }

    @Override
    protected String toString(float[] point) {
        return Arrays.toString(point);
    }

    @Override
    protected AbstractBoundingBox<float[]> getLeafBoxFromLeafNode(Integer pointIndex) {
        return new BoundingBoxFloat(pointStore.get(getPointReference(pointIndex)));
    }

    @Override
    protected AbstractBoundingBox<float[]> getMutableLeafBoxFromLeafNode(Integer nodeReference) {
        // pointStore makes an explicit copy
        float[] leafPoint = pointStore.get(getPointReference(nodeReference));
        return new BoundingBoxFloat(leafPoint, Arrays.copyOf(leafPoint, leafPoint.length), 0);
    }

    @Override
    protected AbstractBoundingBox<float[]> getInternalTwoPointBox(float[] first, float[] second) {
        return new BoundingBoxFloat(first, second);
    }

    @Override
    protected boolean equals(float[] oldPoint, float[] point) {
        return Arrays.equals(oldPoint, point);
    }

    @Override
    protected boolean leftOf(float[] point, int dimension, double val) {
        return point[dimension] <= val;
    }

    @Override
    protected double[] getPoint(Integer nodeOffset) {
        return toDoubleArray(getPointFromLeafNode(nodeOffset));
    }

    @Override
    void recomputePointSum(Integer node) {
        if (pointSum[node] == null) {
            pointSum[node] = new float[pointStore.getDimensions()];
        }
        float[] leftSum = getPointSum(getLeftChild(node));
        float[] rightSum = getPointSum(getRightChild(node));
        for (int i = 0; i < pointStore.getDimensions(); i++) {
            pointSum[node][i] = leftSum[i] + rightSum[i];
        }
    }

}
