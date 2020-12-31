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

    public CompactRandomCutTreeFloat(int maxSize, long seed, IPointStore<float[]> pointStore,
            CompactNodeManager nodeManager, int rootIndex, boolean cacheEnabled) {
        super(maxSize, seed, nodeManager, rootIndex, cacheEnabled);
        checkNotNull(pointStore, "pointStore must not be null");
        super.pointStore = pointStore;
        if (cacheEnabled) {
            cachedBoxes = new BoundingBoxFloat[maxSize - 1];
        }
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
    AbstractBoundingBox<float[]> getMutableLeafBoxFromLeafNode(Integer nodeReference) {
        float[] leafpoint = pointStore.get(nodeManager.getPointIndex(nodeReference));
        return new BoundingBoxFloat(Arrays.copyOf(leafpoint, leafpoint.length),
                Arrays.copyOf(leafpoint, leafpoint.length), 0);
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
    protected double[] getPoint(Integer nodeOffset) {
        return toDoubleArray(getPointFromLeafNode(nodeOffset));
    }

    float[] getPointSum(int ref, float[] point) {
        if (nodeManager.isLeaf(ref)) {
            if (getMass(ref) == 1) {
                return getPointFromLeafNode(ref);
            } else {
                float[] answer = Arrays.copyOf(getPointFromLeafNode(ref), point.length);
                for (int i = 0; i < point.length; i++) {
                    answer[i] *= getMass(ref);
                }
                return answer;
            }
        }
        if (pointSum[ref] == null) {
            reComputePointSum(ref, point);
        }
        return pointSum[ref];
    }

    @Override
    void reComputePointSum(Integer node, float[] point) {
        if (pointSum[node] == null) {
            pointSum[node] = new float[point.length];
        }
        float[] leftSum = getPointSum(nodeManager.getLeftChild(node), point);
        float[] rightSum = getPointSum(nodeManager.getRightChild(node), point);
        for (int i = 0; i < point.length; i++) {
            pointSum[node][i] = leftSum[i] + rightSum[i];
        }
    }

}
