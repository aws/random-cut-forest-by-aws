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

import java.util.Arrays;

import com.amazon.randomcutforest.store.IPointStore;

public class CompactRandomCutTreeDouble extends AbstractCompactRandomCutTree<double[]> {

    public CompactRandomCutTreeDouble(int maxSize, long seed, IPointStore<double[]> pointStore, boolean cacheEnabled,
            boolean centerOfMassEnabled, boolean enableSequenceIndices) {
        super(maxSize, seed, cacheEnabled, centerOfMassEnabled, enableSequenceIndices);
        checkNotNull(pointStore, "pointStore must not be null");
        super.pointStore = pointStore;
        if (cacheEnabled) {
            cachedBoxes = new BoundingBox[maxSize - 1];
        }
        if (centerOfMassEnabled) {
            pointSum = new double[maxSize - 1][];
        }
    }

    public CompactRandomCutTreeDouble(int maxSize, long seed, IPointStore<double[]> pointStore,
            CompactNodeManager nodeManager, int rootIndex, boolean cacheEnabled) {
        super(maxSize, seed, nodeManager, rootIndex, cacheEnabled);
        checkNotNull(pointStore, "pointStore must not be null");
        super.pointStore = pointStore;
        if (cacheEnabled) {
            cachedBoxes = new BoundingBox[maxSize - 1];
        }
    }

    @Override
    protected String toString(double[] point) {
        return Arrays.toString(point);
    }

    @Override
    AbstractBoundingBox<double[]> getInternalTwoPointBox(double[] first, double[] second) {
        return new BoundingBox(first, second);
    }

    @Override
    protected boolean checkEqual(double[] oldPoint, double[] point) {
        return Arrays.equals(oldPoint, point);
    }

    @Override
    AbstractBoundingBox<double[]> getLeafBoxFromLeafNode(Integer nodeReference) {
        return new BoundingBox(pointStore.get(nodeManager.getPointIndex(nodeReference)));
    }

    double[] getPointSum(int ref) {
        return nodeManager.isLeaf(ref) ? getPointFromLeafNode(ref) : pointSum[ref];
    }

    @Override
    protected boolean leftOf(double[] point, int dimension, double val) {
        return point[dimension] <= val;
    }

    // the following is for visitors
    @Override
    protected double[] getPoint(Integer node) {
        double[] internal = pointStore.get(nodeManager.getPointIndex(node));
        return Arrays.copyOf(internal, internal.length);
    }

    double[] getPointSum(int ref, double[] point) {
        if (nodeManager.isLeaf(ref)) {
            if (getMass(ref) == 1) {
                return getPointFromLeafNode(ref);
            } else {
                double[] answer = Arrays.copyOf(getPointFromLeafNode(ref), point.length);
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
    void reComputePointSum(Integer node, double[] point) {
        if (pointSum[node] == null) {
            pointSum[node] = new double[point.length];
        }
        double[] leftSum = getPointSum(nodeManager.getLeftChild(node), point);
        double[] rightSum = getPointSum(nodeManager.getRightChild(node), point);
        for (int i = 0; i < point.length; i++) {
            pointSum[node][i] = leftSum[i] + rightSum[i];
        }
    }
}
